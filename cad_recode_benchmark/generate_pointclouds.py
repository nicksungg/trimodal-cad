#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

try:
    from pytorch3d.ops import sample_farthest_points
except Exception:
    sample_farthest_points = None


def fps_numpy(points: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    selected = np.zeros(n_points, dtype=np.int64)
    distances = np.full(points.shape[0], np.inf, dtype=np.float64)
    farthest = int(rng.integers(0, points.shape[0]))
    for i in range(n_points):
        selected[i] = farthest
        current_point = points[farthest]
        dist = np.sum((points - current_point) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))
    return selected


def mesh_to_point_cloud(
    mesh: trimesh.Trimesh,
    n_points: int,
    n_pre_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    if sample_farthest_points is not None:
        _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
        ids = ids[0].numpy()
    else:
        ids = fps_numpy(vertices, n_points, rng)
    return np.asarray(vertices[ids], dtype=np.float32)


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    max_extent = float(max(mesh.extents))
    if max_extent > 0:
        mesh.apply_scale(2.0 / max_extent)
    return mesh


def process_one(
    stl_path: Path,
    input_root: Path,
    output_root: Path,
    n_points: int,
    n_pre_points: int,
    seed: int,
) -> tuple[Path, str | None]:
    try:
        mesh = trimesh.load_mesh(stl_path)
        mesh = normalize_mesh(mesh)
        rng = np.random.default_rng(seed)
        point_cloud = mesh_to_point_cloud(
            mesh=mesh,
            n_points=n_points,
            n_pre_points=n_pre_points,
            rng=rng,
        )
        rel = stl_path.relative_to(input_root)
        out_path = output_root / rel.with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, point_cloud)
        return out_path, None
    except Exception as exc:
        return Path(), f"{stl_path}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate normalized 256-point clouds from STL files for CAD-Recode."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("cadbench_stls/cadbench_stl_normalized"),
        help="Directory containing input STL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cadbench_stls/cadbench_pointclouds_256"),
        help="Directory where output .npy point clouds are stored.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=256,
        help="Number of points in output point cloud.",
    )
    parser.add_argument(
        "--n-pre-points",
        type=int,
        default=8192,
        help="Number of points sampled from mesh surface before FPS.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducible processing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of STL files to process (0 = all).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    stl_files = sorted(input_dir.rglob("*.stl"))
    if args.limit > 0:
        stl_files = stl_files[: args.limit]

    if not stl_files:
        print(f"No STL files found under: {input_dir}")
        return

    print(f"Found {len(stl_files)} STL files")
    print(f"Writing point clouds to: {output_dir}")
    print(
        "Sampling backend: "
        + ("pytorch3d.ops.sample_farthest_points" if sample_farthest_points else "NumPy FPS fallback")
    )

    ok_count = 0
    errors: list[str] = []

    for idx, stl_path in enumerate(stl_files):
        out_path, error = process_one(
            stl_path=stl_path,
            input_root=input_dir,
            output_root=output_dir,
            n_points=args.n_points,
            n_pre_points=args.n_pre_points,
            seed=args.seed + idx,
        )
        if error is None:
            ok_count += 1
            if ok_count <= 3:
                print(f"OK: {stl_path} -> {out_path}")
        else:
            errors.append(error)

    print(f"Completed: {ok_count}/{len(stl_files)} point clouds generated")
    if errors:
        print(f"Errors: {len(errors)}")
        error_log = output_dir / "pointcloud_generation_errors.txt"
        error_log.parent.mkdir(parents=True, exist_ok=True)
        error_log.write_text("\n".join(errors), encoding="utf-8")
        print(f"Error log: {error_log}")


if __name__ == "__main__":
    main()
