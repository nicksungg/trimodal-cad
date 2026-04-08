import os
import glob
import math
import argparse
import traceback
import sys
import shutil
import inspect
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import trimesh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import cadquery as cq


VIEW_DIRECTIONS = [
    (1, -1, 1), (-1, -1, 1), (1, 1, 1), (-1, 1, 1),
    (1, -1, -1), (-1, -1, -1), (1, 1, -1), (-1, 1, -1),
]


def apply_cadquery_compat_shims() -> None:
    """Bridge common API drift between CadQuery versions used by dataset scripts."""
    text_fn = cq.Workplane.text
    if getattr(text_fn, "_cadevolve_cut_compat", False):
        return

    try:
        params = inspect.signature(text_fn).parameters
    except Exception:
        return

    if "cut" in params or "combine" not in params:
        return

    def text_with_cut_compat(self, *args, cut=None, **kwargs):
        if cut is not None and "combine" not in kwargs:
            kwargs["combine"] = "cut" if cut else True
        return text_fn(self, *args, **kwargs)

    text_with_cut_compat._cadevolve_cut_compat = True
    cq.Workplane.text = text_with_cut_compat


def read_python_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def cadquery_obj_to_shape(obj) -> cq.Shape:
    if isinstance(obj, cq.Shape):
        return obj
    if isinstance(obj, cq.Workplane):
        return obj.val()
    if hasattr(obj, "toCompound"):
        compound = obj.toCompound()
        if isinstance(compound, cq.Shape):
            return compound
    if hasattr(obj, "val"):
        value = obj.val()
        if isinstance(value, cq.Shape):
            return value
    raise TypeError(f"Unsupported CAD object type: {type(obj)}")


def load_shape_from_py(py_path: str) -> cq.Shape:
    code = read_python_file(py_path)
    apply_cadquery_compat_shims()
    ns = {"cq": cq, "cadquery": cq, "__name__": "__main__"}
    exec(code, ns, ns)

    for key in ("result", "shape", "model", "part", "solid"):
        if key in ns:
            return cadquery_obj_to_shape(ns[key])

    raise RuntimeError(
        f"No CAD output variable found in {py_path}. Expected one of: "
        "result/shape/model/part/solid"
    )


def shape_to_mesh(shape: cq.Shape, tol: float = 0.2) -> trimesh.Trimesh:
    verts_raw, faces_raw = shape.tessellate(tol)
    verts = np.array([[v.x, v.y, v.z] for v in verts_raw], dtype=np.float64)
    faces = np.array(faces_raw, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh.faces.shape[0] == 0 or mesh.vertices.shape[0] == 0:
        raise RuntimeError("Empty mesh after tessellation")
    return mesh


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    v = vertices - center
    scale = np.max(np.linalg.norm(v, axis=1))
    if scale > 0:
        v = v / scale
    return v


def render_8_views(mesh: trimesh.Trimesh, out_dir: str, size: int = 448):
    os.makedirs(out_dir, exist_ok=True)
    v = normalize_vertices(mesh.vertices)
    tris = v[mesh.faces]

    dpi = 100
    fig_w = size / dpi
    fig_h = size / dpi

    for i, (x, y, z) in enumerate(VIEW_DIRECTIONS):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        poly = Poly3DCollection(
            tris,
            facecolor=(0.72, 0.72, 0.75, 1.0),
            edgecolor=(0.25, 0.25, 0.25, 0.15),
            linewidths=0.1,
        )
        ax.add_collection3d(poly)

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        ax.set_box_aspect((1, 1, 1))

        r = math.sqrt(x * x + y * y + z * z)
        elev = math.degrees(math.asin(z / r))
        azim = math.degrees(math.atan2(y, x))
        ax.view_init(elev=elev, azim=azim)

        out_path = os.path.join(out_dir, f"view_{i}.png")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def sample_point_cloud(mesh: trimesh.Trimesh, num_points: int = 10000) -> np.ndarray:
    try:
        pts, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    except Exception:
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.asarray(pts, dtype=np.float32)


def build_output_paths(py_file: str, input_root: str, out_pc_root: str, out_img_root: str):
    rel = os.path.relpath(py_file, input_root)
    stem = os.path.splitext(rel)[0]
    root_tag = Path(input_root).name

    pc_path = os.path.join(out_pc_root, root_tag, stem + ".npy")
    img_dir = os.path.join(out_img_root, root_tag, stem)
    return pc_path, img_dir


def has_complete_outputs(pc_path: str, img_dir: str) -> bool:
    if not os.path.exists(pc_path):
        return False
    for i in range(8):
        if not os.path.exists(os.path.join(img_dir, f"view_{i}.png")):
            return False
    return True


def process_one(
    py_file: str,
    input_root: str,
    out_pc_root: str,
    out_img_root: str,
    num_points: int,
    mesh_tol: float,
    img_size: int,
):
    try:
        pc_path, img_dir = build_output_paths(py_file, input_root, out_pc_root, out_img_root)
        os.makedirs(os.path.dirname(pc_path), exist_ok=True)

        tmp_root = os.path.join(out_pc_root, ".tmp")
        os.makedirs(tmp_root, exist_ok=True)
        safe_stem = Path(pc_path).stem.replace(os.sep, "_")
        tmp_pc_path = os.path.join(tmp_root, f"{safe_stem}_{os.getpid()}.tmp.npy")
        tmp_img_dir = os.path.join(tmp_root, f"{safe_stem}_{os.getpid()}_imgs")

        if os.path.exists(tmp_pc_path):
            os.remove(tmp_pc_path)
        if os.path.exists(tmp_img_dir):
            shutil.rmtree(tmp_img_dir, ignore_errors=True)
        os.makedirs(tmp_img_dir, exist_ok=True)

        shape = load_shape_from_py(py_file)
        mesh = shape_to_mesh(shape, tol=mesh_tol)

        pts = sample_point_cloud(mesh, num_points=num_points)
        np.save(tmp_pc_path, pts)

        render_8_views(mesh, tmp_img_dir, size=img_size)

        for i in range(8):
            if not os.path.exists(os.path.join(tmp_img_dir, f"view_{i}.png")):
                raise RuntimeError("Incomplete image set during render")

        os.replace(tmp_pc_path, pc_path)

        if os.path.exists(img_dir):
            shutil.rmtree(img_dir, ignore_errors=True)
        shutil.move(tmp_img_dir, img_dir)

        return True, py_file, ""
    except Exception:
        try:
            if 'tmp_pc_path' in locals() and os.path.exists(tmp_pc_path):
                os.remove(tmp_pc_path)
            if 'tmp_img_dir' in locals() and os.path.exists(tmp_img_dir):
                shutil.rmtree(tmp_img_dir, ignore_errors=True)
        except Exception:
            pass
        return False, py_file, traceback.format_exc()


def collect_py_files(input_roots):
    pairs = []
    for root in input_roots:
        files = glob.glob(os.path.join(root, "**", "*.py"), recursive=True)
        for f in files:
            pairs.append((f, root))
    return pairs


def main():
    ap = argparse.ArgumentParser("CADEvolve: .py CAD code -> point clouds + 8 images")
    ap.add_argument("--input-roots", nargs="+", required=True)
    ap.add_argument(
        "--output-pc-root",
        default="/home/nicksung/Desktop/nicksung/multimodel_class/cadevolve_data/CADEvolve-C_pointcloud",
    )
    ap.add_argument(
        "--output-img-root",
        default="/home/nicksung/Desktop/nicksung/multimodel_class/cadevolve_data/CADEvolve-C_image",
    )
    ap.add_argument("--num-points", type=int, default=10000)
    ap.add_argument("--mesh-tol", type=float, default=0.2)
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--max-files", type=int, default=0, help="For smoke test, e.g. 2")
    ap.add_argument("--force", action="store_true", help="Reprocess files even if outputs already exist")
    ap.add_argument("--max-pool-restarts", type=int, default=200)
    ap.add_argument(
        "--failed-log",
        default="",
        help="Optional path for failed file log. Defaults to <output-pc-root>/failed_files.txt",
    )
    args = ap.parse_args()

    jobs_all = collect_py_files(args.input_roots)

    failed_log_path = args.failed_log or os.path.join(args.output_pc_root, "failed_files.txt")
    os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)

    known_failed = set()
    if os.path.exists(failed_log_path):
        with open(failed_log_path, "r", encoding="utf-8") as f:
            known_failed = {line.strip() for line in f if line.strip()}

    if args.force:
        pending_jobs = jobs_all
        skipped = 0
        skipped_failed = 0
    else:
        pending_jobs = []
        skipped = 0
        skipped_failed = 0
        for py_file, root in jobs_all:
            if py_file in known_failed:
                skipped_failed += 1
                continue
            pc_path, img_dir = build_output_paths(py_file, root, args.output_pc_root, args.output_img_root)
            if has_complete_outputs(pc_path, img_dir):
                skipped += 1
            else:
                pending_jobs.append((py_file, root))

    if args.max_files > 0:
        pending_jobs = pending_jobs[: args.max_files]

    print(f"Found {len(jobs_all)} CAD .py files")
    print(f"Skip existing complete outputs: {skipped}")
    print(f"Skip previously failed: {skipped_failed}")
    print(f"Pending to process: {len(pending_jobs)}")
    if not pending_jobs:
        return

    ok = 0
    fail = 0
    processed = 0
    pool_restarts = 0
    crash_counts = {}

    try:
        with open(failed_log_path, "a", encoding="utf-8") as failed_log:
            max_in_flight = max(8, args.workers * 4)
            pending_queue = list(pending_jobs)

            while pending_queue:
                if pool_restarts > args.max_pool_restarts:
                    raise RuntimeError(
                        f"Exceeded max pool restarts ({args.max_pool_restarts}). "
                        f"Remaining jobs: {len(pending_queue)}"
                    )

                in_flight = {}
                pool_broken = False

                with ProcessPoolExecutor(max_workers=args.workers) as ex:
                    while len(in_flight) < max_in_flight and pending_queue:
                        py_file, root = pending_queue.pop(0)
                        try:
                            fut = ex.submit(
                                process_one,
                                py_file,
                                root,
                                args.output_pc_root,
                                args.output_img_root,
                                args.num_points,
                                args.mesh_tol,
                                args.img_size,
                            )
                            in_flight[fut] = (py_file, root)
                        except BrokenProcessPool:
                            pending_queue.insert(0, (py_file, root))
                            pool_broken = True
                            break

                    while in_flight and not pool_broken:
                        fut = next(as_completed(in_flight))
                        processed += 1
                        src_py, src_root = in_flight.pop(fut)

                        try:
                            success, file_path, err = fut.result()
                            if success:
                                ok += 1
                            else:
                                fail += 1
                                failed_log.write(f"{file_path}\n")
                                failed_log.flush()
                                print(f"\n[FAIL] {file_path}\n{err}")
                        except BrokenProcessPool:
                            pool_broken = True
                            crash_counts[src_py] = crash_counts.get(src_py, 0) + 1
                            if crash_counts[src_py] >= 2:
                                fail += 1
                                failed_log.write(f"{src_py}\n")
                                failed_log.flush()
                                print(f"\n[POOL-CRASH-SKIP] {src_py}")
                            else:
                                pending_queue.insert(0, (src_py, src_root))

                            for rem_py, rem_root in in_flight.values():
                                pending_queue.append((rem_py, rem_root))
                            in_flight.clear()
                            break
                        except Exception:
                            fail += 1
                            failed_log.write(f"{src_py}\n")
                            failed_log.flush()
                            print(f"\n[WORKER-CRASH] {src_py}\n{traceback.format_exc()}")

                        while len(in_flight) < max_in_flight and pending_queue:
                            py_file, root = pending_queue.pop(0)
                            try:
                                fut = ex.submit(
                                    process_one,
                                    py_file,
                                    root,
                                    args.output_pc_root,
                                    args.output_img_root,
                                    args.num_points,
                                    args.mesh_tol,
                                    args.img_size,
                                )
                                in_flight[fut] = (py_file, root)
                            except BrokenProcessPool:
                                pending_queue.insert(0, (py_file, root))
                                pool_broken = True
                                for rem_py, rem_root in in_flight.values():
                                    pending_queue.append((rem_py, rem_root))
                                in_flight.clear()
                                break

                        if processed % 200 == 0:
                            print(
                                f"Progress: processed={processed}/{len(pending_jobs)} "
                                f"success={ok} fail={fail} pending={len(pending_queue)}"
                            )

                if pool_broken:
                    pool_restarts += 1
                    print(f"[POOL-RESTART] count={pool_restarts} pending={len(pending_queue)}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial outputs are kept.")
        sys.exit(130)

    print(f"\nDone. Success={ok}, Fail={fail}, Skipped={skipped}")
    print(f"Failed file log: {failed_log_path}")


if __name__ == "__main__":
    main()
