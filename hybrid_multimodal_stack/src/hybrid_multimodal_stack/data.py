import json
import zlib
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def fps_numpy(points: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= n_points:
        return np.arange(points.shape[0], dtype=np.int64)

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


class CADMultimodalDataset(Dataset):
    def __init__(self, manifest_path: str, pointcloud_num_points: int = 256) -> None:
        self.samples: List[Dict] = []
        self.pointcloud_num_points = pointcloud_num_points
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image_path = Path(sample["image_path"])
        point_path = Path(sample["pointcloud_path"])
        cad_code = sample.get("cad_code")
        cad_path = sample.get("cad_path")

        if cad_code is None:
            if cad_path is None:
                raise ValueError("Sample must contain either 'cad_code' or 'cad_path'.")
            cad_code = Path(cad_path).read_text(encoding="utf-8")

        image = Image.open(image_path).convert("RGB")
        points = np.load(point_path).astype(np.float32)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected point cloud shape [N,3], got {points.shape} from {point_path}")

        rng_seed = zlib.crc32(str(point_path).encode("utf-8"))
        rng = np.random.default_rng(rng_seed)
        point_indices = fps_numpy(points, self.pointcloud_num_points, rng)
        points = points[point_indices]

        return {
            "image": image,
            "points": torch.tensor(points, dtype=torch.float32),
            "cad_code": cad_code,
        }


def _first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_manifest_from_roots(
    image_root: str,
    pointcloud_root: str,
    cad_root: str,
    output_manifest_path: str,
    image_filename: str = "view_*.png",
    max_samples: Optional[int] = None,
) -> int:
    image_root_path = Path(image_root)
    point_root_path = Path(pointcloud_root)
    cad_root_path = Path(cad_root)
    manifest_path = Path(output_manifest_path)

    if not image_root_path.exists():
        raise FileNotFoundError(f"Image root not found: {image_root_path}")
    if not point_root_path.exists():
        raise FileNotFoundError(f"Pointcloud root not found: {point_root_path}")
    if not cad_root_path.exists():
        raise FileNotFoundError(f"CAD root not found: {cad_root_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with manifest_path.open("w", encoding="utf-8") as out:
        for image_path in image_root_path.rglob(image_filename):
            if "/.tmp/" in str(image_path):
                continue

            if not re.fullmatch(r"view_[0-7]\.png", image_path.name):
                continue

            if len(image_path.parts) < 3:
                continue

            case_key = image_path.parent.name
            category = image_path.parent.parent.name

            cad_candidates = [
                cad_root_path / category / f"{case_key}.py",
                cad_root_path / "CADEvolve-C" / category / f"{case_key}.py",
            ]
            point_candidates = [
                point_root_path / category / f"{case_key}.npy",
                point_root_path / "CADEvolve-C" / category / f"{case_key}.npy",
            ]

            cad_path = _first_existing_path(cad_candidates)
            point_path = _first_existing_path(point_candidates)
            if cad_path is None or point_path is None:
                continue

            row = {
                "image_path": str(image_path),
                "pointcloud_path": str(point_path),
                "cad_path": str(cad_path),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

            if max_samples is not None and written >= max_samples:
                break

    return written
