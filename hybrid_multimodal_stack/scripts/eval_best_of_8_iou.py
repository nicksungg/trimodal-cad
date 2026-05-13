#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cadquery as cq
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from hybrid_multimodal_stack.config import HybridConfig
from hybrid_multimodal_stack.model import HybridCADStack


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


def cq_align_shapes(source: cq.Workplane, target: cq.Workplane) -> float:
    c_source = cq.Shape.centerOfMass(source.val())
    c_target = cq.Shape.centerOfMass(target.val())
    I_source = np.array(cq.Shape.matrixOfInertia(source.val()))
    I_target = np.array(cq.Shape.matrixOfInertia(target.val()))
    v_source = cq.Shape.computeMass(source.val())
    v_target = cq.Shape.computeMass(target.val())
    I_p_source, I_v_source = np.linalg.eigh(I_source)
    I_p_target, I_v_target = np.linalg.eigh(I_target)
    s_source = np.sqrt(np.abs(I_p_source).sum() / v_source)
    s_target = np.sqrt(np.abs(I_p_target).sum() / v_target)
    normalized_source = source.translate(-c_source).val().scale(1 / s_source)
    normalized_target = target.translate(-c_target).val().scale(1 / s_target)

    Rs = np.zeros((4, 3, 3))
    Rs[0] = I_v_target @ I_v_source.T
    for i in range(3):
        alignment = 1 - 2 * np.array([i > 0, (i + 1) % 2, i % 3 <= 1])
        Rs[i + 1] = I_v_target @ (alignment[None, :] * I_v_source).T

    best_iou = 0.0
    for i in range(4):
        T = np.zeros([4, 4])
        T[:3, :3] = Rs[i]
        T[-1, -1] = 1
        aligned_source = normalized_source.transformGeometry(cq.Matrix(T.tolist()))
        try:
            intersect = aligned_source.intersect(normalized_target)
            union = aligned_source.fuse(normalized_target)
            iou = intersect.Volume() / union.Volume()
        except Exception:
            iou = 0.0
        best_iou = max(best_iou, float(iou))
    return best_iou


def latest_checkpoint(outputs_root: Path) -> Tuple[Path, Path]:
    run_dirs = [p for p in outputs_root.iterdir() if p.is_dir()]
    candidate_runs: List[Tuple[Path, List[Path]]] = []
    for run_dir in run_dirs:
        ckpts = [p for p in run_dir.glob("checkpoint_step_*.pt") if p.is_file()]
        final_model = run_dir / "final_model.pt"
        if final_model.exists():
            ckpts.append(final_model)
        if ckpts:
            candidate_runs.append((run_dir, ckpts))
    if not candidate_runs:
        raise FileNotFoundError(f"No checkpoints found under {outputs_root}")

    latest_run, ckpts = max(candidate_runs, key=lambda rc: rc[0].stat().st_mtime)

    def ckpt_key(p: Path) -> int:
        if p.name == "final_model.pt":
            return 10**18
        m = re.search(r"checkpoint_step_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    best = max(ckpts, key=ckpt_key)
    return latest_run, best


def load_pointcloud(path: Path, n_points: int, seed: int) -> torch.Tensor:
    points = np.load(path).astype(np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected [N,3] in {path}, got {points.shape}")
    idx = fps_numpy(points, n_points=n_points, rng=np.random.default_rng(seed))
    return torch.tensor(points[idx], dtype=torch.float32).unsqueeze(0)


def extract_code(raw: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, flags=re.S)
    text = m.group(1).strip() if m else raw.strip()
    text = text.replace("<|endoftext|>", "").replace("</s>", "").strip()

    # Prefer the code region beginning at CAD imports/aliases if present.
    starts = []
    for pat in (r"\bimport cadquery as cq\b", r"\bcq\s*=\s*cadquery\b", r"\bwp\d*\s*="):
        hit = re.search(pat, text)
        if hit:
            starts.append(hit.start())
    if starts:
        text = text[min(starts) :]

    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#"):
            lines.append(ln)
            continue
        if re.match(r"^(import\s+|from\s+|[A-Za-z_]\w*\s*=|for\s+|if\s+|elif\s+|else:|try:|except\s+|while\s+|with\s+|return\b)", s):
            lines.append(ln)
    return "\n".join(lines).strip() if lines else text


def try_make_step(code: str, out_step: Path) -> bool:
    wrapped = (
        "import cadquery as cq\n"
        + code
        + "\n_obj = None\n"
        + "for _name in ('solid', 'result', 'model', 'wp', 'part', 'obj'):\n"
        + "    if _name in locals():\n"
        + "        _obj = locals()[_name]\n"
        + "        break\n"
        + "if _obj is None:\n"
        + "    for _v in list(locals().values()):\n"
        + "        if isinstance(_v, cq.Workplane):\n"
        + "            _obj = _v\n"
        + "            break\n"
        + "if _obj is None:\n"
        + "    raise RuntimeError('generated code did not define an exportable cad object')\n"
        + f"\ncq.exporters.export(_obj, r'{str(out_step)}')\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(wrapped)
        tmp_py = Path(tf.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_py)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        return proc.returncode == 0 and out_step.exists()
    finally:
        if tmp_py.exists():
            tmp_py.unlink()


def _image_object_parts(image_root: Path, obj_dir: Path) -> Tuple[str, str, str]:
    rel = obj_dir.relative_to(image_root)
    if len(rel.parts) == 2:
        bench, obj_id = rel.parts
        subgroup = ""
    elif len(rel.parts) == 3:
        bench, subgroup, obj_id = rel.parts
    else:
        raise ValueError(f"Unexpected image object path layout: {obj_dir}")
    return bench, subgroup, obj_id


def build_object_index(image_root: Path, pc_root: Path) -> List[Dict[str, object]]:
    pc_by_key: Dict[Tuple[str, str, str], Path] = {}
    for p in pc_root.rglob("*.npy"):
        rel = p.relative_to(pc_root)
        if len(rel.parts) == 2:
            bench, filename = rel.parts
            subgroup = ""
        elif len(rel.parts) == 3:
            bench, subgroup, filename = rel.parts
        else:
            continue
        pc_by_key[(bench, subgroup, Path(filename).stem)] = p

    objs = []
    for obj_dir in sorted([p for p in image_root.rglob("*") if p.is_dir()]):
        views = sorted(obj_dir.glob("view_*.png"))
        if not views:
            continue
        bench, subgroup, obj_id = _image_object_parts(image_root, obj_dir)
        pc_path = pc_by_key.get((bench, subgroup, obj_id))
        if pc_path is None:
            continue
        subgroup_label = bench if not subgroup else f"{bench}_{subgroup}"
        objs.append(
            {
                "bench": bench,
                "difficulty": subgroup,
                "id": obj_id,
                "subgroup": subgroup_label,
                "views": views,
                "pointcloud": pc_path,
            }
        )
    return objs


def find_gt_step(step_root: Path, bench: str, subgroup: str, obj_id: str) -> Optional[Path]:
    candidates = []
    if subgroup:
        candidates.append(step_root / bench / subgroup / f"{obj_id}.step")
    candidates.append(step_root / bench / f"{obj_id}.step")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    hits = list((step_root / bench).rglob(f"{obj_id}.step"))
    return hits[0] if hits else None


def generate_text_with_sampling(model: HybridCADStack, prompt: str, pixel_values: torch.Tensor, pc: torch.Tensor, max_new_tokens: int) -> str:
    # Keep generation path aligned with eval_forward_and_iou.py, which previously
    # produced feasible CADQuery scripts on this checkpoint.
    return model.generate(
        prompt=prompt,
        pixel_values=pixel_values,
        pointcloud_embeddings=pc,
        max_new_tokens=max_new_tokens,
    )


def load_done_keys(glob_patterns: List[str]) -> set[Tuple[str, str]]:
    done: set[Tuple[str, str]] = set()
    for pat in glob_patterns:
        for p in sorted(Path().glob(pat)):
            if not p.is_file():
                continue
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    subgroup = rec.get("subgroup")
                    obj_id = rec.get("id")
                    if subgroup and obj_id:
                        done.add((str(subgroup), str(obj_id)))
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path("/orcd/data/faez/001/nick/hybrid_multimodal_stack"))
    ap.add_argument("--outputs-root", type=Path, default=Path("/orcd/data/faez/001/nick/hybrid_multimodal_stack/outputs"))
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--pointcloud-root", type=Path, required=True)
    ap.add_argument("--gt-step-root", type=Path, default=Path("/orcd/data/faez/001/nick/cadbench_stls/cadbench_steps_256"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prompt", type=str, default="import cadquery as cq\n")
    ap.add_argument("--pointcloud-num-points", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-objects", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--include-benches", nargs="*", default=None)
    ap.add_argument("--skip-records-glob", nargs="*", default=[])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_dir, ckpt = latest_checkpoint(args.outputs_root)

    cfg = HybridConfig(
        llm_name_or_path="Qwen/Qwen2-1.5B",
        vision_name_or_path="facebook/dinov2-base",
        point_encoder_weights=str(args.project_root / "outputs" / "recode_point_encoder.pt"),
        point_encoder_type="recode_fourier",
    )
    model = HybridCADStack(cfg)
    payload = torch.load(ckpt, map_location="cpu")
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model.load_state_dict(state, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    processor = model.vision_encoder.processor

    objects = build_object_index(args.image_root, args.pointcloud_root)
    if args.include_benches:
        include = set(args.include_benches)
        objects = [o for o in objects if o["bench"] in include]
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    if args.shard_count > 1:
        objects = [o for i, o in enumerate(objects) if (i % args.shard_count) == args.shard_index]
    if args.limit_objects > 0:
        objects = objects[: args.limit_objects]

    done_keys = load_done_keys(args.skip_records_glob)
    if done_keys:
        objects = [o for o in objects if (str(o["subgroup"]), str(o["id"])) not in done_keys]

    gen_code_dir = args.out_dir / "generated_code"
    gen_step_dir = args.out_dir / "generated_steps"
    gen_code_dir.mkdir(exist_ok=True)
    gen_step_dir.mkdir(exist_ok=True)

    records: List[dict] = []
    subgroup_best_ious: Dict[str, List[float]] = defaultdict(list)

    for obj in tqdm(objects, desc="objects"):
        obj_id = obj["id"]
        bench = obj["bench"]
        difficulty = obj["difficulty"]
        subgroup = obj["subgroup"]
        views: List[Path] = obj["views"]
        pc_path: Path = obj["pointcloud"]
        gt_step = find_gt_step(args.gt_step_root, bench, difficulty, obj_id)

        best_iou = None
        best_view = None
        best_step = None

        pc = load_pointcloud(pc_path, args.pointcloud_num_points, args.seed).to(device)

        for img_path in views:
            image = Image.open(img_path).convert("RGB")
            pixel_values = processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
            with torch.inference_mode():
                raw = generate_text_with_sampling(
                    model=model,
                    prompt=args.prompt,
                    pixel_values=pixel_values,
                    pc=pc,
                    max_new_tokens=args.max_new_tokens,
                )
            code = extract_code(raw)

            view_tag = img_path.stem
            code_path = gen_code_dir / f"{subgroup}__{obj_id}__{view_tag}.py"
            code_path.write_text(code, encoding="utf-8")

            step_path = gen_step_dir / f"{subgroup}__{obj_id}__{view_tag}.step"
            generated_step_ok = try_make_step(code, step_path)
            if not generated_step_ok or gt_step is None:
                continue

            try:
                pred_wp = cq.importers.importStep(str(step_path))
                gt_wp = cq.importers.importStep(str(gt_step))
                iou = cq_align_shapes(pred_wp, gt_wp)
            except Exception:
                continue

            if best_iou is None or iou > best_iou:
                best_iou = iou
                best_view = str(img_path)
                best_step = str(step_path)

        rec = {
            "subgroup": subgroup,
            "id": obj_id,
            "pointcloud": str(pc_path),
            "num_views": len(views),
            "gt_step": str(gt_step) if gt_step else None,
            "best_iou": best_iou,
            "best_view": best_view,
            "best_generated_step": best_step,
        }
        records.append(rec)
        if best_iou is not None:
            subgroup_best_ious[subgroup].append(float(best_iou))

    subgroup_summary = {}
    for subgroup, vals in subgroup_best_ious.items():
        subgroup_summary[subgroup] = {
            "num_objects_with_iou": len(vals),
            "avg_best_iou": float(np.mean(vals)) if vals else None,
            "median_best_iou": float(np.median(vals)) if vals else None,
        }

    overall_vals = [r["best_iou"] for r in records if r["best_iou"] is not None]
    summary = {
        "latest_run_dir": str(run_dir),
        "checkpoint_used": str(ckpt),
        "num_objects": len(records),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "num_objects_with_iou": len(overall_vals),
        "num_skipped_from_existing_records": len(done_keys),
        "overall_avg_best_iou": float(np.mean(overall_vals)) if overall_vals else None,
        "overall_median_best_iou": float(np.median(overall_vals)) if overall_vals else None,
        "subgroup_summary": subgroup_summary,
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
