#!/bin/bash
#SBATCH -J hmstack_eval_iou
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /orcd/data/faez/001/nick/hybrid_multimodal_stack/logs/hmstack_eval_iou_%j.out
#SBATCH -e /orcd/data/faez/001/nick/hybrid_multimodal_stack/logs/hmstack_eval_iou_%j.err

set -euo pipefail

cd /orcd/data/faez/001/nick/hybrid_multimodal_stack

PYTHON_BIN="${PYTHON_BIN:-/orcd/data/faez/001/nicksung/conda/envs/cadevolve/bin/python}"
PY_ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
if [[ -d "$PY_ENV_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$PY_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

"$PYTHON_BIN" - <<'PY'
import importlib, sys
mods = ["cadquery", "numpy", "torch", "PIL", "tqdm", "transformers"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print("Missing modules:", missing)
    sys.exit(1)
print("Module preflight passed")
PY

PYTHONPATH=src "$PYTHON_BIN" scripts/eval_forward_and_iou.py \
  --image-root /orcd/data/faez/001/nick/cadbench_stls/from_workstation/cadbench_stl_images \
  --pointcloud-root /orcd/data/faez/001/nick/cadbench_stls/cadbench_pointclouds_256 \
  --gt-step-root /orcd/data/faez/001/nick/cadbench_stls/cadbench_steps_256 \
  --out-dir /orcd/data/faez/001/nick/hybrid_multimodal_stack/outputs/eval_forward_iou
