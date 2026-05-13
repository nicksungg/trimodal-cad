#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/orcd/data/faez/001/nicksung/conda/envs/cadevolve/bin/python}"
PY_ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"

if [[ -d "$PY_ENV_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$PY_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

IMAGE_ROOT="${IMAGE_ROOT:-/orcd/data/faez/001/nick/CADEvolve-C_image_fresh_20260408}"
POINTCLOUD_ROOT="${POINTCLOUD_ROOT:-/orcd/data/faez/001/nick/CADEvolve-C_pointcloud}"
CAD_ROOT="${CAD_ROOT:-/orcd/data/faez/001/nick/CADEvolve-C}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/train_two_adapters}"
MANIFEST_PATH="${MANIFEST_PATH:-$OUTPUT_DIR/train_manifest.jsonl}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAVE_EVERY="${SAVE_EVERY:-100}"
USE_DATA_PARALLEL="${USE_DATA_PARALLEL:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-1}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

ARGS=(
  "$ROOT_DIR/scripts/train_multimodal.py"
  --image-root "$IMAGE_ROOT"
  --pointcloud-root "$POINTCLOUD_ROOT"
  --cad-root "$CAD_ROOT"
  --output-dir "$OUTPUT_DIR"
  --manifest-out "$MANIFEST_PATH"
  --stage projector_warmup
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --learning-rate "$LEARNING_RATE"
  --max-length "$MAX_LENGTH"
  --save-every "$SAVE_EVERY"
  --grad-clip "$GRAD_CLIP"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$USE_DATA_PARALLEL" == "1" ]]; then
  ARGS+=(--use-data-parallel)
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ "$REBUILD_MANIFEST" == "1" ]]; then
  ARGS+=(--rebuild-manifest)
fi

if [[ -n "$TRAIN_MANIFEST" ]]; then
  ARGS+=(--train-manifest "$TRAIN_MANIFEST")
fi

cd "$ROOT_DIR"
PYTHONPATH=src "$PYTHON_BIN" "${ARGS[@]}"
