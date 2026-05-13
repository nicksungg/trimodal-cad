#!/usr/bin/env bash
#SBATCH -J hmstack_continue_latest
#SBATCH -p pi_faez
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=100:00:00
#SBATCH -o /orcd/data/faez/001/nick/hybrid_multimodal_stack/logs/%x_%j.out
#SBATCH -e /orcd/data/faez/001/nick/hybrid_multimodal_stack/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/orcd/data/faez/001/nick/hybrid_multimodal_stack"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_LOG="$LOG_DIR/hmstack_continue_latest_${SLURM_JOB_ID}.log"

# ── Find latest checkpoint across all previous runs ────────────
# Search all train_two_adapters output dirs for the newest checkpoint
RESUME_CKPT=""
for d in "$ROOT_DIR"/outputs/train_two_adapters_*; do
  [[ -d "$d" ]] || continue
  candidate=$(find "$d" -maxdepth 1 -type f -name 'checkpoint_step_*.pt' | sort -V | tail -n1 || true)
  if [[ -n "$candidate" ]]; then
    # Pick the checkpoint with the highest step number
    if [[ -z "$RESUME_CKPT" ]]; then
      RESUME_CKPT="$candidate"
    else
      cur_step=$(echo "$RESUME_CKPT" | grep -oP '\d+(?=\.pt$)')
      new_step=$(echo "$candidate" | grep -oP '\d+(?=\.pt$)')
      if (( new_step > cur_step )); then
        RESUME_CKPT="$candidate"
      fi
    fi
  fi
done

if [[ -z "$RESUME_CKPT" ]]; then
  echo "No checkpoint found to resume from!" >&2
  exit 1
fi

{
  echo "==== job start $(date -Is) ===="
  echo "job_id=$SLURM_JOB_ID"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "node_list=$SLURM_JOB_NODELIST"
  echo "cwd=$PWD"
  echo "host=$(hostname)"
  echo "cpus_per_task=$SLURM_CPUS_PER_TASK"
  echo "resume_checkpoint=$RESUME_CKPT"
  nvidia-smi || true
  echo "==== training start $(date -Is) ===="
} | tee -a "$RUN_LOG"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_DATA_PARALLEL=1
export SAVE_EVERY=10000
export NUM_WORKERS=8
export BATCH_SIZE=4
export OUTPUT_DIR="$ROOT_DIR/outputs/train_two_adapters_continue_latest_${SLURM_JOB_ID}"
export RESUME_CHECKPOINT="$RESUME_CKPT"
export REBUILD_MANIFEST=0
export TRAIN_MANIFEST="$ROOT_DIR/outputs/train_manifest.jsonl"

set +e
cd "$ROOT_DIR"
bash "$ROOT_DIR/scripts/train_two_adapters_orcd.sh" 2>&1 | tee -a "$RUN_LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ $rc -ne 0 ]]; then
  {
    echo "==== training failed $(date -Is) rc=$rc ===="
    echo "collecting diagnostics"
    nvidia-smi || true
    echo "disk usage output dir:"
    du -sh "$OUTPUT_DIR" 2>/dev/null || true
  } | tee -a "$RUN_LOG"
  exit $rc
fi

{
  echo "==== training complete $(date -Is) rc=0 ===="
  echo "artifacts: $OUTPUT_DIR"
  ls -lh "$OUTPUT_DIR" | head -n 50 || true
} | tee -a "$RUN_LOG"