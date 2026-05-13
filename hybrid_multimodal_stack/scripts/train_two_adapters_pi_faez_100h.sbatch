#!/usr/bin/env bash
#SBATCH -J hmstack_pi_faez_100h
#SBATCH -p pi_faez
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=100:00:00
#SBATCH -o /orcd/data/faez/001/nick/hybrid_multimodal_stack/logs/%x_%j.out
#SBATCH -e /orcd/data/faez/001/nick/hybrid_multimodal_stack/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/orcd/data/faez/001/nick/hybrid_multimodal_stack"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_LOG="$LOG_DIR/hmstack_pi_faez_100h_${SLURM_JOB_ID}.log"

{
  echo "==== job start $(date -Is) ===="
  echo "job_id=$SLURM_JOB_ID"
  echo "partition=${SLURM_JOB_PARTITION:-unknown}"
  echo "node_list=$SLURM_JOB_NODELIST"
  echo "cwd=$PWD"
  echo "host=$(hostname)"
  nvidia-smi || true
  echo "==== training start $(date -Is) ===="
} | tee -a "$RUN_LOG"

export PYTHONUNBUFFERED=1
export USE_DATA_PARALLEL=1
export SAVE_EVERY=100
export NUM_WORKERS=8
export BATCH_SIZE=4
export OUTPUT_DIR="$ROOT_DIR/outputs/train_two_adapters_pi_faez_${SLURM_JOB_ID}"

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
