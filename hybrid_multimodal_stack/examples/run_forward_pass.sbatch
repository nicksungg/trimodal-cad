#!/usr/bin/env bash
#SBATCH -J hmstack_fwd
#SBATCH -p pi_faez
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -o /orcd/data/faez/001/nick/hybrid_multimodal_stack/test_v1/logs/%x_%j.out
#SBATCH -e /orcd/data/faez/001/nick/hybrid_multimodal_stack/test_v1/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="/orcd/data/faez/001/nick/hybrid_multimodal_stack"
TEST_DIR="$ROOT_DIR/test_v1"
LOG_DIR="$TEST_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_LOG="$LOG_DIR/hmstack_fwd_${SLURM_JOB_ID}.log"
PYTHON_BIN="/orcd/data/faez/001/nicksung/conda/envs/cadevolve/bin/python"
PY_ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"

if [[ -d "$PY_ENV_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$PY_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

{
  echo "==== job start $(date -Is) ===="
  echo "job_id=$SLURM_JOB_ID"
  echo "node_list=$SLURM_JOB_NODELIST"
  echo "cwd=$PWD"
  echo "host=$(hostname)"
  nvidia-smi || true
  echo "==== forward pass start $(date -Is) ===="
} | tee -a "$RUN_LOG"

cd "$ROOT_DIR"
set +e
PYTHONPATH=src "$PYTHON_BIN" "$TEST_DIR/forward_pass.py" \
  --output-text "$TEST_DIR/generated/hmstack_fwd_${SLURM_JOB_ID}.txt" \
  2>&1 | tee -a "$RUN_LOG"
rc=${PIPESTATUS[0]}
set -e

if [[ $rc -ne 0 ]]; then
  {
    echo "==== forward pass failed $(date -Is) rc=$rc ===="
    nvidia-smi || true
  } | tee -a "$RUN_LOG"
  exit $rc
fi

{
  echo "==== forward pass complete $(date -Is) rc=0 ===="
  echo "artifacts: $TEST_DIR/generated/hmstack_fwd_${SLURM_JOB_ID}.txt"
} | tee -a "$RUN_LOG"
