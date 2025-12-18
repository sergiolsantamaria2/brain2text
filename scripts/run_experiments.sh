#!/usr/bin/env bash
set -euo pipefail

BASE_CFG="${1:-configs/rnn_args.yaml}"
EXP_DIR="${2:-configs/experiments/reslstm}"
GPU="${3:-0}"

echo "Base config: ${BASE_CFG}"
echo "Experiment dir: ${EXP_DIR}"
echo "GPU: ${GPU}"
echo

# W&B grouping for this batch launch
GROUP="reslstm_$(date +%Y%m%d_%H%M%S)"
export WANDB_RUN_GROUP="${GROUP}"
export WANDB_JOB_TYPE="reslstm_ablation"
echo "W&B group: ${WANDB_RUN_GROUP}"
echo "W&B job_type: ${WANDB_JOB_TYPE}"

# Where to store stdout/stderr logs
LOG_DIR="logs/experiments/${GROUP}"
mkdir -p "${LOG_DIR}"

shopt -s nullglob
OVERRIDES=("${EXP_DIR}"/*.yaml)
if [[ ${#OVERRIDES[@]} -eq 0 ]]; then
  echo "No YAML files found in ${EXP_DIR}"
  exit 1
fi

for OV in "${OVERRIDES[@]}"; do
  RUN_ID="$(basename "${OV}" .yaml)"
  LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

  echo
  echo "============================================================"
  echo "Running override: ${OV}"
  echo "Log: ${LOG_FILE}"
  echo "============================================================"

  python -u -m brain2text.model_training.train_model \
    --config "${BASE_CFG}" \
    --override "${OV}" \
    --set "gpu_number=${GPU}" \
    --set "wandb.enabled=true" \
    --set "wandb.group=${GROUP}" \
    --set "wandb.job_type=reslstm_ablation" \
    2>&1 | tee "${LOG_FILE}"
done

echo
echo "All runs finished. Logs in: ${LOG_DIR}"
echo "W&B group: ${GROUP}"