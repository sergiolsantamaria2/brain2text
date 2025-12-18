#!/usr/bin/env bash
set -euo pipefail

BASE_CFG="${1:-configs/rnn_args.yaml}"
EXP_DIR="${2:-configs/experiments/reslstm}"
GPU="${3:-0}"

echo "Base config: ${BASE_CFG}"
echo "Experiment dir: ${EXP_DIR}"
echo "GPU: ${GPU}"
echo

shopt -s nullglob
OVERRIDES=("${EXP_DIR}"/*.yaml)
if [[ ${#OVERRIDES[@]} -eq 0 ]]; then
  echo "No YAML files found in ${EXP_DIR}"
  exit 1
fi

for OV in "${OVERRIDES[@]}"; do
  echo "============================================================"
  echo "Running override: ${OV}"
  echo "============================================================"
  python -m brain2text.model_training.train_model \
    --config "${BASE_CFG}" \
    --override "${OV}" \
    gpu_number="${GPU}"
done
