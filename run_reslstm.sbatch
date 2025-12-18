#!/bin/bash
#SBATCH --job-name=reslstm_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_reslstm_%j.out
#SBATCH --error=logs/slurm_reslstm_%j.err

set -euo pipefail
shopt -s nullglob
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="src:${PYTHONPATH:-}"

BASE="configs/rnn_args.yaml"
DIR="configs/experiments/reslstm"
TS="$(date +%Y-%m-%d_%H%M%S)"
LOGDIR="logs/experiments/reslstm_${TS}"
mkdir -p "$LOGDIR"

for cfg in $(ls -1 "$DIR"/*.yaml | sort); do
  name="$(basename "$cfg" .yaml)"
  echo "============================================================"
  echo "RUN: $name"
  echo "Override: $cfg"
  echo "Log: $LOGDIR/$name.log"
  echo "============================================================"

  python -u -m brain2text.model_training.train_model \
    --config "$BASE" \
    --config "$cfg" \
    2>&1 | tee "$LOGDIR/$name.log"
done