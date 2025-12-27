#!/bin/bash
#SBATCH --job-name=b2t_spkF_hiP_10k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=a-l40s-o-2

set -euo pipefail
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text

cd /home/e12511253/Brain2Text/brain2text
export PYTHONPATH=src

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_DIR="${TMPDIR}/wandb"
mkdir -p logs "$WANDB_DIR"

BASE="configs/rnn_args_best.yaml"
DIR="configs/experiments/sweep_speckle_feature_hiP_10k_tmp"

for CFG in "$DIR"/*.yaml; do
  echo "=== RUN $(basename "$CFG") ==="
  python -u -m brain2text.model_training.train_model \
    --config "$BASE" \
    --config "$CFG"
done
