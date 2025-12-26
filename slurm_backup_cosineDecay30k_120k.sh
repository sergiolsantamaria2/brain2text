#!/bin/bash
#SBATCH --job-name=b2t_bk30k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text

cd /home/e12511253/Brain2Text/brain2text
export PYTHONPATH=src

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_DIR="${TMPDIR}/wandb"
mkdir -p logs "$WANDB_DIR"

BASE="configs/rnn_args.yaml"
CFG="configs/experiments/backup_cosine_decay30k_120k_tmp/gru_cosineDecay30k_120k_seed10_wer1024.yaml"

python -u -m brain2text.model_training.train_model --config "$BASE" --config "$CFG"
