#!/bin/bash
#SBATCH --job-name=b2t_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail
mkdir -p slurm_logs

# --- EDIT THESE PER SWEEP ---
BASE_CFG="configs/rnn_args_best.yaml"
DIR_CFG="configs/experiments/sweep_speckle_feature_hiP_10k_tmp"
# ----------------------------

source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text

cd /home/e12511253/Brain2Text/brain2text
export PYTHONPATH=src

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_DIR="${TMPDIR}/wandb"
mkdir -p "$WANDB_DIR"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi || true
python -c "import torch; print('torch cuda available:', torch.cuda.is_available()); print('device_count:', torch.cuda.device_count())"

mapfile -t CFGS < <(ls -1 "${DIR_CFG}"/*.yaml | sort)
CFG="${CFGS[${SLURM_ARRAY_TASK_ID}]}"
echo "=== Running: ${CFG} ==="

python -u -m brain2text.model_training.train_model \
  --config "${BASE_CFG}" \
  --config "${CFG}"
