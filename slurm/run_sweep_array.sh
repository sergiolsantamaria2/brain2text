#!/bin/bash
#SBATCH --job-name=b2t_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-999%4
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

# --- EDIT THESE TWO LINES PER SWEEP ---
BASE_CFG="configs/rnn_args_best.yaml"
DIR_CFG="configs/experiments/sweep_speckle_time_hiP_10k_tmp"
# -------------------------------------

mkdir -p slurm_logs

source ~/.bashrc
conda activate brain2text

cd /home/e12511253/Brain2Text/brain2text
export PYTHONPATH=src

# Keep wandb/cache on local scratch/tmp to avoid quota issues
export WANDB_DIR="${SLURM_TMPDIR:-/tmp}/wandb"
export TMPDIR="${SLURM_TMPDIR:-/tmp}"

mapfile -t CFGS < <(ls -1 "${DIR_CFG}"/*.yaml | sort)
N=${#CFGS[@]}

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${N}" ]; then
  echo "Task id ${SLURM_ARRAY_TASK_ID} >= number of configs ${N}. Exiting."
  exit 0
fi

CFG="${CFGS[${SLURM_ARRAY_TASK_ID}]}"
echo "=== Running: ${CFG} ==="

python -u -m brain2text.model_training.train_model \
  --config "${BASE_CFG}" \
  --config "${CFG}"
