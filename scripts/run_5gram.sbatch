#!/bin/bash
#SBATCH --job-name=b2t_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclude=a-l40s-o-2

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "ERROR: Missing experiment folder argument(s)."
  echo "Usage: sbatch scripts/run_5gram.sbatch <folder1> [<folder2> ...]"
  echo "Example:"
  echo "  sbatch scripts/run_5gram.sbatch configs/experiments/head_blocks configs/experiments/lr_cosine"
  exit 1
fi

# ---- Env ----
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text
echo "Python exe: $(which python)"
python -V
python -c "import sys; print('sys.executable:', sys.executable)"
python -c "import lm_decoder; print('lm_decoder OK:', lm_decoder.__file__)"


cd /home/e12511253/Brain2Text/brain2text

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_DIR="${TMPDIR}/wandb"
mkdir -p logs "$WANDB_DIR"

BASE="configs/rnn_args.yaml"
OVR="configs/overrides/wer_5gram_only.yaml"

echo "=============================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Base: $BASE"
echo "Global override: $OVR"
echo "Folders: $*"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TMPDIR=$TMPDIR"
echo "WANDB_DIR=$WANDB_DIR"
echo "=============================================="

for EXP_DIR in "$@"; do
  echo
  echo "========== FOLDER: $EXP_DIR =========="

  if [[ ! -d "$EXP_DIR" ]]; then
    echo "ERROR: Folder not found: $EXP_DIR"
    exit 1
  fi

  shopt -s nullglob
  CFG_FILES=("$EXP_DIR"/*.yaml)
  shopt -u nullglob

  if [[ ${#CFG_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No .yaml files found in: $EXP_DIR"
    exit 1
  fi

  echo "Num configs: ${#CFG_FILES[@]}"

  for CFG in "${CFG_FILES[@]}"; do
    echo
    echo "=== RUN $(basename "$CFG") ==="
    python -u -m brain2text.model_training.train_model \
      --config "$BASE" \
      --config "$OVR" \
      --config "$CFG"
  done
done

echo "All runs finished."
