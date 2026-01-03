#!/bin/bash
#SBATCH --job-name=b2t_xlstm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclude=a-l40s-o-2

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "ERROR: Missing experiment folder argument(s)."
  echo "Usage: sbatch scripts/run_xlstm_train.sbatch <folder1> [<folder2> ...]"
  exit 1
fi

# ---- Env (xLSTM) ----
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text_xlstm

python - <<'PY'
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "is_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("ERROR: torch has no CUDA. Install a CUDA-enabled PyTorch build in this env.")
PY


# (Opcional pero recomendable en cluster) cargar CUDA si existe 'module'
if command -v module >/dev/null 2>&1; then
  module load cuda/12.1 || module load cuda/12.0 || module load cuda || true
fi


cd /home/e12511253/Brain2Text/brain2text

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# usar scratch local para compilar extensiones (evita NFS y builds raros)
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export TORCH_EXTENSIONS_DIR="${TMPDIR}/torch_extensions"
export WANDB_DIR="${TMPDIR}/wandb"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p logs "$WANDB_DIR" "$TORCH_EXTENSIONS_DIR"

BASE="configs/rnn_args.yaml"
OVR="configs/overrides/no_wer.yaml"

echo "=============================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Base: $BASE"
echo "Override: $OVR"
echo "Folders: $*"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TMPDIR=$TMPDIR"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
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
