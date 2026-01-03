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
# ---- Runtime linking for lm_decoder (Torch + NEJM OpenFST v8 runtime) ----
# Ensure we DO NOT have a wrong conda-level libfst.so.8 shim (ABI mismatch).
rm -f "${CONDA_PREFIX}/lib/libfst.so.8" 2>/dev/null || true

TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

LM_RT="/home/e12511253/Brain2Text/brain2text/references/nejm-brain-to-text/language_model/runtime"

# Find OpenFST v8 runtime library
FST_SO=$(find "$LM_RT" -type f -name "libfst.so.8.0.0" 2>/dev/null | head -n 1)

if [[ -z "$FST_SO" ]]; then
  echo "ERROR: libfst.so.8.0.0 not found under $LM_RT"
  echo "       Make sure the NEJM language model runtime has been built/downloaded."
  exit 1
fi

FST_DIR=$(dirname "$FST_SO")

# Create the exact SONAME expected by lm_decoder (libfst.so.8) in job-local temp dir
TMP_LIB="${SLURM_TMPDIR:-/tmp}/lm_runtime_libs"
mkdir -p "$TMP_LIB"
ln -sf "$FST_SO" "$TMP_LIB/libfst.so.8"

# Prepend: job-local shim dir, then the NEJM dir, then conda + torch
export LD_LIBRARY_PATH="${TMP_LIB}:${FST_DIR}:${CONDA_PREFIX}/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "TORCH_LIB=$TORCH_LIB"
echo "FST_SO=$FST_SO"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
ls -lah "$TMP_LIB/libfst.so.8" || true

# Fail early if decoder cannot import
python -c "import lm_decoder; print('lm_decoder import: OK')"
# ------------------------------------------------------------------------


cd /home/e12511253/Brain2Text/brain2text
export B2T_DATA_DIR="${PWD}/data/hdf5_data_final"
echo "B2T_DATA_DIR=$B2T_DATA_DIR"


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
