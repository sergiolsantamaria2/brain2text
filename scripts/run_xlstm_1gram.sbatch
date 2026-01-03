#!/bin/bash
#SBATCH --job-name=b2t_xlstm_1g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclude=a-l40s-o-2

set -eo pipefail
set +u  # conda activation scripts sometimes reference unset vars

if [[ $# -lt 1 ]]; then
  echo "ERROR: Missing experiment folder argument(s)."
  echo "Usage: sbatch scripts/run_xlstm_1gram.sbatch <folder1> [<folder2> ...]"
  exit 1
fi

# ---- Env ----
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text_cu126

# Make sure conda libs (e.g., nccl) are visible
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

set -u

# Important for dlopen + some custom extensions
export TORCH_USE_RTLD_GLOBAL=1
export CUDA_MODULE_LOADING=LAZY

# ---- Job-local dirs (avoid quota in /home) ----
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
JOB_TMP="${TMPDIR}/${USER}_b2t_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP}"

export PYTHONFAULTHANDLER=1
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR="${JOB_TMP}/torch_extensions"
rm -rf "${TORCH_EXTENSIONS_DIR}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# A100 = SM80
export TORCH_CUDA_ARCH_LIST="8.0"

# W&B logs on node-local disk
export WANDB_DIR="${JOB_TMP}/wandb"
mkdir -p "${WANDB_DIR}"

# ---- Discover torch CUDA version ----
TORCH_CUDA="$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"
echo "torch CUDA runtime expects: ${TORCH_CUDA}"

# ---- Locate libcudart matching torch.version.cuda and PRELOAD it (fixes __cudaLaunchKernel) ----
export CUDART_SO="$(python - <<'PY'
import os, glob, torch
ver = torch.version.cuda or ""
cp = os.environ.get("CONDA_PREFIX","")
cuda_home = os.environ.get("CUDA_HOME","")

cands = []
# Prefer CUDA_HOME if it exists
if cuda_home and ver:
    cands += glob.glob(os.path.join(cuda_home, "lib64", f"libcudart.so.{ver}*"))
    cands += glob.glob(os.path.join(cuda_home, "lib",   f"libcudart.so.{ver}*"))

# Then conda (pkgs + env)
if cp and ver:
    cands += glob.glob(f"{cp}/pkgs/**/libcudart.so.{ver}*", recursive=True)
    cands += glob.glob(f"{cp}/**/libcudart.so.{ver}*", recursive=True)

cands = [c for c in cands if c and (not c.endswith(".a"))]
print(cands[0] if cands else "")
PY
)"

if [[ -z "${CUDART_SO}" || ! -f "${CUDART_SO}" ]]; then
  echo "ERROR: Could not locate libcudart matching torch CUDA (${TORCH_CUDA})."
  exit 1
fi

export LD_PRELOAD="${CUDART_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="$(dirname "${CUDART_SO}"):${LD_LIBRARY_PATH:-}"

echo "CUDART_SO=${CUDART_SO}"
echo "LD_PRELOAD=${LD_PRELOAD}"

# ---- Repo + data ----
cd /home/e12511253/Brain2Text/brain2text
export B2T_DATA_DIR="${PWD}/data/hdf5_data_final"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p logs

# ---- Runtime linking for lm_decoder (OpenFST v8 runtime) ----
rm -f "${CONDA_PREFIX}/lib/libfst.so.8" 2>/dev/null || true

TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

LM_RT="${PWD}/references/nejm-brain-to-text/language_model/runtime"
FST_SO=$(find "$LM_RT" -type f -name "libfst.so.8.0.0" 2>/dev/null | head -n 1)

if [[ -z "$FST_SO" ]]; then
  echo "ERROR: libfst.so.8.0.0 not found under $LM_RT"
  exit 1
fi

FST_DIR=$(dirname "$FST_SO")
TMP_LIB="${SLURM_TMPDIR:-/tmp}/lm_runtime_libs"
mkdir -p "$TMP_LIB"
ln -sf "$FST_SO" "$TMP_LIB/libfst.so.8"

export LD_LIBRARY_PATH="${TMP_LIB}:${FST_DIR}:${CONDA_PREFIX}/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "TORCH_LIB=$TORCH_LIB"
echo "FST_SO=$FST_SO"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
ls -lah "$TMP_LIB/libfst.so.8" || true

python -c "import lm_decoder; print('lm_decoder import: OK ->', lm_decoder.__file__)" || \
  echo "WARNING: lm_decoder not importable; WER will be disabled by trainer."
# ------------------------------------------------------------------------

# ---- Output: trained_models -> node-local ----
mkdir -p "${JOB_TMP}/trained_models"
rm -rf trained_models
ln -s "${JOB_TMP}/trained_models" trained_models
echo "trained_models -> $(readlink -f trained_models)"

BASE="configs/rnn_args.yaml"
OVR1="configs/overrides/wer_1gram_only.yaml"
OVR2="configs/overrides/disk_conservative.yaml"

echo "=============================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID:-manual}"
echo "Base: $BASE"
echo "Global override 1: $OVR1"
echo "Global override 2: $OVR2"
echo "Folders: $*"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}"
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

    BASE_CFG="$BASE" OVR1_CFG="$OVR1" OVR2_CFG="$OVR2" CFG_PATH="$CFG" \
    python -u - <<'PY'
import os, sys, ctypes, runpy
import os as _os

# Make future dlopen() global
sys.setdlopenflags(_os.RTLD_GLOBAL | _os.RTLD_NOW)

# Preload cudart globally (critical for __cudaLaunchKernel resolution)
ctypes.CDLL(os.environ["CUDART_SO"], mode=ctypes.RTLD_GLOBAL)

sys.argv = [
  "train_model",
  "--config", os.environ["BASE_CFG"],
  "--config", os.environ["OVR1_CFG"],
  "--config", os.environ["OVR2_CFG"],
  "--config", os.environ["CFG_PATH"],
]
runpy.run_module("brain2text.model_training.train_model", run_name="__main__")
PY

  done
done

echo "All runs finished."
