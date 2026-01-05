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

set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "ERROR: Missing experiment folder argument(s)."
  echo "Usage: sbatch scripts/run_xlstm_1gram.sbatch <folder1> [<folder2> ...]"
  exit 1
fi

# ---- Env ----
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text

mkdir -p logs

# ---- Job-local dirs (avoid quota in /home) ----
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
JOB_TMP="${TMPDIR}/${USER}_b2t_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP}"

# Torch extensions: build/load on node-local disk
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

echo "TMPDIR=$TMPDIR"
echo "JOB_TMP=$JOB_TMP"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "WANDB_DIR=$WANDB_DIR"

# ---- Torch / CUDA info ----
TORCH_CUDA="$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"
echo "torch CUDA runtime expects: ${TORCH_CUDA}"

# ---- CUDA toolkit selection (for nvcc) ----
CUDA_HOME_CANDIDATE=""
if [[ -n "${TORCH_CUDA}" && -d "/usr/local/cuda-${TORCH_CUDA}" ]]; then
  CUDA_HOME_CANDIDATE="/usr/local/cuda-${TORCH_CUDA}"
fi

if [[ -z "${CUDA_HOME_CANDIDATE}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME_CANDIDATE="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
fi

if [[ -z "${CUDA_HOME_CANDIDATE}" || ! -d "${CUDA_HOME_CANDIDATE}" ]]; then
  echo "ERROR: No usable CUDA toolkit found. torch expects CUDA=${TORCH_CUDA}."
  exit 1
fi

export CUDA_HOME="${CUDA_HOME_CANDIDATE}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=${CUDA_HOME}"
echo "which nvcc: $(command -v nvcc || echo 'nvcc-not-found')"
nvcc --version || true

NVCC_REL="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | tail -n 1)"
echo "nvcc release: ${NVCC_REL}"

if [[ -n "${TORCH_CUDA}" && -n "${NVCC_REL}" && "${NVCC_REL}" != "${TORCH_CUDA}" ]]; then
  echo "ERROR: nvcc (${NVCC_REL}) != torch CUDA (${TORCH_CUDA})."
  exit 1
fi

# ---- Torch lib dir ----
TORCH_LIB="$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
echo "TORCH_LIB=${TORCH_LIB}"

# ---- Runtime linking for lm_decoder (OpenFST v8 runtime) ----
rm -f "${CONDA_PREFIX}/lib/libfst.so.8" 2>/dev/null || true

LM_RT="/home/e12511253/Brain2Text/brain2text/references/nejm-brain-to-text/language_model/runtime"
FST_SO=$(find "$LM_RT" -type f -name "libfst.so.8.0.0" 2>/dev/null | head -n 1)

if [[ -z "$FST_SO" ]]; then
  echo "ERROR: libfst.so.8.0.0 not found under $LM_RT"
  exit 1
fi

FST_DIR=$(dirname "$FST_SO")
TMP_LIB="${JOB_TMP}/lm_runtime_libs"
mkdir -p "$TMP_LIB"
ln -sf "$FST_SO" "$TMP_LIB/libfst.so.8"

export LD_LIBRARY_PATH="${TMP_LIB}:${FST_DIR}:${CONDA_PREFIX}/lib:${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "FST_SO=$FST_SO"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
ls -lah "$TMP_LIB/libfst.so.8" || true

# ---- Repo + PYTHONPATH ----
cd /home/e12511253/Brain2Text/brain2text

LM_PY="/home/e12511253/Brain2Text/brain2text/references/nejm-brain-to-text/language_model/runtime/server/x86/build/lib.linux-x86_64-cpython-39"
export PYTHONPATH="${LM_PY}:${PWD}/src:${PYTHONPATH:-}"

export B2T_DATA_DIR="${PWD}/data/hdf5_data_final"
export PYTHONUNBUFFERED=1

python -c "import lm_decoder; print('lm_decoder import: OK')" || \
  echo "WARNING: lm_decoder not importable; trainer will disable WER."

# -------------------------------------------------------------------
# CRITICAL FIX: choose a REAL libcudart that exports __cudaLaunchKernel
# -------------------------------------------------------------------
CUDART_SO=""
SYMBOL="__cudaLaunchKernel"

# Prefer conda env locations first
CANDIDATES=()
CANDIDATES+=( "${CONDA_PREFIX}/lib64/libcudart.so"* )
CANDIDATES+=( "${CONDA_PREFIX}/lib/libcudart.so"* )
CANDIDATES+=( "${CONDA_PREFIX}/targets/"*/lib/libcudart.so* )
# Then system CUDA (if exists)
if [[ -n "${TORCH_CUDA}" ]]; then
  CANDIDATES+=( "/usr/local/cuda-${TORCH_CUDA}/lib64/libcudart.so"* )
  CANDIDATES+=( "/usr/local/cuda-${TORCH_CUDA}/lib/libcudart.so"* )
fi

for so in "${CANDIDATES[@]}"; do
  [[ -e "$so" ]] || continue
  # must be ELF shared object (avoid linker scripts)
  file "$so" 2>/dev/null | grep -q "ELF" || continue
  if nm -D "$so" 2>/dev/null | grep -q "${SYMBOL}"; then
    CUDART_SO="$so"
    break
  fi
done

if [[ -z "${CUDART_SO}" ]]; then
  echo "ERROR: Could not find a libcudart that exports ${SYMBOL}."
  echo "Try installing proper cudart dev packages in your env, e.g.:"
  echo "  conda install -n brain2text -c nvidia cuda-cudart=11.7 cuda-cudart-dev=11.7"
  exit 1
fi

export CUDART_SO
export CUDA_LIB="$(dirname "${CUDART_SO}")"

# Make sure THIS CUDA_LIB is what xlstm/cuda_init.py uses for linking
export LD_LIBRARY_PATH="${CUDA_LIB}:${TORCH_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_USE_RTLD_GLOBAL=1

echo "Selected CUDART_SO=${CUDART_SO}"
echo "CUDA_LIB=${CUDA_LIB}"
echo "Symbol check:"
nm -D "${CUDART_SO}" 2>/dev/null | grep "${SYMBOL}" | head -n 1 || true

# ---- Debug (optional but useful) ----
export CUDA_MODULE_LOADING=LAZY
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=1

# ---- Output: trained_models -> node-local ----
OUT_ROOT="${JOB_TMP}/trained_models"
mkdir -p "${OUT_ROOT}"

if [[ -e trained_models && ! -L trained_models ]]; then
  echo "ERROR: trained_models exists and is not a symlink. Refusing to modify it."
  ls -ld trained_models
  exit 1
fi

ln -sfn "${OUT_ROOT}" trained_models
echo "trained_models -> $(readlink trained_models)"
echo "OUT_ROOT=${OUT_ROOT}"

# ---- Configs ----
BASE="configs/rnn_args.yaml"
OVR_XLSTM="configs/overrides/xlstm_only.yaml"
OVR_WER="configs/overrides/wer_1gram_only.yaml"

echo "=============================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Base: $BASE"
echo "Force xLSTM override: $OVR_XLSTM"
echo "WER override: $OVR_WER"
echo "Folders: $*"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
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

    python -u - "$BASE" "$CFG" "$OVR_XLSTM" "$OVR_WER" <<'PY'
import os, sys, ctypes, runpy

base, cfg, ovr_xlstm, ovr_wer = sys.argv[1:5]

# Make future dlopen() global
sys.setdlopenflags(os.RTLD_GLOBAL | os.RTLD_NOW)

# CRITICAL: load cudart globally BEFORE importing the slstm extension
ctypes.CDLL(os.environ["CUDART_SO"], mode=ctypes.RTLD_GLOBAL)

sys.argv = [
  "train_model",
  "--config", base,
  "--config", cfg,
  "--config", ovr_xlstm,
  "--config", ovr_wer,
]

runpy.run_module("brain2text.model_training.train_model", run_name="__main__")
PY

  done
done

echo "All runs finished. Outputs in: ${OUT_ROOT}"
