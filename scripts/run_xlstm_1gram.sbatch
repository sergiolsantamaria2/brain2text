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

mkdir -p logs

# ---- Env ----
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text

# ---- Job-local dirs ----
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
JOB_TMP="${TMPDIR}/${USER}_b2t_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP}"

# ---- Stability knobs ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Debug CUDA ----
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_MODULE_LOADING=LAZY
export PYTHONFAULTHANDLER=1
ulimit -c unlimited || true

# ---- Torch extensions ----
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export TORCH_EXTENSIONS_DIR="${JOB_TMP}/torch_extensions"
rm -rf "${TORCH_EXTENSIONS_DIR}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
export TORCH_CUDA_ARCH_LIST="8.0"

# ---- W&B ----
export WANDB_DIR="${JOB_TMP}/wandb"
mkdir -p "${WANDB_DIR}"

echo "TMPDIR=$TMPDIR"
echo "JOB_TMP=$JOB_TMP"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "WANDB_DIR=$WANDB_DIR"
echo "CONDA_PREFIX=$CONDA_PREFIX"

# ---- Torch / CUDA alignment ----
TORCH_CUDA="$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"
TORCH_VER="$(python - <<'PY'
import torch
print(torch.__version__)
PY
)"
echo "torch version: ${TORCH_VER}"
echo "torch CUDA runtime expects: ${TORCH_CUDA}"

TORCH_LIB="$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
echo "TORCH_LIB=${TORCH_LIB}"

CUDA_HOME_CANDIDATE="$(python - <<'PY'
import torch.utils.cpp_extension as ce
print(ce.CUDA_HOME or "")
PY
)"
if [[ -z "${CUDA_HOME_CANDIDATE}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME_CANDIDATE="$(dirname "$(dirname "$(command -v nvcc)")")"
  fi
fi
if [[ -z "${CUDA_HOME_CANDIDATE}" || ! -d "${CUDA_HOME_CANDIDATE}" ]]; then
  echo "ERROR: Could not determine CUDA_HOME."
  exit 1
fi

export CUDA_HOME="${CUDA_HOME_CANDIDATE}"
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "CUDA_HOME=${CUDA_HOME}"
echo "which nvcc: $(command -v nvcc || echo 'nvcc-not-found')"
nvcc --version || true

NVCC_REL="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | tail -n 1 || true)"
echo "nvcc release: ${NVCC_REL}"
if [[ -n "${TORCH_CUDA}" && -n "${NVCC_REL}" && "${NVCC_REL}" != "${TORCH_CUDA}" ]]; then
  echo "ERROR: nvcc (${NVCC_REL}) != torch CUDA (${TORCH_CUDA})."
  exit 1
fi

# CUDA_LIB EXACT like xlstm cuda_init.py expects
export CUDA_LIB="$(python - <<'PY'
import os
import torch
from packaging import version
import torch.utils.cpp_extension as ce

if version.parse(torch.__version__) >= version.parse("2.6.0"):
    inc = ce.include_paths(device_type="cuda")[-1]
else:
    inc = ce.include_paths(cuda=True)[-1]

cuda_root = os.path.split(inc)[0]
print(os.path.join(cuda_root, "lib"))
PY
)"
echo "CUDA_LIB=${CUDA_LIB}"

if [[ -z "${CUDA_LIB}" || ! -d "${CUDA_LIB}" ]]; then
  echo "ERROR: CUDA_LIB does not exist: ${CUDA_LIB}"
  exit 1
fi
if ! ls "${CUDA_LIB}/libcudart.so"* >/dev/null 2>&1; then
  echo "ERROR: libcudart.so not found under CUDA_LIB=${CUDA_LIB}"
  exit 1
fi
if ! ls "${CUDA_LIB}/libcublas.so"* >/dev/null 2>&1; then
  echo "ERROR: libcublas.so not found under CUDA_LIB=${CUDA_LIB}"
  exit 1
fi

unset LD_PRELOAD || true
export LD_LIBRARY_PATH="${CUDA_LIB}:${TORCH_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH(base)=${LD_LIBRARY_PATH}"

# ---- lm_decoder OpenFST runtime ----
LM_RT="/home/e12511253/Brain2Text/brain2text/references/nejm-brain-to-text/language_model/runtime"
FST_SO="$(find "$LM_RT" -type f -name "libfst.so.8.0.0" 2>/dev/null | head -n 1 || true)"
if [[ -z "$FST_SO" ]]; then
  echo "ERROR: libfst.so.8.0.0 not found under $LM_RT"
  exit 1
fi
FST_DIR="$(dirname "$FST_SO")"
TMP_LIB="${JOB_TMP}/lm_runtime_libs"
mkdir -p "$TMP_LIB"
ln -sf "$FST_SO" "$TMP_LIB/libfst.so.8"
export LD_LIBRARY_PATH="${TMP_LIB}:${FST_DIR}:${LD_LIBRARY_PATH}"
echo "FST_SO=$FST_SO"
echo "LD_LIBRARY_PATH(with_fst)=${LD_LIBRARY_PATH}"
ls -lah "$TMP_LIB/libfst.so.8" || true

# ---- Repo + PYTHONPATH ----
cd /home/e12511253/Brain2Text/brain2text
LM_PY="/home/e12511253/Brain2Text/brain2text/references/nejm-brain-to-text/language_model/runtime/server/x86/build/lib.linux-x86_64-cpython-39"
export PYTHONPATH="${LM_PY}:${PWD}/src:${PYTHONPATH:-}"

python -c "import lm_decoder; print('lm_decoder import: OK')" || \
  echo "WARNING: lm_decoder not importable; trainer will disable WER."

# ---- Data ----
export B2T_DATA_DIR="${PWD}/data/hdf5_data_final"
export PYTHONUNBUFFERED=1

# ---- Output: trained_models -> node-local ----
OUT_ROOT="${JOB_TMP}/trained_models"
mkdir -p "${OUT_ROOT}"
if [[ -e trained_models && ! -L trained_models ]]; then
  echo "ERROR: trained_models exists and is not a symlink."
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
OVR_SAFE="configs/overrides/xlstm_safe.yaml"

echo "=============================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID:-unset}"
echo "Base: $BASE"
echo "Force xLSTM override: $OVR_XLSTM"
echo "WER override: $OVR_WER"
echo "Safe override: $OVR_SAFE"
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

    python -u - "$BASE" "$CFG" "$OVR_XLSTM" "$OVR_WER" "$OVR_SAFE" <<'PY'
import sys, runpy

base, cfg, ovr_xlstm, ovr_wer, ovr_safe = sys.argv[1:6]

sys.argv = [
  "train_model",
  "--config", base,
  "--config", cfg,
  "--config", ovr_xlstm,
  "--config", ovr_wer,
  "--config", ovr_safe,
]

runpy.run_module("brain2text.model_training.train_model", run_name="__main__")
PY

  done
done

echo "All runs finished. Outputs in: ${OUT_ROOT}"
