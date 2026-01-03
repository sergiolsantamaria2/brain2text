#!/bin/bash
#SBATCH --job-name=b2t_exp
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
  echo "Usage: sbatch scripts/run_1gram.sbatch <folder1> [<folder2> ...]"
  exit 1
fi

# ---- Env ----
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text

# ---- CUDA toolkit selection: MUST match torch.version.cuda ----
TORCH_CUDA="$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"
echo "torch CUDA runtime expects: ${TORCH_CUDA}"

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
  echo "This mismatch is a common cause of undefined-symbol issues in custom CUDA extensions (like sLSTM)."
  exit 1
fi
# ---------------------------------------------------------------

# ---- Runtime linking for lm_decoder (Torch + NEJM OpenFST v8 runtime) ----
rm -f "${CONDA_PREFIX}/lib/libfst.so.8" 2>/dev/null || true

TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

LM_RT="/home/e12511253/Brain2Text/brain2text/references/nejm-brain-to-text/language_model/runtime"
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

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "TORCH_LIB=$TORCH_LIB"
echo "FST_SO=$FST_SO"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
ls -lah "$TMP_LIB/libfst.so.8" || true

python -c "import lm_decoder; print('lm_decoder import: OK')" || \
  echo "WARNING: lm_decoder not importable; continuing without WER (trainer will disable WER)."
# ------------------------------------------------------------------------

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

echo "PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER"
echo "MAX_JOBS=$MAX_JOBS"
echo "TMPDIR=$TMPDIR"
echo "JOB_TMP=$JOB_TMP"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "WANDB_DIR=$WANDB_DIR"
# -------------------------------------------------------------------------------

# ---- Locate libcudart that matches torch.version.cuda, export CUDA_LIB + CUDART_SO ----
CUDART_SO="$(python - <<'PY'
import os, glob, torch
ver = torch.version.cuda or ""
cp = os.environ.get("CONDA_PREFIX","")
cuda_home = os.environ.get("CUDA_HOME","")

cands = []
# Prefer CUDA_HOME first (system toolkit if available)
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

export CUDA_LIB="$(dirname "${CUDART_SO}")"
# Important: put CUDA_LIB FIRST to avoid picking wrong cudart from conda/lib64
export LD_LIBRARY_PATH="${CUDA_LIB}:${LD_LIBRARY_PATH:-}"
#!/bin/bash
#SBATCH --job-name=b2t_exp
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
  echo "Usage: sbatch scripts/run_1gram.sbatch <folder1> [<folder2> ...]"
  echo "Example:"
  echo "  sbatch scripts/run_1gram.sbatch configs/experiments/head_blocks configs/experiments/lr_cosine"
  exit 1
fi

# ---- Env ----
source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text

# ---- CUDA toolkit selection: MUST match torch.version.cuda ----
TORCH_CUDA="$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)"
echo "torch CUDA runtime expects: ${TORCH_CUDA}"

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
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME=${CUDA_HOME}"
echo "which nvcc: $(command -v nvcc || echo 'nvcc-not-found')"
nvcc --version || true

NVCC_REL="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | tail -n 1)"
echo "nvcc release: ${NVCC_REL}"

if [[ -n "${TORCH_CUDA}" && -n "${NVCC_REL}" && "${NVCC_REL}" != "${TORCH_CUDA}" ]]; then
  echo "ERROR: nvcc (${NVCC_REL}) != torch CUDA (${TORCH_CUDA})."
  echo "This mismatch is a common cause of undefined-symbol issues in custom CUDA extensions (like sLSTM)."
  exit 1
fi
# ---------------------------------------------------------------

# ---- Runtime linking for lm_decoder (Torch + NEJM OpenFST v8 runtime) ----
rm -f "${CONDA_PREFIX}/lib/libfst.so.8" 2>/dev/null || true

TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

LM_RT="/home/e12511253/Brain2Text/brain2text/references/nejm-brain-to-text/language_model/runtime"
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

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "TORCH_LIB=$TORCH_LIB"
echo "FST_SO=$FST_SO"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
ls -lah "$TMP_LIB/libfst.so.8" || true

python -c "import lm_decoder; print('lm_decoder import: OK')" || \
  echo "WARNING: lm_decoder not importable; continuing without WER (trainer will disable WER)."
# ------------------------------------------------------------------------

# ---- Job-local dirs (avoid quota in /home) ----
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
JOB_TMP="${TMPDIR}/${USER}_b2t_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP}"

# Torch extensions: build/load on node-local disk (prevents HOME cache + quota issues)
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

echo "PYTHONFAULTHANDLER=$PYTHONFAULTHANDLER"
echo "MAX_JOBS=$MAX_JOBS"
echo "TMPDIR=$TMPDIR"
echo "JOB_TMP=$JOB_TMP"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "WANDB_DIR=$WANDB_DIR"
# -------------------------------------------------------------------------------

# ---- Fix __cudaLaunchKernel: LD_PRELOAD libcudart matching torch CUDA ----
CUDART_SO="$(python - <<'PY'
import os, glob, torch
ver = torch.version.cuda or ""
cp = os.environ.get("CONDA_PREFIX","")
cands = []
if cp and ver:
    cands += glob.glob(f"{cp}/pkgs/**/libcudart.so.{ver}*", recursive=True)
    cands += glob.glob(f"{cp}/**/libcudart.so.{ver}*", recursive=True)
cands = [c for c in cands if ".a" not in c]
print(cands[0] if cands else "")
PY
)"

if [[ -z "${CUDART_SO}" || ! -f "${CUDART_SO}" ]]; then
  echo "ERROR: Could not locate libcudart matching torch CUDA (${TORCH_CUDA})."
  exit 1
fi

export LD_PRELOAD="${CUDART_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="$(dirname "${CUDART_SO}"):${LD_LIBRARY_PATH:-}"
export TORCH_USE_RTLD_GLOBAL=1

echo "CUDART_SO=${CUDART_SO}"
echo "LD_PRELOAD=${LD_PRELOAD}"
echo "TORCH_USE_RTLD_GLOBAL=${TORCH_USE_RTLD_GLOBAL}"
# -------------------------------------------------------------------------------

# ---- Repo + data ----
cd /home/e12511253/Brain2Text/brain2text
export B2T_DATA_DIR="${PWD}/data/hdf5_data_final"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p logs

# ---- Output: trained_models -> node-local ----
mkdir -p "${JOB_TMP}/trained_models"
rm -rf trained_models
ln -s "${JOB_TMP}/trained_models" trained_models
echo "trained_models -> $(readlink -f trained_models)"
# ---------------------------------------------

BASE="configs/rnn_args.yaml"
OVR1="configs/overrides/wer_1gram_only.yaml"

echo "=============================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Base: $BASE"
echo "Global override 1: $OVR1"
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
    python -u -m brain2text.model_training.train_model \
      --config "$BASE" \
      --config "$OVR1" \
      --config "$CFG"
  done
done

echo "All runs finished."

# Helps torch dlopen behavior
export TORCH_USE_RTLD_GLOBAL=1

echo "CUDART_SO=${CUDART_SO}"
echo "CUDA_LIB=${CUDA_LIB}"
echo "TORCH_USE_RTLD_GLOBAL=${TORCH_USE_RTLD_GLOBAL}"
# -------------------------------------------------------------------------------

# ---- Repo + data ----
cd /home/e12511253/Brain2Text/brain2text
export B2T_DATA_DIR="${PWD}/data/hdf5_data_final"
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p logs

# ---- Output: trained_models -> node-local (robust symlink) ----
OUT_ROOT="${JOB_TMP}/trained_models"
mkdir -p "${OUT_ROOT}"

# Safety: if trained_models exists and is NOT a symlink, do not delete it
if [[ -e trained_models && ! -L trained_models ]]; then
  echo "ERROR: trained_models exists and is not a symlink. Refusing to modify it."
  ls -ld trained_models
  exit 1
fi

# Force/update the symlink even if an old dangling symlink exists
ln -sfn "${OUT_ROOT}" trained_models
echo "trained_models -> $(readlink trained_models)"
echo "OUT_ROOT=${OUT_ROOT}"
# --------------------------------------------------------------

BASE="configs/rnn_args.yaml"
OVR1="configs/overrides/wer_1gram_only.yaml"

echo "=============================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Base: $BASE"
echo "Global override 1: $OVR1"
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

    # Wrapper: force RTLD_GLOBAL + preload libcudart in-process (this fixes __cudaLaunchKernel)
    python -u - <<PY
import os, sys, ctypes, runpy
import os as _os, sys as _sys

# Make future dlopen() global
_sys.setdlopenflags(_os.RTLD_GLOBAL | _os.RTLD_NOW)

# Preload cudart globally (critical)
ctypes.CDLL(os.environ["CUDART_SO"], mode=ctypes.RTLD_GLOBAL)

# Run the training module with the same CLI args
sys.argv = [
  "train_model",
  "--config", "${BASE}",
  "--config", "${OVR1}",
  "--config", "${CFG}",
]
runpy.run_module("brain2text.model_training.train_model", run_name="__main__")
PY

  done
done

echo "All runs finished."
