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

# ---- Prefer system CUDA toolkit (cluster has CUDA 12.6 under /usr/local/cuda) ----
if [[ -d "/usr/local/cuda" ]]; then
  export CUDA_HOME="/usr/local/cuda"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
  echo "CUDA_HOME=${CUDA_HOME}"
  nvcc --version || true
else
  echo "WARNING: /usr/local/cuda not found. Will try conda-provided CUDA runtime only."
fi

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

# ---- Locate libcudart and PRELOAD it (fixes __cudaLaunchKernel) ----
export CUDART_SO="$(python - <<'PY'
import os, glob, torch

ver = torch.version.cuda or ""
major = ver.split(".")[0] if ver else ""
cp = os.environ.get("CONDA_PREFIX","")
cuda_home = os.environ.get("CUDA_HOME","")

cands = []

def add(patterns):
    for p in patterns:
        cands.extend(glob.glob(p))

# 1) Prefer system toolkit (CUDA_HOME)
if cuda_home and major:
    add([
        os.path.join(cuda_home, "lib64", f"libcudart.so.{ver}*"),
        os.path.join(cuda_home, "lib",   f"libcudart.so.{ver}*"),
        os.path.join(cuda_home, "lib64", f"libcudart.so.{major}*"),
        os.path.join(cuda_home, "lib",   f"libcudart.so.{major}*"),
        os.path.join(cuda_home, "lib64", "libcudart.so*"),
        os.path.join(cuda_home, "lib",   "libcudart.so*"),
    ])

# 2) Then conda env/pkgs
if cp and major:
    add([
        f"{cp}/pkgs/**/libcudart.so.{ver}*",
        f"{cp}/pkgs/**/libcudart.so.{major}*",
        f"{cp}/**/libcudart.so.{ver}*",
        f"{cp}/**/libcudart.so.{major}*",
    ])

cands = [c for c in cands if c and (not c.endswith(".a"))]

best = ""
for x in cands:
    if f"libcudart.so.{ver}" in os.path.basename(x):
        best = x; break
if not best:
    for x in cands:
        if f"libcudart.so.{major}" in os.path.basename(x):
            best = x; break
if not best and cands:
    best = cands[0]

print(best)
PY
)"

if [[ -z "${CUDART_SO}" || ! -f "${CUDART_SO}" ]]; then
  echo "ERROR: Could not locate libcudart for torch CUDA (${TORCH_CUDA})."
  echo "Debug: CUDA_HOME=${CUDA_HOME:-unset}"
  echo "Conda cudart:"
  ls -lah "${CONDA_PREFIX}/lib"/libcudart.so* 2>/dev/null || true
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

# ---- Make lm_decoder importable (built .so) ----
LM_SO=$(find references/nejm-brain-to-text/language_model/runtime/server/x86 -name "lm_decoder.cpython-*.so" | head -n 1)
echo "lm_decoder .so: ${LM_SO:-<not found>}"
if [[ -z "${LM_SO}" ]]; then
  echo "ERROR: lm_decoder .so not found. Did you build/install it in brain2text_cu126?"
  exit 1
fi
export PYTHONPATH="$(dirname "$LM_SO"):${PYTHONPATH:-}"

echo "lm_decoder .so: ${LM_SO}"
python -c "import lm_decoder; print('lm_decoder import: OK ->', lm_decoder.__file__)"
# ---------------------------------------------------------------------------

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
ls -lah "$TMP_LIB/libfst.so.8" || true
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

sys.setdlopenflags(_os.RTLD_GLOBAL | _os.RTLD_NOW)
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
