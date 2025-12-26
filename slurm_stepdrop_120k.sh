#!/bin/bash
#SBATCH --job-name=b2t_stepdrop120k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

source /home/e12511253/miniforge3/etc/profile.d/conda.sh
conda activate brain2text

cd /home/e12511253/Brain2Text/brain2text
export PYTHONPATH=src

export TMPDIR="${SLURM_TMPDIR:-/tmp}"
export WANDB_DIR="${TMPDIR}/wandb"
mkdir -p logs "$WANDB_DIR"

BASE="configs/rnn_args.yaml"
STAGE_A="configs/experiments/compare_120k_stepdrop/gru_stepdrop_stageA_30k_seed10.yaml"

echo "=== STAGE A (0 -> 30k) ==="
python -u -m brain2text.model_training.train_model \
  --config "$BASE" \
  --config "$STAGE_A"

# Discover checkpoint dir from STAGE_A yaml
CKPT_DIR="$(python - <<'PY'
import yaml
p="configs/experiments/compare_120k_stepdrop/gru_stepdrop_stageA_30k_seed10.yaml"
cfg=yaml.safe_load(open(p,"r"))
print(cfg["checkpoint_dir"])
PY
)"
echo "Checkpoint dir: $CKPT_DIR"

# Pick latest checkpoint-like file
CKPT_PATH="$(ls -t "$CKPT_DIR"/* 2>/dev/null | head -n 1 || true)"
if [[ -z "${CKPT_PATH}" ]]; then
  echo "ERROR: No checkpoint found in $CKPT_DIR"
  exit 1
fi
echo "Using checkpoint: $CKPT_PATH"

STAGE_B="configs/experiments/compare_120k_stepdrop/gru_stepdrop_stageB_to120k_lr3e-4_seed10.yaml"
cat > "$STAGE_B" <<YAML
output_dir: ${TMPDIR}/brain2text/trained_models/compare_120k/gru_stepdrop_stageB_seed10
checkpoint_dir: \${output_dir}/checkpoint

init_from_checkpoint: true
init_checkpoint_path: ${CKPT_PATH}

# IMPORTANT: set TOTAL batches so resume continues to 120k
num_training_batches: 120000
lr_scheduler_type: cosine
lr_decay_steps: \${num_training_batches}
lr_decay_steps_day: \${num_training_batches}

# Step drop x10 at ~30k:
lr_max: 0.0003
lr_min: 0.00001
lr_max_day: 0.0001
lr_min_day: 0.00001

# After resume, warmup is usually not needed
lr_warmup_steps: 0
lr_warmup_steps_day: 0

batches_per_val_step: 5000

save_val_logits: false
save_all_val_steps: false
save_best_checkpoint: false
save_final_model: false

seed: 10
dataset:
  seed: 10

wandb:
  enabled: true
  project: brain2text
  run_name: gru_stepdrop_stageB_to120k_lr3e-4_seed10
  tags: ["compare_120k", "stepdrop", "stageB", "gru", "cosine", "lr3e-4", "seed10", "resume"]

eval:
  compute_wer: true
  wer_max_trials: 1024
  wer_every_val_steps: 2
  wer_tag: "1gram"
  lm_dir: "assets/lm/openwebtext_1gram_lm_sil"
  acoustic_scale: 0.35
  blank_penalty: 90.0
  max_active: 7000
  min_active: 200
  beam: 15.0
  lattice_beam: 8.0
  ctc_blank_skip_threshold: 0.95
  length_penalty: 0.0
  nbest: 50
YAML

echo "=== STAGE B (resume @30k, LR drop x10, -> 120k) ==="
python -u -m brain2text.model_training.train_model \
  --config "$BASE" \
  --config "$STAGE_B"
