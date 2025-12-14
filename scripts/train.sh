#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Ajusta si quieres forzar ruta absoluta:
export B2T_DATA_DIR="${B2T_DATA_DIR:-$(pwd)/data/hdf5_data_final}"

CONFIG="${1:-configs/rnn_args.yaml}"

# Recomendación: no reutilizar output_dir por accidente
python -m brain2text.model_training.train_model --config "$CONFIG" 2>&1 | tee train.log
