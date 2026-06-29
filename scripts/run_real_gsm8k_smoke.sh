#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${DICO_CONFIG:-configs/mvp_gsm8k.json}"

python -m src.run_experiment_config \
  --config "$CONFIG_PATH" \
  --experiment smoke
