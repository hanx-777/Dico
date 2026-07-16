#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/dico/gora_bw_r8.yaml "$@"
