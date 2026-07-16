#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/dico/dico_cd_r8.yaml "$@"
