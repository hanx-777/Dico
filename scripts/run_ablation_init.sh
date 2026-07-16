#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml --override dico.init.mode=kaiming_zero_B "$@"
