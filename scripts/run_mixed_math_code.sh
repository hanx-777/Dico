#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/dico/mixed_math_code_r8.yaml "$@"
