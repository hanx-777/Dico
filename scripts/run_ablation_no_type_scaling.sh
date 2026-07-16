#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/ablations/no_type_scaling.yaml "$@"
