#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/ablations/covra_independent.yaml "$@"
