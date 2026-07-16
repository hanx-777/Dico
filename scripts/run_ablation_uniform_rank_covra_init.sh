#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/ablations/uniform_rank_covra_init.yaml "$@"
