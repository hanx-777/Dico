#!/usr/bin/env bash
set -euo pipefail

python scripts/run_experiment.py --config configs/ablations/no_log_compression.yaml "$@"
