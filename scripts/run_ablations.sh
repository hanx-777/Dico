#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source "$SCRIPT_DIR/lib/hf_env.sh"
source "$SCRIPT_DIR/lib/runtime.sh"

SEEDS="${SEEDS:-42 43 44}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_ablations}"
NO_HF_MIRROR=0
HF_ENDPOINT_ARG=""
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --output_dir requires a value" >&2
        exit 2
      fi
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --hf_endpoint)
      if [[ $# -lt 2 ]]; then
        echo "error: --hf_endpoint requires a value" >&2
        exit 2
      fi
      HF_ENDPOINT_ARG="$2"
      shift 2
      ;;
    --no_hf_mirror)
      NO_HF_MIRROR=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage:
  SEEDS="42 43 44" bash scripts/run_ablations.sh [script options] [run_experiment args]

Script options:
  --output_dir DIR     Parent output directory. Default: outputs_ablations.
  --hf_endpoint URL    Set HF_ENDPOINT, overriding the default mirror.
  --no_hf_mirror       Do not set a default HF_ENDPOINT.
  --help               Show this help.

Environment:
  DRY_RUN=1            Print commands without running training.
EOF
      exit 0
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

dico_setup_hf_env "$NO_HF_MIRROR" "$HF_ENDPOINT_ARG"

CONFIGS=(
  configs/experiments/ablations/dico_pre_r8_no_relaxation.yaml
  configs/experiments/ablations/dico_pre_r8_eta100.yaml
  configs/experiments/ablations/dico_pre_r8_answer_full.yaml
  configs/experiments/ablations/dico_pre_r8_random.yaml
  configs/experiments/ablations/dico_predynamic_r8_move20.yaml
)

experiment_name_from_config() {
  python - "$1" <<'PY'
import sys
import yaml
from pathlib import Path
path = Path(sys.argv[1])
payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
print(payload.get("experiment_name") or path.stem)
PY
}

dico_print_run_header "$PROJECT_DIR" "$OUTPUT_DIR"
echo "seeds=$SEEDS"

mkdir -p "$OUTPUT_DIR"

for seed in $SEEDS; do
  for config_path in "${CONFIGS[@]}"; do
    experiment_name="$(experiment_name_from_config "$config_path")"
    run_name="${experiment_name}__seed${seed}"
    save_dir="$OUTPUT_DIR/preallocations/seed${seed}"
    cmd=(
      python scripts/run_experiment.py
      --config "$config_path"
      --override "experiment_name=$run_name"
      --override "project.output_dir=$OUTPUT_DIR"
      --override "seed=$seed"
      --override "calibration.seed=$seed"
      --override "preallocation.sketch_seed=$seed"
      --override "calibration.save_dir=$save_dir"
    )
    if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
      cmd+=("${TRAIN_ARGS[@]}")
    fi
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'run:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      echo "running $run_name from $config_path"
      "${cmd[@]}"
    fi
  done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python scripts/summarize_results.py --output_dir "$OUTPUT_DIR"
fi
