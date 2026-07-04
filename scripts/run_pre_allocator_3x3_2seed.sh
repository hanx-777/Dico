#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source "$SCRIPT_DIR/lib/hf_env.sh"
source "$SCRIPT_DIR/lib/runtime.sh"

SEEDS="${SEEDS:-42 43}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs_pre_allocator_3x3_2seed}"
LOG_DIR="${LOG_DIR:-}"
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
    --log_dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --log_dir requires a value" >&2
        exit 2
      fi
      LOG_DIR="$2"
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
  SEEDS="42 43" bash scripts/run_pre_allocator_3x3_2seed.sh [script options] [run_experiment args]

Script options:
  --output_dir DIR     Parent output directory. Default: outputs_pre_allocator_3x3_2seed.
  --log_dir DIR        Log directory. Default: <output_dir>/logs.
  --hf_endpoint URL    Set HF_ENDPOINT, overriding the default mirror.
  --no_hf_mirror       Do not set a default HF_ENDPOINT.
  --help               Show this help.

Environment:
  SEEDS="42 43"        Space-separated seeds. Default: 42 43.
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

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

dico_setup_hf_env "$NO_HF_MIRROR" "$HF_ENDPOINT_ARG"

CONFIGS=(
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_marginal_curve_budget_guardrails.yaml
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_marginal_curve_layer_diffusion.yaml
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_marginal_curve_concentration_penalty.yaml
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_prototype_bundle_budget_guardrails.yaml
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_prototype_bundle_layer_diffusion.yaml
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_prototype_bundle_concentration_penalty.yaml
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_soft_slot_budget_guardrails.yaml
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_soft_slot_layer_diffusion.yaml
  configs/experiments/allocator_3x3/dico_pre_r8_alloc_soft_slot_concentration_penalty.yaml
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

run_one_seed() {
  local config_path="$1"
  local seed="$2"
  local experiment_name
  experiment_name="$(experiment_name_from_config "$config_path")"
  local run_name="${experiment_name}__seed${seed}"
  local save_dir="$OUTPUT_DIR/preallocations/$experiment_name/seed${seed}"
  local cmd=(
    python scripts/run_experiment.py
    --config "$config_path"
  )
  if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    cmd+=("${TRAIN_ARGS[@]}")
  fi
  cmd+=(
    --override "experiment_name=$run_name"
    --override "project.output_dir=$OUTPUT_DIR"
    --override "seed=$seed"
    --override "calibration.seed=$seed"
    --override "preallocation.sketch_seed=$seed"
    --override "calibration.save_dir=$save_dir"
  )
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'run:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  else
    echo "running $run_name from $config_path"
    "${cmd[@]}"
  fi
}

dico_print_run_header "$PROJECT_DIR" "$OUTPUT_DIR"
echo "seeds=$SEEDS"
echo "experiment_matrix=pre_allocator_3x3"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

for seed in $SEEDS; do
  for config_path in "${CONFIGS[@]}"; do
    run_one_seed "$config_path" "$seed"
  done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python scripts/summarize_results.py --output_dir "$OUTPUT_DIR"
fi
