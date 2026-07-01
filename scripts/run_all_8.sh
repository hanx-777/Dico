#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

source "$SCRIPT_DIR/lib/hf_env.sh"
source "$SCRIPT_DIR/lib/runtime.sh"

NOHUP_MODE=0
NO_HF_MIRROR=0
HF_ENDPOINT_ARG=""
OUTPUT_DIR=""
LOG_DIR=""
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nohup)
      NOHUP_MODE=1
      shift
      ;;
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
      dico_usage
      exit 0
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    OUTPUT_DIR="$(dico_extract_output_dir_from_overrides "outputs" "${TRAIN_ARGS[@]}")"
  else
    OUTPUT_DIR="outputs"
  fi
else
  FILTERED_TRAIN_ARGS=()
  if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    while IFS= read -r item; do
      FILTERED_TRAIN_ARGS+=("$item")
    done < <(dico_filter_project_output_override "${TRAIN_ARGS[@]}")
  fi
  if [[ ${#FILTERED_TRAIN_ARGS[@]} -gt 0 ]]; then
    TRAIN_ARGS=("${FILTERED_TRAIN_ARGS[@]}")
  else
    TRAIN_ARGS=()
  fi
  TRAIN_ARGS+=("--override" "project.output_dir=$OUTPUT_DIR")
  if ! dico_has_override_key "calibration.save_dir" "${TRAIN_ARGS[@]}"; then
    TRAIN_ARGS+=("--override" "calibration.save_dir=$OUTPUT_DIR/preallocations")
  fi
fi

LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
PID_PATH="$OUTPUT_DIR/run_all_8.pid"

dico_setup_hf_env "$NO_HF_MIRROR" "$HF_ENDPOINT_ARG"

run_all_8_foreground() {
  dico_print_run_header "$PROJECT_DIR" "$OUTPUT_DIR"

  run_one_experiment configs/experiments/lora_r4.yaml
  run_one_experiment configs/experiments/lora_r8.yaml

  run_one_experiment configs/experiments/dico_pre_r4.yaml
  run_one_experiment configs/experiments/dico_pre_r8.yaml

  run_one_experiment configs/experiments/dico_dynamic_r4.yaml
  run_one_experiment configs/experiments/dico_dynamic_r8.yaml

  run_one_experiment configs/experiments/dico_predynamic_r4.yaml
  run_one_experiment configs/experiments/dico_predynamic_r8.yaml

  python scripts/summarize_results.py --output_dir "$OUTPUT_DIR"
}

run_one_experiment() {
  local config_path="$1"
  if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    python scripts/run_experiment.py --config "$config_path" "${TRAIN_ARGS[@]}"
  else
    python scripts/run_experiment.py --config "$config_path"
  fi
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  dico_print_run_header "$PROJECT_DIR" "$OUTPUT_DIR" "$LOG_DIR/run_all_8_DRY_RUN.log" "$PID_PATH"
  echo "nohup_mode=$NOHUP_MODE"
  printf 'train_args:'
  if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    printf ' %q' "${TRAIN_ARGS[@]}"
  fi
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ "$NOHUP_MODE" == "1" ]]; then
  if dico_pid_is_running "$PID_PATH"; then
    echo "error: existing run appears active: pid $(cat "$PID_PATH")" >&2
    echo "stop it with: kill \$(cat $PID_PATH)" >&2
    exit 1
  fi

  LOG_PATH="$LOG_DIR/run_all_8_$(date +%Y%m%d_%H%M%S).log"
  CHILD_ARGS=(--output_dir "$OUTPUT_DIR" --log_dir "$LOG_DIR")
  if [[ "$NO_HF_MIRROR" == "1" ]]; then
    CHILD_ARGS+=(--no_hf_mirror)
  fi
  if [[ -n "$HF_ENDPOINT_ARG" ]]; then
    CHILD_ARGS+=(--hf_endpoint "$HF_ENDPOINT_ARG")
  fi
  if [[ ${#TRAIN_ARGS[@]} -gt 0 ]]; then
    CHILD_ARGS+=("${TRAIN_ARGS[@]}")
  fi

  nohup bash "$SCRIPT_DIR/run_all_8.sh" "${CHILD_ARGS[@]}" > "$LOG_PATH" 2>&1 &
  echo "$!" > "$PID_PATH"

  echo "started run_all_8.sh"
  echo "pid: $(cat "$PID_PATH")"
  echo "log: $LOG_PATH"
  echo "tail: tail -f $LOG_PATH"
  exit 0
fi

run_all_8_foreground
