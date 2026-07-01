#!/usr/bin/env bash

dico_usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_all_8.sh [script options] [run_experiment args]

Script options:
  --nohup              Run all 8 experiments in the background with nohup.
  --output_dir DIR     Set project.output_dir for all experiments.
  --log_dir DIR        Set nohup log directory. Default: <output_dir>/logs.
  --hf_endpoint URL    Set HF_ENDPOINT, overriding the default mirror.
  --no_hf_mirror       Do not set a default HF_ENDPOINT.
  --help               Show this help.

Examples:
  bash scripts/run_all_8.sh --override training.max_steps=2
  bash scripts/run_all_8.sh --nohup --override model.name_or_path=/models/qwen
EOF
}

dico_pid_is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

dico_extract_output_dir_from_overrides() {
  local default_output_dir="$1"
  shift
  local output_dir="$default_output_dir"
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--override" && $((i + 1)) -lt ${#args[@]} ]]; then
      local override="${args[$((i + 1))]}"
      if [[ "$override" == project.output_dir=* ]]; then
        output_dir="${override#project.output_dir=}"
      fi
    fi
  done
  echo "$output_dir"
}

dico_filter_project_output_override() {
  local args=("$@")
  local filtered=()
  local i=0
  while [[ $i -lt ${#args[@]} ]]; do
    if [[ "${args[$i]}" == "--override" && $((i + 1)) -lt ${#args[@]} ]]; then
      local override="${args[$((i + 1))]}"
      if [[ "$override" == project.output_dir=* ]]; then
        i=$((i + 2))
        continue
      fi
      filtered+=("${args[$i]}" "$override")
      i=$((i + 2))
      continue
    fi
    filtered+=("${args[$i]}")
    i=$((i + 1))
  done
  if [[ ${#filtered[@]} -gt 0 ]]; then
    printf '%s\n' "${filtered[@]}"
  fi
}

dico_has_override_key() {
  local key="$1"
  shift
  local args=("$@")
  local i
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--override" && $((i + 1)) -lt ${#args[@]} ]]; then
      if [[ "${args[$((i + 1))]}" == "$key="* ]]; then
        return 0
      fi
    fi
  done
  return 1
}

dico_print_run_header() {
  local project_dir="$1"
  local output_dir="$2"
  local log_path="${3:-}"
  local pid_path="${4:-}"

  echo "project_dir=$project_dir"
  echo "output_dir=$output_dir"
  if [[ -n "$log_path" ]]; then
    echo "log_path=$log_path"
  fi
  if [[ -n "$pid_path" ]]; then
    echo "pid_path=$pid_path"
  fi
  dico_print_hf_env
}
