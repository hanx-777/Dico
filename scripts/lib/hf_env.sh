#!/usr/bin/env bash

dico_setup_hf_env() {
  local no_hf_mirror="${1:-0}"
  local hf_endpoint_arg="${2:-}"

  if [[ "$no_hf_mirror" != "1" ]]; then
    if [[ -n "$hf_endpoint_arg" ]]; then
      export HF_ENDPOINT="$hf_endpoint_arg"
    else
      export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    fi
  fi

  export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

  mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
}

dico_print_hf_env() {
  echo "HF_ENDPOINT=${HF_ENDPOINT:-}"
  echo "HF_HOME=${HF_HOME:-}"
  echo "HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-}"
  echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-}"
  echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-}"
}
