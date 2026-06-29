#!/usr/bin/env bash
# Compatibility wrapper. Prefer scripts/lib/hf_env.sh via scripts/run_all_8.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/hf_env.sh"
dico_setup_hf_env 0 "${HF_ENDPOINT:-}"
