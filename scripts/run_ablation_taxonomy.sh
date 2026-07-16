#!/usr/bin/env bash
set -euo pipefail

# legacy alias: historical "taxonomy" ablation was removed from final CovRA.
# Keep this wrapper only for old command references; it now delegates to the
# final-method one-factor ablation "no_sign_split".
python scripts/run_ablation_no_sign_split.sh "$@"
