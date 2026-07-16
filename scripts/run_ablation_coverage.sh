#!/usr/bin/env bash
set -euo pipefail

# legacy alias: historical coverage-objective ablation was removed from final CovRA.
# Keep this wrapper only for old command references; it now delegates to the
# final-method mechanism control "covra_independent" (CovRA-I).
python scripts/run_ablation_covra_independent.sh "$@"
