# Static Acceptance

- status: `PASS_WITH_SKIPS`
- failed_checks: `0`
- skipped_checks: `2`

| check | status | message |
|---|---|---|
| import_core_modules | PASS | core CovRA/training modules import successfully |
| syntax_compile | PASS | src/dico and scripts compile with Python bytecode compiler |
| config_dry_run | PASS | main CovRA r8 config resolves through run_experiment --dry-run |
| cpu_tiny_training_smoke | PASS | tiny CPU path completed calibration, allocation, init, one train step, eval dry-run, checkpoint, and manifest |
| cpu_tiny_manifest_validation | PASS | tiny CPU run_manifest.json passes the formal run manifest validator |
| cpu_tiny_parameter_budget_audit | PASS | tiny CPU manifest parameter budget and active/trainable counts are internally consistent |
| experiment_matrix_generation | PASS | E00-E10 experiment command matrix generates successfully |
| protocol_preflight | PASS | formal configs pass protocol preflight |
| typecheck_tool | SKIP | mypy/pyright is not installed in this environment; typecheck not claimed |
| lint_tool | SKIP | ruff is not installed in this environment; lint not claimed |
