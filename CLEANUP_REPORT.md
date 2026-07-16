# DiCo v0.3 Cleanup Report

## Summary

This cleanup converged the repository to the DiCo v0.3 protocol-aligned
experiment surface. The project now keeps only files needed for DiCo v0.3
methodology, GoRA experiment protocol alignment, reproducible configs, run
entrypoints, tests, and local GSM8K data.

The wording is standardized to "aligned to the GoRA experiment protocol"; GoRA
is not described as the implementation framework.

## Kept Core

- `src/dico/`: v0.3 calibration, sketch/profile, taxonomy, candidate split,
  coverage, procurement, DA-Init, GoRA-BW, cache, analysis, static LoRA,
  masked LoRA equivalence reference, scaling, training, evaluation, config,
  data, and budget utilities.
- `configs/dico/`: LoRA r=8, GoRA-BW, DiCo-CD, DiCo-CD-DA, and mixed math+code.
- `configs/ablations/`: v0.3 ablations for init/taxonomy/coverage/scaling/
  procurement/`r_min`/sketch dimension/atom count.
- `scripts/`: v0.3 run wrappers plus `scripts/run_experiment.py`.
- `tests/`: v0.3 unit/config/script tests, including static-vs-masked LoRA
  equivalence.
- `README.md`, `DiCo_v0.3_AAAI方法论中文初稿.md`, `data/README.md`, and local
  GSM8K JSONL files.

## Deleted

- Legacy configs: `configs/methods/`, `configs/debug/`, old single-factor and
  allocator-grid ablations, old allocator extensions, `gora_original.yaml`, and
  `dico_v027_r8.yaml`.
- Legacy scripts: old matrix/multiseed/preallocator/extension/debug runners and
  old helper wrappers.
- Legacy source files: `src/dico/dynamic_allocation.py` and
  `src/dico/rank_allocator.py`.
- Legacy tests: dico-pre/dynamic/predynamic/rank-allocator/RPCA/old runner tests.
- Legacy docs and reports: v0.2.x drafts, allocator extension docs, migration
  notes, audit reports, generated PDF/HTML reports, and font assets.
- Local generated artifacts: `outputs*`, `logs`, `.hf_cache`, `.pytest_cache`,
  `__pycache__`, `nohup.out`, `*.pid`, `*.log`, and temporary files.

## Verification

Completed checks:

```bash
PYTHONPATH=src pytest tests -q
PYTHONPATH=src python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml --dry-run
DRY_RUN=1 bash scripts/run_dico_cd_da.sh --override training.max_steps=1
for script in scripts/*.sh; do bash -n "$script"; done
```

Result: `78 passed`.

## Recommended Commands

```bash
DRY_RUN=1 bash scripts/run_lora_r8.sh
DRY_RUN=1 bash scripts/run_gora_bw.sh
DRY_RUN=1 bash scripts/run_dico_cd.sh
DRY_RUN=1 bash scripts/run_dico_cd_da.sh

bash scripts/run_lora_r8.sh
bash scripts/run_gora_bw.sh
bash scripts/run_dico_cd.sh
bash scripts/run_dico_cd_da.sh
bash scripts/run_mixed_math_code.sh
```

## Manual Risks

- Full training was not run in this cleanup pass because the environment lacks
  `transformers`; dry-run/config/test validation passed.
- The historical local experiment outputs were deleted by request and are not
  archived in this repository.
- `DiCo_v0.3_AAAI方法论中文初稿.md` is retained as the methodology draft; future
  paper edits should continue to match the implemented sketch-domain profile,
  squared nonnegative coverage, procurement, and DA-Init behavior.
