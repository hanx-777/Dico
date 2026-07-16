# E00 Readiness

- status: `READY_DRY_RUN`
- failed_checks: `0`
- requires_real_gpu_execution: `True`

| check | status | message |
|---|---|---|
| protocol_preflight | PASS | formal configs passed protocol preflight |
| data_files | PASS | training/evaluation data files are present and hashed |
| model_reference | PASS | model reference meta-llama/Llama-3.1-8B-Base looks like a HuggingFace id; download/access must still be confirmed by E00 |
| dependency_versions | WARN | missing runtime packages recorded as warning for local dry-run: ['transformers', 'accelerate', 'datasets'] |
| output_dir | PASS | output directory is writable with 173.07 GB free |
| baseline_statuses | PASS | required local E00/E01 baselines are implemented-not-GPU-run |
| platform_launcher | PASS | platform_train.py exists |
| single_file_platform_launcher | PASS | launch_covra.py exists |
| platform_launcher_dry_run | PASS | platform_train dry-run generated 4 configs × 3 seeds with global batch 64 |
| shell_wrapper_entrypoints | PASS | 18 shell wrapper entrypoints exist, parse, and reference existing files |
| ddp_fallback_launcher | PASS | run_ddp.sh exists |
| ddp_fallback_protocol | PASS | run_ddp.sh fallback protocol is 3 GPUs × per-device batch 3 × grad accum 7 = global batch 63 |
| gpu_count | PASS | visible CUDA GPU count 0 satisfies required 0 |
