# A800 Audit Template

Use this file after running experiments on the A800 server. The automated companion command is:

```bash
python scripts/audit_outputs.py --output_dir outputs
```

## Environment

- Date:
- Host:
- GPU model:
- Driver:
- CUDA:
- Python:
- PyTorch:
- Transformers:
- Conda env:
- Project path:
- Model path:
- Data path:
- Local train data exists:
- Local eval data exists:
- HF_ENDPOINT:
- HF_HOME:
- HUGGINGFACE_HUB_CACHE:
- HF_DATASETS_CACHE:
- Nohup PID:
- Nohup log path:
- Git commit or archive hash:
- Worktree dirty state:

Expected A800 paths:

| Item | Expected Path | OK |
| --- | --- | --- |
| Project | `/ai/lxw/lxw/dico_rank_experiments` | |
| Model | `/ai/lxw/lxw/Qwen3-8B` | |
| Train data | `data/gsm8k/main/train.jsonl` | |
| Eval data | `data/gsm8k/main/test.jsonl` | |
| Output dir | `outputs` | |
| Preallocation cache | `outputs/preallocations` | |
| Nohup PID | `outputs/run_all_8.pid` | |
| Nohup logs | `outputs/logs/run_all_8_*.log` | |
| HF cache | `.hf_cache` or explicit `$HF_HOME` | |

Suggested commands:

```bash
nvidia-smi
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
echo "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
pwd
test -d /ai/lxw/lxw/Qwen3-8B
df -h .
wc -l data/gsm8k/main/train.jsonl data/gsm8k/main/test.jsonl
test -w outputs || mkdir -p outputs
cat outputs/run_all_8.pid 2>/dev/null || true
ls -lh outputs/logs/run_all_8_*.log 2>/dev/null || true
```

## Config Audit

For each of the eight experiments, inspect `outputs/<experiment>/config_resolved.yaml`.

The default A800 configs should use the project-local GSM8K files:

```yaml
data:
  train_path: data/gsm8k/main/train.jsonl
  eval_path: data/gsm8k/main/test.jsonl
```

Expected local dataset counts are 7473 train rows and 1319 test rows. If these
paths are overridden, record the replacement paths and sample counts.

Unless explicitly overridden, `outputs/` and `outputs/preallocations/` should be
inside `/ai/lxw/lxw/dico_rank_experiments`.

| Experiment | Method | Rank | Init | OK |
| --- | --- | ---: | --- | --- |
| lora_r4 | lora | 4 | uniform | |
| lora_r8 | lora | 8 | uniform | |
| dico_pre_r4 | dico_pre | 4 | dico_pre | |
| dico_pre_r8 | dico_pre | 8 | dico_pre | |

## Budget Audit

For each experiment, inspect `budget.json`.

| Experiment | Target Budget | Actual Budget | Error Ratio | Warning |
| --- | ---: | ---: | ---: | --- |
| lora_r4 | | | | |
| lora_r8 | | | | |
| dico_pre_r4 | | | | |
| dico_pre_r8 | | | | |

Acceptance preference: `actual_budget <= target_budget`, `over_budget` is absent or false, and `budget_error_ratio <= 0.01`.
If `over_budget: true` appears, rank lower bounds or stale metadata made the budget infeasible and the run should not be treated as a fair comparison without explanation.

## Rank Audit

Check:

- `rank_allocation_initial.json`
- `rank_allocation_final.json`
- `rank_history.csv`

Questions:

- Does DiCo-Pre keep final allocation equal to initial allocation?
- Do final ranks remain within `r_min` and `r_max`?

## Preallocation Audit

Check `outputs/preallocations/*.json` and DiCo-Pre `rank_allocation_initial.json`.

Required fields:

- `aggregation_mode: weighted_log` for true `atom_mode=svd` DiCo, or `weighted_topk` for the module-proxy baseline/ablation
- `atom_weight_normalization: none`
- `use_cost_aware_allocation: true`
- `atom_mode`
- `atom_mode_limitation` when `atom_mode=module_proxy`
- `profile_norm_mode` when `atom_mode=svd`
- `module_logs`
- `budget_error_ratio`

Important interpretation:

`module_proxy` is not true SVD/rank-one atom weighting. It is a proxy-compatible weighted allocation pipeline based on module-level scores. For `atom_mode=svd`, full signed profiles are stored outside JSONL, typically as `*_profiles.pt`; atom logs should contain profile summary fields and indices only.

## Result Audit

Generate:

```bash
python scripts/summarize_results.py --output_dir outputs
python scripts/audit_outputs.py --output_dir outputs
```

Check:

- `outputs/summary.csv`
- `outputs/summary.md`
- `outputs/audit_report.md`
- `outputs/audit_report.json`
- every experiment `metrics.json` includes `final_eval_accuracy`, `final_exact_match`, `eval_correct`, and `eval_total`
- every experiment `metrics.json` includes `evaluation_protocol: internal_zero_shot`
- every experiment has `eval_predictions.jsonl` when `evaluation.compute_accuracy=true`
- `eval_predictions.jsonl` row count matches `eval_total`

Interpretation note: this project reports internal zero-shot GSM8K exact-match
accuracy with the SFT-style prompt. It is not `lm-evaluation-harness` 8-shot CoT
accuracy. The default A800 config evaluates the full local GSM8K test set of
1319 examples because `data.eval_limit=null` and
`evaluation.accuracy_max_samples=null`.

All eight experiments should be present in summary and audit reports.

## Notes And Exceptions

Record any OOMs, restarts, changed overrides, missing files, or budget warnings here.
