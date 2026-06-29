# DiCo Rank Experiments

Standalone experiment framework for comparing LoRA rank allocation methods without modifying `dico_lite`.

This project is intended to run on the A800 Linux server. Local development in this thread only wrote code, tests, docs, and audit tools; model training, smoke tests, and pytest should be run on the A800 environment.

## Experiments

| Method | r=4 | r=8 |
| --- | --- | --- |
| LoRA | `lora_r4` | `lora_r8` |
| DiCo-Pre | `dico_pre_r4` | `dico_pre_r8` |
| DiCo-Dynamic | `dico_dynamic_r4` | `dico_dynamic_r8` |
| DiCo-PreDynamic | `dico_predynamic_r4` | `dico_predynamic_r8` |

## Method Definitions

`LoRA` is the uniform baseline. Every target module starts with the same active rank and rank does not change during training.

`DiCo-Pre` builds a preallocation before training, initializes active ranks from that allocation, and keeps rank fixed during training.

`DiCo-Dynamic` starts from uniform LoRA. During training it adjusts rank at 20%, 40%, and 60% of total steps using gradient/update scores. Its `move_ratio` is `0.20`.

`DiCo-PreDynamic` starts from the same shared DiCo-Pre preallocation used by DiCo-Pre for the same rank. It adjusts rank at 20%, 40%, and 60%, but only with `move_ratio: 0.10`.

After 60% of training, dynamic ranks are fixed to reduce late-training instability.

## A800 Environment Setup

Create the environment on the A800 server:

```bash
cd /ai/lxw/lxw/dico_rank_experiments
conda create -n dico-rank python=3.10 -y
conda activate dico-rank
```

Install PyTorch for the CUDA version available on the server. Check CUDA first:

```bash
nvidia-smi
```

Example for CUDA 12.1 wheels:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Sanity check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python -c "import transformers, yaml; print('transformers', transformers.__version__)"
df -h .
test -w outputs || mkdir -p outputs
```

## Hugging Face Mirror On A800

For mainland China or unstable direct Hugging Face access, the unified run script enables a Hugging Face mirror by default. To inspect or pre-load the same environment manually:

```bash
source scripts/env_hf_mirror.sh
echo "$HF_ENDPOINT"
echo "$HF_HOME"
```

By default it sets:

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=$PWD/.hf_cache
HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
HF_DATASETS_CACHE=$HF_HOME/datasets
TRANSFORMERS_CACHE=$HF_HOME/transformers
```

Existing environment variables are preserved. You can override cache or endpoint before running:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/ai/lxw/hf_cache
```

You can also pass script flags:

```bash
bash scripts/run_all_8.sh --hf_endpoint https://hf-mirror.com --help
bash scripts/run_all_8.sh --no_hf_mirror --help
```

The default configs use the local GSM8K files vendored in this project, so the
dataset does not need to be downloaded from Hugging Face during normal A800
runs. The mirror is still useful for model/tokenizer files if you point
`model.name_or_path` at a remote repo, but the recommended A800 path is a local
model directory plus the project-local dataset.

## Model And Data Configuration

The default A800 config already points at the expected local model:

```yaml
model:
  name_or_path: /ai/lxw/lxw/Qwen3-8B
```

So the normal single-experiment command does not need a model override:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/lora_r4.yaml
```

By default, full experiments use the project-local GSM8K JSONL files:

```text
data/gsm8k/main/train.jsonl
data/gsm8k/main/test.jsonl
```

These files contain the original GSM8K `question` and `answer` fields, with
7473 train examples and 1319 test examples. `configs/base.yaml` stores these as
relative paths, and the loader resolves them against the project root, so runs
do not need to contact Hugging Face for the dataset.

Optional: to use a different model or custom dataset, override the local paths
explicitly. These example paths are placeholders for custom runs, not the
default A800 setup:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/lora_r4.yaml \
  --override model.name_or_path=/path/to/local/model \
  --override data.train_path=/data/gsm8k_train.jsonl \
  --override data.eval_path=/data/gsm8k_test.jsonl
```

## Path Checklist

On the A800 server, the expected paths are:

```text
project_dir:       /ai/lxw/lxw/dico_rank_experiments
model.name_or_path:/ai/lxw/lxw/Qwen3-8B
train_path:        data/gsm8k/main/train.jsonl
eval_path:         data/gsm8k/main/test.jsonl
output_dir:        outputs
preallocation_dir: outputs/preallocations
nohup_pid:         outputs/run_all_8.pid
nohup_logs:        outputs/logs/run_all_8_*.log
hf_cache:          .hf_cache
```

Check them before a long run:

```bash
cd /ai/lxw/lxw/dico_rank_experiments
pwd
test -d /ai/lxw/lxw/Qwen3-8B
wc -l data/gsm8k/main/train.jsonl data/gsm8k/main/test.jsonl
test -w outputs || mkdir -p outputs
```

All Python entrypoints resolve relative CLI paths against the project root. For
example, `--output_dir outputs`, `--experiment_dir outputs/lora_r4`, and
`--rank_history outputs/dico_dynamic_r4/rank_history.csv` point inside
`/ai/lxw/lxw/dico_rank_experiments` even if the command is launched from
another working directory. Absolute paths are preserved.

## Debug Smoke Tests

Run these first on the A800 server. They use a tiny CPU-compatible model and do not require Qwen, CUDA, downloads, or external datasets:

```bash
python scripts/run_experiment.py --config configs/debug/tiny_lora.yaml
python scripts/run_experiment.py --config configs/debug/tiny_dico_dynamic.yaml
```

Then run unit tests:

```bash
pytest -q
```

## Run Experiments

Single experiment:

```bash
python scripts/run_experiment.py --config configs/experiments/lora_r4.yaml
```

Short dry sanity run of a real config:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/lora_r4.yaml \
  --override training.max_steps=2
```

All eight, foreground/debug mode:

```bash
bash scripts/run_all_8.sh
```

Short all-eight debug run:

```bash
bash scripts/run_all_8.sh --override training.max_steps=2
```

Server production run with `nohup` using the paths already set in
`configs/base.yaml`:

```bash
bash scripts/run_all_8.sh --nohup
```

The default A800 config is tuned for roughly 50GB available GPU memory:

```yaml
model:
  torch_dtype: bfloat16
  load_in_8bit: false
  load_in_4bit: false
training:
  batch_size: 8
  gradient_accumulation_steps: 1
```

If you want to override the model path or force a different precision mode at
launch time:

```bash
bash scripts/run_all_8.sh --nohup \
  --override model.name_or_path=/ai/lxw/lxw/Qwen3-8B \
  --override model.torch_dtype=bfloat16 \
  --override training.batch_size=8
```

The script prints the PID and log path. Check progress with:

```bash
cat outputs/run_all_8.pid
tail -f outputs/logs/run_all_8_*.log
nvidia-smi
```

## Single A800 Vs 4 vGPU A800

The project can run on both, but they are not equivalent from a memory and
placement perspective.

A single physical A800 80GB exposes one CUDA device with about 80GB memory. It
is the simpler and usually faster setup: `device_map=auto` will normally place
the model on one GPU, there is no cross-device model traffic, and OOM debugging
is easier.

The 4 vGPU setup exposes four CUDA devices, but each device may have only about
19.5GB visible memory. The total visible memory is similar, but it is split
across devices. With `device_map=auto`, Transformers may shard the model across
the four vGPUs. This can work for Qwen3-8B, but it may be slower than one 80GB
device because activations and hidden states can move between virtual GPUs.

Recommended physical A800 setting when about 50GB is available:

```yaml
model:
  device_map: auto
  load_in_8bit: false
  load_in_4bit: false
  torch_dtype: bfloat16
training:
  batch_size: 8
  gradient_accumulation_steps: 1
data:
  max_length: 512
```

If a constrained vGPU or smaller-memory run OOMs, first lower
`training.batch_size`, then lower `data.max_length` to `384`. If placement still
fails, enable 8-bit or 4-bit loading as a fallback.

Optional: choose output and log directories explicitly:

```bash
bash scripts/run_all_8.sh --nohup \
  --output_dir outputs_qwen7b \
  --log_dir outputs_qwen7b/logs \
  --override model.name_or_path=/ai/lxw/lxw/Qwen3-8B \
  --override training.batch_size=8
```

When `--output_dir DIR` is used, the script also sets
`calibration.save_dir=DIR/preallocations` unless you explicitly pass your own
`--override calibration.save_dir=...`. This keeps preallocation caches isolated
between server runs.

Compatibility wrappers remain available, but new commands should prefer `run_all_8.sh`:

```bash
bash scripts/run_all_8_experiments.sh
bash scripts/run_all_8_nohup.sh
```

## A800 OOM Controls

If an A800 run OOMs, reduce batch size first while keeping the same effective
batch size if desired:

```bash
--override training.batch_size=4
--override training.gradient_accumulation_steps=2
--override data.max_length=384
```

The default config uses bf16 without bitsandbytes quantization. For a larger
base model or smaller GPU slice that still exceeds memory, enable bitsandbytes
8-bit loading:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/lora_r4.yaml \
  --override model.name_or_path=/ai/lxw/lxw/Qwen3-8B \
  --override model.load_in_8bit=true \
  --override model.load_in_4bit=false \
  --override model.torch_dtype=bfloat16
```

For a more aggressive fallback, use 4-bit NF4 loading:

```bash
python scripts/run_experiment.py \
  --config configs/experiments/lora_r4.yaml \
  --override model.name_or_path=/ai/lxw/lxw/Qwen3-8B \
  --override model.load_in_8bit=false \
  --override model.load_in_4bit=true \
  --override model.torch_dtype=bfloat16
```

Do not set `model.load_in_8bit=true` and `model.load_in_4bit=true` at the same time. This is bitsandbytes 8-bit/4-bit loading, not FP8 training. The base model is quantized for memory savings, while custom LoRA parameters remain floating point and use `model.torch_dtype`.

Keep output directories isolated by `experiment_name`. Do not reuse an output directory for different method/rank settings unless you intentionally want to overwrite or inspect stale artifacts.

Current implementation uses custom `MaskedLoRALinear` rather than PEFT dynamic rank internals. `requirements.txt` includes common ML dependencies, but dynamic rank logic does not depend on PEFT.

## Why Max Rank Plus Mask

Dynamic rank is implemented with a custom `MaskedLoRALinear`. Each target linear-like module is replaced exactly by module name, including standard `nn.Linear` and bitsandbytes quantized linear layers that expose `in_features`, `out_features`, and `weight`. The base linear weights are frozen, and only LoRA parameters are trainable.

For a base rank `r`, the layer allocates `max_rank = r * 2` channels once at injection time:

```text
lora_A: [max_rank, in_features]
lora_B: [out_features, max_rank]
rank_mask: [max_rank]
```

Inactive channels have zero contribution in forward. Their gradients are masked to zero before every optimizer step. The trainer restores inactive channel parameters after the optimizer step, so decoupled weight decay cannot move frozen inactive channels. LoRA optimizer groups default to `weight_decay=0.0`.

Newly activated rank channels are pre-initialized at LoRA injection time and remain frozen while inactive. When activated, they start from their original initialized values. Optimizer state is not reset in this first version.

## Fairness

Fairness is measured by active LoRA parameters, not physical `max_rank` capacity:

```text
active_params(module) = active_rank * (in_dim + out_dim)
```

Every initial and preallocation allocation is repaired so:

```text
actual_active_lora_params <= target_budget
```

The global repair step maximizes feasible active LoRA params without exceeding
the target budget, so it minimizes `target_budget - actual_budget` under rank
bounds. Prefer `budget_error_ratio <= 0.01`. If the ratio is above 1%,
`budget.json` records a warning. If rank lower bounds make the target
impossible, `budget.json` records `over_budget: true` and an explicit warning.

Dynamic updates are stricter: each adjustment is bounded by rank L1 distance,
`sum(abs(new_rank - old_rank)) <= 2 * move_budget`. Dynamic repair may reduce
rank to avoid exceeding the active-parameter budget, but it does not add extra
rank merely to fill unused budget after a dynamic move.

## Weighted Atom-to-Module Aggregation

Early count-based DiCo aggregation treated every selected atom as an equal unit: one selected atom gave its module `+1` rank. The default preallocation path now uses:

```yaml
preallocation:
  aggregation_mode: weighted_topk
  weighted_topk_k: auto
  atom_weight_normalization: none
  use_cost_aware_allocation: true
  atom_utility_floor: 0.0
```

Each atom contributes a continuous utility:

```text
utility = max(atom_utility_floor, importance - coverage_lambda * redundancy)
```

`weighted_topk` aggregates each module by summing its strongest top-k atom utilities. With `weighted_topk_k: auto`, `k = rank * r_max_multiplier`. The default `atom_weight_normalization: none` preserves cross-module magnitude differences.

The module utilities are converted to integer ranks under the active-parameter budget with cost-aware scoring, continuous rank allocation, floor + largest remainder, greedy budget fill, and final budget repair.

`aggregation_mode: count` is retained only for ablation and backward comparison. It is not the default main method.

## Atom Mode Limitation

The current implementation records `atom_mode: module_proxy` when using module-level proxy scores. It does not pretend module-level proxy allocation is true SVD/rank-one atom allocation.

With `atom_mode: module_proxy`, weighted aggregation does **not** truly distinguish SVD/rank-one atom importance inside a module. It only provides a proxy-compatible weighted allocation pipeline based on module-level scores. True atom-level weighting requires `atom_mode: svd`, with real singular values, response norms, and atom profiles.

## Outputs

Each experiment writes:

```text
outputs/<experiment>/
├── config_resolved.yaml
├── metrics.json
├── train_log.jsonl
├── eval_log.jsonl
├── eval_predictions.jsonl
├── rank_allocation_initial.json
├── rank_allocation_final.json
├── rank_history.csv
├── budget.json
├── dynamic_adjustments.jsonl
└── masked_lora_state.pt
```

`eval_log.jsonl` records periodic validation loss during training. Final
evaluation also computes GSM8K generation exact-match accuracy by default and
writes `eval_predictions.jsonl` with question, gold answer, generated answer,
extracted final numbers, and correctness.

`metrics.json` includes `final_eval_loss`, `final_eval_accuracy`,
`final_exact_match`, `eval_correct`, `eval_total`, `eval_sample_count`,
`evaluation_protocol`, `evaluation_prompt_style`, `answer_extraction`, and
`final_metric_name`.
By default `evaluation.metric: gsm8k_accuracy`, so `final_metric` is the final
generation accuracy, not loss.

This project reports **internal zero-shot GSM8K exact-match accuracy** using the
same SFT-style prompt as training. It is not `lm-evaluation-harness` 8-shot CoT
accuracy and should not be used as a public leaderboard number without a
separate lm-eval-compatible evaluation.

For DiCo-Pre and DiCo-PreDynamic, `rank_allocation_initial.json` contains `rank_allocation`, `module_logs`, `aggregation_mode`, `atom_weight_normalization`, `use_cost_aware_allocation`, `atom_mode`, and `budget_error_ratio`.

`rank_history.csv` includes:

```csv
step,module_name,active_rank,max_rank,module_score,total_active_rank,total_active_params,target_budget,budget_error_ratio,rank_distance_from_initial,rank_distance_from_preallocation
```

For `dico_predynamic`, `rank_distance_from_preallocation` shows how far dynamic correction moved from the shared DiCo-Pre allocation. For `dico_dynamic`, that field is empty because it starts from uniform.

## Summary And Audit

Generate summary:

```bash
python scripts/summarize_results.py --output_dir outputs
```

If an older run has `metrics.json` but no accuracy fields, run post-hoc
evaluation for that experiment directory:

```bash
python scripts/evaluate_experiment.py --experiment_dir outputs/lora_r4
python scripts/summarize_results.py --output_dir outputs
```

Inspect final ranks:

```bash
python scripts/inspect_rank_logs.py --rank_history outputs/dico_dynamic_r4/rank_history.csv
```

Generate audit report:

```bash
python scripts/audit_outputs.py --output_dir outputs
```

This writes:

```text
outputs/audit_report.md
outputs/audit_report.json
```

The audit script exits non-zero for critical structural failures. Warnings such as `budget_error_ratio > 0.01` are recorded in the report.

## Audit Checklist

After running experiments on A800, check:

1. `config_resolved.yaml`: method, rank, dynamic settings, `move_ratio`, weighted preallocation fields.
2. `budget.json`: `actual_budget <= target_budget`, preferably `budget_error_ratio <= 0.01`.
3. `rank_allocation_initial.json`: uniform init for DiCo-Dynamic; weighted preallocation metadata for DiCo-Pre and DiCo-PreDynamic.
4. `rank_history.csv`: active ranks, active params, and distance fields are populated.
5. `dynamic_adjustments.jsonl`: only DiCo-Dynamic and DiCo-PreDynamic should contain dynamic adjustments; steps should match configured 20/40/60% thresholds.
6. `metrics.json`: includes method/rank/budget fields, `final_eval_accuracy`, and preallocation metadata for preallocation methods.
7. `outputs/preallocations/*.json`: shared rank-4/rank-8 preallocation files exist for DiCo-Pre and DiCo-PreDynamic reuse.
8. `summary.csv` and `summary.md`: all eight experiments appear.

Fair comparison requires comparing methods at the same rank with aligned active LoRA params. Treat any budget warning as a paper/report caveat.

For result interpretation, also check `evaluation_protocol` and
`eval_sample_count`. The default A800 config uses the full local GSM8K test set
of 1319 examples because `data.eval_limit=null` and
`evaluation.accuracy_max_samples=null`.

## Known Risks For A800 Runs

Current preallocation is `module_proxy` fallback unless true SVD atom mode is implemented later.

Real 8-run reproducibility depends on the A800 server CUDA/PyTorch/transformers stack, local model path, dataset availability, and output directory cleanliness.

This repository was prepared without running local tests in this thread. Run debug smoke tests and `pytest -q` on A800 before launching long experiments.
