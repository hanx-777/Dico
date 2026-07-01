# DiCo Rank Experiments

本仓库是一个独立的 DiCo/LoRA rank 分配实验框架，用于在 GSM8K 数学推理任务上比较 uniform LoRA、训练前 DiCo 预分配、训练中动态分配，以及二者组合后的 rank 策略。

当前代码版本的重点是 **预算受控的 LoRA rank allocation**：在相同或近似相同的可训练参数预算下，比较不同 rank 分配策略对最终 GSM8K exact-match accuracy 和 eval loss 的影响。默认实验使用本地 Qwen3-8B、项目内置 GSM8K JSONL 文件，并在训练结束后统一做 final loss 与 full GSM8K generation accuracy 评估。

## 核心功能

- **Masked LoRA 注入**：对目标线性层注入最大 rank 的 LoRA 参数，并通过 rank mask 控制每个模块实际激活 rank。
- **预算公平约束**：以真实 LoRA 参数量为预算，而不是只看平均 rank；输出 `budget.json` 记录目标预算、实际预算和误差。
- **DiCo-Pre 预分配**：训练前基于校准样本构建 direction atom 证据，生成一次性的 module-level rank pattern。
- **SVD Atom 模式**：通过 streaming randomized sketch 提取 rank-one direction atoms，结合 signed profile、coverage greedy、weighted log aggregation 和 cost-aware allocation。
- **DiCo-Dynamic 动态调整**：训练过程中在指定进度点移动部分 rank，保持总预算约束。
- **DiCo-PreDynamic**：以 DiCo-Pre rank pattern 初始化，再在训练中做较小幅度动态调整。
- **结构化日志与审计**：每个实验写出 JSON/JSONL/CSV，提供 summary、post-hoc evaluation 和 audit 工具。

## 实验矩阵

默认比较 8 组实验：

| 方法 | r=4 | r=8 | 初始化 | 训练中动态调整 |
| --- | --- | --- | --- | --- |
| LoRA baseline | `lora_r4` | `lora_r8` | uniform | 否 |
| DiCo-Pre | `dico_pre_r4` | `dico_pre_r8` | DiCo preallocation | 否 |
| DiCo-Dynamic | `dico_dynamic_r4` | `dico_dynamic_r8` | uniform | 是，`move_ratio=0.20` |
| DiCo-PreDynamic | `dico_predynamic_r4` | `dico_predynamic_r8` | DiCo preallocation | 是，`move_ratio=0.10` |

配置文件位于：

```text
configs/base.yaml
configs/experiments/*.yaml
configs/debug/*.yaml
```

## 方法概览

DiCo 的基本想法是：LoRA 的每一个 rank 对应一个 rank-one 更新方向，因此 rank 分配不应只在“模块”粒度上判断重要性，也应观察模块内部不同 direction atom 的任务响应差异。

当前实现中的 DiCo-Pre 流程如下：

1. 在冻结基座模型上使用校准样本收集一阶响应信号。
2. 对每个目标模块用 streaming randomized sketch 近似提取 top-K SVD direction atoms。
3. 为 atom 构造 signed sample profile，并估计梯度冲突程度。
4. 用 coverage greedy 选择不冗余、冲突较低、效用较高的 atom evidence。
5. 将 selected atoms 的 utility 聚合为 module-level direction demand。
6. 通过 cost-aware soft allocation 和 budget-aware rounding 生成整数 rank pattern。

默认关键配置：

```yaml
preallocation:
  atom_mode: svd
  allocation_method: coverage_evidence_weighted
  aggregation_mode: weighted_log
  top_k_atoms: 8
  sketch_dim: 32
  answer_only: true
  profile_norm_mode: streaming_estimate
  eta: 0.95
  rounding_method: budget_aware_next_atom
  atom_weight_normalization: none
  use_cost_aware_allocation: true
```

更完整的方法草稿见 [v0.2.6.md](v0.2.6.md)。

## 目录结构

```text
dico_rank_experiments/
├── configs/
│   ├── base.yaml
│   ├── debug/
│   └── experiments/
├── data/
│   └── gsm8k/main/
├── scripts/
│   ├── run_experiment.py
│   ├── build_preallocation.py
│   ├── run_all_8.sh
│   ├── evaluate_experiment.py
│   ├── summarize_results.py
│   └── audit_outputs.py
├── src/dico_rank/
│   ├── atom_svd.py
│   ├── preallocation.py
│   ├── rank_budget.py
│   ├── dynamic_allocation.py
│   ├── lora_masked.py
│   ├── trainer.py
│   └── evaluator.py
├── tests/
├── AUDIT.md
├── v0.2.5.md
└── v0.2.6.md
```

主要模块说明：

| 文件 | 作用 |
| --- | --- |
| `src/dico_rank/trainer.py` | 单个实验的主训练流程，包含 preallocation 加载/构建、LoRA 注入、训练、最终评估和文件落盘 |
| `src/dico_rank/lora_masked.py` | Masked LoRA 层、rank mask、梯度 masking、inactive 参数恢复 |
| `src/dico_rank/preallocation.py` | DiCo 预分配主逻辑，兼容 SVD atom 与 module proxy fallback |
| `src/dico_rank/atom_svd.py` | SVD atom 提取、signed profile、coverage greedy、utility aggregation |
| `src/dico_rank/rank_budget.py` | LoRA 参数预算计算、预算修复、weighted allocation、rounding |
| `src/dico_rank/dynamic_allocation.py` | 训练中 rank 动态调整 |
| `src/dico_rank/evaluator.py` | eval loss 与 GSM8K exact-match generation accuracy |
| `src/dico_rank/data.py` | GSM8K JSONL 读取、prompt 格式化、tokenization、tiny dataset |

## 环境准备

推荐服务器路径：

```text
项目目录: /ai/lxw/lxw/dico_rank_experiments
模型目录: /ai/lxw/lxw/Qwen3-8B
训练数据: data/gsm8k/main/train.jsonl
测试数据: data/gsm8k/main/test.jsonl
输出目录: outputs
```

创建环境：

```bash
cd /ai/lxw/lxw/dico_rank_experiments
conda create -n dico-rank python=3.10 -y
conda activate dico-rank
```

安装依赖。下面以 CUDA 12.1 为例，实际服务器请按本机 CUDA 版本选择 PyTorch wheel：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

检查环境：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers, yaml; print('transformers', transformers.__version__)"
```

项目内已经包含 GSM8K JSONL：

```text
data/gsm8k/main/train.jsonl  # 7473 examples
data/gsm8k/main/test.jsonl   # 1319 examples
```

默认仍需要本地模型目录存在：

```bash
test -d /ai/lxw/lxw/Qwen3-8B
wc -l data/gsm8k/main/train.jsonl data/gsm8k/main/test.jsonl
```

## Hugging Face 镜像与缓存

统一脚本默认启用镜像和项目内缓存：

```text
HF_ENDPOINT=https://hf-mirror.com
HF_HOME=$PWD/.hf_cache
HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
HF_DATASETS_CACHE=$HF_HOME/datasets
TRANSFORMERS_CACHE=$HF_HOME/transformers
```

手动启用：

```bash
source scripts/env_hf_mirror.sh
```

如果模型和数据都在本地，镜像通常只是备用。运行 8 组实验时可禁用镜像：

```bash
bash scripts/run_all_8.sh --no_hf_mirror
```

## 快速验证

tiny 配置不需要 Qwen3-8B，也不需要 CUDA，适合先检查代码路径：

```bash
python scripts/run_experiment.py --config configs/debug/tiny_lora.yaml
python scripts/run_experiment.py --config configs/debug/tiny_dico_dynamic.yaml
```

运行测试：

```bash
pytest -q
```

如果需要只跑正式配置的极短 smoke test：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/lora_r4.yaml \
  --override training.max_steps=2 \
  --override evaluation.accuracy_max_samples=8
```

## 运行单个实验

例如运行 LoRA r=4：

```bash
python scripts/run_experiment.py --config configs/experiments/lora_r4.yaml
```

运行 DiCo-Pre r=4：

```bash
python scripts/run_experiment.py --config configs/experiments/dico_pre_r4.yaml
```

常用 override 示例：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/dico_pre_r4.yaml \
  --override project.output_dir=outputs_single_test \
  --override calibration.save_dir=outputs_single_test/preallocations \
  --override model.name_or_path=/ai/lxw/lxw/Qwen3-8B \
  --override training.max_steps=100 \
  --override evaluation.accuracy_max_samples=128
```

只构建或复用 DiCo preallocation cache：

```bash
python scripts/build_preallocation.py --config configs/experiments/dico_pre_r4.yaml
```

## 一次运行 8 组实验

前台运行：

```bash
bash scripts/run_all_8.sh
```

短跑 8 组：

```bash
bash scripts/run_all_8.sh \
  --output_dir outputs_smoke \
  --override training.max_steps=2 \
  --override evaluation.accuracy_max_samples=8
```

服务器后台运行：

```bash
bash scripts/run_all_8.sh --nohup \
  --output_dir outputs_bs4_full \
  --log_dir outputs_bs4_full/logs \
  --override model.name_or_path=/ai/lxw/lxw/Qwen3-8B \
  --override model.torch_dtype=bfloat16 \
  --override training.batch_size=4 \
  --override training.gradient_accumulation_steps=2 \
  --override calibration.batch_size=4
```

`--output_dir DIR` 会自动设置：

```text
project.output_dir=DIR
calibration.save_dir=DIR/preallocations
```

除非你显式传入 `--override calibration.save_dir=...`。

## 查看运行状态

查看后台 PID：

```bash
cat outputs_bs4_full/run_all_8.pid
ps -p "$(cat outputs_bs4_full/run_all_8.pid)" -o pid,etime,command
```

查看日志和 GPU：

```bash
tail -f outputs_bs4_full/logs/run_all_8_*.log
nvidia-smi
```

训练日志中的常见事件：

```text
experiment_start
data_loaded
lora_injected
train_step
dynamic_adjustment
training_complete
final_loss_start
final_loss_complete
final_accuracy_start
final_accuracy_progress
final_accuracy_complete
experiment_complete
```

## 输出文件

每个实验目录通常包含：

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

重点文件：

| 文件 | 含义 |
| --- | --- |
| `config_resolved.yaml` | 合并继承与 override 后的最终配置 |
| `metrics.json` | 最终 loss、GSM8K accuracy、预算、rank 和评估协议 |
| `eval_predictions.jsonl` | 每条测试样本的生成文本、抽取答案和对错 |
| `rank_allocation_initial.json` | 初始 rank pattern |
| `rank_allocation_final.json` | 训练结束后的最终 rank pattern |
| `rank_history.csv` | rank 随训练过程变化的历史 |
| `budget.json` | target budget、actual budget、budget error |
| `dynamic_adjustments.jsonl` | dynamic 方法的 rank 移动记录 |
| `masked_lora_state.pt` | 已训练的 LoRA 参数状态 |

DiCo-Pre 还会在 preallocation cache 目录写出：

```text
outputs/preallocations/dico_pre_rank4_seed42.json
outputs/preallocations/dico_pre_rank4_seed42_atom_logs.jsonl
outputs/preallocations/dico_pre_rank8_seed42.json
outputs/preallocations/dico_pre_rank8_seed42_atom_logs.jsonl
```

`atom_mode=svd` 时，完整 signed profile 可能保存为 `*_profiles.pt`；JSONL 中只保存 profile 摘要和索引，避免日志文件过大。

## 评估协议

默认配置：

```yaml
evaluation:
  metric: gsm8k_accuracy
  protocol: internal_zero_shot
  prompt_style: sft_cot_hash
  answer_extraction: strict_then_flexible
  max_batches: 4
  compute_accuracy: true
  accuracy_during_training: false
  accuracy_max_samples: null
  generation_max_new_tokens: 256
```

说明：

- 训练过程中默认不做 generation accuracy。
- 训练结束后统一计算 final eval loss 和 GSM8K generation accuracy。
- `max_batches` 只影响 final loss 的 batch 数。
- `accuracy_max_samples: null` 表示评估完整本地 GSM8K test set，即 1319 条。
- 当前 accuracy 是项目内部 zero-shot GSM8K exact-match，使用 SFT-style prompt，不等价于 `lm-evaluation-harness` 的 8-shot CoT leaderboard 分数。

## 汇总、审计与补评估

生成结果汇总：

```bash
python scripts/summarize_results.py --output_dir outputs_bs4_full
```

输出：

```text
outputs_bs4_full/summary.csv
outputs_bs4_full/summary.md
```

生成审计报告：

```bash
python scripts/audit_outputs.py --output_dir outputs_bs4_full
```

输出：

```text
outputs_bs4_full/audit_report.md
outputs_bs4_full/audit_report.json
```

查看某个实验最终 rank：

```bash
python scripts/inspect_rank_logs.py \
  --rank_history outputs_bs4_full/dico_dynamic_r4/rank_history.csv
```

如果旧实验目录已有 checkpoint 和 rank allocation，但缺少 accuracy，可做 post-hoc evaluation：

```bash
python scripts/evaluate_experiment.py --experiment_dir outputs_bs4_full/lora_r4
python scripts/summarize_results.py --output_dir outputs_bs4_full
```

## 实验完成检查清单

跑完 8 组后建议至少检查：

1. `summary.md` 中 8 个实验是否都出现。
2. 每个 `metrics.json` 是否包含 `final_eval_accuracy`、`final_exact_match`、`eval_correct`、`eval_total`。
3. 如果评估完整测试集，`eval_total` 是否为 1319。
4. 每个 `budget.json` 是否满足 `actual_budget <= target_budget`。
5. `budget_error_ratio` 是否小于或等于 `0.01`。
6. DiCo-Pre / DiCo-PreDynamic 是否记录 `aggregation_mode=weighted_log`、`atom_weight_normalization=none`、`use_cost_aware_allocation=true`。
7. DiCo-Dynamic / DiCo-PreDynamic 是否有 `dynamic_adjustments.jsonl`。
8. `audit_report.md` 是否没有 Critical 问题。

公平比较时按相同 reference rank 比较：

```text
lora_r4 vs dico_pre_r4 vs dico_dynamic_r4 vs dico_predynamic_r4
lora_r8 vs dico_pre_r8 vs dico_dynamic_r8 vs dico_predynamic_r8
```

## OOM 与资源调整

优先降低 calibration batch：

```bash
--override calibration.batch_size=2
```

再降低训练 batch，并用梯度累积保持有效 batch size：

```bash
--override training.batch_size=2
--override training.gradient_accumulation_steps=4
```

必要时降低最大长度：

```bash
--override data.max_length=384
```

最后再考虑量化加载：

```bash
--override model.load_in_8bit=true
--override model.load_in_4bit=false
```

或 4-bit：

```bash
--override model.load_in_8bit=false
--override model.load_in_4bit=true
```

不要同时设置 `model.load_in_8bit=true` 和 `model.load_in_4bit=true`。

## 中断和重跑

如果后台任务还在运行：

```bash
cat outputs_bs4_full/run_all_8.pid
kill "$(cat outputs_bs4_full/run_all_8.pid)"
```

正式实验建议每次使用新的输出目录：

```bash
bash scripts/run_all_8.sh --nohup --output_dir outputs_$(date +%Y%m%d_%H%M%S)
```

如果复用同一个实验目录，当前实验的日志和结果文件可能被覆盖。为了避免混入旧结果，正式对比建议新建输出目录。

## 推荐命令

第一次短跑：

```bash
cd /ai/lxw/lxw/dico_rank_experiments
conda activate dico-rank
python scripts/run_experiment.py --config configs/debug/tiny_lora.yaml
bash scripts/run_all_8.sh \
  --output_dir outputs_smoke \
  --override training.max_steps=2 \
  --override evaluation.accuracy_max_samples=8
```

正式后台跑：

```bash
cd /ai/lxw/lxw/dico_rank_experiments
conda activate dico-rank
bash scripts/run_all_8.sh --nohup \
  --output_dir outputs_bs4_full \
  --log_dir outputs_bs4_full/logs \
  --override model.name_or_path=/ai/lxw/lxw/Qwen3-8B \
  --override training.batch_size=4 \
  --override training.gradient_accumulation_steps=2 \
  --override calibration.batch_size=4
```

汇总和审计：

```bash
python scripts/summarize_results.py --output_dir outputs_bs4_full
python scripts/audit_outputs.py --output_dir outputs_bs4_full
cat outputs_bs4_full/summary.md
```

## 备注

- `AUDIT.md` 是 A800 服务器实验审计模板。
- `v0.2.5.md` 和 `v0.2.6.md` 是方法说明草稿，不是运行入口。
- `outputs/`、`outputs_old_*` 和 `*.pid` 属于实验产物，复现实验时应以新的输出目录为准。
