# DiCo Rank Experiments 技术文档

版本：v0.2.7  
更新时间：2026-07-02

## v0.2.7 Changelog

- 增加预算双口径：paramcount 是公平比较主口径，ranksum 仅作辅助诊断。
- 增加 multi-seed 主实验脚本：`scripts/run_all_multiseed.sh`，默认 `SEEDS="42 43 44"`。
- 增加 LoRA eta98 baseline：`lora_r4_eta98` / `lora_r8_eta98`，active paramcount ratio 控制在 `[0.97, 0.98]`。
- 增加 r8 ablation 配置：no relaxation、eta100、answer full、random at budget、PreDynamic move20。
- 增加 evidence relaxation 顶层报告，显式记录 rank beyond selected evidence 的比例。
- 增加训练中轻量 loss-only eval：只写 `train_log.jsonl(event=mid_eval_loss)`，不做 generation，不写 `eval_log.jsonl`。
- cache compatibility 现在记录完整不兼容原因 list，同时保留旧的单字段 alias。
- `masked_lora_state.pt` 明确只保存 `lora_A`、`lora_B`、`rank_mask`，默认不保存 optimizer state。

## 1. 项目定位

`dico_rank_experiments` 是一个 LoRA rank allocation 实验框架，用于在 GSM8K 上比较：

- `lora`：uniform active rank baseline。
- `dico_pre`：训练前使用 DiCo calibration / atom evidence 进行一次 rank preallocation。
- `dico_dynamic`：从 uniform rank 开始，训练过程中按 score 移动 rank。
- `dico_predynamic`：先 DiCo preallocation，再做较小幅度 dynamic rank movement。

当前主线不是 sparse DiCo，而是 **DiCo-98 budget-fair preallocation**：

```text
0.98 * target_budget_paramcount <= actual_budget_paramcount <= target_budget_paramcount
```

对 `dico_pre` / `dico_predynamic`，`trainer.py` 不再用通用 `BudgetManager.repair(...)` 覆盖 DiCo allocator 输出。trainer 只做预算校验和 diagnostics 记录。

## 2. 预算口径

每个模块 active LoRA 参数量：

```text
P_m(r_m) = r_m * (in_dim_m + out_dim_m)
```

总预算：

```text
P(r) = sum_m P_m(r_m)
```

`target_budget_paramcount` 来自同 rank uniform LoRA。v0.2.7 起，预算输出同时包含：

- `target_budget_paramcount`
- `actual_budget_paramcount`
- `budget_ratio_paramcount`
- `target_budget_ranksum`
- `actual_budget_ranksum`
- `budget_ratio_ranksum`

兼容字段仍保留：

- `target_budget`
- `actual_budget`
- `budget_ratio`

这些兼容字段等价于 paramcount 口径。`budget_error` 是 signed error：

```text
actual_budget_paramcount - target_budget_paramcount
```

audit 的公平区间判断统一使用 `budget_ratio_paramcount`。

## 3. 默认实验矩阵

| 实验 | method | rank | 初始化 | 训练中 rank 调整 |
| --- | --- | ---: | --- | --- |
| `lora_r4` | `lora` | 4 | uniform | 否 |
| `lora_r8` | `lora` | 8 | uniform | 否 |
| `dico_pre_r4` | `dico_pre` | 4 | DiCo-98 | 否 |
| `dico_pre_r8` | `dico_pre` | 8 | DiCo-98 | 否 |
| `dico_dynamic_r4` | `dico_dynamic` | 4 | uniform | 是，`move_ratio=0.20` |
| `dico_dynamic_r8` | `dico_dynamic` | 8 | uniform | 是，`move_ratio=0.20` |
| `dico_predynamic_r4` | `dico_predynamic` | 4 | DiCo-98 | 是，`move_ratio=0.10` |
| `dico_predynamic_r8` | `dico_predynamic` | 8 | DiCo-98 | 是，`move_ratio=0.10` |

额外 baseline：

- `lora_r4_eta98`
- `lora_r8_eta98`

额外 ablation：

- `configs/experiments/ablations/dico_pre_r8_no_relaxation.yaml`
- `configs/experiments/ablations/dico_pre_r8_eta100.yaml`
- `configs/experiments/ablations/dico_pre_r8_answer_full.yaml`
- `configs/experiments/ablations/dico_pre_r8_random.yaml`
- `configs/experiments/ablations/dico_predynamic_r8_move20.yaml`

## 4. 单实验流程

入口：

```bash
python scripts/run_experiment.py --config configs/experiments/dico_pre_r8.yaml
```

主流程：

```text
load config
  -> load tokenizer/model
  -> locate target linear modules
  -> load/tokenize GSM8K
  -> compute uniform target_budget_paramcount
  -> build initial allocation
  -> inject Masked LoRA
  -> train max_steps
  -> optional mid loss-only eval
  -> save final allocation/state
  -> final loss eval
  -> final GSM8K generation accuracy
  -> write metrics/logs/audit artifacts
```

`training.eval_steps` 保留兼容旧配置，但不再触发 generation eval。v0.2.7 的 mid eval 是 loss-only，并且只进入 `train_log.jsonl`。

## 5. DiCo-Pre 语义

当前 DiCo-Pre 主线配置：

```yaml
preallocation:
  atom_mode: svd
  allocation_method: coverage_evidence_weighted
  aggregation_mode: weighted_log
  eta: 0.98
  allow_rank_beyond_selected_evidence: true
  rounding_method: budget_aware_next_atom
  use_soft_tail: true
  lambda_next: 1.0
  use_cost_aware_allocation: true
```

关键点：

- DiCo allocator 自己负责预算公平分配。
- `rank_allocation_initial.json` 保存 DiCo allocator 原始输出。
- 无 dynamic 的 `dico_pre`，`rank_allocation_final.json` 与 initial allocation 一致。
- `dico_predynamic` 的 reference 使用未被 generic repair 改写的 DiCo allocation。
- 如果 DiCo allocation 低于 eta，不由 trainer repair，只记录 warning。

## 6. Evidence Relaxation

因为 DiCo-98 允许：

```yaml
allow_rank_beyond_selected_evidence: true
```

所以部分 final rank 可能超过 selected evidence count。v0.2.7 在 `metrics.json` 顶层记录：

```json
{
  "evidence_relaxation": {
    "selected_evidence_total": 0,
    "final_rank_total": 0,
    "rank_beyond_evidence_total": 0,
    "rank_beyond_evidence_ratio": 0.0,
    "modules_with_beyond": 0,
    "modules_total": 0
  }
}
```

audit 对 `rank_beyond_evidence_ratio > 0.30` 给 warning，不作为 error。

## 7. Multi-Seed 运行

服务器推荐路径：

```bash
cd /ai/lxw/lxw/dico_rank_experiments
conda activate dico-rank
```

默认 3 seed：

```bash
SEEDS="42 43 44" bash scripts/run_all_multiseed.sh --output_dir outputs_multiseed
```

包含 LoRA eta98 baseline：

```bash
INCLUDE_LORA_ETA=1 SEEDS="42 43 44" bash scripts/run_all_multiseed.sh --output_dir outputs_multiseed_eta
```

检查命令展开，不训练：

```bash
DRY_RUN=1 SEEDS="42 43" bash scripts/run_all_multiseed.sh
```

输出目录形式：

```text
outputs_multiseed/lora_r4__seed42/
outputs_multiseed/dico_pre_r8__seed43/
```

每个 seed 同时覆盖：

- `seed`
- `calibration.seed`
- `preallocation.sketch_seed`

## 8. Ablation 运行

```bash
SEEDS="42 43 44" bash scripts/run_ablations.sh --output_dir outputs_ablations
```

dry run：

```bash
DRY_RUN=1 SEEDS="42" bash scripts/run_ablations.sh
```

## 9. Summary 与 Audit

生成 summary：

```bash
python scripts/summarize_results.py --output_dir outputs_multiseed
```

输出：

- `summary_per_run.csv`：每个 run 一行。
- `summary.csv`：multi-seed 时按 experiment 聚合 mean/std/n。
- `summary.md`：简表。

审计输出：

```bash
python scripts/audit_outputs.py --output_dir outputs_multiseed
```

audit 会检查：

- 主实验 seed coverage。
- budget interval。
- DiCo-Pre 是否使用 `[eta, 1.0]` paramcount ratio。
- LoRA eta98 是否使用 `[0.97, 0.98]` paramcount ratio。
- dynamic adjustment 文件与 step。
- evaluation/prediction 行数。
- cache compatibility diagnostics。
- evidence relaxation warning。

## 10. 关键输出文件

每个实验目录包含：

```text
config_resolved.yaml
metrics.json
budget.json
rank_allocation_initial.json
rank_allocation_final.json
rank_history.csv
train_log.jsonl
eval_log.jsonl
eval_predictions.jsonl
masked_lora_state.pt
```

其中：

- `train_log.jsonl`：训练 step、dynamic adjustment、mid loss eval。
- `eval_log.jsonl`：只写最终 `event=final_eval`。
- `masked_lora_state.pt`：只保存 LoRA A/B 和 rank mask，不保存 optimizer state。
- `budget.json`：最终 budget 口径和 policy diagnostics。

## 11. 验证命令

不运行完整 GSM8K 训练的工程验证：

```bash
python -m pytest -q
bash -n scripts/run_all_8.sh scripts/run_all_multiseed.sh scripts/run_ablations.sh
DRY_RUN=1 SEEDS="42 43" bash scripts/run_all_multiseed.sh
python scripts/run_experiment.py \
  --config configs/experiments/ablations/dico_pre_r8_no_relaxation.yaml \
  --override training.max_steps=2 \
  --override evaluation.accuracy_max_samples=2
```

最后一条会启动 tiny/真实配置取决于当前 config/model override。服务器上跑前请确认 `model.name_or_path` 指向可用模型。
