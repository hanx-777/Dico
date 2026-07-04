# DiCo Rank Allocator 实现技术文档

本文档说明当前代码中 DiCo-Pre rank allocator 的实现方式，重点覆盖 SVD atom logs 之后到最终 rank pattern 的路径。该实现不改变 SVD atom 提取、signed response profile 构造、coverage certification 等上游逻辑，只重构最终 rank budget allocation。

## 1. 总体架构

当前 DiCo-Pre 的 rank 分配链路分为四层：

1. `atom_svd.py` 负责从校准数据中提取 SVD direction atoms，并输出 atom logs。
2. `preallocation.py` 负责调度 DiCo-Pre 流程，在 SVD 模式下调用新的 rank allocator。
3. `rank_budget.py` 保留预算 API 表面，提供 `allocate_by_rank_allocator(...)` 作为新入口，同时保留旧的 `allocate_by_directional_evidence(...)`。
4. `rank_allocator.py` 实现可组合 rank allocator：先通过 `atom_to_rank` 把 atom evidence 转换为 rank 边际证据，再通过 `smoothing` 调整候选增量，最后贪心生成最终 rank allocation。

核心设计是两个独立实验轴：

```text
atom logs
  -> atom normalization
  -> atom_to_rank strategy
  -> optional smoothing strategy
  -> budget-safe greedy allocation
  -> WeightedAllocationResult
```

当前默认组合在 `configs/base.yaml` 中定义为：

```yaml
preallocation:
  rank_allocator:
    atom_to_rank: marginal_curve
    smoothing: layer_diffusion
```

旧版方向直购行为仍可通过以下组合复现：

```yaml
preallocation:
  rank_allocator:
    atom_to_rank: legacy_atom_purchase
    smoothing: none
```

## 2. 入口与数据流

### 2.1 SVD preallocation 入口

在 `DiCoPreAllocator._allocate_svd(...)` 中，代码先调用：

```python
atoms, diagnostics = extract_svd_atom_records(...)
atom_logs = [atom.to_log_dict() for atom in atoms]
```

随后进入新 allocator：

```python
allocation = allocate_by_rank_allocator(
    atom_logs=atom_logs,
    module_dims=self.module_dims,
    target_budget=int(rank_budget),
    eta=float(self.pre_cfg.get("eta", 0.98)),
    r_min=max(0, int(rank * float(self.pre_cfg.get("r_min_multiplier", 0.0)))),
    r_max=r_max,
    config=self.pre_cfg.get("rank_allocator"),
    allow_rank_beyond_selected_evidence=bool(
        self.pre_cfg.get("allow_rank_beyond_selected_evidence", True)
    ),
    budget_mode=self.config.get("budget", {}).get("mode", "equal_trainable_params"),
    warning_threshold=float(self.config.get("budget", {}).get("warning_threshold", 0.01)),
)
```

这里 `rank_budget` 是与 uniform LoRA 对齐后的目标参数预算。`r_min` 和 `r_max` 仍由现有 `rank`、`r_min_multiplier`、`r_max_multiplier` 决定。allocator 只负责在 `actual_budget <= target_budget` 的约束下生成整数 rank。

### 2.2 Budget API 兼容层

`rank_budget.py` 新增：

```python
def allocate_by_rank_allocator(...) -> WeightedAllocationResult
```

该函数负责：

1. 根据 `module_dims` 计算每个模块单 rank 成本。
2. 调用 `rank_allocator.allocate_rank_pattern(...)`。
3. 用 `_budget_info(...)` 生成统一的 `BudgetInfo`。
4. 将 allocator 的 `allocation`、`module_logs`、`diagnostics` 包装成 `WeightedAllocationResult`。

`WeightedAllocationResult` 现在包含可选字段：

```python
diagnostics: dict[str, Any] | None = None
```

因此上层 trainer/preallocation 仍然可以读取原有字段，不需要大规模改写。

## 3. Rank Allocator 内部实现

主要实现位于 `src/dico_rank/rank_allocator.py`。

### 3.1 数据结构

`NormalizedAtom` 是 allocator 内部的 atom 表示：

```python
NormalizedAtom(
    module_name: str,
    atom_index: int,
    utility: float,
    raw_utility: float,
    selected: bool,
    module_type: str,
    layer: int,
    profile: tuple[float, ...] | None,
)
```

其中：

- `utility` 是参与 rank allocation 的归一化效用。
- `raw_utility` 保留原始 atom log 中的 `utility`。
- `module_type` 优先读取 atom log 中的 `type/module_type/tau`，否则从模块名最后一段推断。
- `layer` 优先读取 `layer/layer_idx/ell`，否则从 `layers.<idx>` 模块名模式推断。
- `profile` 从 `pi/response_profile/signed_response` 中读取，用于 prototype bundle。

`RankEvidence` 表示每个模块可购买 rank slot 的边际证据：

```python
RankEvidence(
    values: dict[str, list[float]],
    metadata: dict[str, dict[str, Any]],
)
```

其中 `values[module][k]` 表示该模块第 `k+1` 个可购买 rank 的边际值。

`RankAllocatorResult` 是 allocator 的内部返回：

```python
RankAllocatorResult(
    allocation: dict[str, int],
    module_logs: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    warning: str | None,
)
```

### 3.2 Atom normalization

`normalize_atoms(...)` 会过滤掉：

- 不在 `module_dims` 中的模块。
- `selected=False` 的 atom。

对于非 legacy 模式，基础效用计算为：

```text
utility = log1p(gain * align ** align_gamma)
```

其中 `gain` 的字段别名包括：

```text
g_sel, selected_gain, coverage_gain, gain
```

`align` 的字段别名包括：

```text
align, alignment
```

如果没有 gain 字段，则退回使用 atom log 中的 `utility`。

当前支持 `type_normalization`：

- `none`：不做类型归一化。
- `median`：同 type 内除以 median。
- `zscore_iqr`：同 type 内减 median，再除以 IQR。

legacy 路径有特殊保护：当 `atom_to_rank=legacy_atom_purchase` 时，`normalize_atoms(..., use_raw_utility=True)` 会直接使用 raw `utility`，并跳过 type normalization，避免新默认配置污染旧版方向直购 baseline。

## 4. Atom-to-Rank 策略

当前支持四种策略，其中前三种用于 3×3 主实验，第四种用于 legacy baseline。

### 4.1 `marginal_curve`

实现函数：

```python
build_marginal_curve_evidence(...)
```

流程：

1. 按模块收集 selected atom 的 utility。
2. 每个模块内按 utility 降序排列。
3. 如果允许 evidence relaxation 且模块存在 utility，则用模块平均 utility 补齐到 `r_max`。
4. 对第 `j` 个 rank slot 施加 decay。

当前 decay 支持：

- `none`：不衰减。
- `sqrt`：第 `j` 个 rank slot 乘以 `1 / sqrt(j)`。
- `geometric`：第 `j` 个 rank slot 乘以 `geometric_lambda ** (j - 1)`。

默认参数：

```yaml
marginal_curve:
  decay: sqrt
  geometric_lambda: 0.75
```

该策略对应 importance-based adaptive rank allocation：把 atom evidence 变成模块级 rank 边际曲线，再按预算购买边际收益最高的 rank。

### 4.2 `prototype_bundle`

实现函数：

```python
build_prototype_bundle_evidence(...)
```

流程：

1. 按模块收集 selected atoms。
2. 每个模块内按 `utility` 降序、`atom_index` 升序处理 atom。
3. 对每个 atom，与已存在 bundle center 的 `profile` 计算 cosine similarity。
4. 如果相似度高于阈值，则加入该 bundle；否则创建新 bundle。
5. 每个 bundle 转换成一个 rank evidence item。

bundle value 计算为：

```text
bundle_value = max(member.utility)
             + residual_weight * sum(member.utility * (1 - cosine(member.profile, center.profile)))
```

默认参数：

```yaml
prototype_bundle:
  similarity_threshold: 0.8
  residual_weight: 0.25
```

如果 atom 缺少 profile，则不会强制失败；该 atom 独立成 bundle，并在 `module_logs` 的 `prototype_warnings` 中记录 `missing_profile`。

### 4.3 `soft_slot`

实现函数：

```python
build_soft_slot_evidence(...)
```

流程：

1. 构造 `1..r_max` 的 rank slot。
2. 每个 atom 对所有 slot 分配 soft support。
3. slot logit 为：

```text
logit(slot) = (atom.utility - slot_decay * slot_index) / temperature
```

4. softmax 后把 atom utility 分摊到各 slot。
5. 贪心分配阶段只能购买 `current_rank + 1`，因此仍保持 slot precedence。

默认参数：

```yaml
soft_slot:
  temperature: 1.0
  slot_decay: 0.15
```

该策略对应 ordered / prefix rank slot 思路：后续 rank slot 可以得到证据，但必须按 rank 前缀顺序购买。

### 4.4 `legacy_atom_purchase`

实现函数：

```python
_legacy_allocate(...)
```

该路径复现旧版 direct atom purchase：

1. 只使用 selected atoms。
2. 直接按 atom 粒度排序购买。
3. 每买中一个 atom，就给该 atom 所在模块 rank 加 1。
4. 若 selected evidence 不足以达到 eta，下游再做 evidence relaxation。

排序键为：

```text
utility / cost ** beta,
utility,
cheaper cost,
module name,
atom index
```

注意：实现里通过 `reverse=True` 和负号字段实现等价排序。该路径使用 raw utility，并且跳过新归一化。

## 5. Smoothing 策略

### 5.1 `layer_diffusion`

实现函数：

```python
apply_layer_diffusion(...)
```

默认 kernel：

```yaml
layer_diffusion:
  kernel: [0.25, 0.50, 0.25]
```

实现逻辑：

1. 按 `(module_type, layer)` 建立模块索引。
2. 对同 type 的 `layer-1/layer/layer+1` 做局部扩散。
3. 边界层只使用可用邻居，并重新归一化 kernel mass。
4. 不跨 module type 扩散。

### 5.2 `budget_guardrails`

实现函数：

```python
_violates_guardrails(...)
```

该策略不改变 evidence value，而是在 greedy 购买 rank 时拒绝违反 caps 的候选增量。

当前支持：

- `max_rank_per_module`
- `layer_cap_multiplier`
- `type_cap_multiplier`
- `type_budget_bounds`

默认参数：

```yaml
budget_guardrails:
  max_rank_per_module: null
  layer_cap_multiplier: 1.8
  type_cap_multiplier: 2.0
  type_budget_bounds: null
```

如果 guardrails 过强导致无法达到 `eta * target_budget`，allocator 不会突破 `target_budget`，而是在 diagnostics/warning 中记录未达到 eta。

### 5.3 `concentration_penalty`

该策略在 greedy scoring 阶段引入软 HHI 惩罚：

```text
score = base_score - lambda * delta_hhi
```

其中：

```text
base_score = marginal_value / cost ** cost_beta
```

默认参数：

```yaml
concentration_penalty:
  lambda: 0.02
```

这是 soft penalty，不是 hard cap；如果某个模块 evidence 足够强，仍可以继续获得 rank。

## 6. Greedy Allocation 与 Evidence Relaxation

非 legacy 策略统一进入 `_generic_allocate(...)`。

### 6.1 第一阶段：按边际证据购买 rank

初始化：

```python
allocation = {module: r_min}
```

循环选择最佳候选模块：

1. 候选模块不能超过 `r_max`。
2. 增加一个 rank 后不能超过 `target_budget`。
3. 如果使用 `budget_guardrails`，候选不能违反 guardrails。
4. 候选的 rank offset 为 `allocation[module] - r_min`。
5. 从 `RankEvidence.next_value(module, rank_offset)` 读取边际值。
6. 计算 cost-aware score：

```text
score = marginal_value / cost ** cost_beta
```

如果启用 concentration penalty，则额外减去 `lambda * delta_hhi`。

选择 tie-break key：

```text
score,
-layer,
module_type,
module_name
```

该阶段每次只增加一个 rank，并记录：

- `purchased_evidence_rank`
- `purchased_slots`

### 6.2 第二阶段：Evidence Relaxation

如果第一阶段结束后：

```text
actual_budget < eta * target_budget
```

则进入 relaxation 阶段。候选模块仍必须满足：

- 不超过 `target_budget`
- 不超过 `r_max`
- 如 `allow_rank_beyond_selected_evidence=False`，不能超过 selected evidence count
- 如启用 guardrails，不能违反 caps

relaxation 阶段使用模块平均 density：

```text
avg_density = mean(selected_utilities[module]) / cost ** cost_beta
```

如果仍无法达到 eta，会记录 warning：

```text
selected evidence constraints prevented reaching eta target
```

## 7. Diagnostics 与日志

allocator diagnostics 由 `_allocation_diagnostics(...)` 生成，主要字段包括：

- `actual_budget`
- `budget_ratio`
- `atom_to_rank`
- `smoothing`
- `hhi`
- `gini`
- `layer_total_variation`
- `type_budget_share`
- `num_nonzero_modules`
- `max_rank`
- `warnings`

在 `preallocation.py` 中，这些 diagnostics 会并入 SVD diagnostics：

```python
full_diagnostics = {
    **diagnostics,
    "allocation_method": "directional_budgeted",
    **(allocation.diagnostics or {}),
    ...
}
```

最终会进入 `PreallocationResult.diagnostics`，并在 `to_dict(...)` 时被写入 preallocation payload 顶层。

每个 module log 会记录：

- `final_rank`
- `rank_cost`
- `final_budget`
- `selected_evidence_count`
- `purchased_evidence_rank`
- `evidence_relaxation_rank`
- `rank_beyond_selected_evidence`
- `rank_beyond_evidence_ratio`
- `selected_atom_utilities`
- `purchased_slots`

prototype bundle 还会额外记录：

- `bundle_count`
- `prototype_warnings`

## 8. Cache 兼容性

`build_preallocation_cache_context(...)` 已将 allocator 配置纳入 cache context：

```python
"preallocation": {
    "allocation_method": ...,
    "rank_allocator": pre_cfg.get("rank_allocator"),
    ...
}
```

因此只要更改：

- `atom_to_rank`
- `smoothing`
- utility 参数
- marginal/prototype/soft-slot/smoothing 子参数

旧 preallocation cache 都会被判定为不兼容。3×3×2 runner 还进一步按组合和 seed 隔离 cache 目录：

```text
<output_dir>/preallocations/<experiment_name>/seed<seed>
```

这样可以避免 9 个 allocator 组合覆盖同名 cache 文件。

## 9. 3×3×2 实验实现

实验配置目录：

```text
configs/experiments/allocator_3x3/
```

共 9 个 YAML，覆盖：

```text
marginal_curve      × budget_guardrails
marginal_curve      × layer_diffusion
marginal_curve      × concentration_penalty
prototype_bundle    × budget_guardrails
prototype_bundle    × layer_diffusion
prototype_bundle    × concentration_penalty
soft_slot           × budget_guardrails
soft_slot           × layer_diffusion
soft_slot           × concentration_penalty
```

每个配置均为：

```yaml
method: dico_pre
rank: 8
rank_strategy:
  init: dico_pre
preallocation:
  aggregation_mode: weighted_log
  atom_weight_normalization: none
  use_cost_aware_allocation: true
  rank_allocator:
    atom_to_rank: ...
    smoothing: ...
```

LoRA 和 allocator 主参数继承自 `configs/base.yaml`：

```yaml
lora:
  alpha: 16
  dropout: 0.05
preallocation:
  eta: 0.98
  allocation_method: directional_budgeted
```

专用 runner：

```text
scripts/run_pre_allocator_3x3_2seed.sh
```

默认行为：

- `SEEDS="42 43"`
- `OUTPUT_DIR=outputs_pre_allocator_3x3_2seed`
- 只枚举 9 个 allocator configs
- 每个实验名追加 `__seed<seed>`
- 固定 override 写在命令末尾，防止用户透传参数覆盖 seed/cache 语义

运行命令：

```bash
SEEDS="42 43" bash scripts/run_pre_allocator_3x3_2seed.sh \
  --output_dir outputs_pre_allocator_3x3_2seed
```

后台运行命令：

```bash
cd /ai/lxw/lxw/dico_rank_experiments
conda activate dico-rank
RUN_DIR=outputs_pre_allocator_3x3_2seed_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR/logs"
SEEDS="42 43" nohup bash scripts/run_pre_allocator_3x3_2seed.sh \
  --output_dir "$RUN_DIR" \
  --log_dir "$RUN_DIR/logs" \
  --override model.name_or_path=/ai/lxw/lxw/Qwen3-8B \
  --override model.torch_dtype=bfloat16 \
  --override training.batch_size=4 \
  --override training.gradient_accumulation_steps=2 \
  --override calibration.batch_size=4 \
  > "$RUN_DIR/logs/pre_allocator_3x3_2seed.log" 2>&1 &
echo $! > "$RUN_DIR/pre_allocator_3x3_2seed.pid"
```

## 10. 测试覆盖

当前测试覆盖分为三类。

### 10.1 Allocator 单元测试

文件：

```text
tests/test_rank_allocator.py
```

覆盖内容：

- 3×3 组合均能运行。
- 所有组合都满足 `actual_budget <= target_budget`。
- rank 为整数并位于 `[r_min, r_max]`。
- `legacy_atom_purchase + none` 使用 raw utility，不受新默认 normalization 影响。
- `marginal_curve` 在 many-weak-atoms fixture 上与 legacy 行为不同。
- `prototype_bundle` 会合并高 cosine profile。
- `soft_slot` 保持 slot precedence。
- `layer_diffusion` 会降低孤立层 spike。
- `concentration_penalty` 会降低 HHI。
- guardrails 在 eta 不可达时返回 warning。

### 10.2 Preallocation/cache 测试

文件：

```text
tests/test_preallocation.py
```

覆盖内容：

- cache context 记录 `rank_allocator`。
- allocator 设置变化会导致 cache incompatibility。
- 原有 preallocation cache 语义仍兼容。

### 10.3 实验配置与 runner 测试

文件：

```text
tests/test_debug_configs.py
tests/test_run_all_8_scripts.py
```

覆盖内容：

- 3×3 配置目录恰好包含 9 个 YAML。
- 每个配置可加载，且为 `method=dico_pre`、`rank=8`。
- 9 个 `(atom_to_rank, smoothing)` 组合完整且无重复。
- 主流参数继承正确。
- runner dry-run 生成 18 条命令。
- seed、experiment_name、preallocation cache 路径均正确隔离。

推荐本地验证：

```bash
PYTHONPATH=src pytest -q tests/test_debug_configs.py tests/test_run_all_8_scripts.py tests/test_rank_allocator.py
PYTHONPATH=src pytest -q tests/test_budget*.py tests/test_*preallocation*.py tests/test_random_allocation.py
SEEDS="42 43" DRY_RUN=1 bash scripts/run_pre_allocator_3x3_2seed.sh \
  --output_dir outputs_pre_allocator_3x3_2seed_dryrun \
  --no_hf_mirror \
  --override training.max_steps=1
git diff --check
```

## 11. 当前边界与注意事项

1. rank allocator 只接管 SVD atom logs 之后的最终预算分配，不修改上游 atom 提取和 coverage 选择。
2. `legacy_atom_purchase + none` 是旧行为复现路径，但不包含在当前 3×3×2 主实验矩阵中。
3. `module_proxy` fallback 仍走旧的 module utility allocation，不进入 SVD rank allocator 路径。
4. 3×3×2 runner 不包含 LoRA baseline、DiCo-Dynamic 或 PreDynamic。
5. cache 文件名本身仍为原有 `dico_pre_rank{rank}_seed{seed}.json` 形式，因此 runner 必须使用组合级独立 `calibration.save_dir`。
6. `type_budget_bounds` 当前只实现上界检查；如果后续需要强制下界，需要单独扩展 guardrails 逻辑。
7. `prototype_bundle` 依赖 response profile；缺失 profile 时会退化为 one atom per bundle，并记录 warning。
