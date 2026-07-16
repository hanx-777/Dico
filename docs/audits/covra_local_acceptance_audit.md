# CovRA 本地验收审计

日期：2026-07-15

## 本地阶段结论

`本地实现与静态验收已完成，可以上传服务器并进入E00 GPU pilot。`

这个结论只适用于本地开发阶段。它表示代码、配置、CPU 单元测试、小模型集成测试、dry-run、manifest 校验和服务器启动脚本已经准备好；它不表示真实 Llama-3.1-8B 训练、A800 显存行为、tokens/s、PyTorch SDPA、BF16/FP32 混合精度、完整评测或 MTBench-local judge 已经完成 GPU 验证。

## 状态标记规则

本地阶段只使用：

- `IMPLEMENTED_AND_CPU_VERIFIED`
- `IMPLEMENTED_NOT_GPU_RUN`
- `BLOCKED_BY_UNRESOLVED_PROTOCOL`
- `NOT_IMPLEMENTED`

服务器阶段才允许使用：

- `IMPLEMENTED_AND_GPU_VERIFIED`
- `GPU_RUN_FAILED`
- `BLOCKED_BY_ENVIRONMENT`

不能把 CPU 验证和 GPU 验证合并成同一种状态。

## 当前证据快照

| 项目 | 状态 | 证据 | 边界 |
|---|---:|---|---|
| 单元测试与集成测试 | `IMPLEMENTED_AND_CPU_VERIFIED` | `reports/final_delivery.md` 中记录最新 `pytest -q` 结果 | 仅代表本地/CPU |
| 静态验收 | `IMPLEMENTED_AND_CPU_VERIFIED` | `reports/static_acceptance.{json,md}` 为 `PASS_WITH_SKIPS` | typecheck/lint 工具缺失时可为 skip；不代表 GPU |
| 协议预检 | `IMPLEMENTED_AND_CPU_VERIFIED` | `reports/protocol_preflight.{json,md}` | 检查 config 约束，不检查真实显存 |
| E00 readiness dry-run | `IMPLEMENTED_NOT_GPU_RUN` | `reports/e00_readiness.{json,md}` 为 `READY_DRY_RUN`，且 `requires_real_gpu_execution=true` | 必须在 A800 服务器用 GPU gate 重跑 |
| baseline registry | `IMPLEMENTED_AND_CPU_VERIFIED` / `IMPLEMENTED_NOT_GPU_RUN` / `BLOCKED_BY_UNRESOLVED_PROTOCOL` | `reports/baseline_status.{json,md}` | GoRA-public 已锁定 commit 并接入；EVA 仍有协议未决项 |
| run manifest 契约 | `IMPLEMENTED_AND_CPU_VERIFIED` | `scripts/validate_run_manifest.py`，`tests/scripts/test_validate_run_manifest.py` | E00/E01 真实产物仍需再校验 |
| 单文件无参数启动器 | `IMPLEMENTED_AND_CPU_VERIFIED` | `launch_covra.py`，`tests/scripts/test_launch_covra.py` | 本地只验证启动器行为；未在平台队列真实运行 |
| LoRA/AdaLoRA/GoRA 对齐 | `IMPLEMENTED_AND_CPU_VERIFIED` | `tests/unit/test_aligned_lora_protocol.py`、`test_adalora_official.py`、`test_gora_aligned.py`、tiny trainer 集成测试 | GoRA 正式路径已改为 direct weight-grad hook；真实 Llama/A800 为 `IMPLEMENTED_NOT_GPU_RUN` |

最新回归：`pytest -q` 为 **331 passed，4 warnings**。warning 均来自 tiny/退化输入下 scikit-learn 聚类簇数不足，不是训练或协议测试失败。`compileall` 与正式配置 protocol preflight 通过；当前本地环境未安装 `ruff`，因此 lint 工具项记录为跳过，不冒充 PASS。

## 本地验收清单

| 要求 | 状态 | 证据 |
|---|---:|---|
| CovRA 主链路接入正式训练入口 | `IMPLEMENTED_AND_CPU_VERIFIED` | `src/dico/trainer.py`；`scripts/static_acceptance.py` 的 CPU tiny 路径；trainer 集成测试 |
| CovRA-I 与 CovRA-M 定义正确并可启动 | `IMPLEMENTED_AND_CPU_VERIFIED` | `configs/ablations/covra_independent.yaml`；`configs/ablations/covra_module_scalar.yaml`；preflight 与单测 |
| 必需消融配置可解析、可预检 | `IMPLEMENTED_AND_CPU_VERIFIED` | `scripts/protocol_preflight.py`；`reports/protocol_preflight.json` |
| 历史机制默认关闭且不污染最终路径 | `IMPLEMENTED_AND_CPU_VERIFIED` | config schema / preflight / README；legacy 字段限定在 `dico.legacy_covra_v05.*` |
| DP 预算求解通过暴力枚举一致性测试 | `IMPLEMENTED_AND_CPU_VERIFIED` | `tests/unit/test_budget.py` |
| strict budget 在已测场景下不超预算 | `IMPLEMENTED_AND_CPU_VERIFIED` | budget 测试与 manifest validator |
| 方向初始化经过前向输出一致性测试 | `IMPLEMENTED_AND_CPU_VERIFIED` | `tests/unit/test_dico_init_v03.py`；trainer 集成测试 |
| 小模型端到端路径走共享 trainer | `IMPLEMENTED_AND_CPU_VERIFIED` | `scripts/static_acceptance.py`；`tests/unit/test_dico_da_init_trainer_integration.py` |
| 正式配置可展开且未知字段会报错 | `IMPLEMENTED_AND_CPU_VERIFIED` | `scripts/protocol_preflight.py`；config 测试 |
| 受限平台无参数启动器已准备 | `IMPLEMENTED_AND_CPU_VERIFIED` | `launch_covra.py`；`tests/scripts/test_launch_covra.py` |
| 正式 config 不依赖本地绝对路径 | `IMPLEMENTED_AND_CPU_VERIFIED` | config 使用相对路径或 HF id；服务器绝对路径集中在 `launch_covra.py` 顶部 |
| E00 单卡和 DDP fallback 脚本存在 | `IMPLEMENTED_NOT_GPU_RUN` | `scripts/platform_train.py`；`scripts/run_ddp.sh`；`launch_covra.py`；E00 readiness 报告 |
| GPU 事项均明确标记未运行 | `IMPLEMENTED_NOT_GPU_RUN` | `reports/final_delivery.md`；`reports/e00_readiness.md` |
| 外部协议未决项已记录 | `BLOCKED_BY_UNRESOLVED_PROTOCOL` | `reports/baseline_status.md`；GoRA 最终官方评测脚本与 EVA 说明 |

## 当前默认模型与数据

| 类别 | 当前设置 |
|---|---|
| 正式主模型 | Llama-3.1-8B base；配置里默认 `meta-llama/Llama-3.1-8B-Base`，服务器推荐改为本地模型目录 |
| tokenizer | 与 base model 同源；revision 需在 E00 记录 |
| 数学主训练集 | MetaMathQA-100K，默认 `data/metamathqa/train.jsonl`；固定前 100,000 个成员并以 `dataset_seed=42` 统一打乱；正式训练曝光 100,032 条，末尾重复 32 条 |
| 数学主评测集 | GSM8K test，默认 `data/gsm8k/main/test.jsonl`，1319 条 |
| Code 组合 | CodeFeedback code-only → HumanEval，作为 GoRA 原始组合/推荐扩展 |
| Chat 组合 | WizardLM → MTBench，作为 GoRA 原始组合/推荐扩展 |

## 上传服务器前需要修改的位置

在服务器上执行前，编辑 `launch_covra.py` 顶部“手动配置区”：

- `CONDA_ENV_PYTHON`
- `PROJECT_ROOT`
- `MODEL_PATH`
- `DATA_ROOT` / `DATA_PATHS_TO_CHECK`
- `CONFIG_FILES`
- `SEEDS`
- `GPU_IDS`
- `OUTPUT_DIR`
- `LOG_DIR`

受限平台最终命令：

```bash
python launch_covra.py
```

如果平台允许设置环境变量但不允许传命令行参数，也可以用 `launch_covra.py` 顶部列出的 `COVRA_*` 环境变量覆盖这些路径。

## 仍必须在服务器验证的项目

以下项目在本地只能标记为 `IMPLEMENTED_NOT_GPU_RUN`，直到 3×A800 服务器真实执行并归档证据：

- Llama-3.1-8B 真实加载；
- 1024 样本真实校准；
- A800 峰值显存和 tokens/s；
- PyTorch SDPA 运行时行为；
- frozen BF16 base + FP32 adapter；
- 单卡 batch 4、gradient accumulation 16；
- 3 卡 DDP fallback 的 global batch 63；
- E00 LoRA/CovRA pilot；
- GSM8K、HumanEval、MTBench 完整评测；
- E01 及后续正式实验。

## SDPA v4 公平性与性能修复证据

- 标准 LoRA 保持 r8、alpha16、`alpha/r=2`、Kaiming A、零 B、FP32 adapter。
- AdaLoRA 使用 A/E/B、全局预算竞争、裁剪前重要度、可恢复 E 梯度和未平方 Frobenius 正交正则；不再强制每模块最低 target rank。
- GoRA-public/GoRA-BM 每个 calibration batch 只做一次完整反传，直接收集目标基础权重梯度；旧 CPU 重建路径仅保留为 `gora_bw` legacy。
- 正式 CovRA 明确要求 calibration/SVD/方向恢复/响应块/条件残差投影在 CUDA 上执行，不允许 `auto` 静默退回 CPU。
- launcher 默认输出升级为 v3，并将三个 seed 固定为三个独立单进程任务；所有旧输出仍为 `legacy_protocol_pilot`。
- token cache 的键包含有序数据 hash、tokenizer 指纹、prompt 版本和 max length；缓存命中不能改变 token 或样本顺序。
- 本地仅验证 `stage_metrics.jsonl` 的 CPU/null-CUDA 语义；A800 allocated/reserved/peak 与 `gpu_monitor.csv` 仍是 `IMPLEMENTED_NOT_GPU_RUN`。
