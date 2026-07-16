# CovRA 基线公平性与 GPU 性能修复审计

日期：2026-07-15

## 结论

本轮已完成代码与 CPU/tiny 验证，但尚未在目标 3×A800 上执行。因此当前结论是：

`本地实现与静态验收已完成，可以上传服务器并进入E00 GPU pilot。`

不得据此声明正式实验已经完成。所有真实 Llama-3.1-8B、PyTorch SDPA、BF16/FP32 混合精度、显存、GPU 利用率和完整 GSM8K 结果均为 `IMPLEMENTED_NOT_GPU_RUN`。

## 已修复的公平性问题

| 项目 | 修复后实际行为 | 本地状态 |
|---|---|---|
| 标准 LoRA | r8、alpha16、`alpha/r=2`、Kaiming A、B=0、A/B 同 LR 5e-5、精确预算 6,815,744 | `IMPLEMENTED_AND_CPU_VERIFIED` |
| AdaLoRA | A/E/B，init rank12、目标平均 rank8、全局无 floor 裁剪、E 置零但可重新获梯度、裁剪前 EMA 重要度、未平方正交正则、LR 5e-4 | `IMPLEMENTED_AND_CPU_VERIFIED` |
| GoRA-public | 固定 1024 个训练样本；每 batch 一次反传；直接 target-weight gradient hook；无 answer-only mask；GPU FP32 importance/伪逆；B/A LR 16× | `IMPLEMENTED_AND_CPU_VERIFIED` |
| GoRA-BM | 与 GoRA-public 唯一方法差异为 strict budget repair，不超 6,815,744 | `IMPLEMENTED_AND_CPU_VERIFIED` |
| 数据顺序 | 固定 100K 成员，以公共 `dataset_seed=42` 打乱；1563×64=100,032 次曝光，重复 32 条 | `IMPLEMENTED_AND_CPU_VERIFIED` |
| 校准集合 | `calibration.seed=42` 固定，launcher 不随训练 seed 覆盖；CovRA sketch/init seed 随 42/43/44 | `IMPLEMENTED_AND_CPU_VERIFIED` |
| 评测 | final checkpoint 单次 greedy GSM8K，batch4、max new tokens256、统一提取与原始预测 artifact | `IMPLEMENTED_AND_CPU_VERIFIED` |
| PyTorch SDPA | 正式配置统一显式传入 `sdpa`；目标平台不需要 `flash-attn` 二进制包 | `IMPLEMENTED_NOT_GPU_RUN` |

## 参数预算报告规则

- method-faithful：LoRA、AdaLoRA、GoRA-public、CovRA 分别报告 physical/peak/final/optimizer state。
- strict equal-budget：LoRA、GoRA-BM、CovRA 不得超出 6,815,744；可精确时误差为零。
- AdaLoRA 的 Llama3 q/k/v/o physical/peak 为 10,225,152，final active 为 6,816,768；不得称为与 LoRA 物理参数相同。
- GoRA-public 保留实际 method-faithful 预算；不得把其结果伪装为 strict equal-budget。

## GPU 性能修复

- GoRA 从旧路径每个 calibration batch 按模块块重复整模型反传，改为每 batch 一次 direct weight-gradient backward。1024 样本、batch4 应为 256 次反传。
- GoRA importance、伪逆和 B 初始化逐模块在目标 GPU FP32 执行；梯度累计缓冲按配置卸载 CPU，处理完及时释放。
- CovRA 正式配置使用 `compute_device=cuda` 和 `allocation_device=cuda`；随机投影、SVD、方向恢复、响应块、QR 与残差投影不再因 profile 提前 CPU 化而全部落到 CPU。
- CovRA 校准期间冻结全部 8B 基础权重，只保留输入/中间激活图和目标模块输出梯度；不再为冻结底座分配无意义的参数梯度。校准结束恢复原始 `requires_grad` 状态，随后由 adapter 注入统一冻结底座。
- 正式 CovRA 同时执行 CPU reference 核验；候选选择、逐秩曲线或最终 rank allocation 若超过 `atol=1e-6, rtol=1e-5` 容差或离散结果不同，会自动采用 CPU reference 结果，并在诊断中记录 `cpu_reference_fallback=true`。
- 内层模块块循环不再反复 `gc.collect()` / `torch.cuda.empty_cache()`；只在阶段边界强制释放。
- token cache 避免 4 个方法×3 seeds 重复 tokenize 100K 数据，但不改变样本或 token。
- 每个 run 写 `stage_metrics.jsonl`；服务器有 `nvidia-smi` 时，launcher 同时写独立 `gpu_monitor.csv`。

## 三张 A800 调度

默认不是 DDP，也不是 DeepSpeed。GPU0/1/2 分别运行 seed42/43/44；同一方法的三个 seed 并行，方法之间按 LoRA、AdaLoRA、GoRA-public、CovRA 顺序执行。每个 child 强制单进程，因此单 run global batch 保持 `4×16=64`。

只有单卡 E00 pilot OOM 时才验证 3 卡 DDP global batch63 fallback。DDP 不得与默认主表结果混用而不披露。

## 仍无法完全对齐的项目

- GoRA 仓库没有公开最终 GSM8K/HumanEval/MTBench 完整评测脚本；当前 greedy GSM8K 是统一内部协议，不能直接对照论文绝对值。
- GoRA MetaMath gamma 论文 `.08` 与脚本 `.05` 冲突；主配置按代码/脚本来源规则使用 `.05` 语义，`.08` 仅限附录敏感性。
- GoRA N 的论文64、旧脚本32、parser 默认8冲突；正式配置锁定 1024 个全局校准样本，N64 的准确含义仍为 `[待核对]`。
- 模型/tokenizer immutable revision 需在服务器用本地目录指纹或 revision 锁定。
- Microsoft LoRA 官方 commit 仍为 `[待网络恢复后锁定]`。

## 服务器验证指标

E00 必须核对每阶段 wall time、CUDA allocated/reserved、allocated peak、reserved peak，以及每秒 GPU utilization、memory、power。重点比较 GoRA direct 与 legacy 的 backward 次数和校准耗时，并确认 rank、预算与 tiny reference 语义一致。
