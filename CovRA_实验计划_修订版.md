# CovRA 实验计划修订版：GoRA 对齐、方法同步与 3×A800 调度

> 版本：v3.0（2026-07-15，基线公平性与 GPU 路径修复已接入）
> 硬件：3 × A800（单机）
> 调度原则：三张卡都参与实验调度；默认三卡并行三个单卡 run，而非强制每个 run 使用 3 卡 DDP。
> 目标：在尽可能复用 GoRA 官方实验协议的基础上，公平、统一、可复现地比较 CovRA、GoRA 与关键 LoRA 基线。
> 标记约定：`[待核对]` 表示论文、代码或配置仍无法从公开材料中确认；不得用猜测填补。

> 当前实现状态：E01 启动 profile 已接入 Uniform LoRA、完整 AdaLoRA、GoRA-public 与 CovRA，3 seeds 共 12 runs，正式输出为 `outputs/e01_llama3_r8_aligned_sdpa_v4`；E02 profile 已接入 LoRA、GoRA-BM 与 CovRA，3 seeds 共 9 runs，输出为 `outputs/e02_llama3_r8_strict_budget_sdpa_v4`。三个 seed 是三个独立单进程任务，不是 DDP。目标服务器因 GLIBC/CUDA 工具链限制无法加载 FlashAttention2，所有正式方法统一使用 PyTorch SDPA；旧 FlashAttention2/v3 结果不得与 SDPA v4 混合。EVA、Code/Chat 组合和 r32/r128 仍按后文计划保留，但不在当前 launcher 范围；旧 `gora_bw` 和 v2 输出统一标记为 `legacy_protocol_pilot`。

---

## 1. 实验目标与研究问题

本计划的核心目标不是盲目扩大实验规模，而是把 CovRA 与 GoRA 的比较压到同一条实验轨道上：相同基础模型、相同数据、相同训练语义、相同评测规则、相同参数预算报告口径，并用少量机制实验解释 CovRA 为什么有效。

| RQ | 研究问题 | 对应论文主张 | 主要实验 |
|---|---|---|---|
| RQ1 | 在已对齐的数学主任务上，CovRA 是否优于 Uniform LoRA、AdaLoRA 与 GoRA-public；EVA 接入后是否仍成立？ | 主性能贡献 | E01 |
| RQ2 | 在相同实际可训练参数量下，CovRA 与 GoRA 的差异是否仍成立？ | 公平预算贡献 | E02 |
| RQ3 | GoRA 论文、历史脚本与当前代码配置冲突对结果有多敏感？ | 可复现性审查 | E03 |
| RQ4 | 条件边际估值是否优于固定独立能量估值？ | CovRA 核心机制 | E04 |
| RQ5 | 方向级逐秩价值是否优于模块级单一标量？ | 方向级结构贡献 | E05 |
| RQ6 | rank 分配与方向初始化各自贡献是否可分离？ | 机制可解释性 | E06 |
| RQ7 | 全局/分组草图、正负拆分、类型缩放、对数压缩和 DP 求解是否必要？ | 支撑组件贡献 | E07-E08 |
| RQ8 | CovRA 对 calibration 样本选择是否稳定，额外开销是否可接受？ | 稳定性与效率 | E09-E10 |

---

## 2. Source Manifest 与优先级

执行前必须把下列来源写入实验 manifest；未能确认的条目保留 `[待核对]`。

| 来源 | 当前审查结论 |
|---|---|
| GoRA paper | `He et al. 2025, GoRA: Gradient-driven Adaptive Low Rank Adaptation`；PDF SHA256：`852ae8a82a6c80c283d64dab5664a1c9d9021e4b42e08b5c488755a7ad41f39c` |
| GoRA official repo | 已审查 commit：`4037d4d6ba67ff88de87f90b943ff4e3a3649b67`；公开脚本与当前 parser 存在不兼容 |
| AdaLoRA official repo | 已审查 commit：`d10f5ebee16c478fa2f41a44a237b38e8c9b0338` |
| Microsoft LoRA repo | commit `[待网络恢复后锁定]`；标准 LoRA 公式与初始化已有本地回归测试 |
| GoRA historical scripts | `cc716875cc4f0e54a38e1b2bf09b718019640a04`、`2581e52490796fb94ad497e3aa7460c972d704f9`、`980a554facfa7bcfb4e5e9b80b56236e55c78666` |
| LoRA-GA upstream | 已审查 commit：`c4cd5372c75b290924214b348008891f744512ef`；作为 GoRA 缺失评测脚本时的公开 fallback |
| FastChat | MTBench 本地 judge 计划锁定 `v0.2.36` / `b21d0f780ca4472a13714262a0790f2ee1ade659` |
| CovRA manuscript | `/Users/lxw/Downloads/CovRA.md`；当前方法论以该文件为准 |
| Fable5 原始计划 | SHA256：`29902a95a0dd4781ceec864325edc078f0ec0607f9765b45f78b6951a450f7fe`；只读保留，不覆盖 |

优先级规则改为：

1. 论文实验对应的冻结 release 或历史 commit。
2. 该 commit 下可执行的 launch script、生成 config 与运行日志。
3. 论文附录和正文。
4. 当前仓库默认值仅作为参考，不得与历史脚本拼接后称为“官方行为”。

GoRA-public 主配置必须按上述来源规则预注册，不能根据 GSM8K/HumanEval/MTBench 测试结果选择。

---

## 3. 阻塞问题与修订动作

| 阻塞点 | 风险 | 修订动作 |
|---|---|---|
| CovRA 方法字段与当前论文不同步 | `beta,h,delta,permutation,BH-FDR` 属于历史机制；当前论文强调 `rho,type_scaling,log_compression,solver=dp` | 重写方法-配置字段映射表；历史字段单列为当前实现遗留字段 |
| `K=8` 与自适应秩预算冲突 | 若 `K <= r_ref`，strict budget 下容易退化为均匀秩；r32/r128 更不可行 | 主档设 `K>=32,r_max=32`；r32 需 `K>=128,r_max=128`；r128 暂停 |
| E04/E05 对照定义不清 | `marginal-only` 未正式定义，CovRA-M 缺失 | E04 改为 CovRA vs CovRA-I；E05 改为 CovRA-I vs CovRA-M |
| 强制 3 卡 DDP 牺牲 batch=64 | global batch 63 与 GoRA 不完全一致，seed 串行导致周期变长 | 默认三卡并行三个单卡 run，保持 `1×4×16=64`；OOM 才用 3 卡 DDP fallback |
| E03 可能用测试集选配置 | 根据 GSM8K 选择 GoRA N/gamma 会污染测试集 | E03 只做附录敏感性，不用于选择主配置 |
| EVA 被放到推荐档 | EVA 是最接近的微调前激活谱/方向初始化基线之一 | EVA 升为 E01 必须基线 |
| GSM8K 随机采样评测不稳定 | 单样本 temperature `.8` 会引入评测随机性 | 主表使用 greedy；GoRA fallback 随机协议放附录 |
| GoRA 旧路径重复整模型反传并在 CPU 重建梯度 | 1024 样本会放大为 2048 次反传，且不等价于官方直接权重梯度 | 正式 GoRA 改为 direct target-weight gradient hook：batch4 共 256 次反传；旧路径只保留 legacy |
| training seed 覆盖 calibration seed | 各方法/seed 使用不同校准集合，形成数据混杂 | calibration seed 固定 42；launcher 只改变训练 seed 与 CovRA sketch/init seed |
| `NUM_GPUS` 环境变量可能误触发 DDP | 单 seed 进程意外占用多卡并改变 global batch | child 显式 `--num-processes 1` 且 `NUM_GPUS=1` |

---

## 4. GoRA 对齐原则

1. 主实验保留 GoRA 原始 Llama3.1 三个组合：MetaMathQA→GSM8K、CodeFeedback→HumanEval、WizardLM→MTBench。
2. 所有方法共享同一训练协议；方法专属配置只允许出现在方法专属表中。
3. 硬件适配优先保持训练语义；单卡可跑时保持 GoRA global batch 64。
4. 所有报告以实际可训练参数量为准，不以名义 rank 为准。
5. 单 seed 只用于 pilot、配置敏感性或低成本筛查，不用于正文主结论。
6. 测试集不得用于调参、checkpoint 选择或多次评测择优。
7. E03 只解释配置敏感性，不决定主配置。

---

## 5. 公共训练协议

### 5.1 Llama3.1 主协议：优先单卡并行

| 配置 | 取值 |
|---|---|
| base model/tokenizer | 与 GoRA 一致的 Llama-3.1-8B base/tokenizer，具体 revision `[待核对并锁定 hash]` |
| target modules | attention `wq,wk,wv,wo`；若本地 HF 名为 `q_proj,k_proj,v_proj,o_proj`，需在 manifest 中记录映射 |
| epoch | 1 |
| 默认调度 | 3 张 A800 同时跑 3 个单卡 run：GPU0 seed42，GPU1 seed43，GPU2 seed44 |
| 默认 global batch | `64 = 1 GPU × batch 4 × grad_accum 16` |
| 数据顺序与曝光 | 固定前 100K 个唯一成员，公共 `dataset_seed=42` 打乱；1563×64=100,032 次曝光，最后重复 32 条并写入 manifest |
| optimizer steps | 100K 数据：1563；52K 数据：`ceil(52000/64)=813`；实际以日志确认 |
| optimizer | 当前统一 AdamW，betas `(0.9,0.999)`，eps `1e-8`；不把未启用的 FusedAdam 写成已生效 |
| LR | 默认 `5e-5`；AdaLoRA `5e-4`；LoRA+/GoRA B 矩阵 `16×` |
| scheduler | cosine，min LR ratio `.1` |
| warmup | `int(steps × .03) + 1`；前 10 step auto-warmup rate `.05` |
| weight decay / clip | `5e-4` / `1.0` |
| LoRA dropout | `0` |
| precision | frozen base BF16，adapter FP32 |
| memory | 所有方法统一 `attn_implementation=sdpa`，不依赖 `flash-attn`；默认不开 gradient checkpointing |
| checkpoint | final checkpoint only |
| max sequence length | 1024，按 GoRA 脚本行为核对后更新 `[待核对]` |

### 5.2 3 卡 DDP fallback

只有 E00 单卡 pilot OOM 时启用：

| 配置 | 取值 |
|---|---|
| world size | 3 |
| Llama3 global batch | `63 = 3 GPUs × per-device batch 3 × grad_accum 7` |
| steps | 不预先写死；必须从 smoke test 日志确认 dataloader length、`drop_last`、DistributedSampler padding 与最后一个 accumulation 周期 |
| OOM 二级方案 | `per-device batch=1, grad_accum=21`，保持 global batch 63 |
| gradient checkpointing | 仍 OOM 时统一对所有相关方法打开，并在正文与效率表标注 |

### 5.3 Llama2 附录协议

| 配置 | 取值 |
|---|---|
| base model/tokenizer | Llama-2-7B base/tokenizer，具体 revision `[待核对并锁定 hash]` |
| 默认调度 | 优先单卡并行，保持 official batch 32 的最近可行配置；若 OOM 再 DDP |
| LR/WD | LR `2e-5`，weight decay `0` |
| warmup/scheduler | warmup `.03`，cosine，decay floor 按官方脚本 `[待核对]` |
| target modules | math/code：q/k/v/o + MLP；chat：q/k/v/o |
| precision/max length | BF16，max length 1024 |

---

## 6. CovRA 方法-配置字段映射

### 6.1 论文最终方法字段

| 当前方法组件 | 配置字段 | 主档默认值 | 对应实验 |
|---|---|---:|---|
| 校准样本 | `N` / `calibration.num_samples` | 1024 | E00/E01/E10 |
| 全局/分组草图平衡 | `lambda` / `preallocation.lambda_cov` | 1.0 | E07 |
| 分组数 | `B` / `preallocation.response_agg_groups` | 4 或 8；最终由 pilot 显存确认 | E07/E10 |
| 随机草图维度 | `d_s` / `preallocation.sketch_dim` | 主档至少 64；需验证支持 K=32 | E00/E10 |
| 候选数 | `K` / `preallocation.top_k_atoms` | r8 主档 `K>=32` | E00/E01 |
| 正负响应拆分 | `rho` / sign split threshold | `[待定并预注册]` | E08 |
| 类型尺度归一化 | `type_scaling` | enabled | E08 |
| 对数压缩 | `log_compression` | enabled | E08 |
| 条件边际覆盖 | `conditional_marginal` | enabled | E04 |
| 离散求解 | `solver=dp` | enabled | E08 |
| 方向初始化 | `subspace_init` | enabled，`B0=0` | E06 |
| 秩边界 | `r_min,r_max` | r8：`r_min=2或4,r_max=32` | E00/E01 |
| 预算下界 | `eta` | strict 主表要求尽量精确；fallback 记录实际误差 | E02 |

### 6.2 当前实现遗留字段

| 当前字段 | 位置 | 处理方式 |
|---|---|---|
| `dico.taxonomy.permutation_count=1000` | `configs/base.yaml`, `configs/dico/base.yaml` | 不写作最终方法核心超参；若代码仍读取，标为当前实现遗留机制 |
| `dico.taxonomy.fdr.method=BH` | 同上 | 不写作最终方法核心超参 |
| `dico.procurement.beta` / `preallocation.beta` | 同上 | 与论文 DP 求解不一致；若继续使用，必须列为实现偏差 |
| `coverage.relative_stop_delta` / 历史 `delta` | README/历史配置 | 不写作最终方法核心超参 |
| 历史 `h` | 历史窗口/平滑描述 | 不写作最终方法核心超参 |

下一步若要真正执行实验，必须先新增或修改 config，使 `top_k_atoms>=r_max`、`dropout=0`、`gradient_checkpointing=false`、`calibration.num_samples=1024`、adapter FP32 与本计划一致。

---

## 7. 方法专属超参数

| 方法 | 专属配置 | 主实验地位 |
|---|---|---|
| Uniform LoRA | rank 8，alpha 16，B=0，dropout 0 | 必须 |
| AdaLoRA | A/E/B；init rank 12，target average rank 8，`tinit=150,tfinal=900,deltaT=1`；全局无 floor 裁剪、E 可恢复梯度、EMA `.85/.85`、未平方正交正则权重 `.5`、LR `5e-4`；报告 physical/peak/final | 必须 |
| GoRA-public | 1024 固定样本、batch4、256 次 direct weight-gradient backward；r=8，rmin=4，rmax=32，gamma 脚本语义 `.05`，union_mean，moderate rounding，rank-stabilized dynamic scaling，GPU FP32 pseudo-inverse init，B LR 16× | 必须 |
| GoRA-BM | GoRA rank 分配投影到 strict equal-budget；标明为修改版，不代表官方 GoRA | 必须，E02 |
| EVA | 官方实现与 calibration 规则核对后执行；实际参数量必须报告 | 必须 |
| CovRA | 使用第 6 节最终方法字段；主档必须满足 `K>=r_max>r_ref` | 必须 |
| LoRA-GA | stable gamma 16，按官方/上游初始化 | 推荐 |
| rsLoRA / LoRA+ | 缩放或学习率策略基线，不是静态自适应秩核心对照 | 推荐/附录 |
| DoRA/OLoRA/PiSSA | 仅在算力充足档加入 | 可选 |

---

## 8. 模型与数据集矩阵

| 编号 | 模型 | 训练数据 | 样本数 | 测试集 | 正文/附录 | 处理要求 |
|---|---|---|---:|---|---|---|
| MD1 | Llama-3.1-8B | MetaMathQA | 100,000 | GSM8K test | 正文 | 使用 GoRA/LoRA-GA prompt fallback；过滤、截断、答案格式需锁 hash |
| MD2 | Llama-3.1-8B | CodeFeedback code-only | 100,000 | HumanEval | 正文 | 只训练 code label；抽样与过滤规则 `[待核对]` |
| MD3 | Llama-3.1-8B | WizardLM | 52,000 | MTBench | 正文 | prompt 与对话格式 `[待核对]` |
| MD4 | Llama-2-7B | MetaMathQA | 100,000 | GSM8K test | 推荐附录 | 跟随 LoRA-GA/GoRA Llama2 协议 |
| MD5 | Llama-2-7B | CodeFeedback code-only | 100,000 | HumanEval | 推荐附录 | 跟随 LoRA-GA/GoRA Llama2 协议 |
| MD6 | Llama-2-7B | WizardLM | 52,000 | MTBench | 推荐附录 | 跟随 LoRA-GA/GoRA Llama2 协议 |

扩展模型与混合任务组合不进入正文主表。

---

## 9. 参数预算匹配规则

Llama3 q/k/v/o uniform rank 预算：

| 档位 | r_ref | 目标预算 | CovRA 候选与边界 |
|---|---:|---:|---|
| 主档 | 8 | 6,815,744 | `r_min=2或4,r_max=32,K>=32` |
| 高档 | 32 | 27,262,976 | `r_min=8或16,r_max=128,K>=128`；执行前先 pilot |
| 极高档 | 128 | 109,051,904 | 暂停；需先证明 `K>=256`、草图维度和显存可行 |

strict budget 规则：

- 不得超出预算。
- 若无法精确匹配，绝对误差不得超过一个最小可行参数增量，并同时报告相对误差。
- 可精确匹配时要求误差为零。
- 每次报告 `requires_grad 参数量、final active 参数量、peak active 参数量、optimizer state 估计、预算误差`。
- AdaLoRA 单列 peak/final；不强行同时匹配。
- GoRA-public 与 GoRA-BM 分开报告，不能混写。

---

## 10. Seed、重复实验与 calibration 稳定性

| 用途 | seed 设置 |
|---|---|
| 正文主实验 | 42/43/44，报告 mean ± sample std |
| E00 pilot | seed 42 |
| GoRA 配置敏感性 | seed 42；不用于选择主配置 |
| CovRA 核心机制消融 | 42/43/44 |
| calibration manifest | 主结果固定 training-only 1024 samples；calibration seed 42，与训练 seed 解耦 |
| calibration 稳定性 | calibration seed 42-51，5-10 次，仅分配不训练 |
| 高成本扩展 | 先 seed 42，若进入正文或关键附录再补 42/43/44 |

calibration 稳定性报告：

- rank 分配 Spearman 相关系数。
- 预算加权 L1 距离。
- 各模块类型平均秩波动。
- 候选子空间相似度。
- 选差异最大的两种配置做少量训练确认，放推荐档。

当前主 profile 的训练 seed 为 42/43/44；`calibration.seed` 始终保持 42。CovRA 的 `preallocation.sketch_seed` 与方向初始化仍随 run seed 变化，用于估计方法随机性。三个 run 的 calibration `selection_hash` 必须相同。

---

## 11. 评测协议

| 任务 | 主表稳定评测 | GoRA fallback/附录复现 | 指标 |
|---|---|---|---|
| GSM8K | greedy decoding，temperature 0，固定 prompt 与 regex | 若确认 GoRA 使用随机采样，再单列 vLLM BF16、temperature `.8`、top_p `.95`、max tokens 1024 | accuracy |
| HumanEval | 官方 harness，明确使用 pass@1 无偏估计公式 | 可补 LoRA-GA fallback：5 samples/task，temperature `.8`，top_p `.95` | pass@1 |
| MTBench | FastChat `v0.2.36`，本地 Llama3.1-70B-Instruct judge | 不与 GoRA 三 judge 平均分做绝对值比较 | MTBench-local |

MTBench-local 必须锁定并落盘：

- judge prompt 版本。
- conversation template。
- judge decoding temperature。
- judge seed。
- position bias 处理。
- 是否交换回答顺序。
- 失败重试规则。
- judge 模型 revision 与 tensor-parallel 设置。

同一 checkpoint 只评测一次并归档 raw generations、extracted answers 和 scores；禁止测试集选优。

---

## 12. 主实验矩阵

| ID | 目的 | 模型 | 数据集 | 方法 | 参数预算 | 训练超参数 | seed | GPU | 运行次数 | 评测指标 | 对应主张 | 输出图表 | 优先级 |
|---|---|---|---|---|---|---|---|---|---:|---|---|---|---|
| E00 | 调度与可行性 pilot | Llama3.1-8B | MD1 | LoRA, CovRA | r8 | 单卡 `4×16=64`；OOM 后 DDP fallback | 42 | 1/run；fallback 3/run | 2 | peak mem, steps, params | 验证调度与配置可跑 | pilot 日志 | 必须 |
| E01 | 公平主结果 | Llama3.1-8B | MD1-MD3 | LoRA, AdaLoRA, GoRA-public, EVA, CovRA | r8 | 默认单卡并行协议 | 42/43/44 | 1/run，并行占满3卡 | 45 | GSM8K acc, HumanEval pass@1, MTBench-local | CovRA 主性能 | Table 1 | 必须 |
| E02 | strict-budget 核验 | Llama3.1-8B | MD1-MD2 | LoRA, CovRA, GoRA-BM | r8 strict | 默认单卡并行协议 | 42/43/44 | 1/run | 6-18 | acc/pass@1 | 同预算公平性 | Table 2 | 必须 |
| E03 | GoRA 配置敏感性 | Llama3.1-8B | MD1 | GoRA N64, gamma .08, N64+.08 | r8 | 默认单卡并行协议 | 42 | 1/run | 3 | GSM8K acc | 配置差异敏感性 | Appendix table | 必须 |
| E04 | 条件估值对照 | Llama3.1-8B | MD1 | CovRA vs CovRA-I | r8 | 唯一差异：条件残差价值 vs 固定独立能量 | 42/43/44 | 1/run | 3 新跑 | GSM8K acc | 条件边际贡献 | Table 3 | 必须 |
| E05 | 方向级结构对照 | Llama3.1-8B | MD1 | CovRA-I vs CovRA-M | r8 | 唯一差异：方向级逐秩结构 vs 模块级单一标量 | 42/43/44 | 1/run | 3 新跑 | GSM8K acc | 方向级价值 | Table 3 | 必须 |
| E06 | rank/init 2×2 | Llama3.1-8B | MD1 | Uniform+Random, CovRA-rank+Random, Uniform+CovRA-init, CovRA-full | r8 | 复用 E01 的 Uniform/CovRA-full；新增两个格子 | 42/43/44 | 1/run | 6 新跑 | GSM8K acc | rank 与 init 分离 | 2×2 table | 必须 |
| E07 | 全局/分组草图消融 | Llama3.1-8B | MD1 | Global Only, Grouped Only | r8 | 仅改 `lambda`/分组草图组件 | 42/43/44 | 1/run | 6 | acc + rank dist | 草图组件必要性 | Ablation table | 必须 |
| E08 | 支撑组件消融 | Llama3.1-8B | MD1 | w/o Sign Split, w/o Type Scaling, w/o Log Compression, Proportional Rounding | r8 | 每次只关一个组件 | 42/43/44 | 1/run | 12 | acc + budget utilization | 支撑组件贡献 | Ablation table | 必须 |
| E09 | 效率与预算报告 | Llama3.1-8B | MD1-MD3 | LoRA, AdaLoRA, GoRA-public, EVA, CovRA | r8 | piggyback E01 | 42/43/44 | 1/run | 0 新跑 | time, memory, params | 开销可接受 | Table 4 | 必须 |
| E10 | calibration 稳定性 | Llama3.1-8B | MD1 | CovRA | r8 | 仅分配，不训练 | cal 42-51 | 1/run | 0 训练 | Spearman, weighted L1, subspace similarity | 分配稳定性 | Appendix/正文简表 | 推荐 |
| E11 | 补充基线 | Llama3.1-8B | MD1-MD3 | LoRA-GA, rsLoRA, LoRA+ | r8 | 默认单卡并行协议 | 42/43/44 | 1/run | 27 | 同 E01 | 补充强基线 | Appendix | 推荐 |
| E12 | Code 机制确认 | Llama3.1-8B | MD2 | E04-E06 关键机制 | r8 | 默认单卡并行协议 | 42/43/44 | 1/run | 12 | HumanEval pass@1 | 机制非数学特例 | Appendix | 推荐 |
| E13 | r32 预算扩展 | Llama3.1-8B | MD1-MD2 | LoRA, GoRA, CovRA | r32 | 必须先通过 `K>=128` pilot | 42/43/44 | 1/run 或 fallback | 18 | acc/pass@1 | 高预算趋势 | Appendix | 推荐 |
| E14 | Llama2 原始组合 | Llama2-7B | MD4-MD6 | LoRA, GoRA, CovRA | r8 | Llama2 附录协议 | 42/43/44 | 1/run 或 fallback | 27 | 同任务指标 | 跨模型泛化 | Appendix | 推荐 |
| E15 | r128 高预算 | Llama3.1-8B | MD1-MD2 | LoRA, GoRA, CovRA | r128 | 暂停；需证明 `K>=256` 可行 | 42/43/44 | 待定 | 18 | acc/pass@1 | 极高预算趋势 | Appendix | 可选 |
| E16 | 更多基线 | Llama3.1-8B | MD1-MD3 | DoRA, OLoRA, PiSSA | r8 | 默认单卡并行协议 | 42/43/44 | 1/run | 27 | 同 E01 | 补充比较 | Appendix | 可选 |
| E17 | DDP batch 差异检查 | Llama3.1-8B | MD1 | LoRA, GoRA, CovRA | r8 | DDP global batch 63 | 42 | 3/run | 3 | GSM8K acc | DDP fallback 影响 | Appendix | 条件触发 |

E02 复用规则：若 E01 的 LoRA 与 CovRA 已是 strict-budget 配置，则 E02 只新增 GoRA-BM 的 6 个 run；若 E01 的 CovRA 是 method-faithful 非 strict 配置，则 E02 必须补跑 LoRA/CovRA strict 版本，总计最多 18 个结果。

必须档 formal training runs：E01 45 + E02 6-18 + E03 3 + E04 3 + E05 3 + E06 6 + E07 6 + E08 12 = 84-96；另有 E00 pilot 2 个 run。E09 不新增训练。

上述 84-96 是完整论文计划，不等于当前唯一启动命令会立即运行的数量。当前 `python launch_covra.py` 默认只运行 E01 数学主组合的 12 个 v3 runs；E02 需设置 `COVRA_PROFILE=e02_strict_budget` 后另启 9 个 runs。EVA、MD2/MD3 与 E03-E17 尚未接入默认 launcher，不得误报为已经执行。

---

## 13. CovRA 核心机制与诊断

| 分析 | 来源 | 输出 |
|---|---|---|
| 条件估值 | E04 | CovRA vs CovRA-I mean±std，rank 分布差异 |
| 方向级结构 | E05 | CovRA-I vs CovRA-M，模块级标量损失 |
| 响应重叠 | E04/E05 artifacts，无新增训练 | overlap heatmap；overlap 与 `r_m^CovRA-r_m^CovRA-I` 的相关分析；边际增益衰减曲线 |
| rank/init 交互 | E06 | 2×2 控制表 |
| 全局/分组草图 | E07 | rank histogram，候选恢复率，任务分组响应差异 |
| 支撑组件 | E08 | sign split、type scaling、log compression、DP 的单因素消融 |
| rank 分布 | E01/E13 | layer × module heatmap，rank concentration 指标 |
| 逐秩效用 | E04/E05 | utility per rank 曲线 |

这些实验用于解释 CovRA 机制，不与 GoRA 主结果混为一谈。

---

## 14. 效率分析

效率指标从 E01/E13 piggyback 采集，不新增训练 run：

- calibration/init wall time。
- training wall time。
- peak GPU memory。
- tokens/s。
- adapter 参数量与 optimizer state 估计。
- rank allocation overhead。
- MTBench 本地 judge wall time `[待核对]`。

效率表至少包括 LoRA、AdaLoRA、GoRA-public、EVA、CovRA。

---

## 15. 正文与附录图表规划

| 位置 | 图表 |
|---|---|
| 正文 Figure 1 | CovRA 方法总览 |
| 正文 Table 1 | E01 主结果，三任务 mean±std，含实际参数量 |
| 正文 Table 2 | E02 strict-budget 与预算误差 |
| 正文 Table 3 | E04-E08 核心消融 |
| 正文 Figure 2 | 预算水平或主性能曲线 |
| 正文 Figure 3 | 响应重叠与逐秩效用 |
| 正文 Figure 4 | CovRA/GoRA/EVA rank 分布 heatmap |
| 正文 Table 4 | 效率与开销 |
| 附录 Table A | 完整公共与方法专属超参 |
| 附录 Table B | GoRA 论文/代码/config 冲突清单 |
| 附录 Table C | 全部 seed 原始结果 |
| 附录 Table D | 当前实现字段与论文最终字段映射 |
| 附录 Figure E | r32/r128 与 Llama2 扩展 |
| 附录 Figure F | calibration 稳定性与敏感性 |

GoRA 对齐协议示意不占正文图号，放入附录或表格。

---

## 16. 3×A800 调度方案

默认调度：

- 三张 A800 全部使用，但每个 run 默认单卡。
- 同一配置的三个 seed 并行：GPU0 seed42，GPU1 seed43，GPU2 seed44。
- Llama3 默认 `batch=4, grad_accum=16, global batch=64`。
- 日志目录、checkpoint 名、manifest 必须包含 experiment id、method、dataset、budget、seed、global batch、world size。
- 每个 run 生成 `stage_metrics.jsonl`；服务器检测到 `nvidia-smi` 时同步生成约 1 秒采样的 `gpu_monitor.csv`。

fallback 调度：

- 单卡 OOM 后才启用 3 卡 DDP。
- DDP 使用 `per-device batch=3, grad_accum=7, global batch=63`。
- DDP 下 optimizer steps 必须来自 smoke test 日志，不按除法写死。
- 若 DDP 仍 OOM，先 `per-device batch=1, grad_accum=21`；仍 OOM 时统一开启 gradient checkpointing。

推荐先跑 E00，再决定后续所有实验使用默认单卡并行还是 DDP fallback。

---

## 17. 实验执行顺序

1. 建立 source manifest：GoRA paper SHA、GoRA repo commit、历史脚本 commit、LoRA-GA commit、FastChat tag、数据 hash、模型/tokenizer revision。
2. 建立数据 manifest：训练、calibration、验证、测试 split 的样本数、hash、prompt 示例与 token length 统计。
3. 跑 E00 单卡 pilot，确认显存、global batch 64、steps、warmup、LR schedule、参数量日志。
4. 若 E00 单卡通过，全部必须档默认用三卡并行单卡 run；若失败，切 DDP fallback 并重跑 smoke test。
5. 预注册 GoRA-public 主配置，不看测试集结果。
6. 跑 E01 主结果。
7. 跑 E02 strict-budget 核验。
8. 跑 E03 配置敏感性，仅入附录。
9. 跑 E04-E08 机制与支撑组件消融。
10. 汇总 E09 效率表与正文图表。
11. 主结论稳定后进入推荐档 E10-E14。
12. 算力充足时进入 E15-E17。

---

## 18. 必须、推荐和可选清单

| 档位 | 实验 | formal training runs | 用途 |
|---|---|---:|---|
| 最小可投稿 | E00-E09 | 84-96 + 2 pilot | 主结果、预算公平、GoRA 敏感性、CovRA 核心机制、支撑组件、效率 |
| 推荐完整 | E00-E14 | 168-180 + 2 pilot | 增加补充基线、Code 机制确认、r32、Llama2、calibration 稳定性 |
| 算力充足 | E00-E17 | ≤216 + 2 pilot | 增加 r128、更多基线、DDP batch 差异检查 |

---

## 19. 测试与验收

| 检查 | 验收标准 |
|---|---|
| 文档自查 | 不再出现 `CovRA candidates=8` 作为 r8 主配置；不再把 `beta,h,delta,permutation,BH-FDR` 写作最终方法核心超参 |
| 方法对照 | E04/E05/E06 每个变体均有唯一差异定义 |
| 调度语义 | 所有主实验明确 single-GPU global batch 64；DDP fallback 明确 global batch 63 与日志确认要求 |
| 配置一致性 | 对照 `configs/base.yaml` 与 `configs/dico/base.yaml`，列出当前实现字段与论文最终字段差异 |
| 参数量检查 | 所有方法训练前后输出 `requires_grad`、active rank、budget error；strict-budget 误差合格 |
| 数据检查 | 每个 split 输出样本数、hash、prompt 示例、token length 分布 |
| 训练检查 | 日志确认 world size、global batch、optimizer steps、warmup steps、LR group、dropout、precision、gradient checkpointing 状态 |
| 性能检查 | 分阶段记录 allocated/reserved/allocated peak/reserved peak；GoRA 1024 样本应记录 256 backward；CovRA 不允许 CUDA 配置静默 fallback CPU |
| 评测检查 | greedy 主表与 GoRA fallback 附录分开；HumanEval 使用官方 pass@1 估计公式；MTBench-local judge 配置完整 |
| 结论检查 | 正文每个 claim 均能映射到 E01/E02/E04-E09 |
| 可复现检查 | 所有 run 的 config、git commit、数据 hash、模型 hash、seed、命令行参数写入 manifest |

---

## 20. 尚待核对的 GoRA 与 CovRA 配置

- Llama-3.1-8B base model 的确切 HuggingFace revision 与 tokenizer revision。
- GoRA final benchmark 使用的 GSM8K/HumanEval/MTBench generation 参数与答案后处理脚本。
- GoRA 训练数据的 exact subset、shuffle、filter 与 prompt 模板。
- GoRA 论文中 N=64、历史脚本中 N=32、当前 parser 默认 N=8 的最终解释。
- MetaMath gamma `.08` 与历史脚本 `.05` 的来源差异。
- LoRA adapter dtype、DeepSpeed FusedAdam、gradient clipping、FlashAttention 与 activation checkpointing 的最终生效状态。
- EVA 公平复现所需版本、calibration 数据、target modules 与实际参数量。
- CovRA 的 `rho` 默认值、type scaling 与 log compression 在当前代码中的准确字段名和生效状态。
- 本地 Llama3.1-70B-Instruct judge 与 GoRA 三 judge 平均分之间的偏差。

---

## 21. 默认假设

- “3卡训练”解释为三张 A800 都参与实验调度，而不是强制每个 run 必须 3 卡 DDP。
- 单卡 pilot 通过时，主实验优先保持 GoRA global batch 64。
- GoRA-public 主配置按预注册来源规则确定，不根据 GSM8K/HumanEval/MTBench 测试结果选择。
- r32 与 r128 在候选数、草图维度、显存和时间 pilot 通过前不进入正式训练队列。
- GoRA final benchmark 脚本缺失时，主表使用稳定 greedy 内部评测，GoRA fallback 随机协议放附录。
- Fable5 原始计划只作为审查输入，不覆盖、不改写。
