# CovRA 公平对齐实验工程

本仓库实现 **CovRA**（代码中的历史方法名仍为 `dico_cd_da`），并在同一个训练器中提供正式对齐的 Uniform LoRA、AdaLoRA、GoRA-public 与 GoRA-BM。GoRA 行为锁定到官方仓库 commit `4037d4d6ba67ff88de87f90b943ff4e3a3649b67`；代码/脚本行为优先于论文冲突项。旧 `gora_bw` 仅作为 legacy pilot 保留，不进入正式启动队列，也不得标记为 GoRA-public。

方法论文档见仓库根目录 `CovRA_v0_6_2_AAAI方法论中文初稿_主文压缩_复现细节保留版.md`（以此版本为准；方法在 v0.5 改名为 **CovRA**，早期 DiCo_v0.3 系列草稿已被后续版本部分推翻并从仓库移除，见文中说明）。

> **当前验证边界（SDPA v4，2026-07-15）**：LoRA、AdaLoRA、GoRA-public、GoRA-BM 的组件测试和 tiny 模型正式入口集成测试已完成；目标服务器因 GLIBC/CUDA 工具链不兼容无法加载 FlashAttention2，正式协议现统一使用 PyTorch SDPA。真实 Llama-3.1-8B/A800 仍需重新执行 E00，因此只能标记为 `IMPLEMENTED_AND_CPU_VERIFIED` 或 `IMPLEMENTED_NOT_GPU_RUN`，不能标记为 GPU 已验证。旧 v3/FlashAttention2 输出一律视为 `legacy_protocol_pilot`。正式新结果必须写入 `outputs/e01_llama3_r8_aligned_sdpa_v4`。

---

## 当前默认实验模型与数据集

当前仓库的正式 r8 主配置默认使用以下模型与数据：

| 位置 | 当前默认值 | 说明 |
|---|---|---|
| 基础模型 | `meta-llama/Llama-3.1-8B-Base` / 本地等价路径（常见目录名如 `Meta-Llama-3.1-8B-Base`） | 与 GoRA Llama3.1 主协议对齐；具体 model/tokenizer revision 仍需在 A800 服务器 E00 中锁定并写入 manifest |
| tokenizer | 与基础模型同源 | 若使用本地模型目录，目录内必须包含 tokenizer 文件 |
| 主训练集 | `data/metamathqa/train.jsonl`，MetaMathQA-100K | 先固定前 100,000 个唯一成员，再以公共 `dataset_seed=42` 生成所有方法共享的训练顺序；1563×64 实际曝光 100,032 条，末尾重复 32 条并写入 manifest |
| 主评测集 | `data/gsm8k/main/test.jsonl`，GSM8K test | 1319 条，随仓库携带，主表使用 greedy 生成式 exact-match |
| Code 扩展组合 | CodeFeedback code-only → HumanEval | 作为 GoRA 原始组合之一保留在实验计划中；当前需要先准备 CodeFeedback 数据 |
| Chat 扩展组合 | WizardLM → MTBench | 作为 GoRA 原始组合之一保留在实验计划中；MTBench-local 需要服务器端本地 judge 验证 |
| 本地 tiny 路径 | `data.source=tiny` 或测试内置 tiny 模型 | 只用于 CPU 单测/小模型集成/dry-run，不代表正式实验 |

本地阶段的当前结论是：

`本地实现与静态验收已完成，可以上传服务器并进入E00 GPU pilot。`

这句话只表示代码、配置、CPU/小模型测试、manifest、启动脚本和本地 dry-run readiness 已通过；真实 Llama-3.1-8B、3×A800、1024 样本校准、显存、tokens/s、完整 GSM8K/HumanEval/MTBench 评测仍然必须标记为 `IMPLEMENTED_NOT_GPU_RUN`，不能写成 GPU 已验证。

---

## 目录

1. [项目结构](#1-项目结构)
2. [环境配置](#2-环境配置)
3. [数据集准备](#3-数据集准备)
4. [模型准备](#4-模型准备)
5. [与 GoRA 对齐的实验协议](#5-与-gora-对齐的实验协议)
6. [DiCo 与 GoRA 的超参数对齐规则](#6-dico-与-gora-的超参数对齐规则)
7. [配置组织方式](#7-配置组织方式)
8. [路径配置说明](#8-路径配置说明)
9. [启动命令](#9-启动命令)
10. [pytest / 配置合法性检查](#10-pytest--配置合法性检查)
11. [结果保存与日志](#11-结果保存与日志)
12. [常见问题排查](#12-常见问题排查)
13. [从零复现实验的完整命令流程](#13-从零复现实验的完整命令流程)

---

## 1. 项目结构

```text
dico_rank_experiments/
├── src/dico/                 # 核心算法与训练代码（唯一的实现，无 peft 依赖）
│   ├── calibration.py        # 监督 token 校准样本、样本 id/hash 与响应统计辅助
│   ├── atom_svd.py           # 候选方向提取：随机草图、全局/分组草图、SVD 方向原子
│   ├── covra_core.py         # Final CovRA/CovRA-I/CovRA-M：sign split、条件边际、逐秩效用
│   ├── preallocation.py      # 编排最终 CovRA 预分配，产出 rank 分配 + direction_bank
│   ├── rank_budget.py        # 真实参数成本 DP、多选背包、strict budget 审计
│   ├── init.py               # 方向锚定初始化(Gram-Schmidt, B=0)
│   ├── taxonomy.py           # reference CovRA taxonomy / 置换检验
│   ├── procurement.py        # reference CovRA quota-aware procurement / relaxation
│   ├── pseudo_groups.py      # 无任务组标签时的伪组谱聚类
│   ├── gora.py                # 正式 GoRA-public/GoRA-BM、伪逆初始化和 strict repair
│   ├── gora_bw.py             # legacy：旧 GoRA-BW pilot，不进入正式队列
│   ├── adalora.py             # 正式 A/E/B、EMA 重要度、全局预算与正交正则
│   ├── lora_static.py        # 默认 LoRA 注入路径：固定 rank 的 StaticLoRALinear(自研，非 peft)
│   ├── lora_masked.py        # 历史/等价性参考：动态 rank 的 MaskedLoRALinear
│   ├── lora_scaling.py       # LoRA scaling 公式(alpha_over_sqrt_r 即 rsLoRA-style / alpha_over_r / alpha_over_max_rank)
│   ├── model_loader.py       # 模型/分词器加载、目标模块发现、tiny 合成模型(测试用)
│   ├── data.py                # 数据集加载(本地文件优先，其次 HF datasets)、GSM8K prompt/answer 构造
│   ├── evaluator.py          # loss 评估 + GSM8K 生成式 exact-match 评估
│   ├── trainer.py            # 训练主循环：预分配→LoRA注入→训练→评估→落盘
│   ├── config.py             # YAML 加载、inherits 继承合并、--override 应用
│   ├── logging_utils.py      # train/eval jsonl 日志、rank_history.csv
│   └── utils.py, path_utils.py, cache.py, ...
├── configs/
│   ├── base.yaml              # 全局默认值(模型/数据/训练/LoRA/预算/校准/预分配/dico/评估)
│   ├── dico/                  # 正式配置：lora_r8 / adalora_r8 / gora_public_r8 / gora_bm_r8 / dico_cd_da_r8
│   └── ablations/             # 消融 config：全部 inherits 自 dico_cd_da_r8.yaml，只覆盖 1-2 个字段
├── scripts/
│   ├── run_experiment.py      # 唯一的 Python 入口：加载config→apply overrides→train()
│   └── run_*.sh                # 对应每个方法/消融的 shell 包装，透传所有额外参数
├── data/
│   ├── README.md              # 说明本地已随仓库携带 GSM8K jsonl 的原因
│   └── gsm8k/main/{train,test}.jsonl   # 随仓库携带的 GSM8K 数据(7473/1319条)
├── outputs/                    # 训练产出（被 .gitignore 忽略，需要自己创建/写入）
│   └── <project.output_dir 相对路径>/<experiment_name>/...
├── tests/
│   ├── unit/                   # 组件级单测(CovRA core/DP/init/manifest/evaluator/gora_bw/lora等)
│   ├── configs/                # 配置合法性检查(legacy路径必须不存在、method白名单)
│   └── scripts/                # 入口/脚本级 smoke test(dry-run)
├── pytest.ini                  # pythonpath=src, testpaths=tests
├── requirements.txt
└── CovRA_v0_6_2_AAAI方法论中文初稿_主文压缩_复现细节保留版.md   # 方法论权威文档
```

**逻辑意义**：`src/dico/` 是唯一实现；`configs/dico/*.yaml` 是"一个方法 = 一个可直接跑的 config"，`configs/ablations/*.yaml` 全部只从 `dico_cd_da_r8.yaml` 派生、只改一两个字段；`scripts/run_*.sh` 提供最终方法脚本和保留的 legacy alias，脚本本身不包含任何超参数，所有参数都来自 config + `--override`；`outputs/` 是运行时产物目录，不进仓库（`.gitignore` 里 `outputs/*` 被忽略，只保留 `outputs/.gitkeep`）。

---

## 2. 环境配置

### Python 与 CUDA

- Python：建议 **3.10 或 3.11**（`transformers`/`bitsandbytes`/`accelerate` 生态在这两个版本上最稳定；仓库开发时也在 3.13 下跑通过纯 CPU 单测，但 GPU 训练服务器建议用 3.10/3.11 以避免 bitsandbytes 的 wheel 兼容性问题）。
- CUDA：与训练服务器的驱动匹配的 CUDA 11.8 或 12.1 均可（决定了要装哪个 torch 编译版本）；`bf16` 训练建议 Ampere 及以上架构（A100/A800/RTX 30/40 系列）。
- 本仓库**不依赖 HuggingFace `peft`**：LoRA 是自研实现（`src/dico/lora_static.py` 默认路径，`lora_masked.py` 为历史等价参考），`requirements.txt` 里没有 `peft`，不需要装。

### 依赖版本建议

`requirements.txt` 目前只列包名、未锁版本，按下表补齐版本号（与代码实际用到的 API 对应，例如 `AutoModelForCausalLM.from_pretrained(..., dtype=...)`、`BitsAndBytesConfig`、`gradient_checkpointing_enable`、`sklearn.cluster.SpectralClustering`）：

| 包 | 建议版本 | 用途 |
|---|---|---|
| `torch` | `>=2.2,<2.6`（按 CUDA 版本选对应 wheel） | 训练/推理主框架 |
| `transformers` | `>=4.43`（Llama-3.1 需要较新版本；若用 Qwen2.5/Qwen3 也建议 `>=4.43`） | 模型/分词器加载 |
| `accelerate` | `>=0.30` | `device_map="auto"` 多卡/自动放置 |
| `bitsandbytes` | `>=0.43`（仅 `load_in_8bit`/`load_in_4bit` 时需要，CPU/纯 bf16 训练可不装） | 8bit/4bit 量化加载 |
| `datasets` | `>=2.19` | 走 HuggingFace 数据源分支时使用（默认走本地 jsonl 可不依赖） |
| `scikit-learn` | `>=1.3` | `pseudo_groups.py` 的谱聚类 + silhouette score |
| `numpy` / `pandas` | 与 torch/transformers 兼容的最新稳定版即可 | 通用数值/表格处理 |
| `pyyaml` | `>=6.0` | config 解析 |
| `tqdm` | 任意较新版本 | 无强依赖 |
| `pytest` | `>=7.0` | 单测 |

安装：

```bash
python -m venv .venv && source .venv/bin/activate   # 或使用 conda
pip install -r requirements.txt
# 如需 8bit/4bit 量化加载，确认 bitsandbytes 已装且与 CUDA 版本匹配
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 3. 数据集准备

### 训练集：MetaMathQA-100K（与 GoRA 协议对齐）

默认训练集是 MetaMathQA-100K，对应 GoRA 论文（arXiv 2502.12171）的 Appendix C.4 实验协议。
本地路径：`data/metamathqa/train.jsonl`（**不随仓库携带**，首次运行前须执行下载脚本）。

**下载命令：**

```bash
# 标准下载（需要访问 HuggingFace）
python scripts/download_data.py

# 服务器无法访问 hf.co 时，使用国内镜像
python scripts/download_data.py --hf-endpoint https://hf-mirror.com

# 自定义 HF 缓存目录
python scripts/download_data.py --hf-cache /your/hf_cache

# 仅检查文件是否存在（不下载）
python scripts/download_data.py --check-only
```

下载脚本是幂等的：文件已存在时直接跳过。下载完成后文件约 100MB，约 100K 条，每行一个 JSON 对象，字段为 `question` / `answer`。

**对应 config 字段（`configs/base.yaml`）：**

```yaml
data:
  train_path: data/metamathqa/train.jsonl       # MetaMathQA-100K 本地路径
  eval_path: data/gsm8k/main/test.jsonl         # GSM8K 测试集（随仓库携带）
  train_dataset: MetaMathQA-100K                # 描述性字段，不驱动加载
```

### 评估集：GSM8K（已随仓库携带）

`data/gsm8k/main/test.jsonl` 已随仓库携带（1319 条），**不可修改**，所有实验均在此测试集评估以保证结果可比性。

`data/gsm8k/main/train.jsonl`（7473 条原始 GSM8K 训练集）也随仓库携带，但**默认不再用于主实验训练**。如需在 GSM8K-only 条件下做消融，在对应 config 中 override：

```yaml
data:
  train_path: data/gsm8k/main/train.jsonl
```

### 数据加载优先级

`src/dico/data.py::load_raw_datasets` 的实际加载逻辑：

1. `data.source == "tiny"`（或 `train_path == "tiny"`）→ 内置 4 条合成样例，仅用于单测/冒烟；
2. 若配置了 `data.train_sources`（多数据源列表，见下方"混合数据集"）→ 依次加载每个源、按 `limit` 截断、打上 `group` 标签后拼接，`eval_path` 仍是单独一份；
3. 若 `data.train_path` 指向的文件/目录**存在** → 用 `_read_records` 读取本地 `.jsonl` / `.json` / `.parquet`；
4. 若本地路径**不存在** → 抛出 `FileNotFoundError` 并提示运行 `python scripts/download_data.py`，**不再静默联网下载**。

### 数据格式要求

任何自制数据集只要符合以下 jsonl 格式，即可通过 `data.train_path` 直接接入：

```json
{"question": "题目文本", "answer": "解题步骤\n#### 42"}
```

`data.py::build_sft_example` 会把 `question`/`answer` 拼成 SFT 格式，`labels` 对 prompt 部分做 `-100` mask（只在答案 token 上算 loss）。

### 混合数据集（`mixed_math_code_r8.yaml`）

`data.train_sources` 是一份 `{path, group, limit}` 列表，`load_raw_datasets` 会依次加载每个源、
按 `limit` 截断、给每条记录打上 `_group` 标签（`data.py::tokenize_records` 会把它带进 tokenized
example 的 `group` 字段）后拼接。数学源用现成的 GoRA prompt 模板（`#### 答案` 约定），代码源
（`group: code`）会跳过这条数学专用指令、用纯 instruction/response 模板。CodeFeedback 数据需要先
运行 `python scripts/download_codefeedback.py`（默认下载 `m-a-p/CodeFeedback-Filtered-Instruction`
并子采样到 5 万条，写到 `data/codefeedback/train.jsonl`）。

`calibration.group_sampling: balanced` 会在采样校准池时按组分层抽样（各组各取
`num_samples // 组数`），而不是对拼接后的全集做一次性随机/前缀采样,避免两个源文件大小差异导致
校准池里数学/代码样本比例失衡。`trainer.py` 会把采样到的校准样本的真实 `group` 标签（只要不止一
个 distinct group）写入 `config["data"]["group_labels"]`，供 `dico.split.mode: group` 使用真实
任务标签做组拆分（而不是回退到从 profile 几何结构聚类出的伪组）——`mixed_math_code_r8.yaml` 已经
把 `dico.split.mode` 设成了 `group`，因为它是仓库里唯一真正有逐样本任务标签的 config。

---


## 4. 模型准备

### 默认模型

```yaml
model:
  name_or_path: meta-llama/Llama-3.1-8B-Base
  torch_dtype: bfloat16
  device_map: auto
  load_in_8bit: false
  load_in_4bit: false
```

`meta-llama/Llama-3.1-8B-Base` 是 **gated 模型**，需要先在 HuggingFace 网站申请访问权限，再执行：

```bash
huggingface-cli login   # 填入你的 HF token（需要有该模型的访问权限）
```

### 下载方式与本地缓存

- `src/dico/model_loader.py::load_tokenizer_and_model` 直接调用 `AutoTokenizer.from_pretrained(name)` / `AutoModelForCausalLM.from_pretrained(name, ...)`，**没有显式传 `cache_dir`**，因此缓存路径完全由标准 HuggingFace 环境变量决定：
  ```bash
  export HF_HOME=/your/cache/path          # 新版 transformers 推荐
  # 或者老版本:
  export TRANSFORMERS_CACHE=/your/cache/path
  export HF_HUB_CACHE=/your/cache/path/hub
  ```
  不设置时默认落在 `~/.cache/huggingface/`。
- 如果网络访问 HuggingFace Hub 不稳定，可以设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`，或者提前用 `huggingface-cli download meta-llama/Llama-3.1-8B-Base --local-dir /path/to/local_model` 下载到本地目录。
- **推荐做法（尤其是内网/无外网的训练服务器）**：把模型下载到本地目录后，直接把 `model.name_or_path` 改成本地路径，`model_loader.py` 对本地路径和 Hub id 一视同仁：
  ```bash
  --override model.name_or_path=/path/to/local/Llama-3.1-8B-Base
  ```
  `data/README.md` 里也记录了团队在实际服务器上用本地路径（如 `/ai/lxw/lxw/Qwen3-8B`）跑实验的惯例，就是为了避免反复联网下载。

### 量化 / 多卡

- `model.load_in_8bit` / `model.load_in_4bit` 二选一，需要 `bitsandbytes`；同时打开会直接报错。
- `model.device_map: auto` 依赖 `accelerate` 自动切分到可见 GPU；单卡场景可以设 `device_map: null` 强制整模型放一张卡。

### 对应 config 字段

`model.name_or_path` / `model.torch_dtype` / `model.device_map` / `model.load_in_8bit` / `model.load_in_4bit` / `model.bnb_4bit_quant_type` / `model.bnb_4bit_use_double_quant`。

---

## 5. 与 GoRA 对齐的实验协议

### 已经在 config 层面对齐、并且代码真正执行的部分

| 维度 | 对齐方式 | 字段 |
|---|---|---|
| 目标模块 | attention 全部四个投影 | `lora.target_modules: [q_proj, k_proj, v_proj, o_proj]` |
| global batch | 64（单卡 micro batch 4 × accumulation 16） | `training.batch_size` / `training.gradient_accumulation_steps` |
| 峰值学习率 | 5e-5 | `training.learning_rate` |
| 权重衰减 | 5e-4（GoRA 附录表 Llama-3.1-8B-Base 数值） | `training.weight_decay` |
| 训练规模 | 固定 MetaMathQA 前 100,000 个唯一成员，公共 `dataset_seed=42` 打乱；1563 optimizer steps × global batch 64 = 100,032 次曝光，循环末尾重复 32 条 | `data.train_limit` / `data.shuffle` / `data.dataset_seed` / `training.max_steps` |
| cosine LR 衰减下限 | 衰减到峰值 LR 的 10%（而非 0），配合 warmup | `training.lr_decay_ratio`（自定义 `build_cosine_schedule_with_warmup_and_floor`，不再依赖 `transformers.get_cosine_schedule_with_warmup`） |
| seed | reference CovRA 的训练、校准与 sketch seed 同步使用 42/43/44；其他 baseline 保持原协议 | `seed` / `calibration.seed` / `preallocation.sketch_seed` |
| 优化器 | AdamW；LoRA/CovRA 的 A/B 同 LR；正式 GoRA-public/GoRA-BM 的 B/A LR 比固定为 16，参数组写入 manifest | `training.*` / `gora.b_lr_multiplier` |
| 目标参数预算 | `B* = Σ_m 8·(d_in+d_out)`，即目标模块集合上 r=8 的真实参数量 | `rank: 8` + `budget.mode: equal_trainable_params` |
| 预算窗 | `[η·B*, B*]`，默认 η=0.98 | `budget.enforce_min_ratio` / `preallocation.eta` |
| LoRA scaling | 标准 LoRA 使用 `alpha/r`；rsLoRA 独立配置使用 `alpha/sqrt(r)`；GoRA 使用 rank-stabilized dynamic scaling | `lora.scaling` |
| 评测指标 | GSM8K 生成式 exact-match（`####` 后数字，strict-then-flexible 抽取） | `evaluation.metric: gsm8k_accuracy` |

### GoRA 论文协议对齐现状（历史上有过"尚未打通"的项，现已基本补齐）

| GoRA 协议要求 | 当前仓库现状 | 影响 |
|---|---|---|
| MetaMathQA-100K 训练 | 已接入 `data/metamathqa/train.jsonl`；先固定前 100K 成员，再以 `dataset_seed=42` 统一打乱 | 所有方法共享完全相同的成员、顺序和曝光策略 |
| MetaMathQA-50K + CodeFeedback-50K 混合训练（§6.5） | 已接入：`data.train_sources` 加载+拼接两个源，`calibration.group_sampling: balanced` 分层采样校准池，真实 `group` 标签驱动 `dico.split.mode: group` 的任务组拆分（见上方"混合数据集"一节） | `mixed_math_code_r8.yaml` 可直接跑；需要先 `python scripts/download_codefeedback.py` |
| GSM8K + HumanEval 评测 | GSM8K 评测器已实现；HumanEval（代码生成+执行判分）也已实现（`evaluate_humaneval_pass_at_1`），但只在 `data.eval_datasets` 包含 `humaneval` 时触发 | 数学主实验（3×3）默认只跑 GSM8K；如需 HumanEval 需要显式在 config 里加 `humaneval` |
| 3 个 seed 取均值 | 默认 `e01_aligned` 为 LoRA / AdaLoRA / GoRA-public / CovRA，各 42/43/44，共 12 runs；`e02_strict_budget` 单独运行 LoRA / GoRA-BM / CovRA | `launch_covra.py` / `COVRA_PROFILE` |
| 多卡训练 | `scripts/platform_train.py` 默认 `--num-gpus 3`：**不是** DDP，而是把每个 config 的 3 个 seed 各自绑定一张卡、并行跑独立的单卡 `train()`（训练+各自的最终评估），`training.batch_size=4, grad_accum=16 = effective batch 64`（和 GoRA 单卡协议完全一致，不用再凑 63） | 12 组默认核心实验按 3 卡分批打满：训练阶段 3 卡同时训、评估阶段也 3 卡同时评（不再有 rank-0 单卡评估、另外 2 张卡空等的问题）；单卡顺序调试传 `--num-gpus 1` |
| GoRA 对 B 矩阵使用 16 倍学习率 | 已按方法专属参数组实现；scheduler 同比例缩放所有组，因此 warmup、峰值和 decay 全程维持 16× | E00 需在真实 GPU 日志复核 |

GoRA-public 使用 1024 条固定训练样本，每个 calibration batch 只做一次完整 forward/backward；仅 q/k/v/o 基础权重临时启用梯度，并通过直接权重梯度 hook 立即卸载和清空，不再使用旧实现的 activation/output-gradient 重建或 answer-only mask。之后使用 `mean(abs(W⊙G_avg))`、`union_mean`、moderate rounding、rmin=4/rmax=32、GPU FP32 伪逆 B 初始化、rank-stabilized scaling、`scale_by_lr=true/init_lr=.05` 和 B/A=16×。GoRA-BM 只额外执行 strict budget repair。正式数学配置关闭训练中途与最终的 GSM8K loss pass，只在 final checkpoint 做一次 batch=4 greedy accuracy 评测。

`protocol.unresolved_fields` 会机器可读地记录参考运行没有锁定的模型与 tokenizer revision。正式验收必须保存本地模型目录 fingerprint，不能把模型目录差异误归因为训练随机性。

### Baseline registry 与协议状态

基线状态由 `src/dico/baselines.py` 统一维护，并可用下面命令生成机器可读 JSON 与 Markdown：

```bash
python scripts/baseline_status.py \
  --json-output reports/baseline_status.json \
  --markdown-output reports/baseline_status.md
```

本地阶段状态分四类：`IMPLEMENTED_AND_CPU_VERIFIED`、`IMPLEMENTED_NOT_GPU_RUN`、`BLOCKED_BY_UNRESOLVED_PROTOCOL`、`NOT_IMPLEMENTED`。服务器阶段才允许使用 `IMPLEMENTED_AND_GPU_VERIFIED`、`GPU_RUN_FAILED`、`BLOCKED_BY_ENVIRONMENT`。截至本版本，Uniform LoRA、完整 AdaLoRA A/E/B、GoRA-public、GoRA-BM、CovRA、CovRA-I 与 CovRA-M 均已接入正式入口并完成 CPU/tiny 验证，但尚未在目标 A800 环境重新验收。EVA 仍为 `BLOCKED_BY_UNRESOLVED_PROTOCOL`，且不在本轮实现范围。

CovRA-I 与 CovRA-M 是基于 `covra_full` conditional-coverage/DP 实验协议的机制消融，不是默认 reference CovRA。二者通过 `dico_cd_da_r8_covra_full_experimental.yaml` 继承旧协议；CovRA-M 继续使用显式 `module_scalar_template`（`w_j = 1 / j`，`j=1..r_max`，`sum_to_module_energy`），该模板不得根据测试集结果或下游性能调节。

E00–E10 的当前可执行命令与 blocked 项由 `scripts/experiment_matrix.py` 生成：

```bash
python scripts/experiment_matrix.py \
  --json-output reports/experiment_matrix.json \
  --markdown-output reports/experiment_matrix.md
```

该矩阵只给已接入配置真实命令；外部协议未核对或 wrapper 未实现的项会保留在 `blocked_items`，不得手工改成可跑状态。

启动 E00 前建议先生成总 readiness 报告；在本地 CPU/无 GPU 环境用 `--require-gpu-count 0` 只做 dry-run readiness，在 A800 服务器上改成 `--require-gpu-count 3` 作为硬门槛：

```bash
python scripts/e00_readiness.py \
  --json-output reports/e00_readiness.json \
  --markdown-output reports/e00_readiness.md \
  --model-path /ai/lxw/lxw/Meta-Llama-3.1-8B \
  --output-dir outputs/e00_llama3_pilot \
  --min-free-gb 200 \
  --require-runtime-deps \
  --require-gpu-count 3
```

该报告会汇总协议预检、MetaMathQA/GSM8K 数据文件存在性与 SHA256、Llama3 模型路径或 HF id（本地路径会检查 `config.json` 和 tokenizer 文件）、Python/torch/transformers/accelerate/datasets/numpy/scipy/sklearn/pandas 等运行依赖、输出目录可写性与剩余空间、关键本地 baseline 状态、`platform_train.py` / DDP fallback launcher 是否存在，且会对 `platform_train.py` 做一次 dry-run，确认默认 3 卡调度会生成 4 个配置 × 3 个 seed 的单卡命令并保持 effective batch 64；最后检查可见 CUDA GPU 数量。它只证明“可以开始 E00 冒烟”，不代表 E00 或正式训练已经执行。本地 CPU dry-run 可以不加 `--require-runtime-deps`，此时缺少 GPU 训练依赖只会以 WARN 记录；A800 正式开跑前建议加上该参数作为硬门槛。

在启动 reference CovRA 前，先运行 dry-run 与 reference 配置测试。旧 `protocol_preflight.py` 继续服务于 `covra_full` experimental 协议，不再定义默认 CovRA：

```bash
python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml --dry-run
pytest -q tests/configs/test_covra_reference_alignment.py tests/unit/test_seed_propagation.py
```

reference 金标要求 global batch `4×16=64`、`top_k_atoms=8`、`r_max=32`、`sketch_dim=16`、dropout 0.05、gradient checkpointing 开启、`compute_device=auto`、`allocation_device=cpu`，并由 taxonomy/NSW coverage/physical utility/quota procurement 完成分配。`top_k_atoms < r_max` 是已运行参考行为，只对 `covra_v05` 放行；`covra_full` experimental 仍要求 `top_k_atoms >= r_max`。

每个 E00/E01 run 完成后，先用独立校验器检查 `run_manifest.json` 是否真的能支撑审计，再做结果汇总：

```bash
python scripts/validate_run_manifest.py \
  --output-dir outputs/e01_llama3_r8_main \
  --json-output reports/run_manifest_validation.json \
  --markdown-output reports/run_manifest_validation.md
```

该脚本会校验 manifest 核心字段、single-GPU/DDP batch 语义、adapter dtype/dropout/target modules、预算与参数量一致性、optimizer-state 估算、已记录 artifact 的路径/SHA256/大小/JSONL 行数，以及 prediction JSONL 是否包含 raw/processed/score 等必备字段。任何 FAIL 都表示该 run 暂不能进入论文表格或 seed 汇总。

高预算 r32 目前只开放 pilot 配置：`configs/dico/dico_cd_da_r32_pilot.yaml`。该配置满足 `rank=32`、`r_max=128`、`top_k_atoms=128`、`sketch_dim=192`，并把 `training.max_steps=1`、`evaluation.compute_accuracy=false` 固定为 smoke/pilot 语义。正式 r32 训练仍属于推荐扩展，必须等 E10 GPU pilot 通过后再进入正式队列。

方法实现、实验协议和剩余阻塞项的审计报告由 `scripts/audit_status.py` 生成：

```bash
python scripts/audit_status.py --output-dir reports/audit
```

它会写出三组文件：`method_implementation_audit.{json,md}`、`experiment_protocol_audit.{json,md}` 和 `status_matrix.{json,md}`。这些报告用于验收“哪些已经实现、哪些只是 dry-run ready、哪些仍被协议或 GPU 执行阻塞”；它们不包含训练分数，也不能替代 E00 pilot 或正式 GPU run。

最终交付/归档摘要由 `scripts/final_delivery_report.py` 汇总已有报告生成；它不会自己跑测试或 GPU，只记录你传入的最新测试证据，并把 README、baseline 状态、E00 readiness、E00–E10 命令、方法/协议审计、状态矩阵、未执行 GPU 项和外部协议阻塞项汇总到一个机器可读 JSON 与 Markdown：

```bash
python scripts/final_delivery_report.py \
  --json-output reports/final_delivery.json \
  --markdown-output reports/final_delivery.md \
  --test-result "pytest -q :: 278 passed, 4 warnings"
```

执行入口现在会做显式配置字段校验：`scripts/run_experiment.py` 在 dry-run 和真实训练前都会拒绝未知字段，避免 `training.typo_batch_size` 这类拼写错误静默进入实验。每个真实 run 的 `run_manifest.json` 也会记录 train/eval split 的样本数与 hash，以及实际 calibration 样本的 `sample_id`/`sample_hash`/`selection_hash`，用于后续复现实验和审稿追溯。manifest 的 `source_control` 与 `command` 字段会记录 repo root、git commit、branch、dirty 状态、dirty 文件数、status hash、启动 argv、cwd 和 Python executable，确保实验 artifact 能追溯到真实代码状态和启动命令。manifest 的 `seeds` 字段会分开记录 `base_seed`、`model_and_lora_init_seed`、rank-local `training_rng_seed`、`calibration_seed` 和 `preallocation_sketch_seed`；DDP fallback 下 `training_rng_seed=base_seed+local_rank`，与 trainer 实际行为一致。manifest 的 `config` 字段会记录 `config_resolved.yaml` 的路径与 SHA256，确保 manifest 能绑定到本次实际生效的完整展开配置。manifest 的 `scheduler` 字段会结构化记录 `cosine_with_warmup_and_floor`、warmup ratio/steps、LR floor ratio 和 optimizer-step 来源，方便核对 warmup/衰减是否真实生效。`module_budget` 会逐模块记录 `d_in/d_out`、`rank_cost=d_in+d_out`、初始/最终 rank 以及初始/最终 active LoRA 参数量，用于核对真实参数预算而不是名义 rank。manifest 的 `method_artifacts` 字段会记录预算、rank allocation、初始化摘要、诊断和 utility JSON 的路径、SHA256、字节大小和格式；`run_artifacts` 字段会记录 train/eval logs、`metrics.json`、`evaluation_protocol.json` 和 `run_summary.md` 的路径、SHA256、字节大小、格式，以及 JSONL 行数；`checkpoint_artifacts` 字段会记录最终 LoRA adapter checkpoint 的路径、SHA256、字节大小、格式以及 final-checkpoint-only 选择规则；`evaluation_artifacts` 字段会记录已写出的 GSM8K/HumanEval prediction JSONL 的路径、SHA256、行数和 raw/processed/score 必备字段，避免评测分数与原始 generation 脱钩。`run_summary.md` 从同一 payload 抽取 config hash、source/command、seeds、scheduler、method/run/checkpoint/评测 artifact、预算、参数量和 timing 字段，供人眼快速核对。runtime 字段会记录 Python/PyTorch/CUDA 版本、`CUDA_VISIBLE_DEVICES`、可见 GPU 型号列表、当前 CUDA 设备和 peak CUDA memory；`dependency_versions` 会记录 torch、transformers、accelerate、datasets、numpy/scipy/sklearn/pandas 以及可选 vLLM 的版本，E00 时可直接核对是否为预期的 A800 调度环境和软件栈。主协议使用 `model.torch_dtype: bfloat16` 加载冻结底座，并按已执行的 CovRA 参考版本将 `lora.adapter_dtype` 对齐为 `bfloat16`；既有基线仍保持各自原协议中的 FP32 adapter 设置。trainer 不再让 adapter dtype 静默跟随 base dtype，manifest 的 `precision.adapter_dtype` 来自实际 LoRA 参数。训练协议中的 `training.max_grad_norm: 1.0` 会在 `optimizer.step()` 前真实执行梯度裁剪，并写入 run manifest；训练日志同时记录每个 optimizer step 的 `grad_norm_before_clip`，供 E00 和正式 run 审计。manifest 的 `optimizer` 字段会记录实际 optimizer 名称、AdamW betas/eps、每个参数组的当前 lr、initial lr、weight decay 和参数量；`optimizer_state_estimate` 会按可训练 adapter 参数估算 AdamW 一阶/二阶动量状态字节数，参数本体存储仍由参数量与 dtype 字段分开审计；`timing` 字段会写入 calibration/allocation/initialization/training 秒数、训练 token 数和 tokens/s；其中 calibration 秒数优先来自 SVD sketch/basis/profile 三次 pass 的内部计时，allocation 秒数为预分配 wall time 扣除这些校准 pass 后的剩余。LoRA adapter checkpoint 由 `src/dico/lora_checkpoint.py` 统一保存/恢复，trainer 写出的 `masked_lora_state.pt` 已有 fresh tiny LoRA 模型恢复测试；E00 后仍需在真实 Llama3/A800 artifact 上做一次恢复 smoke。

Rank/init 分离消融目前对应两个可启动配置：

- `configs/ablations/uniform_rank_covra_init.yaml`：仍运行 CovRA 候选提取与 direction bank，但用 `preallocation.rank_override: uniform_ref` 将最终 rank 强制回统一参考秩，用于隔离 CovRA 子空间初始化收益。
- `configs/ablations/covra_rank_random_init.yaml`：保留 CovRA rank 分配，但设置 `dico.init.mode: kaiming_zero_B`，跳过 direction-anchored 初始化，用于隔离 rank 分配收益。

评测协议当前状态：

- GSM8K 主表使用 greedy decoding（`do_sample=false, temperature=0, top_p=1`），每条 raw generation、截断后文本、答案提取、`metric=exact_match` 和逐样本 `score` 写入 `eval_predictions.jsonl`。
- HumanEval 使用 greedy completion，并在 metrics/predictions 中记录 `official_unbiased` pass@1 估计器；每条 completion 归档 raw/截断文本、`metric=task_success` 和逐样本 `score`。当前实现是每题 1 个 completion，因此 pass@1 数值等同于单样本正确率，但公式和字段不是含混的 accuracy alias。
- `evaluation_protocol.json` 会在每个 run 开始写出 checkpoint 选择规则、GSM8K、HumanEval 和 MTBench-local 的协议配置。`checkpoint_selection.rule=final_checkpoint_only`，且 `uses_test_metric_for_selection=false`：trainer 只评估最终 adapter 状态一次，不按验证集/测试集指标择优。MTBench-local judge 不在 trainer 内部直接执行；外部执行器是 `scripts/mtbench_local_judge.py`，会读取 FastChat-style question/answer JSONL，按锁定的本地 judge 配置写出 `mtbench_local_protocol.json`、`mtbench_local_judgments.jsonl` 和 `mtbench_local_metrics.json`。在真实 MTBench answer set 和目标本地 70B judge 跑完前，不能报告 MTBench-local 分数；`--dry-run` 只冻结协议，不产出分数。

MTBench-local 执行器 dry-run 示例：

```bash
python scripts/mtbench_local_judge.py \
  --questions-jsonl data/mtbench/questions.jsonl \
  --answers-jsonl outputs/mtbench/model_answer/covra.jsonl \
  --output-dir outputs/mtbench/local_judge/covra \
  --judge-model meta-llama/Llama-3.1-70B-Instruct \
  --dry-run
```

多 seed / 多 run 的 manifest 汇总使用：

```bash
python scripts/collect_run_manifests.py \
  --output-dir outputs/e01_llama3_r8_main \
  --json-output reports/run_manifest_summary.json \
  --markdown-output reports/run_manifest_summary.md
```

该脚本扫描每个 run 的 `run_manifest.json`，按 `_seedN` 后缀分组，汇总 final metric、预算误差、实际参数量、global batch、optimizer steps 和耗时等字段。JSON 报告保留完整字段统计；Markdown 表也直接展示 `requires_grad`、`active_final`、`active_peak` 均值，便于正文/附录快速核对参数预算。它只汇总已经真实存在的 run manifest；不会替代 GPU smoke test，也不会生成不存在的训练结果。

### Reference CovRA 与 experimental CovRA 边界

默认 `configs/dico/dico_cd_da_r8.yaml` 使用 `allocation_method: covra_v05`，执行 taxonomy → virtual candidates → NSW coverage（含 κ residual）→ physical utility → quota-aware procurement → direction-anchored init。正式参数来自代码实际读取的 `dico.taxonomy`、`dico.pseudo_group`、`dico.split`、`dico.coverage`、`dico.procurement` 与 `dico.init`；`dico.legacy_covra_v05.*` 只为旧配置解析兼容，不是正式参数源。

`configs/dico/dico_cd_da_r8_covra_full_experimental.yaml` 保留 conditional response blocks、逐秩效用曲线与 DP 求解器。CovRA-I、CovRA-M、sign/type/log/DP、rank/init 分离消融和 r32 pilot 都继承这条 experimental 配置链，不得与 reference CovRA 三 seed 结果混合统计。启动 reference CovRA 前运行：

```bash
python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml --dry-run
pytest -q tests/configs/test_covra_reference_alignment.py tests/unit/test_seed_propagation.py
```

配置测试会锁定 reference resolved 金标并确认共享 baseline 配置未受 CovRA 专属继承链影响。

---

## 6. DiCo 与 GoRA 的超参数对齐规则

### 必须完全一致的参数（否则预算/scaling 不可比）

| 参数 | 值 | 原因 |
|---|---|---|
| `rank` | 8（作为目标预算 r_ref） | 定义 `B* = Σ 8(d_in+d_out)`，是所有方法的共同预算基准 |
| `lora.target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | 模块集合不同，`B*` 和 rank 分配含义都不可比 |
| scaling | 方法忠实设置 | 标准 LoRA/CovRA 使用固定 `alpha/r` 比例；GoRA 使用其官方 rank-stabilized dynamic scaling；不得为了表面统一破坏方法语义 |
| `lora.alpha` | 16 | 与 scaling 公式配套 |
| `budget.mode` | `equal_trainable_params` | 保证 `B*` 按真实参数量而非"平均 rank"计算 |
| `preallocation.eta` / `budget.enforce_min_ratio` | 0.98 | 预算窗下界比例，所有方法必须落入同一 `[ηB*, B*]` |
| `training.batch_size` / `gradient_accumulation_steps` / `learning_rate` / `weight_decay` | 4 / 16 / 5e-5 / 5e-4（AdaLoRA 专属 LR 5e-4） | global batch 64；保留 AdaLoRA 官方专属 LR |
| `seed` / `calibration.seed` | reference CovRA 的 training/calibration/sketch seed 同步为 42/43/44；其他 baseline 保持现有协议 | 复现已运行 reference launcher，不改变 baseline |
| `model.name_or_path` / `model.torch_dtype` | 同一底座模型、同一精度 | 最基本的公平性前提 |
| `data.train_path` / `data.eval_path` | 所有方法必须用同一份数据 | 数据不同则结果不可比，这条比任何超参数都重要 |

### CovRA 方法专属参数（GoRA 没有对应概念，不写入公共训练协议）

- `preallocation.top_k_atoms=8`、`sketch_dim=16`、`compute_device=auto`、`allocation_device=cpu`：与已运行 reference profile/SVD 和 CPU allocation 对齐；允许 `8 < r_max(32)` 并记录方向证据不足诊断。
- `dico.taxonomy`、`dico.pseudo_group`、`dico.split`：分类、伪任务组和 virtual candidate 参数。
- `dico.coverage.window_h`、`relative_stop_delta` 与 `kappa_calibration`：NSW coverage 和局部 κ residual 参数。
- `preallocation.beta=1.0` 是参考 base 值；实际采购由 `dico.procurement.beta=0.5` 覆盖，诊断记录最终有效值。
- `dico.init.mode: direction_anchored` / `zero_B: true`：direction bank 只消费实际购买方向且初始 `delta_w_zero=true`。
- `covra_full` / `covra_independent` / `covra_module_scalar` 及 sign/type/log/DP 字段只属于 experimental 配置链。

### 只能在消融/拓展实验里修改的参数

- `lora.scaling`（`alpha_over_r` vs `alpha_over_sqrt_r`，对应 `configs/ablations/scaling_alpha_over_r.yaml`）——主表对比必须固定，消融才允许换。
- `preallocation.r_min_multiplier`（如 `configs/ablations/rmin_0.yaml`，属于附录敏感性，不进入主表）。
- `preallocation.top_k_atoms`、`preallocation.sketch_dim`（如 `top_k_atoms_64.yaml`、`sketch_dim_32.yaml`，属于候选/草图敏感性）。
- `preallocation.use_sign_split`、`preallocation.use_type_scaling`、`preallocation.use_log_compression`、`preallocation.solver`（分别由 `no_sign_split`、`no_type_scaling`、`no_log_compression`、`proportional_rounding` 消融控制）。
- `dico.init.mode=kaiming_zero_B` 或 `preallocation.rank_override=uniform_ref`：只用于 `random_init`、`covra_rank_random_init`、`uniform_rank_covra_init` 等 rank/init 分离实验。

---

## 7. 配置组织方式

### 继承机制

`src/dico/config.py::load_yaml` 支持顶层 `inherits: <相对路径>` 字段，会先递归加载父 config，再用 `deep_merge`（dict 递归合并，list/标量整体覆盖）叠加当前文件的内容：

```yaml
# configs/dico/dico_cd_da_r8.yaml
inherits: covra_reference_base.yaml

experiment_name: dico_cd_da_r8_protocol_aligned
method: dico_cd_da
rank: 8
dico:
  version: cd_da
  init:
    mode: direction_anchored
```

继承链：`configs/dico/dico_cd_da_r8.yaml` → `configs/dico/covra_reference_base.yaml`。reference base 不继承 `configs/dico/base.yaml` 或共享 `configs/base.yaml`，因此 LoRA、AdaLoRA、GoRA-public、GoRA-BM 的 resolved 配置不会随 CovRA 对齐而变化。

### 四类 config 的关系

- **基础配置**：baseline 继续使用 `configs/base.yaml` + `configs/dico/base.yaml`；正式 CovRA 单独使用完整的 `configs/dico/covra_reference_base.yaml`。
- **方法配置**（等价于"基线配置"和"方法配置"，仓库没有区分这两个概念，统一放在 `configs/dico/`）：
  - `lora_r8.yaml`：固定 rank=8 的普通 LoRA baseline（`calibration.enabled: false`，不跑预分配）。
  - `rs_lora_r8.yaml`：独立 rsLoRA scaling 对照，不进入默认 E01。
  - `adalora_r8.yaml`：完整 A/E/B AdaLoRA，`init_rank=12 → target_rank=8`，全局裁剪并报告 physical/peak/final 参数量。
  - `gora_public_r8.yaml`：锁定官方 commit 的 method-faithful GoRA，保留实际预算。
  - `gora_bm_r8.yaml`：只比 GoRA-public 多 strict budget repair。
  - `gora_bw_r8.yaml`：legacy pilot，不进入正式启动器。
  - `dico_cd_r8.yaml`：DiCo 完整流水线，但初始化用 `kaiming_zero_B`（即不加方向锚定初始化）。
  - `dico_cd_da_r8.yaml`：默认 reference CovRA（`covra_v05`）+ 方向锚定初始化。
  - `dico_cd_da_r8_covra_full_experimental.yaml`：旧 conditional coverage + DP 的 experimental CovRA。
  - `mixed_math_code_r8.yaml`：在 `dico_cd_da_r8.yaml` 基础上换成 math+code 混合数据源（`data.train_sources`）+ `dico.split.mode: group`，用真实任务标签测试任务组拆分（见第 3 节"混合数据集"）。
- **拓展配置**：仓库目前没有单独的 `configs/extensions/` 目录（早期版本有过，v0.3 精简后被移除，`tests/configs/test_v03_slim_layout.py` 会断言这类 legacy 目录不应该存在）；如果需要做拓展实验（例如 LoRA+ 学习率比例消融），建议新建 `configs/extensions/xxx.yaml`，`inherits: ../dico/dico_cd_da_r8.yaml`，只覆盖需要拓展的字段，参照 `configs/ablations/*.yaml` 的写法。
- **消融配置**：reference 参数敏感性可继承 `dico_cd_da_r8.yaml`；依赖 conditional/DP、CovRA-I/M 或 rank/init 分离的消融继承 `dico_cd_da_r8_covra_full_experimental.yaml`。

### 用 `--override` 做一次性改动（不新建文件）

```bash
python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml \
  --override seed=43 \
  --override calibration.seed=43 \
  --override preallocation.sketch_seed=43 \
  --override project.output_dir=outputs/dico_v03_seed43 \
  --override training.max_steps=500
```

`--override key.path=value` 可重复传，按 `.` 分隔的路径做递归合并（`apply_overrides`），value 会尝试按 YAML 语法解析（所以 `true`/`42`/`1.0e-3` 会被解析成对应类型，字符串需要保证不会被误解析成数字/布尔值）。

---

## 8. 路径配置说明

| 路径 | 由谁决定 | 默认值 | 建议 |
|---|---|---|---|
| 项目根目录 `_project_root` | `config.py::load_yaml` 自动推导：从 config 文件路径向上找到名为 `configs` 的目录，取其父目录 | 仓库根目录 | 不需要手动设置；`scripts/run_experiment.py` 也会把 `ROOT/src` 加入 `sys.path` |
| 输出目录 | `project.output_dir`（相对 `_project_root`，也可写绝对路径） | `outputs/dico_v03` | 每个实验最终落盘在 `<output_dir>/<experiment_name>/`，**不同实验必须用不同的 `experiment_name` 或 `output_dir`**，否则会互相覆盖 |
| 校准/预分配缓存 | `calibration.save_dir` | `outputs/dico_v03/preallocations` | 存放 `dico_v03_rank{r}_seed{seed}_profiles.pt`、`..._direction_bank.pt` 等预分配缓存；同一 `rank`+`seed`+config 指纹会命中缓存跳过重新校准，改了影响 rank 分配的字段后缓存会自动判定为不兼容并重算 |
| 数据集路径 | `data.train_path` / `data.eval_path` | `data/gsm8k/main/{train,test}.jsonl` | 相对路径相对于 `_project_root`；绝对路径原样使用 |
| 模型路径/HF Hub id | `model.name_or_path` | `meta-llama/Llama-3.1-8B-Base` | 可以是本地目录，见第4节 |
| HuggingFace 模型缓存 | 环境变量 `HF_HOME` / `TRANSFORMERS_CACHE` / `HF_HUB_CACHE` | `~/.cache/huggingface/` | 代码里没有覆盖，纯粹走标准 HF 环境变量 |
| HuggingFace 数据集缓存 | 环境变量 `HF_DATASETS_CACHE` | 代码里写死回退值 `/root/hf_cache/datasets`（仅在 `data.train_path` 不存在、需要回退到 `datasets.load_dataset` 时才用得到） | 建议显式 `export HF_DATASETS_CACHE=/your/path`，不要依赖那个写死的默认值 |
| 检查点 | 训练结束时自动保存 `<output_dir>/<experiment_name>/masked_lora_state.pt`（`trainer.py::save_masked_lora_state`，只筛选 `state_dict()` 里名字包含 `lora_A`/`lora_B`/`rank_mask` 的参数，对 static 和 masked 两种注入方式都适用，尽管函数名叫"masked"）；base 模型权重不保存，只有 LoRA 增量 | — | 只在训练结束时保存一次，没有 `training.save_steps` 中间检查点（该字段和 `save_optimizer_state` 目前都未被读取，是预留字段）；如需中间检查点或保存 optimizer 状态，需要自己在 `trainer.py` 训练循环里补 |
| 日志 | `<output_dir>/<experiment_name>/{train_log.jsonl, eval_log.jsonl}` | 同上 | 纯文件日志，没有 wandb/tensorboard 集成，见第11节 |

---

## 9. 启动命令

### 受限算力平台：单文件、无参数启动

如果算力平台只能执行一个 Python 文件，并且启动命令不能附带任何参数，请使用仓库根目录的 `launch_covra.py`。

先在 `launch_covra.py` 顶部的“手动配置区 / MANUAL CONFIGURATION”修改：

- `CONDA_ENV_PYTHON`：服务器 conda 环境中的 Python；
- `PROJECT_ROOT`：上传后的项目根目录；
- `MODEL_PATH`：本地 Llama-3.1-8B-Base 模型目录；
- `DATA_ROOT` / `DATA_PATHS_TO_CHECK`：MetaMathQA、GSM8K 等数据路径；
- `PROFILE`：默认 `e01_aligned`；也可通过平台环境变量 `COVRA_PROFILE=e02_strict_budget` 切换；
- `SEEDS`：默认 `(42, 43, 44)`；
- `GPU_IDS`：默认 `("0", "1", "2")`；
- `OUTPUT_DIR` / `LOG_DIR`：输出和日志目录。

平台最终只需要执行：

```bash
python launch_covra.py
```

`launch_covra.py` 只负责平台启动：它会检查关键路径、创建输出和日志目录、设置工作目录与环境变量，然后通过 `subprocess.Popen` 调用现有正式入口 `scripts/platform_train.py`。训练输出会同时显示在控制台并保存到 `logs/launch_covra.log`；任何子训练进程失败时，启动器会返回对应非零错误码。它不会修改 CovRA 方法逻辑，也不会在这里改实验超参数；如需改变协议，应先修改 config 和实验计划。

服务器首次安装环境时执行 `bash scripts/setup_server_env.sh`。该脚本安装 `requirements.txt`；正式配置统一使用 PyTorch 内置 SDPA，不需要安装 `flash-attn`。安装完成后仍须运行 E00 readiness，并在真实 Llama/A800 日志中确认 effective attention implementation 为 `sdpa`。

默认 `e01_aligned` 顺序执行 4 个方法，每个方法内部并行运行 3 个独立单卡 seed，共 12 runs，输出到 `outputs/e01_llama3_r8_aligned_sdpa_v4`。顺序为 Uniform LoRA、AdaLoRA、GoRA-public、CovRA。每个 child 都显式使用 `--num-processes 1` 和 `NUM_GPUS=1`，不会继承平台变量而误入 DDP。E02 profile 为 LoRA、GoRA-BM、CovRA，共 9 runs，输出到 `outputs/e02_llama3_r8_strict_budget_sdpa_v4`。SDPA v4 与旧 FlashAttention2/v3 结果不得混合统计。

### 单个实验（单卡）

```bash
# 用 config 直接跑（最基础的方式）
python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml

# 用 shell 包装跑（等价，脚本内部就是上面这行）
bash scripts/run_dico_cd_da.sh

# 带 override
bash scripts/run_dico_cd_da.sh --override training.max_steps=100 --override seed=43
```

### 多卡 DDP 启动（Accelerate，3×A800）

> **默认不建议把主实验跑成 DDP。** 新版 3×A800 调度优先使用 `scripts/platform_train.py`：
> 三张卡同时跑三个独立单卡 seed，保持 `batch_size=4 × grad_accum=16 = global batch 64`，与 GoRA 主协议精确对齐。
> 本节 DDP 只作为单卡 pilot OOM 时的 fallback。有效 batch size =
> `per_gpu_batch_size × num_gpus × gradient_accumulation_steps`；fallback 统一使用
> `3 × 3 × 7 = 63`，并在实验表中标记为硬件适配差异。

```bash
# DDP fallback：仅当单卡 global batch 64 pilot OOM 时使用
NUM_GPUS=3 bash scripts/run_ddp.sh configs/dico/dico_cd_da_r8.yaml \
    --override training.per_gpu_batch_size=3 \
    --override training.gradient_accumulation_steps=7

# 也可以跳过 run_ddp.sh，直接用 accelerate 读取同一份 configs/accelerate_3gpu.yaml
# （run_ddp.sh 内部就是这条命令，加了 NUM_GPUS/MASTER_PORT 两个可覆盖的环境变量）
accelerate launch --config_file configs/accelerate_3gpu.yaml \
    scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml

# 如果你的启动环境只能调用 scripts/run_experiment.py 这一个文件（不能用 shell 包装、
# 不能用 accelerate launch 前缀），加 --num-processes 让它自己内部用
# accelerate.notebook_launcher 拉起多进程，效果和上面等价：
python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml \
    --num-processes 3 \
    --override training.per_gpu_batch_size=3 \
    --override training.gradient_accumulation_steps=7
```

**DDP 模式下的注意事项：**
- 训练日志、checkpoint、metrics.json、以及所有 Appendix-B 产物文件（`budget.json`/`rank_dict.json`/
  `diagnostics.json`/`seeds.json`/`resolved_config.yaml` 等）都只由 rank-0（`is_main`）写入，其余进程不写文件
- 预分配/校准（DiCo preallocation / GoRA-BW calibration）只在 rank-0 执行，且**采样池是全量训练集**
  （不是 rank-0 的 DDP 分片），结果通过 `broadcast_object_list` 广播给其他进程——这保证了同一份 config
  在单卡/多卡下算出的 rank allocation 完全一致
- 训练数据分片当前不是 PyTorch `DistributedSampler`，而是显式 `records[local_rank::world_size]` 手动 stride
  分片；`batch_iter` 是无限循环迭代器，`drop_last=false`，最后一个 gradient accumulation 周期也会从循环迭代器
  补足完整 micro-batch。每个 run 的 `run_manifest.json` 会在 `data_loading` 字段记录
  `sampler_type`、`distributed_sampler_used`、`drop_last`、`dataloader_length_batches` 和
  `last_accumulation_behavior`，用于审计 DDP fallback 的真实训练语义。
- 最终评估（GSM8K accuracy / HumanEval）只在 rank-0 上跑，避免 DDP generate 死锁
- **`training.ddp_timeout_minutes`（默认 180）**：最终 GSM8K accuracy 评估默认跑全量 1319 条
  `data/gsm8k/main/test.jsonl`（`evaluation.accuracy_max_samples: null`），按约 2.2s/条估算要 **~49 分钟**，
  且只有 rank-0 在跑；其余 rank 在训练结束后的 `accelerator.wait_for_everyone()` barrier 上空等。NCCL 进程组
  默认 watchdog 超时只有 **10 分钟**，遇到全量评估必然导致其余 rank 判定超时、直接把整个进程组杀掉
  （`ProcessGroupNCCL Watchdog caught collective operation timeout` + `SIGABRT`）——训练本身其实已经正常跑完，
  死在最后收尾这一步。`trainer.py` 构造 `Accelerator` 时通过 `InitProcessGroupKwargs(timeout=...)` 把这个
  超时改成了 `training.ddp_timeout_minutes`（默认 3 小时），需要更长可以在 config 里调大；如果关掉全量评估
  （设置 `evaluation.accuracy_max_samples` 为一个小数字）也能规避，但默认值已经足够覆盖全量评估
- 单卡模式（`python scripts/run_experiment.py`）完全向后兼容；主实验优先用单卡 global batch 64，而不是 DDP fallback
- **`model.device_map: auto`（`configs/base.yaml` 默认值）会在多进程 DDP 下自动被忽略**：
  `device_map=auto` 是单进程模型并行切分（把一个模型摊到它能看到的所有 GPU 上），如果每个 DDP 进程都各自
  这么干，3 个进程会同时把整个模型摊到同一批 GPU 上抢显存，直接冲突。`trainer.py` 检测到
  `num_processes > 1` 时会跳过 `device_map`，改为让每个 rank 各自加载一份完整模型副本、放到自己独占的
  GPU 上（`accelerator.device`），并打印一条 warning 日志说明发生了什么——**不需要手动在 DDP 启动命令里
  加 `--override model.device_map=null`**。

**直接 `python scripts/run_experiment.py --config ...`（不加 `--num-processes`，也不经过 `accelerate launch`）只是单进程**：
如果 `model.device_map: auto` 生效且机器上有多张可见 GPU，这个单进程会把模型按层切到多张卡上（模型并行，
前向/反向要跨卡串行传递，通常比单卡+DDP慢很多），不是这里说的 DDP 数据并行。要用 3 张 A800 跑数据并行，
三选一：`scripts/run_ddp.sh`、`accelerate launch --config_file configs/accelerate_3gpu.yaml`，或者
`python scripts/run_experiment.py ... --num-processes 3`（唯一不需要外部命令包装的方式，内部通过
`accelerate.notebook_launcher` 拉起进程，仍然需要环境里装了 `accelerate` 包）。

**`scripts/platform_train.py`（4 个配置 × 3 seed 主实验批量启动）默认是 3 卡并行跑 3 个 seed，不是 DDP**：
`--num-gpus` 默认 3；对每个 config，把它的 3 个 seed 各自分配一张卡（`CUDA_VISIBLE_DEVICES` 单卡），用
`subprocess.Popen` 同时拉起 3 个完全独立的单卡进程（`python scripts/run_experiment.py ...`，不经过
`accelerate launch`），每个进程各自完成训练和自己的最终评估；等这 3 个 seed 全部跑完（成功或失败）再进入
下一个 config。每个进程的输出重定向到 `logs/<experiment_name>.log`，方便 `tail -f` 单独一个 seed。
Llama-3.1-8B-Base + LoRA 单卡（A800）跑得下，没必要为了训练用 DDP，换成"3 个 seed 各占一张卡"之后训练和
评估两个阶段都能吃满 3 卡，也彻底避开了 NCCL/`accelerate` 相关的复杂度和超时风险。同一 config 内某个 seed
失败不会打断同批次里还在跑的其他 seed，但该批次结束后会汇总报错并**不会**再启动下一个 config。
如需退回单卡顺序执行（调试、显存不足等场景）传 `--num-gpus 1` 即可。如果未来某个模型大到单卡装不下、
真的需要 DDP 数据并行，用上面"多卡 DDP 启动"一节里独立的 `scripts/run_ddp.sh` /
`accelerate launch --config_file configs/accelerate_3gpu.yaml` 手动跑单个实验（`platform_train.py`
本身不再走这条路径）。


### legacy 单方法脚本

以下脚本保留给单项调试；正式 E01 不使用旧 `run_gora_bw.sh`，而由 `launch_covra.py` 选择 `gora_public_r8.yaml`：

```bash
bash scripts/run_lora_r8.sh
bash scripts/run_gora_bw.sh
bash scripts/run_dico_cd.sh
bash scripts/run_dico_cd_da.sh
bash scripts/run_mixed_math_code.sh
```

reference CovRA 三 seed 使用专用 launcher；它会同步 training/calibration/sketch seed 并隔离输出目录：

```bash
python scripts/platform_train.py \
  --config configs/dico/dico_cd_da_r8.yaml \
  --seeds 42,43,44 \
  --batch-size 4 \
  --grad-accum 16 \
  --calibration-batch-size 4
```

### 拓展实验

仓库暂无内置拓展 config；按第7节方法新建 `configs/extensions/xxx.yaml`（`inherits: ../dico/dico_cd_da_r8.yaml`），然后：

```bash
python scripts/run_experiment.py --config configs/extensions/xxx.yaml
```

### 消融实验

```bash
bash scripts/run_ablation_covra_independent.sh
bash scripts/run_ablation_covra_module_scalar.sh
bash scripts/run_ablation_no_sign_split.sh
bash scripts/run_ablation_no_type_scaling.sh
bash scripts/run_ablation_no_log_compression.sh
bash scripts/run_ablation_proportional_rounding.sh
bash scripts/run_ablation_global_only.sh
bash scripts/run_ablation_grouped_only.sh
bash scripts/run_ablation_random_init.sh
bash scripts/run_ablation_uniform_rank_covra_init.sh
bash scripts/run_ablation_covra_rank_random_init.sh
```

### 冒烟测试（不加载模型、不跑训练，只验证配置能正确解析）

```bash
# 单个配置
DRY_RUN=1 python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml
# 或
python scripts/run_experiment.py --config configs/dico/dico_cd_da_r8.yaml --dry-run

# 所有方法脚本一次性验证
for s in run_lora_r8 run_gora_bw run_dico_cd run_dico_cd_da run_mixed_math_code \
         run_ablation_covra_independent run_ablation_covra_module_scalar \
         run_ablation_no_sign_split run_ablation_no_type_scaling \
         run_ablation_no_log_compression run_ablation_proportional_rounding \
         run_ablation_global_only run_ablation_grouped_only \
         run_ablation_random_init run_ablation_uniform_rank_covra_init \
         run_ablation_covra_rank_random_init; do
  DRY_RUN=1 bash scripts/${s}.sh
done
```

`--dry-run`（或环境变量 `DRY_RUN=1`）只会打印 config 解析摘要（`experiment_name`/`method`/`rank`/`seed`/`output_dir`/`lora_injection`/`lora_scaling`/`dico_version`），**不会加载模型、不需要 GPU、不需要联网**，是提交任何真实实验前的第一道检查。

---

## 10. pytest / 配置合法性检查

```bash
# 全量单测(组件级 + 配置合法性 + 入口冒烟测试)，pytest.ini 已设置 pythonpath=src
pytest tests -q
# 等价的显式写法
PYTHONPATH=src pytest tests -q

# 只跑配置合法性检查(method白名单、legacy路径必须不存在)
pytest tests/configs -q

# 只跑入口/脚本级冒烟测试(内部用 DRY_RUN=1,不需要模型/GPU)
pytest tests/scripts -q

# 只跑某个方法/阶段相关的单测,例如 DA-init 或覆盖认证
pytest tests/unit/test_dico_init_v03.py tests/unit/test_dico_da_init_trainer_integration.py -v
pytest tests/unit/test_dico_coverage_v03.py tests/unit/test_dico_coverage_module_type_isolation.py -v

# shell 脚本语法检查(不执行,只检查bash语法)
for script in scripts/*.sh; do bash -n "$script"; done
```

`tests/configs/test_v03_slim_layout.py` 会检查精简目录布局和正式 method 白名单；`scripts/protocol_preflight.py` 还会拒绝“method=lora 但实际使用 rsLoRA scaling”、错误 GoRA commit、校准样本不一致、训练中途评测或 eval batch 非 4。

`tests/scripts/test_v03_entrypoints.py` 用子进程真正执行 `python scripts/run_experiment.py --dry-run`，并 dry-run 覆盖最终方法脚本和保留的 legacy alias（配合 `DRY_RUN=1`），确认入口脚本本身没有语法错误、没有 `ModuleNotFoundError`、返回码为 0——这是"跑实验前最后一道检查"，建议每次改动配置/脚本后都跑一遍。

---

## 11. 结果保存与日志

每次真实训练（非 dry-run）都会在 `<project.output_dir>/<experiment_name>/` 下写出：

```text
config_resolved.yaml          # 本次实际生效的完整config(含所有inherits+override解析结果)
rank_allocation_initial.json  # 训练前的初始rank分配(含module_logs等诊断字段)
rank_allocation_final.json    # 训练结束时的最终rank分配
rank_dict.json                # 精简版 {module_name: rank}
rank_history.csv              # 每一步的rank/budget/loss等时间序列(见 logging_utils.RANK_HISTORY_FIELDS)
budget.json                   # 预算校验结果(target/actual/budget_error等)
diagnostics.json              # CovRA allocation_method、逐秩效用、选择索引、DP诊断等方法诊断
physical_utility.json         # 物理候选/逐秩 utility 摘要（若本次方法产生）
normalization_stats.json      # type scaling / log compression 等归一化统计（若本次方法产生）
init_summary.json             # 初始化摘要(mode、delta_w_zero、direction_bank_path等)
train_log.jsonl               # 每 logging_steps 步一条:loss/吞吐/时间
eval_log.jsonl                # 评估记录(loss、GSM8K accuracy等)
metrics.json                  # 最终汇总指标
masked_lora_state.pt          # 训练结束时保存的LoRA权重(仅lora_A/lora_B/rank_mask,不含base模型权重)
```

预分配缓存额外写在 `calibration.save_dir` 下（默认 `outputs/dico_v03/preallocations/`）：`dico_v03_rank{r}_seed{seed}_profiles.pt`（画像缓存）、`..._direction_bank.pt`（方向锚定初始化用的全维方向向量 sidecar，普通 JSON 存不了 Tensor 所以单独存）、以及主预分配结果 json。

**没有 wandb / tensorboard 集成**——全仓库搜索确认没有任何 `wandb`/`tensorboard`/`SummaryWriter` 引用，全部是上面这些 JSON/JSONL/CSV 文件。如果你需要可视化，需要自己写脚本读取 `train_log.jsonl`/`eval_log.jsonl`/`rank_history.csv` 后接入 wandb/tensorboard/matplotlib，或者直接用 `pandas.read_csv`/`json.loads` 读取分析。

**只在训练结束时保存一次 LoRA 权重**（`masked_lora_state.pt`），没有中间检查点、不保存 base 模型权重、不保存 optimizer 状态——`training.save_steps`/`save_optimizer_state` 字段目前都未被读取。如果需要中间检查点或者需要保存/恢复 optimizer 状态以支持断点续训，需要自己在 `trainer.py` 训练循环里补。

---

## 12. 常见问题排查

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| 模型下载失败 / 403 | `meta-llama/Llama-3.1-8B-Base` 是 gated 模型，没有申请权限或没有登录 | `huggingface-cli login`，确认 HF 账号已被授权访问该模型；或改用本地路径 `model.name_or_path=/path/to/model` |
| 下载卡住/超时 | 国内网络访问 HF Hub 不稳定 | `export HF_ENDPOINT=https://hf-mirror.com`，或提前用 `huggingface-cli download` 下载到本地再指向本地路径 |
| 数据集路径报错 `ValueError: No supported dataset file found` | `data.train_path` 指向的目录下没有 `data.jsonl`/`train.jsonl`/`train.json`/parquet 中任何一种 | 确认路径拼写、确认是相对 `_project_root` 还是绝对路径；用 `python scripts/run_experiment.py --config ... --dry-run` 先看打印出的 `config_path` 和路径是否符合预期 |
| 想用 MetaMathQA 却报"路径不存在"回退到了 HF 下载 | `data.train_path` 没有正确指向你准备好的本地文件，代码静默回退到 `datasets.load_dataset` 分支 | 检查 `data.train_path` 是否确实存在；见第3节数据准备说明 |
| 显存不足 (OOM) | 8B 模型 + BF16 + micro batch4 在单卡上未通过 E00 | 主协议不能自行量化或改变 accumulation。按 E00 清单统一验证 3 卡 DDP global batch63 fallback；量化、checkpointing 或 micro-batch 改动都属于需要对所有方法一致披露并重跑的协议变化，不能混入当前主表 |
| Config 改了但好像没生效 | 参见第5节"配置字段存在但代码未读取"列表；或者 override 路径写错(`--override` 用 `.` 分隔嵌套key，例如 `dico.init.mode=...`而不是`dico_init_mode=...`) | 用 `--dry-run` 打印的摘要和 `config_resolved.yaml` 确认最终生效值；对不确定是否被消费的字段直接 `grep -rn "字段名" src/dico/` |
| reference CovRA 三个 seed 的校准样本不同 | 这是参考 launcher 的预期行为：training/calibration/sketch seed 都随 42/43/44 同步 | 检查 `seed`、`calibration.seed`、`preallocation.sketch_seed` 在每个 run 内一致；其他 baseline 不套用该覆盖 |
| 输出目录冲突/结果被覆盖 | 两个实验用了同一个 `project.output_dir` + `experiment_name` 组合 | 每个实验换不同的 `experiment_name` 或 `project.output_dir`；训练开始前脚本会把 `config_resolved.yaml` 写入该目录，可以先检查该目录是否已存在旧结果 |
| `ModuleNotFoundError: No module named 'dico'` | 没有把 `src/` 加入 `PYTHONPATH`，且不是通过 `scripts/run_experiment.py`（它自己会插入 sys.path）或 pytest（`pytest.ini` 已配 `pythonpath=src`）运行 | 用 `scripts/run_experiment.py` 或 `pytest` 运行；手动跑 python 脚本时 `export PYTHONPATH=src` |
| `nohup bash -lc '...' &` 后台跑实验时报 `ModuleNotFoundError: No module named 'yaml'`（或其他明明已装的包） | `-l`（login shell）会重新 source `~/.bash_profile`/`~/.profile`；如果 conda 的初始化钩子只写在 `~/.bashrc` 里（`conda init bash` 默认行为），login shell 不会自动 source 它，导致新开的 bash 进程没有激活你当前的 conda 环境，`python`/`pip` 退回系统默认解释器 | 把 `bash -lc '...'` 改成 `bash -c '...'`（去掉 `-l`，不重新加载登录脚本，直接继承当前 shell 已经 export 好的 `PATH`）；如果还不行，在脚本开头显式 `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env名>` 再跑；提交后先看日志前几行是不是正常的 `[run] experiment_name=...`，不是的话说明环境没激活对 |
| `bitsandbytes` 相关报错 | CUDA版本与bitsandbytes wheel不匹配，或纯CPU环境尝试用了`load_in_8bit/4bit` | 确认GPU环境且bitsandbytes版本与CUDA匹配；CPU/dry-run场景直接关掉 `load_in_8bit`/`load_in_4bit` |
| GoRA 校准仍显示 CPU 矩阵重建或每 batch 多次 backward | 启动了 legacy `gora_bw`/v2 config，而不是 v3 formal GoRA | 正式日志必须显示 `gradient_collection=official_weight_grad_hook`，1024 样本、batch4 共 256 次 backward；旧结果标记 `legacy_protocol_pilot` |
| CovRA 分配阶段 CPU 利用率高 | reference 配置按已运行版本使用 `compute_device=auto` 做 profile/SVD、`allocation_device=cpu` 做 taxonomy/coverage/procurement | 检查 resolved config 是否为 `auto/cpu`；smoke 可缩小样本数，但正式校准为 256 |

---

## 13. 从零复现实验的完整命令流程

以下流程假设你在一台有 GPU 的 Linux 服务器上从空环境开始：

```bash
# 1. 克隆仓库并进入目录
git clone <your-repo-url> dico_rank_experiments
cd dico_rank_experiments

# 2. 建立 Python 环境并安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 按第2节表格核实/补齐版本，尤其 torch 要选对应本机 CUDA 的 wheel

# 3. 配置 HuggingFace 访问(模型是gated,需要先申请权限)
huggingface-cli login
# 如网络不稳定:
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$HOME/.cache/huggingface   # 或改成你自己的缓存路径

# 4. (可选,但强烈建议)如果要脱离联网/使用本地模型:
# huggingface-cli download meta-llama/Llama-3.1-8B-Base --local-dir /data/models/Llama-3.1-8B-Base
# 之后所有命令加 --override model.name_or_path=/data/models/Llama-3.1-8B-Base

# 5. 确认数据集:默认GSM8K已随仓库携带,无需下载
ls data/gsm8k/main/
# 如果要严格对齐GoRA的MetaMathQA/CodeFeedback协议,先完成第3节"需要你自己准备的数据"

# 6. 跑单测,确认代码本身没有问题(不需要GPU/模型/数据集,几十秒内完成)
pytest tests -q

# 7. dry-run 检查所有主配置能正确解析(不加载模型)
for s in run_lora_r8 run_gora_bw run_dico_cd run_dico_cd_da run_mixed_math_code; do
  DRY_RUN=1 bash scripts/${s}.sh
done

# 8. 先用极小 max_steps 冒烟跑一次真实训练,确认模型能加载、显存够用、流程能走完
bash scripts/run_dico_cd_da.sh \
  --override training.max_steps=2 \
  --override calibration.num_samples=8 \
  --override experiment_name=smoke_dico_cd_da \
  --override project.output_dir=outputs/smoke

# 9. 检查冒烟产出是否完整
ls outputs/smoke/smoke_dico_cd_da/
cat outputs/smoke/smoke_dico_cd_da/metrics.json

# 10. 冒烟通过后,跑完整主实验矩阵(会比较耗时,建议 nohup/tmux 后台运行)
bash scripts/run_lora_r8.sh
bash scripts/run_gora_bw.sh
bash scripts/run_dico_cd.sh
bash scripts/run_dico_cd_da.sh
bash scripts/run_mixed_math_code.sh

# 11. (可选)多seed重复主方法,用于取均值/方差
for seed in 42 43 44; do
  bash scripts/run_dico_cd_da.sh \
    --override seed=${seed} --override calibration.seed=${seed} \
    --override preallocation.sketch_seed=${seed} \
    --override experiment_name=dico_cd_da_r8_seed${seed} \
    --override project.output_dir=outputs/dico_v03_multiseed
done

# 12. (可选)跑消融
bash scripts/run_ablation_covra_independent.sh
bash scripts/run_ablation_covra_module_scalar.sh
bash scripts/run_ablation_no_sign_split.sh
bash scripts/run_ablation_no_type_scaling.sh
bash scripts/run_ablation_no_log_compression.sh
bash scripts/run_ablation_proportional_rounding.sh
bash scripts/run_ablation_global_only.sh
bash scripts/run_ablation_grouped_only.sh
bash scripts/run_ablation_random_init.sh
bash scripts/run_ablation_uniform_rank_covra_init.sh
bash scripts/run_ablation_covra_rank_random_init.sh

# 13. 汇总结果:所有 metrics.json / rank_dict.json / run_manifest.json 都在各自的
#     outputs/.../<experiment_name>/ 目录下,可以自己写脚本按 experiment_name 遍历汇总,
#     或者用 pandas 读取 rank_history.csv / train_log.jsonl / eval_log.jsonl 做曲线分析。
```

---

如果你在复现过程中发现本 README 描述的行为与代码实际行为不一致（尤其是第5节列出的"未打通"项，代码后续可能会补齐），请以 `grep -rn "<config字段名>" src/dico/*.py` 的搜索结果为准，并考虑更新本 README。
