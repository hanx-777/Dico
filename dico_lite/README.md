# DiCo-lite (v2) 🚀

**DiCo-lite** 是一个针对大语言模型 (LLM) 参数高效微调 (PEFT) 设计的极简、超高性能的模块级 Rank 分配框架。

本项目是对原有 `DiCo-LoRA` 的彻底重构。我们剥离了所有厚重的历史遗留代码（例如 SVD 分解、Direction-level 原子计算、复杂的缓存机制等），并将架构升级为**纯正的单进程流水线**，实现了极低的内存消耗和极致的 GPU 计算效率。

---

## ✨ 核心特性 (Key Features)

1. **极致性能 (Gram-trick Calibration)**
   - 彻底废弃了原有的庞大协方差矩阵（$4096 \times 4096$）计算。
   - 采用 Gram 技巧，在 GPU 侧直接对维度极小的 $T \times T$ 样本特征矩阵进行原地的 `bfloat16` 累加。
   - **效果**：消除了每样本高达 14GB 的 PCIe 数据回传，GPU 算力利用率拉满，校准速度获得数量级提升！

2. **单进程优雅架构 (Unified Single-Process Pipeline)**
   - 旧版本在校准结束后，需要通过 `subprocess` 唤起新的训练进程，导致 8B 模型被重复加载，引发高达 40GB 的内存泄漏与极其缓慢的 CPU-Offload 前向传播。
   - **DiCo-lite** 在同一个 Python 进程内完成 **`模型读取 -> 模块特征校准 -> 预算分配 -> 原地 LoRA 注入 -> SFT 训练`**。模型仅驻留显存一次，全程无缝衔接。

3. **双基线对比测试套件 (Automated Comparative Suite)**
   - 自动化支持 `uniform`（均匀分配）与 `dico`（模块级特征分配）两种模式。
   - 一键串行执行对比实验，输出目录完全隔离，方便后续进行无污染的指标对比。

---

## 🛠️ 环境配置 (Setup)

请确保你的工作站已经安装了 Anaconda。在根目录下执行以下命令来构建纯净环境：

```bash
conda create -n dico-lora python=3.10 -y
conda activate dico-lora

# 请根据你机器的实际 CUDA 版本替换 cu121 (例如 cu118)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 安装依赖库
pip install transformers peft trl datasets tqdm
```

*(注：如果你需要使用 4-bit 量化加载，请确保正确安装了对应系统的 `bitsandbytes` 库)*

---

## ⚙️ 配置说明 (Configuration)

所有的超参数都被集中收敛到了 `configs/config.json` 文件中。
核心参数说明：

- `target_modules`：用逗号分隔的目标线性层名称。
- `load_in_4bit`：开启 4-bit 加载（单卡 24G 显存跑 8B 模型的利器）。
- `calibration_size`：用于计算模块重要性的校验集样本数。
- `avg_rank`：目标基准平均 Rank（Budget 会根据此值进行换算）。
- `coverage_eps` 与 `budget_floor_ratio`：贪婪分配算法的覆盖率阈值和预算下限。

---

## 🚀 运行实验 (How to Run)

我们为你准备了开箱即用的自动化实验脚本，它会自动先后运行 `Uniform` 基线与 `Module-DiCo-lite` 基线。

### 1. Windows 环境 (PowerShell)

在 PowerShell 中进入项目目录并执行：

```powershell
cd C:\Users\admin\Desktop\刘新武\Dico\dico_lite
conda activate dico-lora

# 模式 A: 仅生成分配策略（Dry-run 校验，不触发耗时的训练）
.\scripts\run.ps1

# 模式 B: 执行完整的自动化对比实验（校准 + 训练）
.\scripts\run.ps1 -RunTrain "true"
```

### 2. Linux / macOS 环境 (Bash)

```bash
cd /path/to/Dico/dico_lite
conda activate dico-lora

# 模式 A: 仅生成分配策略
bash scripts/run.sh

# 模式 B: 执行完整的自动化对比实验
bash scripts/run.sh true
```

---

## 📁 目录结构与架构 (Architecture)

```text
dico_lite/
├── configs/
│   └── config.json       # 唯一的配置文件
├── scripts/
│   ├── run.ps1           # Windows 自动化运行套件
│   └── run.sh            # Linux/Mac 自动化运行套件
├── src/
│   ├── main.py           # 核心协调器 (Orchestrator)
│   ├── calibration.py    # 基于 Gram-trick 的高速 Hook 校准器
│   ├── allocator.py      # 贪婪预算分配算法 (Coverage -> Tail completion)
│   ├── train.py          # HuggingFace SFTTrainer 与 PEFT LoRA 注入逻辑
│   ├── data.py           # 数据集处理
│   ├── model_utils.py    # 包含量化和硬件优化的模型加载器
│   └── utils.py          # 辅助工具
└── README.md
```

---

## 💡 常见排错指南 (Troubleshooting)

1. **缓存跳过 (Skip Calibration)**：
   如果在对应的输出目录（如 `outputs/gsm8k_dico`）中检测到已经存在的 `rank_pattern.json`，系统会**跳过校准阶段**，直接使用该策略进行训练。如果你修改了配置需要重新校准，请手动删除 `outputs` 文件夹或该 JSON 文件。

2. **显存溢出 (CUDA Out of Memory)**：
   尽管 DiCo-lite 已经将内存和显存消耗降到了最低，但在实际的 SFT 训练阶段，由于梯度检查点或 batch 大小，仍可能导致 OOM。如果发生 OOM，请在 `config.json` 中适当降低 `per_device_train_batch_size`。
