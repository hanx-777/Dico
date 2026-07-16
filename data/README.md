# 本地数据

本目录存放实验所需的全部数据集，使用相对路径管理，保证项目离线可复现。

---

## 目录结构

```text
data/
  gsm8k/
    main/
      train.jsonl        ← GSM8K 原始训练集（7473 条）
      test.jsonl         ← GSM8K 测试集（1319 条，评估集，不可修改）
  metamathqa/
    train.jsonl          ← MetaMathQA-100K 训练集（~100K 条，需手动下载）
  README.md
```

所有文件保留统一字段格式：`question`（题目）和 `answer`（答案，包含 `#### <number>` 格式的最终答案）。

---

## 数据集说明

### GSM8K（已随仓库携带）

- **训练集**：`data/gsm8k/main/train.jsonl`，7473 条（原始 GSM8K）
- **测试集**：`data/gsm8k/main/test.jsonl`，1319 条
- 来源：[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- 评估集 (`test.jsonl`) **不可修改**，所有实验均在此测试集上评估，以保证结果可比性。

### MetaMathQA-100K（需下载，GoRA 协议对齐训练集）

- **路径**：`data/metamathqa/train.jsonl`，约 100K 条
- 来源：[meta-math/MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)
- 原始字段为 `query` / `response`，下载脚本自动转换为项目约定的 `question` / `answer` 格式。
- 这是与 GoRA 论文（arXiv 2502.12171）对齐的训练集，也是 `configs/base.yaml` 的默认 `train_path`。

---

## 下载 MetaMathQA-100K

### 方法一：使用下载脚本（推荐）

```bash
# 从 HuggingFace 官方下载（需要网络访问）
python scripts/download_data.py

# 使用镜像（服务器无法直连 hf.co 时）
python scripts/download_data.py --hf-endpoint https://hf-mirror.com

# 指定 HF 缓存目录
python scripts/download_data.py --hf-cache /your/hf_cache

# 只检查文件是否存在，不下载
python scripts/download_data.py --check-only
```

下载脚本是**幂等的**：如果目标文件已存在，会直接跳过，不会重复下载。

### 方法二：手动准备

若以上方式均不可用，可手动准备数据，将文件放置于：

```
data/metamathqa/train.jsonl
```

文件格式要求：每行一个 JSON 对象，包含 `question` 和 `answer` 字段：

```json
{"question": "题目文本", "answer": "解题步骤\n#### 42"}
```

---

## 关于绝对路径

所有配置文件均使用相对路径，相对于项目根目录（即包含 `configs/` 的目录）。
`data.py` 的 `_resolve_data_path` 函数负责将相对路径解析为绝对路径，无需手动指定。
