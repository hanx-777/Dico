# DiCo-LoRA MVP

这是一个可运行的 DiCo-lite 研究代码库，用于在 decoder-only HuggingFace causal LM 上测试 LoRA rank allocation。当前配置目标模块为模型的所有线性层（`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`）。

本仓库只包含代码，不包含 Python 环境、模型权重、GSM8K 数据缓存或实验输出。请在工作站上用 conda 自行安装依赖并运行实验。

## 实验内容

MVP 对比三种 LoRA rank 分配方法：

- `uniform`：每个目标模块使用相同整数 rank。
- `module_coverage`：基于校准样本的模块级 norm profile 做 greedy rank-1 模块选择。
- `dico`：基于 SVD atom 的方向级 greedy allocation，并带 prefix constraint。

默认设置有意保持 prompt-neutral，因为当前实验重点是 rank allocation，而不是 Qwen3 prompt 策略：

- `--use_chat_template false`
- `--enable_thinking false`
- `--target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

## 工作站 Conda 环境配置

进入代码目录：

```bash
cd dico_lora_mvp
```

创建并激活 conda 环境：

```bash
conda create -n dico-lora python=3.10 -y
conda activate dico-lora
```

安装 PyTorch。请根据工作站的 CUDA 版本，到 PyTorch 官网选择对应命令。示例：

```bash
# 示例：CUDA 12.1，请按你的工作站实际 CUDA 版本调整
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

安装其余依赖：

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

注意：

- Qwen3 需要较新的 `transformers`。如果模型 config 加载失败，程序会提示升级 `transformers`。
- 当前 README 默认走全量微调，不使用 4-bit 量化；请确认工作站显存足够加载和训练本地 Qwen3-8B。
- `bitsandbytes` 只在你以后显式启用 `--load_in_4bit true` 或 `--load_in_8bit true` 时需要。全量微调路径不依赖量化加载。
- 本仓库不会在本地预装 `.venv` 或 conda 环境。

## 统一配置文件

所有实验参数集中在一个文件里：

```text
configs/mvp_gsm8k.json
```

你主要改这里：

```json
{
  "common": {
    "model_name_or_path": "../Qwen3-8B",
    "finetune_mode": "full",
    "load_in_4bit": false,
    "load_in_8bit": false,
    "gradient_checkpointing": true,
    "torch_dtype": "bf16"
  }
}
```

常用字段：

- `common.model_name_or_path`：本地模型路径，默认已配置为 `../Qwen3-8B`。
- `common.finetune_mode`：默认 `full`，表示全量微调。
- `common.load_in_4bit` / `common.load_in_8bit`：全量微调必须保持 `false`。
- `common.learning_rate`、`common.num_train_epochs`、`common.gradient_accumulation_steps`：训练超参。
- `experiments.smoke.defaults`：真实 smoke 的样本数和长度。
- `experiments.mvp.defaults`：完整 MVP 对比实验的样本数和长度。
- `experiments.*.runs[].output_dir`：每个方法的输出目录。

如果要使用另一份配置文件：

```bash
export DICO_CONFIG=/path/to/your_config.json
```

## 真实模型 + 真实 GSM8K Smoke

这是主 pipeline 检查。它使用真实模型路径和真实 `openai/gsm8k` 数据集，只把样本数量设得很小，用来先检查 hook、mask、SVD、allocation、全量微调参数检查、训练和评估流程是否能跑通。

```bash
cd dico_lora_mvp
conda activate dico-lora

bash scripts/run_real_gsm8k_smoke.sh

或者在 Windows/跨平台环境下直接运行：
python -m src.run_experiment_config --experiment smoke
```

默认 smoke 参数：

- `calibration_size=4`
- `train_limit=8`
- `eval_limit=8`
- `top_k_atoms=2`
- `max_length=256`
- `avg_rank=1`
- `finetune_mode=full`
- `load_in_4bit=false`

这些值都在 `configs/mvp_gsm8k.json` 的 `experiments.smoke.defaults` 里调整。

## 完整对比实验

模型路径和超参数已在配置文件中完成对齐配置。可以直接运行全量实验：

```bash
cd dico_lora_mvp
conda activate dico-lora
bash scripts/run_mvp_gsm8k.sh

或者在 Windows/跨平台环境下直接运行：
python -m src.run_experiment_config --experiment mvp
```

该脚本会依次运行：

- `uniform`
- `module_coverage`
- `dico`

每个方法默认使用：

- `calibration_size=384`
- `train_limit=7473`
- `eval_limit=1319`
- `avg_rank=4`
- `finetune_mode=full`
- `num_train_epochs=1`
- `max_length=512`
- `load_in_4bit=false`
- `gradient_checkpointing=true`

说明：`finetune_mode=full` 会训练模型全部参数，不注入 LoRA adapter，因此不会生成 `lora_rank_verification.json`，而是生成 `full_finetune_verification.json`。如果之后要重新做 DiCo-LoRA rank allocation 对比，需要把脚本里的 `--finetune_mode full` 改回 `--finetune_mode lora`。

脚本结束后会打印汇总表，包括：

- `method`
- `exact_match`
- `trainable_params`
- `used_budget`
- `output_dir`

## 测试

轻量语法检查：

```bash
python -m compileall src tests
```

单元检查：

```bash
pytest tests/test_data_gsm8k.py tests/test_svd_allocator.py -v
```

真实集成测试：

```bash
pytest tests/test_real_gsm8k_smoke.py -v -s
```

如果 `configs/mvp_gsm8k.json` 里的 `model_name_or_path` 无效，runner 会报错提醒你先改配置；它不会用假数据或构造样本替代真实模型和真实 GSM8K。

## 输出文件

每次运行会在对应 `--output_dir` 下写入：

- `config.json`：CLI 参数、`peft`/`transformers`/torch 版本和模型 config 摘要。
- `calibration_pass1.pt`
- `dico_atoms.pt`
- `calibration_profiles.pt`
- `calibration_summary.json`
- `mask_debug/sample_000.json` 到最多 `sample_002.json`
- `rank_pattern.json`
- `full_finetune_verification.json`：全量微调模式下生成，确认所有参数可训练。
- `lora_rank_verification.json`：仅在 `--finetune_mode lora` 时生成。
- `eval_results.json`
- `eval_predictions.jsonl`
- `run_summary.json`
- DiCo diagnostics 的 CSV/JSON/PNG 文件

其中 `mask_debug` 用于检查 DiCo 统计是否真的落在 answer/loss-bearing token，而不是 prompt token。

## Go / No-Go 标准

继续推进 DiCo，如果满足任一条件：

- `avg_rank=1` 下，DiCo-lite 比 `module_coverage` 高约 0.3 到 0.5 个 GSM8K exact-match 点。
- diagnostics 显示明显的 direction-level redundancy pair。
- DiCo 的 rank pattern 和 module-level coverage 有有意义的差异。

暂停或重新设计，如果出现：

- DiCo rank pattern 几乎等同于 `module_coverage`。
- atom-level 和 module-level similarity matrix 没有明显差异。
- DiCo 稳定低于 `module_coverage`。
- calibration 对样本非常不稳定。

## 已知限制

这是 MVP，不是完整论文代码：

- 只支持 GSM8K。
- 默认单 seed。
- 不实现 RoBERTa/GLUE。
- 不实现 `tau` 或 `d_m^*`。
- 不做大规模 sweep。
- 校准不会跨样本保存完整 `G_i,m` 矩阵，但 Qwen3-8B 仍然需要足够 GPU 显存。
