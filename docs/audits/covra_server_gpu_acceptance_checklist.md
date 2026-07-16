# CovRA 服务器 GPU 验收清单

本清单用于项目上传到 3×A800 服务器后的第二阶段验收。任何项目只有在目标服务器真实运行并归档日志/报告后，才能标记为 `IMPLEMENTED_AND_GPU_VERIFIED`。

## 受限平台启动入口

如果算力平台只能执行一个 Python 文件且不能附带参数，最终命令为：

```bash
python launch_covra.py
```

运行前请先编辑 `launch_covra.py` 顶部“手动配置区 / MANUAL CONFIGURATION”，使 conda Python、项目目录、模型目录、数据目录、config、seed、GPU 和输出目录都指向服务器真实路径。

默认 profile 为 `e01_aligned`，顺序执行 4 个方法配置，每个方法内部并行运行 3 个独立单卡 seed，共 12 runs：Uniform LoRA、AdaLoRA、GoRA-public、CovRA。输出根目录为 `outputs/e01_llama3_r8_aligned_sdpa_v4`。launcher 对每个 child 强制 `--num-processes 1` 与 `NUM_GPUS=1`，不会因平台继承变量意外启动 DDP。如需 E02 strict-budget，设置环境变量 `COVRA_PROFILE=e02_strict_budget` 后仍执行同一条 Python 命令，输出到 `outputs/e02_llama3_r8_strict_budget_sdpa_v4`。

## E00-A：环境检查

需要记录：

- `nvidia-smi` 输出；
- GPU 型号、数量和显存；
- CUDA driver/runtime；
- PyTorch CUDA 可用性；
- BF16 支持情况；
- PyTorch SDPA 是否可用于 Llama-3.1-8B；
- 模型路径是否可访问；
- 数据路径是否可访问；
- 输出目录和日志目录是否可写。

建议先运行：

```bash
bash scripts/setup_server_env.sh
```

该脚本安装 `requirements.txt`。正式协议使用 PyTorch SDPA，不安装或要求 FlashAttention2；若平台环境已经准备好，可跳过安装，但仍须执行下面的 readiness 硬门禁。

```bash
python scripts/e00_readiness.py \
  --json-output reports/e00_readiness_server.json \
  --markdown-output reports/e00_readiness_server.md \
  --require-gpu-count 3 \
  --model-path /path/to/Meta-Llama-3.1-8B-Base \
  --require-runtime-deps \
  --min-free-gb 100
```

训练前期望状态：`READY_DRY_RUN`。

## E00-B：单卡 LoRA pilot

目的：确认一张 A800 能加载 Llama-3.1-8B，并跑通 GoRA 对齐的单卡 batch 协议。

必须检查：

- 单卡模型加载成功；
- batch size 为 `4`；
- gradient accumulation 为 `16`；
- global batch 为 `64`；
- 至少完成一次 forward/backward/update；
- checkpoint 保存和恢复可用；
- 参数量与预算日志完整；
- 标准 LoRA scaling 为 `alpha/r=2`，B 初始为零；
- 记录峰值显存；
- `run_manifest.json` 可通过独立校验。

建议命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  --config configs/dico/lora_r8.yaml \
  --override experiment_name=e00_lora_pilot_seed42 \
  --override seed=42 \
  --override calibration.seed=42 \
  --override preallocation.sketch_seed=42 \
  --override training.batch_size=4 \
  --override training.gradient_accumulation_steps=16 \
  --override training.max_steps=1
```

## E00-C：单卡 CovRA pilot

目的：确认 CovRA 的校准、候选提取、条件覆盖、DP 分配、方向初始化和一步训练都能在一张 A800 上跑通。

必须检查：

- 两次校准遍历；
- `K >= r_max`；
- 先跑 smoke-scale calibration，再跑完整 `N=1024`；
- 候选方向提取成功；
- 条件边际覆盖成功；
- DP 分配成功；
- 方向初始化成功；
- 至少完成一步训练；
- manifest 完整；
- 显存和时间字段完整。
- `preallocation.compute_device=cuda` 与 `preallocation.allocation_device=cuda` 真正生效，不能出现 CPU fallback；
- `stage_metrics.jsonl` 包含校准、分配、初始化、训练、评测等阶段的 allocated/reserved/peak 指标。

smoke 建议命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  --config configs/dico/dico_cd_da_r8.yaml \
  --override experiment_name=e00_covra_smoke_seed42 \
  --override seed=42 \
  --override calibration.seed=42 \
  --override preallocation.sketch_seed=42 \
  --override calibration.num_samples=8 \
  --override training.batch_size=4 \
  --override training.gradient_accumulation_steps=16 \
  --override training.max_steps=1
```

完整校准 pilot 建议命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  --config configs/dico/dico_cd_da_r8.yaml \
  --override experiment_name=e00_covra_fullcal_seed42 \
  --override seed=42 \
  --override calibration.seed=42 \
  --override preallocation.sketch_seed=42 \
  --override calibration.num_samples=1024 \
  --override training.batch_size=4 \
  --override training.gradient_accumulation_steps=16 \
  --override training.max_steps=1
```

## E00-D：三卡调度检查

优先验证三张卡并行三个单卡 run：

- GPU0 跑 seed 42；
- GPU1 跑 seed 43；
- GPU2 跑 seed 44；
- 无 GPU 分配冲突；
- 输出目录互相隔离；
- 每个 seed 日志完整；
- 任意子任务失败时，启动器返回非零错误码。
- 每个 run 目录生成独立 `gpu_monitor.csv`，采样周期约 1 秒；
- 三个 child 日志均显示 `--num-processes 1`，不存在 NCCL/DDP 初始化。

受限平台命令：

```bash
python launch_covra.py
```

普通平台 dry-run：

```bash
python scripts/platform_train.py \
  --dry-run \
  --skip-model-check \
  --num-gpus 3 \
  --seeds 42,43,44 \
  --batch-size 4 \
  --grad-accum 16 \
  --calibration-batch-size 4
```

只有当单卡 global batch 64 OOM 时，才验证 3 卡 DDP fallback：

```bash
bash scripts/run_ddp.sh configs/dico/dico_cd_da_r8.yaml
```

DDP fallback 日志必须记录 world size `3` 和有效 global batch `63`。

## E00-E：真实配置验收

每个真实 run 完成后，检查日志和 `run_manifest.json`：

- world size；
- global batch；
- optimizer steps；
- warmup steps；
- adapter dtype；
- frozen model dtype；
- dropout；
- gradient checkpointing；
- target modules；
- actual parameter budget；
- rank 配置；
- fallback 状态；
- calibration/allocation/initialization/training 时间；
- 峰值显存；
- tokens/s。
- `data_loading.sample_exposures=100032`、`repeated_exposures=32`、`dataset_seed=42`；
- model effective attention implementation 为 `sdpa`；
- 三个训练 seed 的 calibration selection hash 相同，sketch/init seed 分别为 42/43/44；
- GoRA-public/GoRA-BM 的 A/B 参数组 LR 比在 warmup、峰值和 decay 阶段均为 16；
- GoRA calibration manifest 显示 `gradient_collection=official_weight_grad_hook`、`backward_passes=256`、`answer_only=false`；
- AdaLoRA manifest 同时包含 physical、peak active、final active 参数量；
- 最终 GSM8K 使用左 padding、batch 4、greedy，仅评测 final checkpoint 一次。

然后运行：

```bash
python scripts/validate_run_manifest.py \
  --output-dir outputs/e01_llama3_r8_aligned_sdpa_v4 \
  --json-output reports/run_manifest_validation_server.json \
  --markdown-output reports/run_manifest_validation_server.md

python scripts/collect_run_manifests.py \
  --output-dir outputs/e01_llama3_r8_aligned_sdpa_v4 \
  --json-output reports/run_manifest_summary_server.json \
  --markdown-output reports/run_manifest_summary_server.md
```

## 服务器验收报告

E00-A 到 E00-E 完成后，生成独立报告：

```text
docs/audits/covra_server_gpu_acceptance_report.md
```

只有该报告通过，才能声明项目准备进入 E01 正式实验。报告通过前，所有 GPU 相关事项仍保持 `IMPLEMENTED_NOT_GPU_RUN`，或根据服务器证据标记为 `GPU_RUN_FAILED` / `BLOCKED_BY_ENVIRONMENT`。
