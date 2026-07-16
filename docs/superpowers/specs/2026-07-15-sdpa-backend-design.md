# 统一 SDPA 注意力后端设计

## 目标

在目标服务器无法加载 FlashAttention2 的条件下，将所有正式 CovRA 对比方法统一切换到 PyTorch SDPA，同时保持模型、数据、参数预算、训练步数、有效批大小、学习率、seed 和评测协议不变。

## 设计

- 公共配置使用 `model.attn_implementation: sdpa`，并关闭 `runtime.require_flash_attention_2`。
- LoRA、AdaLoRA、GoRA-public、GoRA-BM 与 CovRA 均从同一公共配置继承该后端，禁止方法级覆盖。
- E01/E02 输出升级到独立的 `aligned_sdpa_v4` 目录，避免与 FlashAttention2 pilot 或失败的 v3 目录混合。
- 服务器 readiness 不再把 `flash-attn` 当作必需依赖；安装脚本不再安装或导入它。
- 协议预检必须验证正式配置统一为 SDPA，并继续验证原有 batch、步数、dtype、预算和评测约束。
- manifest 继续通过 resolved config 记录实际注意力后端。

## 公平性边界

SDPA 与 FlashAttention2 实现相同的缩放点积注意力语义，但浮点归约顺序可能不同，因此不承诺逐 bit 或最终指标完全相同。所有方法和 seed 必须统一重跑；SDPA v4 结果不得与 FlashAttention2 v3 结果混合统计。该变化作为统一的环境适配差异披露。

## 验收

- 五个正式配置解析后均为 SDPA，且不要求 FlashAttention2。
- 模型加载器把 `sdpa` 传给 Transformers。
- readiness 严格依赖列表不包含 `flash-attn`。
- 启动 dry-run 输出新的 v4 目录及原有 12 个单卡任务。
- 协议预检、CPU smoke 与全量 pytest 通过。
