# LoRA、AdaLoRA、GoRA 公平对齐实施记录

本实施记录对应用户确认的“LoRA、AdaLoRA、GoRA 公平对齐实施方案”。修改遵循测试先行，且不覆盖既有输出。

1. 以单元测试锁定标准 LoRA 初始化、scaling、官方式学习率调度和方法参数组。
2. 将简化 AdaLoRA 替换为 A/E/B 三元组、全局预算裁剪、EMA 重要度和正交正则实现。
3. 新增 GoRA-public 与 GoRA-BM，实现官方重要度、秩映射、伪逆初始化、动态 scaling 和 B/A 16 倍学习率。
4. 将 GSM8K 最终评测改为左 padding 的批量 greedy generation，并保持原始预测归档。
5. 更新正式配置、唯一启动器、协议预检、manifest 与中文文档。
6. 运行 CPU 单元测试、小模型端到端测试和配置/启动验收；GPU 项只标记为未运行。

边界：EVA、CodeFeedback/HumanEval、WizardLM/MTBench、r32/r128 不在本轮实现范围；旧 `gora_bw` 与已有 pilot 输出均保留为 legacy。
