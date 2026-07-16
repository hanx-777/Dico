from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_states_current_default_model_datasets_and_local_readiness_boundary():
    readme = _read("README.md")

    assert "当前默认实验模型与数据集" in readme
    assert "Meta-Llama-3.1-8B-Base" in readme
    assert "MetaMathQA-100K" in readme
    assert "GSM8K test" in readme
    assert "CodeFeedback" in readme
    assert "HumanEval" in readme
    assert "WizardLM" in readme
    assert "MTBench" in readme
    assert "本地实现与静态验收已完成，可以上传服务器并进入E00 GPU pilot。" in readme
    assert "任务已完整完成，可以开始正式实验" not in readme


def test_launch_covra_top_configuration_is_documented_in_chinese():
    text = _read("launch_covra.py")

    assert "手动配置区" in text
    assert "需要在 A800 服务器上修改" in text
    assert "不要在这里根据结果调参" in text
    assert "只负责平台启动" in text


def test_acceptance_docs_are_chinese_and_keep_cpu_gpu_statuses_separate():
    local = _read("docs/audits/covra_local_acceptance_audit.md")
    server = _read("docs/audits/covra_server_gpu_acceptance_checklist.md")

    for forbidden in (
        "Local-stage conclusion",
        "Evidence snapshot",
        "Server GPU Acceptance Checklist",
        "Required evidence",
        "Suggested command",
        "Launch entrypoint",
    ):
        assert forbidden not in local
        assert forbidden not in server

    assert "本地阶段结论" in local
    assert "服务器 GPU 验收清单" in server
    assert "IMPLEMENTED_AND_CPU_VERIFIED" in local
    assert "IMPLEMENTED_NOT_GPU_RUN" in local
    assert "IMPLEMENTED_AND_GPU_VERIFIED" in server
    assert "不能把 CPU 验证和 GPU 验证合并成同一种状态" in local


def test_chinese_docs_cover_environment_paths_launch_evaluation_and_faq():
    readme = _read("README.md")
    server = _read("docs/audits/covra_server_gpu_acceptance_checklist.md")

    for phrase in (
        "环境配置",
        "路径配置说明",
        "启动命令",
        "评测协议当前状态",
        "常见问题排查",
    ):
        assert phrase in readme

    for phrase in (
        "环境检查",
        "单卡 LoRA pilot",
        "单卡 CovRA pilot",
        "三卡调度检查",
        "真实配置验收",
        "服务器验收报告",
    ):
        assert phrase in server
