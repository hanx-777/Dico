import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_final_delivery_report_summarizes_current_artifacts_and_blockers(tmp_path: Path):
    json_path = tmp_path / "final_delivery.json"
    md_path = tmp_path / "final_delivery.md"

    result = subprocess.run(
        [
            "python",
            "scripts/final_delivery_report.py",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
            "--test-result",
            "pytest -q :: 278 passed, 4 warnings",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["summary"]["status"] == "NOT_COMPLETE_GPU_AND_EXTERNAL_PROTOCOL_PENDING"
    assert payload["summary"]["e00_readiness_status"] == "READY_DRY_RUN"
    assert payload["test_evidence"]["latest"] == "pytest -q :: 278 passed, 4 warnings"
    assert "gpu_e00_pilot" in payload["gpu_not_executed"]
    assert "EVA" in set(payload["external_protocol_blockers"])
    assert "GoRA-public" not in set(payload["external_protocol_blockers"])
    assert "E00" in payload["experiment_commands"]
    assert "E10" in payload["experiment_commands"]
    assert any("scripts/platform_train.py" in command for command in payload["experiment_commands"]["E01"])
    assert payload["delivery_artifacts"]["readme"] == "README.md"
    assert payload["delivery_artifacts"]["directory_structure"] == "reports/directory_structure.md"
    assert payload["delivery_artifacts"]["changed_files"] == "reports/changed_files.md"
    assert payload["delivery_artifacts"]["static_acceptance"] == "reports/static_acceptance.md"
    assert payload["summary"]["static_acceptance_status"] in {"PASS", "PASS_WITH_SKIPS"}
    skipped_static = {row["id"] for row in payload["static_acceptance_skipped_checks"]}
    assert {"typecheck_tool", "lint_tool"} <= skipped_static
    skipped_protocol = {row["id"] for row in payload["protocol_preflight_skipped_checks"]}
    assert "covra_candidate_budget" in skipped_protocol
    assert payload["delivery_artifacts"]["method_audit"] == "reports/audit/method_implementation_audit.md"
    assert payload["delivery_artifacts"]["protocol_audit"] == "reports/audit/experiment_protocol_audit.md"
    assert payload["delivery_artifacts"]["status_matrix"] == "reports/audit/status_matrix.md"
    assert "modified_files" in payload
    assert "E00 single-GPU LoRA/CovRA/AdaLoRA/GoRA-public/GoRA-BM pilot on A800" in payload["unexecuted_gpu_tests"]
    assert "E01-E10 formal/recommended GPU training matrix" in payload["unexecuted_gpu_tests"]
    assert "DDP fallback runtime validation after single-GPU OOM" in payload["unexecuted_gpu_tests"]

    markdown = md_path.read_text(encoding="utf-8")
    assert "# Final Delivery Report" in markdown
    assert "NOT_COMPLETE_GPU_AND_EXTERNAL_PROTOCOL_PENDING" in markdown
    assert "pytest -q :: 278 passed, 4 warnings" in markdown
    assert "GoRA-public" in markdown
    assert "Static acceptance skipped checks" in markdown
    assert "typecheck_tool" in markdown
    assert "lint_tool" in markdown
    assert "Protocol preflight skipped checks" in markdown
    assert "covra_candidate_budget" in markdown
    assert "Unexecuted GPU tests" in markdown
    assert "E01-E10 formal/recommended GPU training matrix" in markdown
    assert "DDP fallback runtime validation after single-GPU OOM" in markdown
    assert "directory_structure" in markdown
    assert "changed_files" in markdown
    assert "E00" in markdown and "E10" in markdown
