import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_static_acceptance_report_covers_stage4_gates(tmp_path: Path):
    json_path = tmp_path / "static_acceptance.json"
    md_path = tmp_path / "static_acceptance.md"

    result = subprocess.run(
        [
            "python",
            "scripts/static_acceptance.py",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    check_ids = {check["id"] for check in payload["checks"]}
    assert {
        "import_core_modules",
        "syntax_compile",
        "config_dry_run",
        "cpu_tiny_training_smoke",
        "cpu_tiny_manifest_validation",
        "cpu_tiny_parameter_budget_audit",
        "experiment_matrix_generation",
        "protocol_preflight",
        "typecheck_tool",
        "lint_tool",
    } <= check_ids
    assert payload["summary"]["failed_checks"] == 0
    assert payload["summary"]["status"] in {"PASS", "PASS_WITH_SKIPS"}
    assert "# Static Acceptance" in md_path.read_text(encoding="utf-8")
