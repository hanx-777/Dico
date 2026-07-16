import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_baseline_status_script_writes_json_and_markdown(tmp_path):
    json_path = tmp_path / "baseline_status.json"
    md_path = tmp_path / "baseline_status.md"

    result = subprocess.run(
        [
            "python",
            "scripts/baseline_status.py",
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
    payload = json.loads(json_path.read_text())
    assert {row["method"] for row in payload["baselines"]} >= {"gora_public", "gora_bm", "covra"}
    assert payload["status_values"]
    covra = next(row for row in payload["baselines"] if row["method"] == "covra")
    assert {
        "requires_grad_params",
        "peak_active_params",
        "final_active_params",
        "budget_error",
    } <= set(covra["parameter_metrics"])
    markdown = md_path.read_text()
    assert "parameter_metrics" in markdown
    assert "requires_grad_params" in markdown
    assert "| gora_public | GoRA-public | IMPLEMENTED_NOT_GPU_RUN |" in markdown
    assert "| covra | CovRA | IMPLEMENTED_NOT_GPU_RUN | configs/dico/dico_cd_da_r8.yaml |" in markdown
