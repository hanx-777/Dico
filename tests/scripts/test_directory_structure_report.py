import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_directory_structure_report_records_deliverable_tree(tmp_path: Path):
    json_path = tmp_path / "directory_structure.json"
    md_path = tmp_path / "directory_structure.md"

    result = subprocess.run(
        [
            "python",
            "scripts/directory_structure.py",
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
    roots = {entry["path"] for entry in payload["roots"]}
    assert {
        "configs",
        "scripts",
        "src/dico",
        "tests",
        "reports",
    } <= roots
    assert "outputs" in payload["excluded_roots"]
    assert ".git" in payload["excluded_roots"]
    assert any(entry["path"] == "configs/dico/dico_cd_da_r8.yaml" for entry in payload["entries"])
    assert any(entry["path"] == "src/dico/covra_core.py" for entry in payload["entries"])
    assert any(entry["path"] == "scripts/run_experiment.py" for entry in payload["entries"])
    assert all("__pycache__" not in entry["path"] for entry in payload["entries"])
    assert all(".DS_Store" not in entry["path"] for entry in payload["entries"])

    markdown = md_path.read_text(encoding="utf-8")
    assert "# Final Directory Structure" in markdown
    assert "configs/dico" in markdown
    assert "src/dico" in markdown
    assert "reports/audit" in markdown
