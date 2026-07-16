import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_changed_files_report_structures_git_status(tmp_path: Path):
    json_path = tmp_path / "changed_files.json"
    md_path = tmp_path / "changed_files.md"

    result = subprocess.run(
        [
            "python",
            "scripts/changed_files_report.py",
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
    assert payload["summary"]["total_changed_files"] > 0
    assert {
        "added",
        "modified",
        "deleted",
        "renamed",
        "untracked",
        "other",
    } <= set(payload["by_category"])
    assert any(row["path"].endswith("README.md") for row in payload["files"])
    assert all("raw_status" in row for row in payload["files"])

    markdown = md_path.read_text(encoding="utf-8")
    assert "# Changed Files Report" in markdown
    assert "| category | count |" in markdown
    assert "README.md" in markdown
