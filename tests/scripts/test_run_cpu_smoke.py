from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_cpu_smoke_runs_all_formal_methods_through_shared_trainer(tmp_path):
    report = tmp_path / "cpu_smoke.json"
    result = subprocess.run(
        ["python", "scripts/run_cpu_smoke.py", "--output-root", str(tmp_path / "runs"), "--report", str(report)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "IMPLEMENTED_AND_CPU_VERIFIED"
    assert set(payload["methods"]) == {"lora", "adalora", "gora_public", "gora_bm", "dico_cd_da"}
    assert all(row["optimizer_steps"] == 1 for row in payload["methods"].values())
    assert all(row["gpu_status"] == "IMPLEMENTED_NOT_GPU_RUN" for row in payload["methods"].values())
