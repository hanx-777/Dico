import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_experiment_matrix_script_writes_e00_to_e10_commands_and_blockers(tmp_path):
    json_path = tmp_path / "experiment_matrix.json"
    md_path = tmp_path / "experiment_matrix.md"

    result = subprocess.run(
        [
            "python",
            "scripts/experiment_matrix.py",
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
    ids = {row["id"] for row in payload["experiments"]}
    assert {f"E{idx:02d}" for idx in range(11)} <= ids

    e00 = next(row for row in payload["experiments"] if row["id"] == "E00")
    assert e00["status"] == "READY_DRY_RUN"
    assert any("training.max_steps=1" in command for command in e00["commands"])
    assert any("training.batch_size=4" in command for command in e00["commands"])
    assert any("training.gradient_accumulation_steps=16" in command for command in e00["commands"])

    e01 = next(row for row in payload["experiments"] if row["id"] == "E01")
    assert e01["priority"] == "must"
    assert any("scripts/platform_train.py" in command for command in e01["commands"])
    assert all("--child-num-processes 1" in command for command in e01["commands"])
    assert any("configs/dico/adalora_r8.yaml" in command for command in e01["commands"])
    assert not any("configs/ablations/covra_independent.yaml" in command for command in e01["commands"])
    assert not any("configs/ablations/covra_module_scalar.yaml" in command for command in e01["commands"])
    assert any("configs/ablations/covra_independent.yaml" in command for command in next(row for row in payload["experiments"] if row["id"] == "E04")["commands"])
    assert any("configs/ablations/covra_module_scalar.yaml" in command for command in next(row for row in payload["experiments"] if row["id"] == "E05")["commands"])
    assert "adalora" not in e01["blocked_items"]
    assert "eva" in e01["blocked_items"]

    e02 = next(row for row in payload["experiments"] if row["id"] == "E02")
    assert all("outputs/e02_llama3_r8_strict_budget_sdpa_v4" in command for command in e02["commands"])

    e09 = next(row for row in payload["experiments"] if row["id"] == "E09")
    assert any("scripts/e00_readiness.py" in command for command in e09["commands"])
    assert any("scripts/collect_run_manifests.py" in command for command in e09["commands"])
    assert any("scripts/validate_run_manifest.py" in command for command in e09["commands"])
    assert any("scripts/protocol_preflight.py" in command for command in e09["commands"])
    assert any("scripts/static_acceptance.py" in command for command in e09["commands"])
    assert any("scripts/directory_structure.py" in command for command in e09["commands"])
    assert any("scripts/changed_files_report.py" in command for command in e09["commands"])
    assert any("scripts/mtbench_local_judge.py" in command for command in e09["commands"])
    assert any("scripts/final_delivery_report.py" in command for command in e09["commands"])
    assert "mtbench_local_70b_judge_execution:NOT_EXECUTED" in e09["blocked_items"]
    assert "mtbench_local_judge_executor:NOT_IMPLEMENTED" not in e09["blocked_items"]
    assert "full run-manifest aggregator:NOT_IMPLEMENTED" not in e09["blocked_items"]

    e10 = next(row for row in payload["experiments"] if row["id"] == "E10")
    assert any("dico_cd_da_r32_pilot.yaml" in command for command in e10["commands"])
    assert "r32 configs require K>=128 pilot" not in e10["blocked_items"]

    markdown = md_path.read_text()
    assert "| id | title | priority | status | purpose | commands | blocked_items | notes |" in markdown
    assert "| E00 |" in markdown
    assert "| E10 |" in markdown
    assert "BLOCKED_BY_UNRESOLVED_PROTOCOL" in markdown
    assert "DDP command is fallback only" in markdown
