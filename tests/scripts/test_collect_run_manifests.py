import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _write_manifest(root: Path, experiment: str, seed: int, metric: float, budget_error: int = 0) -> None:
    run_dir = root / experiment
    run_dir.mkdir(parents=True)
    payload = {
        "experiment_name": experiment,
        "method": "dico_cd_da",
        "rank": 8,
        "seed": seed,
        "world_size": 1,
        "global_batch_size": 64,
        "optimizer_steps": 1563,
        "warmup_steps": 47,
        "budget": {
            "target_budget": 6815744,
            "actual_budget": 6815744 + budget_error,
            "budget_error": budget_error,
            "budget_error_ratio": budget_error / 6815744,
        },
        "parameter_metrics": {
            "requires_grad_params": 6815744 + budget_error,
            "peak_active_params": 6815744 + budget_error,
            "final_active_params": 6815744 + budget_error,
            "budget_target": 6815744,
            "budget_actual": 6815744 + budget_error,
            "budget_error": budget_error,
        },
        "metrics": {
            "final_metric": metric,
            "final_eval_accuracy": metric,
            "target_budget": 6815744,
        },
        "parameter_counts": {
            "requires_grad": 6815744 + budget_error,
            "active_final": 6815744 + budget_error,
            "active_peak": 6815744 + budget_error,
        },
        "timing": {"run_elapsed_sec": 100.0 + seed},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_collect_run_manifests_groups_seeds_and_writes_reports(tmp_path):
    output_dir = tmp_path / "outputs"
    _write_manifest(output_dir, "covra_seed42", 42, 0.60)
    _write_manifest(output_dir, "covra_seed43", 43, 0.66)
    _write_manifest(output_dir, "lora_seed42", 42, 0.50, budget_error=-16)
    json_path = tmp_path / "summary.json"
    md_path = tmp_path / "summary.md"

    result = subprocess.run(
        [
            "python",
            "scripts/collect_run_manifests.py",
            "--output-dir",
            str(output_dir),
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
    covra = payload["groups"]["covra"]
    assert covra["seeds"] == [42, 43]
    assert covra["n"] == 2
    assert abs(covra["fields"]["final_metric"]["mean"] - 0.63) < 1e-12
    assert covra["fields"]["final_metric"]["std"] > 0
    assert covra["fields"]["global_batch_size"]["mean"] == 64
    assert covra["fields"]["budget_error"]["mean"] == 0
    lora = payload["groups"]["lora"]
    assert lora["fields"]["budget_error"]["mean"] == -16

    markdown = md_path.read_text()
    assert (
        "| group | method | seeds | final_metric mean±std | budget_error mean | "
        "requires_grad mean | active_final mean | active_peak mean | global_batch |"
    ) in markdown
    assert "| covra | dico_cd_da | 42,43 |" in markdown
    assert "6815744" in markdown


def test_collect_run_manifests_can_summarize_baseline_parameter_metric_names(tmp_path):
    output_dir = tmp_path / "outputs"
    _write_manifest(output_dir, "covra_seed42", 42, 0.60)
    _write_manifest(output_dir, "covra_seed43", 43, 0.66)
    json_path = tmp_path / "summary.json"

    result = subprocess.run(
        [
            "python",
            "scripts/collect_run_manifests.py",
            "--output-dir",
            str(output_dir),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(tmp_path / "summary.md"),
            "--fields",
            "requires_grad_params",
            "peak_active_params",
            "final_active_params",
            "budget_target",
            "budget_actual",
            "budget_error",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    covra = json.loads(json_path.read_text())["groups"]["covra"]
    fields = covra["fields"]
    assert fields["requires_grad_params"]["mean"] == 6815744
    assert fields["peak_active_params"]["mean"] == 6815744
    assert fields["final_active_params"]["mean"] == 6815744
    assert fields["budget_target"]["mean"] == 6815744
    assert fields["budget_actual"]["mean"] == 6815744
    assert fields["budget_error"]["mean"] == 0
