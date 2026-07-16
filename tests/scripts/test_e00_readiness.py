import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_e00_readiness_report_checks_configs_data_baselines_and_launchers(tmp_path):
    json_path = tmp_path / "e00_readiness.json"
    md_path = tmp_path / "e00_readiness.md"

    result = subprocess.run(
        [
            "python",
            "scripts/e00_readiness.py",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
            "--require-gpu-count",
            "0",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "READY_DRY_RUN"
    assert payload["summary"]["failed_checks"] == 0
    check_ids = {check["id"] for check in payload["checks"]}
    assert {
        "protocol_preflight",
        "data_files",
        "model_reference",
        "dependency_versions",
        "output_dir",
        "baseline_statuses",
        "platform_launcher",
        "single_file_platform_launcher",
        "platform_launcher_dry_run",
        "shell_wrapper_entrypoints",
        "ddp_fallback_launcher",
        "ddp_fallback_protocol",
        "gpu_count",
    } <= check_ids
    data_check = next(check for check in payload["checks"] if check["id"] == "data_files")
    assert data_check["details"]["train"]["exists"] is True
    assert data_check["details"]["train"]["sha256"]
    assert data_check["details"]["eval"]["line_count"] == 1319
    model_check = next(check for check in payload["checks"] if check["id"] == "model_reference")
    assert model_check["status"] == "PASS"
    assert model_check["details"]["name_or_path"]
    dependency_check = next(check for check in payload["checks"] if check["id"] == "dependency_versions")
    assert dependency_check["status"] in {"PASS", "WARN"}
    assert dependency_check["details"]["python"]
    assert dependency_check["details"]["packages"]["torch"]["installed"] is True
    assert "flash-attn" not in dependency_check["details"]["required_packages"]
    output_check = next(check for check in payload["checks"] if check["id"] == "output_dir")
    assert output_check["status"] == "PASS"
    assert output_check["details"]["writable"] is True
    launcher_dry_run = next(check for check in payload["checks"] if check["id"] == "platform_launcher_dry_run")
    assert launcher_dry_run["status"] == "PASS"
    assert launcher_dry_run["details"]["command_count"] == 12
    assert launcher_dry_run["details"]["effective_batch"] == 64
    assert launcher_dry_run["details"]["num_gpus"] == 3
    wrapper_check = next(check for check in payload["checks"] if check["id"] == "shell_wrapper_entrypoints")
    assert wrapper_check["status"] == "PASS"
    assert "scripts/run_ablation_covra_independent.sh" in wrapper_check["details"]["checked_wrappers"]
    assert "scripts/run_ablation_no_sign_split.sh" in wrapper_check["details"]["checked_wrappers"]
    assert "scripts/run_ablation_random_init.sh" in wrapper_check["details"]["checked_wrappers"]
    assert wrapper_check["details"]["failed_wrappers"] == []
    ddp_protocol = next(check for check in payload["checks"] if check["id"] == "ddp_fallback_protocol")
    assert ddp_protocol["status"] == "PASS"
    assert ddp_protocol["details"]["default_num_gpus"] == 3
    assert ddp_protocol["details"]["default_per_gpu_batch_size"] == 3
    assert ddp_protocol["details"]["default_gradient_accumulation_steps"] == 7
    assert ddp_protocol["details"]["effective_global_batch"] == 63
    assert ddp_protocol["details"]["accelerate_config_exists"] is True
    baseline_check = next(check for check in payload["checks"] if check["id"] == "baseline_statuses")
    assert "adalora" not in baseline_check["details"]["not_ready_or_blocked"]
    assert "# E00 Readiness" in md_path.read_text()


def test_e00_readiness_can_fail_when_required_gpu_count_is_unmet(tmp_path):
    json_path = tmp_path / "e00_readiness_gpu_fail.json"
    md_path = tmp_path / "e00_readiness_gpu_fail.md"

    result = subprocess.run(
        [
            "python",
            "scripts/e00_readiness.py",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
            "--require-gpu-count",
            "999",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    gpu_check = next(check for check in payload["checks"] if check["id"] == "gpu_count")
    assert gpu_check["status"] == "FAIL"
    assert "requires at least 999 visible CUDA GPU" in gpu_check["message"]
    assert "gpu_count" in result.stderr


def test_e00_readiness_fails_when_local_model_path_is_incomplete(tmp_path):
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    json_path = tmp_path / "e00_readiness_bad_model.json"
    md_path = tmp_path / "e00_readiness_bad_model.md"

    result = subprocess.run(
        [
            "python",
            "scripts/e00_readiness.py",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
            "--model-path",
            str(model_dir),
            "--require-gpu-count",
            "0",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    model_check = next(check for check in payload["checks"] if check["id"] == "model_reference")
    assert model_check["status"] == "FAIL"
    assert "missing config.json" in model_check["message"]
    assert "model_reference" in result.stderr


def test_e00_readiness_can_require_runtime_dependencies(tmp_path):
    json_path = tmp_path / "e00_readiness_bad_deps.json"
    md_path = tmp_path / "e00_readiness_bad_deps.md"

    result = subprocess.run(
        [
            "python",
            "scripts/e00_readiness.py",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
            "--require-gpu-count",
            "0",
            "--require-runtime-deps",
            "--require-package",
            "definitely-missing-covra-test-package",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    dependency_check = next(check for check in payload["checks"] if check["id"] == "dependency_versions")
    assert dependency_check["status"] == "FAIL"
    assert "definitely-missing-covra-test-package" in dependency_check["details"]["missing_required"]
    assert "dependency_versions" in result.stderr


def test_e00_readiness_fails_when_output_dir_is_not_a_directory(tmp_path):
    bad_output = tmp_path / "not_a_directory"
    bad_output.write_text("not a dir", encoding="utf-8")
    json_path = tmp_path / "e00_readiness_bad_output.json"
    md_path = tmp_path / "e00_readiness_bad_output.md"

    result = subprocess.run(
        [
            "python",
            "scripts/e00_readiness.py",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
            "--output-dir",
            str(bad_output),
            "--require-gpu-count",
            "0",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    output_check = next(check for check in payload["checks"] if check["id"] == "output_dir")
    assert output_check["status"] == "FAIL"
    assert "not a directory" in output_check["message"]
    assert "output_dir" in result.stderr
