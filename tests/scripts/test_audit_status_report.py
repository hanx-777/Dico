import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_audit_status_script_writes_method_protocol_and_status_reports(tmp_path):
    output_dir = tmp_path / "audit"

    result = subprocess.run(
        [
            "python",
            "scripts/audit_status.py",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    method = json.loads((output_dir / "method_implementation_audit.json").read_text())
    protocol = json.loads((output_dir / "experiment_protocol_audit.json").read_text())
    status = json.loads((output_dir / "status_matrix.json").read_text())

    method_ids = {row["id"] for row in method["requirements"]}
    assert {
        "covra_full",
        "covra_independent",
        "covra_module_scalar",
        "ablation_single_factor_metadata",
        "dp_solver",
        "legacy_isolation",
    } <= method_ids
    assert method["requirements_by_status"]["IMPLEMENTED_NOT_GPU_RUN"] >= 1
    legacy = next(row for row in method["requirements"] if row["id"] == "legacy_isolation")
    assert "legacy artifacts" in legacy["requirement"]
    assert "tests/unit/test_dico_da_init_trainer_integration.py" in legacy["evidence"]

    protocol_ids = {row["id"] for row in protocol["requirements"]}
    assert {
        "single_gpu_global_batch_64",
        "e00_readiness_report",
        "ddp_fallback_global_batch_63",
        "r32_pilot_config",
        "ddp_data_loading_manifest",
        "strict_config_schema",
        "protocol_preflight",
        "source_control_and_command_manifest",
        "model_revision_manifest",
        "resolved_config_manifest",
        "seed_manifest",
        "unresolved_protocol_fields",
        "data_calibration_manifest",
        "runtime_hardware_manifest",
        "dependency_versions_manifest",
        "adapter_fp32_protocol",
        "optimizer_lr_group_manifest",
        "optimizer_state_estimate_manifest",
        "scheduler_protocol_manifest",
        "module_budget_manifest",
        "gradient_clipping_protocol",
        "evaluation_artifact_manifest",
        "timing_and_throughput_manifest",
        "method_artifact_manifest",
        "run_artifact_manifest",
        "run_manifest_validation",
        "checkpoint_artifact_manifest",
        "checkpoint_restore",
        "checkpoint_selection_protocol",
        "gsm8k_greedy",
        "mtbench_local",
    } <= protocol_ids
    model_revision = next(row for row in protocol["requirements"] if row["id"] == "model_revision_manifest")
    assert "UNRESOLVED" in model_revision["requirement"]
    mtbench = next(row for row in protocol["requirements"] if row["id"] == "mtbench_local")
    assert mtbench["status"] == "IMPLEMENTED_NOT_GPU_RUN"

    status_ids = {row["id"] for row in status["requirements"]}
    assert "gpu_e00_pilot" in status_ids
    assert next(row for row in status["requirements"] if row["id"] == "gpu_e00_pilot")["status"] == "NOT_EXECUTED"
    assert next(row for row in status["requirements"] if row["id"] == "MTBench-local executor")["status"] == "IMPLEMENTED_NOT_GPU_RUN"

    method_md = (output_dir / "method_implementation_audit.md").read_text()
    protocol_md = (output_dir / "experiment_protocol_audit.md").read_text()
    status_md = (output_dir / "status_matrix.md").read_text()
    assert "# Method Implementation Audit" in method_md
    assert "# Experiment Protocol Audit" in protocol_md
    assert "# Status Matrix" in status_md
    assert "GoRA-public" in status_md
