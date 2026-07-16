#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.baselines import REQUIRED_PARAMETER_METRICS


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: str
    message: str


def _pass(check_id: str, message: str) -> CheckResult:
    return CheckResult(check_id, "PASS", message)


def _fail(check_id: str, message: str) -> CheckResult:
    return CheckResult(check_id, "FAIL", message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _has_nested(payload: dict[str, Any], path: str) -> bool:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    return True


def check_required_fields(manifest: dict[str, Any]) -> CheckResult:
    required = [
        "experiment_name",
        "method",
        "rank",
        "seed",
        "source_control.git_commit",
        "source_control.git_dirty",
        "command.argv",
        "command.cwd",
        "command.python_executable",
        "python",
        "dependency_versions.torch",
        "dependency_versions.python",
        "dependency_versions.transformers",
        "dependency_versions.accelerate",
        "cuda_available",
        "cuda_version",
        "cuda_device_count",
        "cuda_device_names",
        "model.name_or_path",
        "model.model_revision",
        "model.tokenizer_revision",
        "model.model_revision_status",
        "model.tokenizer_revision_status",
        "config.resolved_config_path",
        "config.resolved_config_sha256",
        "seeds.base_seed",
        "seeds.calibration_seed",
        "seeds.preallocation_sketch_seed",
        "data.dataset_name",
        "data.train_path",
        "data.eval_path",
        "data.train_count",
        "data.eval_count",
        "data.train_hash",
        "data.eval_hash",
        "calibration.num_selected_samples",
        "calibration.sample_ids",
        "calibration.sample_hashes",
        "calibration.sample_indices",
        "calibration.selection_hash",
        "data_loading.sampler_type",
        "data_loading.drop_last",
        "data_loading.last_accumulation_behavior",
        "data_loading.dataloader_length_batches",
        "data_loading.optimizer_steps_source",
        "world_size",
        "global_batch_size",
        "optimizer_steps",
        "warmup_steps",
        "training.batch_size_per_process",
        "training.gradient_accumulation_steps",
        "training.gradient_checkpointing",
        "training.learning_rate",
        "training.weight_decay",
        "training.max_grad_norm",
        "training.lr_decay_ratio",
        "optimizer.name",
        "optimizer.param_groups",
        "scheduler.name",
        "scheduler.optimizer_steps_source",
        "precision.model_torch_dtype",
        "precision.adapter_dtype",
        "lora.dropout",
        "lora.target_modules",
        "budget.target_budget",
        "budget.actual_budget",
        "budget.budget_error",
        "parameter_counts.requires_grad",
        "parameter_counts.active_final",
        "parameter_counts.active_peak",
        "timing.calibration_sec",
        "timing.allocation_sec",
        "timing.initialization_sec",
        "timing.training_sec",
        "timing.train_tokens",
        "timing.train_tokens_per_sec",
        "hardware.cuda_peak_memory_allocated_bytes",
    ]
    missing = [path for path in required if not _has_nested(manifest, path)]
    if missing:
        return _fail("required_fields", f"missing required manifest fields: {missing}")
    return _pass("required_fields", "core manifest fields are present")


def check_config_hash(manifest: dict[str, Any]) -> CheckResult:
    config_path = Path(str(_nested(manifest, "config.resolved_config_path")))
    expected = _nested(manifest, "config.resolved_config_sha256")
    if not config_path.exists():
        return _fail("config_hash", f"resolved config path does not exist: {config_path}")
    actual = _sha256(config_path)
    if actual != expected:
        return _fail("config_hash", f"config sha256 mismatch for {config_path}: expected {expected}, got {actual}")
    return _pass("config_hash", "resolved config SHA256 matches file contents")


def check_batch_semantics(manifest: dict[str, Any]) -> CheckResult:
    world_size = int(manifest.get("world_size", 0))
    batch = int(_nested(manifest, "training.batch_size_per_process") or 0)
    grad_accum = int(_nested(manifest, "training.gradient_accumulation_steps") or 0)
    global_batch = int(manifest.get("global_batch_size", 0))
    expected = world_size * batch * grad_accum
    if global_batch != expected:
        return _fail("batch_semantics", f"global_batch_size={global_batch} but world_size*batch*grad_accum={expected}")
    protocol_scope = str(manifest.get("protocol_scope", "formal_e01"))
    if protocol_scope == "cpu_tiny_smoke":
        return _pass(
            "batch_semantics",
            f"CPU tiny smoke scope: world_size={world_size}, global_batch_size={global_batch}",
        )
    if world_size == 1 and global_batch != 64:
        return _fail("batch_semantics", f"single-GPU protocol expects global_batch_size=64, got {global_batch}")
    if world_size == 3 and global_batch != 63:
        return _fail("batch_semantics", f"DDP fallback protocol expects global_batch_size=63, got {global_batch}")
    return _pass("batch_semantics", f"world_size={world_size}, global_batch_size={global_batch}")


def check_precision_and_lora(manifest: dict[str, Any]) -> CheckResult:
    adapter_dtype = str(_nested(manifest, "precision.adapter_dtype"))
    dropout = float(_nested(manifest, "lora.dropout"))
    modules = list(_nested(manifest, "lora.target_modules") or [])
    failures: list[str] = []
    if adapter_dtype != "float32":
        failures.append(f"adapter dtype must be float32, got {adapter_dtype}")
    if dropout != 0.0:
        failures.append(f"lora.dropout must be 0, got {dropout}")
    if modules != ["q_proj", "k_proj", "v_proj", "o_proj"]:
        failures.append(f"target_modules must be q/k/v/o projections, got {modules}")
    if failures:
        return _fail("precision_and_lora", "; ".join(failures))
    return _pass("precision_and_lora", "adapter dtype/dropout/target modules match protocol")


def check_model_revision_status(manifest: dict[str, Any]) -> CheckResult:
    allowed = {"LOCKED", "UNRESOLVED"}
    model_status = str(_nested(manifest, "model.model_revision_status"))
    tokenizer_status = str(_nested(manifest, "model.tokenizer_revision_status"))
    model_revision = str(_nested(manifest, "model.model_revision"))
    tokenizer_revision = str(_nested(manifest, "model.tokenizer_revision"))
    failures: list[str] = []
    if model_status not in allowed:
        failures.append(f"model_revision_status must be one of {sorted(allowed)}, got {model_status}")
    if tokenizer_status not in allowed:
        failures.append(f"tokenizer_revision_status must be one of {sorted(allowed)}, got {tokenizer_status}")
    for label, status, revision in (
        ("model", model_status, model_revision),
        ("tokenizer", tokenizer_status, tokenizer_revision),
    ):
        if status == "LOCKED" and revision == "UNRESOLVED":
            failures.append(f"{label} revision status is LOCKED but revision value is UNRESOLVED")
        if status == "UNRESOLVED" and revision != "UNRESOLVED":
            failures.append(f"{label} revision status is UNRESOLVED but revision value is {revision!r}")
    if failures:
        return _fail("model_revision_status", "; ".join(failures))
    return _pass(
        "model_revision_status",
        f"model_revision_status={model_status}, tokenizer_revision_status={tokenizer_status}",
    )


def check_budget_consistency(manifest: dict[str, Any]) -> CheckResult:
    target = int(_nested(manifest, "budget.target_budget"))
    actual = int(_nested(manifest, "budget.actual_budget"))
    error = int(_nested(manifest, "budget.budget_error"))
    active_final = int(_nested(manifest, "parameter_counts.active_final"))
    module_total = _nested(manifest, "module_budget.total_final_params")
    failures: list[str] = []
    if actual - target != error:
        failures.append(f"budget_error={error} but actual-target={actual - target}")
    if str(manifest.get("method")) == "adalora":
        adalora_final = _nested(manifest, "budget.adalora_final_active_params")
        if adalora_final is None or active_final != int(adalora_final):
            failures.append(
                f"AdaLoRA active_final={active_final} but adalora_final_active_params={adalora_final}"
            )
        if module_total is not None and int(module_total) != actual:
            failures.append(
                f"AdaLoRA module A/B budget={module_total} but actual_budget={actual}"
            )
    else:
        if active_final != actual:
            failures.append(f"active_final={active_final} but actual_budget={actual}")
        if module_total is not None and int(module_total) != active_final:
            failures.append(f"module_budget.total_final_params={module_total} but active_final={active_final}")
    if failures:
        return _fail("budget_consistency", "; ".join(failures))
    return _pass("budget_consistency", "budget, parameter counts, and module totals agree")


def check_parameter_metrics(manifest: dict[str, Any]) -> CheckResult:
    metrics = manifest.get("parameter_metrics")
    if not isinstance(metrics, dict):
        return _fail("parameter_metrics", "parameter_metrics section is missing or not an object")
    expected_values = {
        "requires_grad_params": _nested(manifest, "parameter_counts.requires_grad"),
        "peak_active_params": _nested(manifest, "parameter_counts.active_peak"),
        "final_active_params": _nested(manifest, "parameter_counts.active_final"),
        "budget_target": _nested(manifest, "budget.target_budget"),
        "budget_actual": _nested(manifest, "budget.actual_budget"),
        "budget_error": _nested(manifest, "budget.budget_error"),
    }
    failures: list[str] = []
    for metric_name in REQUIRED_PARAMETER_METRICS:
        if metric_name not in metrics:
            failures.append(f"missing parameter_metrics.{metric_name}")
            continue
        expected = expected_values.get(metric_name)
        actual = metrics.get(metric_name)
        if expected is None:
            failures.append(f"cannot cross-check parameter_metrics.{metric_name}: source field is missing")
            continue
        if int(actual) != int(expected):
            failures.append(
                f"parameter_metrics.{metric_name}={actual} but source field value is {expected}"
            )
    if failures:
        return _fail("parameter_metrics", "; ".join(failures))
    return _pass("parameter_metrics", "baseline parameter-metric contract is present and consistent")


def check_optimizer_state_estimate(manifest: dict[str, Any]) -> CheckResult:
    estimate = manifest.get("optimizer_state_estimate")
    if not isinstance(estimate, dict):
        return _fail("optimizer_state_estimate", "optimizer_state_estimate is missing")
    trainable = int(_nested(manifest, "parameter_counts.requires_grad"))
    state_trainable = int(estimate.get("trainable_params", -1))
    bytes_per_param = estimate.get("state_bytes_per_param")
    estimated = estimate.get("estimated_state_bytes")
    failures: list[str] = []
    if state_trainable != trainable:
        failures.append(f"trainable_params={state_trainable} but requires_grad={trainable}")
    if bytes_per_param is not None and estimated != trainable * int(bytes_per_param):
        failures.append(
            f"estimated_state_bytes={estimated} but requires_grad*state_bytes_per_param={trainable * int(bytes_per_param)}"
        )
    if failures:
        return _fail("optimizer_state_estimate", "; ".join(failures))
    return _pass("optimizer_state_estimate", "optimizer-state estimate is consistent with trainable params")


def check_calibration_selection(manifest: dict[str, Any]) -> CheckResult:
    calibration = manifest.get("calibration", {})
    if not isinstance(calibration, dict):
        return _fail("calibration_selection", "calibration section is missing or not an object")
    num_selected = int(calibration.get("num_selected_samples", -1))
    sample_ids = list(calibration.get("sample_ids") or [])
    sample_hashes = [str(value) for value in list(calibration.get("sample_hashes") or [])]
    sample_indices = list(calibration.get("sample_indices") or [])
    recorded_hash = str(calibration.get("selection_hash"))
    failures: list[str] = []
    for label, values in (
        ("sample_ids", sample_ids),
        ("sample_hashes", sample_hashes),
        ("sample_indices", sample_indices),
    ):
        if len(values) != num_selected:
            failures.append(f"{label} length={len(values)} but num_selected_samples={num_selected}")
    digest = hashlib.sha256()
    for value in sample_hashes:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    expected_hash = digest.hexdigest()
    if recorded_hash != expected_hash:
        failures.append(f"selection_hash mismatch: expected {expected_hash}, got {recorded_hash}")
    if failures:
        return _fail("calibration_selection", "; ".join(failures))
    return _pass(
        "calibration_selection",
        f"num_selected_samples={num_selected}, selection_hash verified",
    )


def _artifact_entries(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for group_name in ("method_artifacts", "run_artifacts", "checkpoint_artifacts", "evaluation_artifacts"):
        group = manifest.get(group_name, {})
        if isinstance(group, dict):
            for artifact_name, record in group.items():
                if isinstance(record, dict) and record.get("path"):
                    entries.append((f"{group_name}.{artifact_name}", record))
    return entries


def check_artifact_hashes(manifest: dict[str, Any]) -> CheckResult:
    failures: list[str] = []
    for name, record in _artifact_entries(manifest):
        path = Path(str(record["path"]))
        if not path.exists():
            failures.append(f"{name}: missing artifact {path}")
            continue
        expected_sha = record.get("sha256")
        if expected_sha is not None:
            actual_sha = _sha256(path)
            if actual_sha != expected_sha:
                failures.append(f"{name}: sha256 mismatch for {path}: expected {expected_sha}, got {actual_sha}")
        expected_size = record.get("size_bytes")
        if expected_size is not None and int(expected_size) != path.stat().st_size:
            failures.append(f"{name}: size_bytes mismatch for {path}")
        if record.get("format") == "jsonl" and record.get("num_rows") is not None:
            actual_rows = len(_jsonl_rows(path))
            if actual_rows != int(record["num_rows"]):
                failures.append(f"{name}: num_rows mismatch for {path}: expected {record['num_rows']}, got {actual_rows}")
    if failures:
        return _fail("artifact_hashes", "; ".join(failures))
    return _pass("artifact_hashes", "all recorded artifact paths, SHA256 values, sizes, and row counts match")


def check_prediction_required_fields(manifest: dict[str, Any]) -> CheckResult:
    failures: list[str] = []
    evaluation_artifacts = manifest.get("evaluation_artifacts", {})
    if not isinstance(evaluation_artifacts, dict):
        return _pass("prediction_required_fields", "no evaluation prediction artifacts recorded")
    for name, record in evaluation_artifacts.items():
        if not isinstance(record, dict) or record.get("format") != "jsonl" or not record.get("path"):
            continue
        required = set(record.get("required_fields") or [])
        if not required:
            continue
        rows = _jsonl_rows(Path(str(record["path"])))
        for index, row in enumerate(rows):
            missing = required - set(row)
            if missing:
                failures.append(f"{name}: row {index} missing fields {sorted(missing)}")
                break
    if failures:
        return _fail("prediction_required_fields", "; ".join(failures))
    return _pass("prediction_required_fields", "prediction JSONL rows contain their required fields")


CHECKS = (
    check_required_fields,
    check_config_hash,
    check_batch_semantics,
    check_precision_and_lora,
    check_model_revision_status,
    check_budget_consistency,
    check_parameter_metrics,
    check_optimizer_state_estimate,
    check_calibration_selection,
    check_artifact_hashes,
    check_prediction_required_fields,
)


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checks = [check(manifest) for check in CHECKS]
    return {
        "path": str(path),
        "experiment_name": manifest.get("experiment_name"),
        "status": "FAIL" if any(check.status == "FAIL" for check in checks) else "PASS",
        "checks": [asdict(check) for check in checks],
    }


def _manifest_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in args.manifest]
    if args.output_dir:
        output_dir = Path(args.output_dir)
        paths.extend(sorted(output_dir.glob("*/run_manifest.json")))
    if not paths:
        raise SystemExit("Provide --manifest or --output-dir")
    return paths


def run_validation(paths: list[Path]) -> dict[str, Any]:
    manifests = [validate_manifest(path) for path in paths]
    failed_checks = sum(
        1
        for manifest in manifests
        for check in manifest["checks"]
        if check["status"] == "FAIL"
    )
    return {
        "summary": {
            "status": "FAIL" if failed_checks else "PASS",
            "validated_manifests": len(manifests),
            "failed_checks": failed_checks,
            "total_checks": sum(len(row["checks"]) for row in manifests),
        },
        "manifests": manifests,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Run Manifest Validation",
        "",
        f"- status: `{payload['summary']['status']}`",
        f"- validated_manifests: `{payload['summary']['validated_manifests']}`",
        f"- failed_checks: `{payload['summary']['failed_checks']}`",
        "",
        "| manifest | status | failed_checks |",
        "|---|---|---|",
    ]
    for row in payload["manifests"]:
        failed = [check for check in row["checks"] if check["status"] == "FAIL"]
        failed_text = "<br>".join(f"`{check['id']}`: {check['message']}" for check in failed) if failed else "-"
        lines.append(f"| {row['path']} | {row['status']} | {failed_text} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate completed run_manifest.json artifacts.")
    parser.add_argument("--manifest", action="append", default=[], help="Path to a run_manifest.json file.")
    parser.add_argument("--output-dir", help="Directory containing */run_manifest.json files.")
    parser.add_argument("--json-output", default="reports/run_manifest_validation.json")
    parser.add_argument("--markdown-output", default="reports/run_manifest_validation.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _manifest_paths(args)
    payload = run_validation(paths)

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.markdown_output:
        md_path = Path(args.markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(payload), encoding="utf-8")

    if payload["summary"]["status"] == "FAIL":
        for manifest in payload["manifests"]:
            for check in manifest["checks"]:
                if check["status"] == "FAIL":
                    print(f"{manifest['path']}: {check['id']}: {check['message']}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[validate_run_manifest] validated {payload['summary']['validated_manifests']} manifest(s): PASS")


if __name__ == "__main__":
    main()
