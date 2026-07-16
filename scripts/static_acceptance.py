#!/usr/bin/env python
from __future__ import annotations

import argparse
import compileall
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

_TINY_SMOKE_CACHE: dict[str, Any] | None = None


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _pass(check_id: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(check_id, "PASS", message, details)


def _fail(check_id: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(check_id, "FAIL", message, details)


def _skip(check_id: str, message: str, **details: Any) -> CheckResult:
    return CheckResult(check_id, "SKIP", message, details)


def _run(command: list[str], timeout_sec: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=False,
    )


def check_import_core_modules() -> CheckResult:
    modules = [
        "dico.config",
        "dico.covra_core",
        "dico.rank_budget",
        "dico.trainer",
        "dico.baselines",
    ]
    imported: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - message is the useful artifact.
            return _fail("import_core_modules", f"failed importing {module}: {exc}", imported=imported)
        imported.append(module)
    return _pass("import_core_modules", "core CovRA/training modules import successfully", imported=imported)


def check_syntax_compile() -> CheckResult:
    targets = [ROOT / "src" / "dico", ROOT / "scripts"]
    failures: list[str] = []
    for target in targets:
        ok = compileall.compile_dir(str(target), quiet=1, maxlevels=20)
        if not ok:
            failures.append(str(target.relative_to(ROOT)))
    if failures:
        return _fail("syntax_compile", f"Python syntax compilation failed for {failures}", failed_targets=failures)
    return _pass("syntax_compile", "src/dico and scripts compile with Python bytecode compiler")


def _command_check(check_id: str, command: list[str], pass_message: str, timeout_sec: int = 120) -> CheckResult:
    result = _run(command, timeout_sec=timeout_sec)
    details = {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }
    if result.returncode != 0:
        return _fail(check_id, f"command failed with return code {result.returncode}", **details)
    return _pass(check_id, pass_message, **details)


def check_config_dry_run() -> CheckResult:
    return _command_check(
        "config_dry_run",
        ["python", "scripts/run_experiment.py", "--config", "configs/dico/dico_cd_da_r8.yaml", "--dry-run"],
        "main CovRA r8 config resolves through run_experiment --dry-run",
    )


def _tiny_covra_config(work_dir: Path) -> dict[str, Any]:
    return {
        "_project_root": str(work_dir),
        "seed": 42,
        "experiment_name": "static_acceptance_tiny_covra",
        "method": "dico_cd_da",
        "rank": 1,
        "project": {"output_dir": str(work_dir / "outputs")},
        "model": {
            "type": "tiny",
            "name_or_path": "tiny",
            "hidden_size": 8,
            "vocab_size": 64,
            "torch_dtype": "float32",
        },
        "data": {
            "source": "tiny",
            "train_path": "tiny",
            "eval_path": "tiny",
            "max_length": 16,
            "train_limit": 2,
            "eval_limit": 2,
        },
        "training": {
            "max_steps": 1,
            "batch_size": 4,
            "gradient_accumulation_steps": 16,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.03,
            "lr_decay_ratio": 0.1,
            "weight_decay": 5e-4,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": False,
        },
        "lora": {
            "injection": "static",
            "alpha": 16,
            "dropout": 0.0,
            "adapter_dtype": "float32",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "max_rank_multiplier": 4,
        },
        "budget": {
            "mode": "equal_trainable_params",
            "warning_threshold": 0.01,
            "enforce_target_ratio": 1.0,
            "enforce_min_ratio": 0.5,
        },
        "calibration": {
            "enabled": True,
            "num_samples": 2,
            "batch_size": 1,
            "seed": 42,
            "save_dir": str(work_dir / "preallocations"),
        },
        "preallocation": {
            "atom_mode": "svd",
            "allocation_method": "covra_full",
            "top_k_atoms": 4,
            "sketch_dim": 4,
            "sketch_seed": 42,
            "answer_only": False,
            "profile_norm_mode": "streaming_estimate",
            "lambda_cov": 1.0,
            "response_agg_groups": 2,
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
            "solver": "dp",
            "subspace_init": "direction_anchored",
            "eta": 0.5,
            "r_min_multiplier": 0.0,
            "r_max_multiplier": 4.0,
            "allow_rank_beyond_selected_evidence": True,
        },
        "dico": {
            "version": "cd_da",
            "init": {"mode": "direction_anchored", "zero_B": True},
        },
    }


def _run_tiny_covra_smoke() -> dict[str, Any]:
    global _TINY_SMOKE_CACHE
    if _TINY_SMOKE_CACHE is not None:
        return _TINY_SMOKE_CACHE
    try:
        from dico.trainer import train
    except Exception as exc:  # pragma: no cover - import check above should catch this too.
        _TINY_SMOKE_CACHE = {"error": f"could not import trainer: {exc}"}
        return _TINY_SMOKE_CACHE

    try:
        work_dir = Path(tempfile.mkdtemp(prefix="covra_static_tiny_train_"))
        config = _tiny_covra_config(work_dir)
        metrics = train(config)
        output_dir = work_dir / "outputs" / "static_acceptance_tiny_covra"
        manifest_path = output_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        _TINY_SMOKE_CACHE = {
            "work_dir": str(work_dir),
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "metrics": metrics,
            "manifest": manifest,
        }
    except Exception as exc:
        _TINY_SMOKE_CACHE = {"error": f"tiny CPU training failed: {exc}"}
    return _TINY_SMOKE_CACHE


def check_cpu_tiny_training_smoke() -> CheckResult:
    result = _run_tiny_covra_smoke()
    if result.get("error"):
        return _fail("cpu_tiny_training_smoke", str(result["error"]))
    output_dir = Path(str(result["output_dir"]))
    required = [
        output_dir / "run_manifest.json",
        output_dir / "run_summary.md",
        output_dir / "rank_dict.json",
        output_dir / "init_summary.json",
        output_dir / "masked_lora_state.pt",
        output_dir / "eval_predictions.jsonl",
    ]
    missing = [str(path.name) for path in required if not path.exists()]
    if missing:
        return _fail("cpu_tiny_training_smoke", f"tiny CPU training missed artifact(s): {missing}")
    manifest = dict(result.get("manifest") or {})
    failures: list[str] = []
    if manifest.get("model", {}).get("name_or_path") != "tiny":
        failures.append("manifest model.name_or_path is not tiny")
    if int(manifest.get("optimizer_steps", 0)) != 1:
        failures.append(f"optimizer_steps={manifest.get('optimizer_steps')}, expected 1")
    if int(manifest.get("parameter_counts", {}).get("requires_grad", 0)) <= 0:
        failures.append("requires_grad parameter count is not positive")
    if manifest.get("checkpoint_artifacts", {}).get("adapter_checkpoint", {}).get("contains_base_model_weights") is not False:
        failures.append("adapter checkpoint artifact does not prove base weights are excluded")
    if failures:
        return _fail("cpu_tiny_training_smoke", "; ".join(failures))
    metrics = dict(result.get("metrics") or {})
    return _pass(
        "cpu_tiny_training_smoke",
        "tiny CPU path completed calibration, allocation, init, one train step, eval dry-run, checkpoint, and manifest",
        final_metric=metrics.get("final_metric"),
        requires_grad=manifest.get("parameter_counts", {}).get("requires_grad"),
        active_final=manifest.get("parameter_counts", {}).get("active_final"),
        train_tokens=manifest.get("timing", {}).get("train_tokens"),
    )


def check_cpu_tiny_manifest_validation() -> CheckResult:
    result = _run_tiny_covra_smoke()
    if result.get("error"):
        return _fail("cpu_tiny_manifest_validation", str(result["error"]))
    manifest_path = Path(str(result["manifest_path"]))
    if not manifest_path.exists():
        return _fail("cpu_tiny_manifest_validation", f"tiny manifest is missing: {manifest_path}")
    validation = _run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(Path(str(result["work_dir"])) / "manifest_validation.json"),
            "--markdown-output",
            str(Path(str(result["work_dir"])) / "manifest_validation.md"),
        ],
        timeout_sec=120,
    )
    if validation.returncode != 0:
        return _fail(
            "cpu_tiny_manifest_validation",
            f"validate_run_manifest failed for tiny manifest with return code {validation.returncode}",
            stdout_tail=validation.stdout[-2000:],
            stderr_tail=validation.stderr[-2000:],
            manifest_path=str(manifest_path),
        )
    return _pass(
        "cpu_tiny_manifest_validation",
        "tiny CPU run_manifest.json passes the formal run manifest validator",
        manifest_path=str(manifest_path),
        stdout_tail=validation.stdout[-2000:],
    )


def check_cpu_tiny_parameter_budget_audit() -> CheckResult:
    result = _run_tiny_covra_smoke()
    if result.get("error"):
        return _fail("cpu_tiny_parameter_budget_audit", str(result["error"]))
    manifest = dict(result.get("manifest") or {})
    budget = dict(manifest.get("budget") or {})
    counts = dict(manifest.get("parameter_counts") or {})
    module_budget = dict(manifest.get("module_budget") or {})
    target = int(budget.get("target_budget", -1))
    actual = int(budget.get("actual_budget", -1))
    error = int(budget.get("budget_error", 0))
    requires_grad = int(counts.get("requires_grad", -1))
    active_final = int(counts.get("active_final", -1))
    active_peak = int(counts.get("active_peak", -1))
    module_total = int(module_budget.get("total_final_params", -1))
    failures: list[str] = []
    if actual - target != error:
        failures.append(f"budget_error={error} but actual-target={actual - target}")
    if active_final != actual:
        failures.append(f"active_final={active_final} but actual_budget={actual}")
    if module_total != active_final:
        failures.append(f"module_budget.total_final_params={module_total} but active_final={active_final}")
    if active_peak < active_final:
        failures.append(f"active_peak={active_peak} is smaller than active_final={active_final}")
    if requires_grad <= 0:
        failures.append(f"requires_grad={requires_grad} is not positive")
    if failures:
        return _fail("cpu_tiny_parameter_budget_audit", "; ".join(failures))
    return _pass(
        "cpu_tiny_parameter_budget_audit",
        "tiny CPU manifest parameter budget and active/trainable counts are internally consistent",
        target_budget=target,
        actual_budget=actual,
        budget_error=error,
        requires_grad=requires_grad,
        active_final=active_final,
        active_peak=active_peak,
        module_total_final_params=module_total,
    )


def check_experiment_matrix_generation() -> CheckResult:
    with tempfile.TemporaryDirectory(prefix="covra_static_matrix_") as tmp:
        tmp_dir = Path(tmp)
        return _command_check(
            "experiment_matrix_generation",
            [
                "python",
                "scripts/experiment_matrix.py",
                "--json-output",
                str(tmp_dir / "experiment_matrix.json"),
                "--markdown-output",
                str(tmp_dir / "experiment_matrix.md"),
            ],
            "E00-E10 experiment command matrix generates successfully",
        )


def check_protocol_preflight() -> CheckResult:
    with tempfile.TemporaryDirectory(prefix="covra_static_preflight_") as tmp:
        tmp_dir = Path(tmp)
        return _command_check(
            "protocol_preflight",
            [
                "python",
                "scripts/protocol_preflight.py",
                "--json-output",
                str(tmp_dir / "protocol_preflight.json"),
                "--markdown-output",
                str(tmp_dir / "protocol_preflight.md"),
            ],
            "formal configs pass protocol preflight",
        )


def check_typecheck_tool() -> CheckResult:
    tool = shutil.which("mypy") or shutil.which("pyright")
    if tool is None:
        return _skip(
            "typecheck_tool",
            "mypy/pyright is not installed in this environment; typecheck not claimed",
            checked_tools=["mypy", "pyright"],
        )
    if Path(tool).name == "mypy":
        command = [tool, "src/dico"]
    else:
        command = [tool, "src/dico"]
    return _command_check("typecheck_tool", command, f"{Path(tool).name} typecheck completed")


def check_lint_tool() -> CheckResult:
    tool = shutil.which("ruff")
    if tool is None:
        return _skip("lint_tool", "ruff is not installed in this environment; lint not claimed", checked_tools=["ruff"])
    return _command_check("lint_tool", [tool, "check", "src/dico", "scripts", "tests"], "ruff lint completed")


CHECKS = (
    check_import_core_modules,
    check_syntax_compile,
    check_config_dry_run,
    check_cpu_tiny_training_smoke,
    check_cpu_tiny_manifest_validation,
    check_cpu_tiny_parameter_budget_audit,
    check_experiment_matrix_generation,
    check_protocol_preflight,
    check_typecheck_tool,
    check_lint_tool,
)


def run_acceptance() -> dict[str, Any]:
    checks = [check() for check in CHECKS]
    failed = sum(1 for check in checks if check.status == "FAIL")
    skipped = sum(1 for check in checks if check.status == "SKIP")
    status = "FAIL" if failed else ("PASS_WITH_SKIPS" if skipped else "PASS")
    return {
        "summary": {
            "status": status,
            "total_checks": len(checks),
            "failed_checks": failed,
            "skipped_checks": skipped,
        },
        "checks": [asdict(check) for check in checks],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Static Acceptance",
        "",
        f"- status: `{payload['summary']['status']}`",
        f"- failed_checks: `{payload['summary']['failed_checks']}`",
        f"- skipped_checks: `{payload['summary']['skipped_checks']}`",
        "",
        "| check | status | message |",
        "|---|---|---|",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['id']} | {check['status']} | {check['message']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU/local static acceptance gates before GPU experiments.")
    parser.add_argument("--json-output", default="reports/static_acceptance.json")
    parser.add_argument("--markdown-output", default="reports/static_acceptance.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_acceptance()
    json_path = Path(args.json_output)
    md_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    if payload["summary"]["status"] == "FAIL":
        for check in payload["checks"]:
            if check["status"] == "FAIL":
                print(f"{check['id']}: {check['message']}", file=sys.stderr)
        raise SystemExit(1)
    print(
        f"[static_acceptance] {payload['summary']['status']}; "
        f"failed={payload['summary']['failed_checks']}, skipped={payload['summary']['skipped_checks']}"
    )


if __name__ == "__main__":
    main()
