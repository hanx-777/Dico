#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch

from dico.baselines import baseline_status_matrix
from dico.config import load_yaml
from protocol_preflight import _default_config_paths, run_preflight


@dataclass(frozen=True)
class ReadinessCheck:
    id: str
    status: str
    message: str
    details: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _status_from_failed(failed: bool) -> str:
    return "FAIL" if failed else "PASS"


def check_protocol_preflight() -> ReadinessCheck:
    payload = run_preflight(_default_config_paths())
    failed = payload["summary"]["failed_checks"] > 0
    return ReadinessCheck(
        id="protocol_preflight",
        status=_status_from_failed(failed),
        message=(
            "formal configs passed protocol preflight"
            if not failed
            else "formal configs failed protocol preflight"
        ),
        details=payload["summary"],
    )


def _file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        record["sha256"] = _sha256(path)
        record["size_bytes"] = path.stat().st_size
        record["line_count"] = _line_count(path)
    else:
        record["sha256"] = None
        record["size_bytes"] = None
        record["line_count"] = None
    return record


def check_data_files() -> ReadinessCheck:
    config = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    train_path = ROOT / str(config["data"]["train_path"])
    eval_path = ROOT / str(config["data"]["eval_path"])
    train = _file_record(train_path)
    eval_record = _file_record(eval_path)
    failed = not train["exists"] or not eval_record["exists"]
    return ReadinessCheck(
        id="data_files",
        status=_status_from_failed(failed),
        message="training/evaluation data files are present and hashed" if not failed else "required data file is missing",
        details={"train": train, "eval": eval_record},
    )


def check_model_reference(model_path_override: str | None = None) -> ReadinessCheck:
    config = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    name_or_path = str(model_path_override or config["model"]["name_or_path"])
    candidate = Path(name_or_path).expanduser()
    if candidate.exists():
        config_json = candidate / "config.json"
        tokenizer_files = [
            candidate / "tokenizer.json",
            candidate / "tokenizer.model",
            candidate / "tokenizer_config.json",
        ]
        missing: list[str] = []
        if not config_json.exists():
            missing.append("config.json")
        if not any(path.exists() for path in tokenizer_files):
            missing.append("tokenizer file")
        failed = bool(missing)
        return ReadinessCheck(
            id="model_reference",
            status=_status_from_failed(failed),
            message=(
                f"local model path is present: {candidate}"
                if not failed
                else f"local model path {candidate} is incomplete: missing {', '.join(missing)}"
            ),
            details={
                "name_or_path": name_or_path,
                "source": "override" if model_path_override else "config",
                "kind": "local_path",
                "exists": True,
                "config_json": str(config_json),
                "config_json_exists": config_json.exists(),
                "tokenizer_file_exists": any(path.exists() for path in tokenizer_files),
                "tokenizer_candidates": [str(path) for path in tokenizer_files],
            },
        )
    if "/" in name_or_path and not name_or_path.startswith("."):
        return ReadinessCheck(
            id="model_reference",
            status="PASS",
            message=(
                f"model reference {name_or_path} looks like a HuggingFace id; "
                "download/access must still be confirmed by E00"
            ),
            details={
                "name_or_path": name_or_path,
                "source": "override" if model_path_override else "config",
                "kind": "hf_id_or_remote_reference",
                "exists": False,
                "note": "No local model files were checked for this non-local reference.",
            },
        )
    return ReadinessCheck(
        id="model_reference",
        status="FAIL",
        message=f"model path/reference is neither an existing path nor a clear HF id: {name_or_path}",
        details={
            "name_or_path": name_or_path,
            "source": "override" if model_path_override else "config",
            "kind": "unknown",
            "exists": False,
        },
    )


DEFAULT_REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
]
OPTIONAL_PACKAGES = ["vllm", "flash-attn"]


def _package_record(package: str) -> dict[str, Any]:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "installed": version is not None,
        "version": version,
    }


def check_dependency_versions(
    *,
    strict: bool = False,
    extra_required: list[str] | None = None,
) -> ReadinessCheck:
    required = list(DEFAULT_REQUIRED_PACKAGES)
    for package in extra_required or []:
        if package not in required:
            required.append(package)
    package_names = required + [package for package in OPTIONAL_PACKAGES if package not in required]
    packages = {package: _package_record(package) for package in package_names}
    missing_required = [package for package in required if not packages[package]["installed"]]
    status = "FAIL" if strict and missing_required else ("WARN" if missing_required else "PASS")
    message = (
        "required runtime packages are installed"
        if not missing_required
        else (
            f"missing required runtime packages: {missing_required}"
            if strict
            else f"missing runtime packages recorded as warning for local dry-run: {missing_required}"
        )
    )
    return ReadinessCheck(
        id="dependency_versions",
        status=status,
        message=message,
        details={
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
            "required_packages": required,
            "optional_packages": OPTIONAL_PACKAGES,
            "missing_required": missing_required,
            "strict": bool(strict),
            "packages": packages,
        },
    )


def check_output_dir(output_dir_override: str | None = None, min_free_gb: float = 0.0) -> ReadinessCheck:
    config = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    output_dir_raw = output_dir_override or config.get("project", {}).get("output_dir", "outputs/dico_v03")
    output_dir = Path(str(output_dir_raw)).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    details: dict[str, Any] = {
        "path": str(output_dir),
        "source": "override" if output_dir_override else "config",
        "exists": output_dir.exists(),
        "writable": False,
        "free_bytes": None,
        "free_gb": None,
        "min_free_gb": float(min_free_gb),
    }
    if output_dir.exists() and not output_dir.is_dir():
        return ReadinessCheck(
            id="output_dir",
            status="FAIL",
            message=f"output path exists but is not a directory: {output_dir}",
            details=details,
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".e00_readiness_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        usage = shutil.disk_usage(output_dir)
        details.update(
            {
                "exists": output_dir.exists(),
                "writable": True,
                "free_bytes": int(usage.free),
                "free_gb": float(usage.free / (1024**3)),
            }
        )
    except OSError as exc:
        return ReadinessCheck(
            id="output_dir",
            status="FAIL",
            message=f"output directory is not writable: {output_dir}: {exc}",
            details=details,
        )
    if float(details["free_gb"]) < float(min_free_gb):
        return ReadinessCheck(
            id="output_dir",
            status="FAIL",
            message=f"output directory free space {details['free_gb']:.2f} GB is below required {min_free_gb:.2f} GB",
            details=details,
        )
    return ReadinessCheck(
        id="output_dir",
        status="PASS",
        message=f"output directory is writable with {details['free_gb']:.2f} GB free",
        details=details,
    )


def check_baseline_statuses() -> ReadinessCheck:
    rows = baseline_status_matrix()
    by_method = {str(row["method"]): row for row in rows}
    required_ready = ["uniform_lora", "adalora", "gora_public", "gora_bm", "covra", "covra_independent", "covra_module_scalar"]
    missing_or_not_ready = [
        method
        for method in required_ready
        if by_method.get(method, {}).get("status") not in {"IMPLEMENTED_NOT_GPU_RUN", "IMPLEMENTED_AND_VERIFIED"}
    ]
    not_ready_or_blocked = [
        str(row["method"])
        for row in rows
        if row["status"] in {"NOT_IMPLEMENTED", "BLOCKED_BY_UNRESOLVED_PROTOCOL"}
    ]
    failed = bool(missing_or_not_ready)
    return ReadinessCheck(
        id="baseline_statuses",
        status=_status_from_failed(failed),
        message=(
            "required local E00/E01 baselines are implemented-not-GPU-run"
            if not failed
            else f"required local baselines are not ready: {missing_or_not_ready}"
        ),
        details={
            "required_ready": required_ready,
            "missing_or_not_ready": missing_or_not_ready,
            "not_ready_or_blocked": not_ready_or_blocked,
        },
    )


def check_launcher(script_name: str, check_id: str) -> ReadinessCheck:
    path = ROOT / "scripts" / script_name
    failed = not path.exists()
    return ReadinessCheck(
        id=check_id,
        status=_status_from_failed(failed),
        message=f"{script_name} exists" if not failed else f"{script_name} is missing",
        details={"path": str(path), "exists": path.exists()},
    )


def check_root_launcher(script_name: str, check_id: str) -> ReadinessCheck:
    path = ROOT / script_name
    failed = not path.exists()
    return ReadinessCheck(
        id=check_id,
        status=_status_from_failed(failed),
        message=f"{script_name} exists" if not failed else f"{script_name} is missing",
        details={"path": str(path), "exists": path.exists()},
    )


def _extract_shell_default_int(text: str, name: str) -> int | None:
    match = re.search(rf"\b{name}=\$\{{{name}:-([0-9]+)\}}", text)
    return int(match.group(1)) if match else None


def check_ddp_fallback_protocol() -> ReadinessCheck:
    script_path = ROOT / "scripts" / "run_ddp.sh"
    accelerate_config_path = ROOT / "configs" / "accelerate_3gpu.yaml"
    details: dict[str, Any] = {
        "script_path": str(script_path),
        "script_exists": script_path.exists(),
        "bash_syntax_ok": False,
        "accelerate_config_path": str(accelerate_config_path),
        "accelerate_config_exists": accelerate_config_path.exists(),
        "default_num_gpus": None,
        "default_per_gpu_batch_size": None,
        "default_gradient_accumulation_steps": None,
        "accelerate_num_processes": None,
        "effective_global_batch": None,
        "stderr_excerpt": [],
    }
    failures: list[str] = []
    if not script_path.exists():
        failures.append("scripts/run_ddp.sh is missing")
    else:
        syntax = subprocess.run(
            ["bash", "-n", str(script_path)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        details["bash_syntax_ok"] = syntax.returncode == 0
        details["stderr_excerpt"] = syntax.stderr.splitlines()[:10]
        if syntax.returncode != 0:
            failures.append("scripts/run_ddp.sh has invalid bash syntax")
        text = script_path.read_text(encoding="utf-8")
        details["default_num_gpus"] = _extract_shell_default_int(text, "NUM_GPUS")
        per_gpu_match = re.search(r"training\.per_gpu_batch_size=([0-9]+)", text)
        grad_accum_match = re.search(r"training\.gradient_accumulation_steps=([0-9]+)", text)
        details["default_per_gpu_batch_size"] = int(per_gpu_match.group(1)) if per_gpu_match else None
        details["default_gradient_accumulation_steps"] = int(grad_accum_match.group(1)) if grad_accum_match else None
        if "configs/accelerate_3gpu.yaml" not in text:
            failures.append("run_ddp.sh must launch with configs/accelerate_3gpu.yaml")
    if accelerate_config_path.exists():
        accelerate_config = load_yaml(accelerate_config_path)
        details["accelerate_num_processes"] = int(accelerate_config.get("num_processes") or 0)
    else:
        failures.append("configs/accelerate_3gpu.yaml is missing")
    if (
        details["default_num_gpus"] is not None
        and details["default_per_gpu_batch_size"] is not None
        and details["default_gradient_accumulation_steps"] is not None
    ):
        details["effective_global_batch"] = int(details["default_num_gpus"]) * int(
            details["default_per_gpu_batch_size"]
        ) * int(details["default_gradient_accumulation_steps"])
    expected = {
        "default_num_gpus": 3,
        "default_per_gpu_batch_size": 3,
        "default_gradient_accumulation_steps": 7,
        "accelerate_num_processes": 3,
        "effective_global_batch": 63,
    }
    for key, value in expected.items():
        if details.get(key) != value:
            failures.append(f"{key} expected {value}, got {details.get(key)}")
    return ReadinessCheck(
        id="ddp_fallback_protocol",
        status="FAIL" if failures else "PASS",
        message=(
            "run_ddp.sh fallback protocol is 3 GPUs × per-device batch 3 × grad accum 7 = global batch 63"
            if not failures
            else "; ".join(failures)
        ),
        details=details,
    )


SHELL_WRAPPER_ENTRYPOINTS = [
    "scripts/run_lora_r8.sh",
    "scripts/run_gora_bw.sh",
    "scripts/run_dico_cd.sh",
    "scripts/run_dico_cd_da.sh",
    "scripts/run_mixed_math_code.sh",
    "scripts/run_ablation_covra_independent.sh",
    "scripts/run_ablation_covra_module_scalar.sh",
    "scripts/run_ablation_no_sign_split.sh",
    "scripts/run_ablation_no_type_scaling.sh",
    "scripts/run_ablation_no_log_compression.sh",
    "scripts/run_ablation_proportional_rounding.sh",
    "scripts/run_ablation_global_only.sh",
    "scripts/run_ablation_grouped_only.sh",
    "scripts/run_ablation_random_init.sh",
    "scripts/run_ablation_uniform_rank_covra_init.sh",
    "scripts/run_ablation_covra_rank_random_init.sh",
    "scripts/run_ablation_taxonomy.sh",
    "scripts/run_ablation_coverage.sh",
]


def _wrapper_references(text: str) -> list[str]:
    refs = set(re.findall(r"\b(?:configs|scripts)/[A-Za-z0-9_./-]+", text))
    return sorted(ref for ref in refs if not ref.endswith("/"))


def check_shell_wrapper_entrypoints() -> ReadinessCheck:
    records: list[dict[str, Any]] = []
    failed_wrappers: list[str] = []
    for wrapper in SHELL_WRAPPER_ENTRYPOINTS:
        path = ROOT / wrapper
        record: dict[str, Any] = {
            "path": wrapper,
            "exists": path.exists(),
            "bash_syntax_ok": False,
            "references": [],
            "missing_references": [],
            "stderr_excerpt": [],
        }
        if path.exists():
            result = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            record["bash_syntax_ok"] = result.returncode == 0
            record["stderr_excerpt"] = result.stderr.splitlines()[:10]
            references = _wrapper_references(path.read_text(encoding="utf-8"))
            record["references"] = references
            record["missing_references"] = [ref for ref in references if not (ROOT / ref).exists()]
        if (
            not record["exists"]
            or not record["bash_syntax_ok"]
            or record["missing_references"]
        ):
            failed_wrappers.append(wrapper)
        records.append(record)
    return ReadinessCheck(
        id="shell_wrapper_entrypoints",
        status="FAIL" if failed_wrappers else "PASS",
        message=(
            f"{len(SHELL_WRAPPER_ENTRYPOINTS)} shell wrapper entrypoints exist, parse, and reference existing files"
            if not failed_wrappers
            else f"shell wrapper entrypoint failures: {failed_wrappers}"
        ),
        details={
            "checked_wrappers": SHELL_WRAPPER_ENTRYPOINTS,
            "failed_wrappers": failed_wrappers,
            "records": records,
        },
    )


def check_platform_launcher_dry_run(
    *,
    model_path: str | None = None,
    output_dir: str | None = None,
) -> ReadinessCheck:
    model_ref = model_path or str(load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")["model"]["name_or_path"])
    output_ref = output_dir or "outputs/e00_readiness_platform_dry_run"
    command = [
        sys.executable,
        "scripts/platform_train.py",
        "--dry-run",
        "--skip-model-check",
        "--model-path",
        model_ref,
        "--output-dir",
        output_ref,
        "--num-gpus",
        "3",
        "--seeds",
        "42,43,44",
        "--batch-size",
        "4",
        "--grad-accum",
        "16",
        "--calibration-batch-size",
        "4",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout = result.stdout
    experiment_commands = re.findall(r"(?:^|\s)(?:python\s+)?scripts/run_experiment\.py\b", stdout)
    failed_reasons: list[str] = []
    if result.returncode != 0:
        failed_reasons.append(f"platform_train dry-run exited {result.returncode}: {result.stderr.strip()}")
    if "effective_batch=64" not in stdout:
        failed_reasons.append("dry-run output did not report effective_batch=64")
    if "num_gpus=3 (parallel single-GPU workers)" not in stdout:
        failed_reasons.append("dry-run output did not report 3 parallel single-GPU workers")
    if len(experiment_commands) != 12:
        failed_reasons.append(f"expected 12 run_experiment commands, got {len(experiment_commands)}")
    for seed in ("seed=42", "seed=43", "seed=44"):
        if seed not in stdout:
            failed_reasons.append(f"missing {seed} in dry-run output")
    return ReadinessCheck(
        id="platform_launcher_dry_run",
        status="FAIL" if failed_reasons else "PASS",
        message=(
            "platform_train dry-run generated 4 configs × 3 seeds with global batch 64"
            if not failed_reasons
            else "; ".join(failed_reasons)
        ),
        details={
            "command": command,
            "returncode": result.returncode,
            "command_count": len(experiment_commands),
            "effective_batch": 64 if "effective_batch=64" in stdout else None,
            "num_gpus": 3 if "num_gpus=3 (parallel single-GPU workers)" in stdout else None,
            "stdout_excerpt": stdout.splitlines()[:20],
            "stderr_excerpt": result.stderr.splitlines()[:20],
        },
    )


def check_gpu_count(required: int) -> ReadinessCheck:
    available = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    failed = available < int(required)
    return ReadinessCheck(
        id="gpu_count",
        status=_status_from_failed(failed),
        message=(
            f"visible CUDA GPU count {available} satisfies required {required}"
            if not failed
            else f"E00 requires at least {required} visible CUDA GPU(s), got {available}"
        ),
        details={
            "cuda_available": bool(torch.cuda.is_available()),
            "visible_cuda_gpu_count": available,
            "required_gpu_count": int(required),
            "note": (
                "Use --require-gpu-count 3 on the A800 server as the hard gate. "
                "Local CPU checks may use --require-gpu-count 0."
            ),
        },
    )


def run_readiness(
    required_gpu_count: int,
    model_path: str | None = None,
    *,
    require_runtime_deps: bool = False,
    extra_required_packages: list[str] | None = None,
    output_dir: str | None = None,
    min_free_gb: float = 0.0,
) -> dict[str, Any]:
    checks = [
        check_protocol_preflight(),
        check_data_files(),
        check_model_reference(model_path),
        check_dependency_versions(
            strict=require_runtime_deps,
            extra_required=extra_required_packages,
        ),
        check_output_dir(output_dir, min_free_gb),
        check_baseline_statuses(),
        check_launcher("platform_train.py", "platform_launcher"),
        check_root_launcher("launch_covra.py", "single_file_platform_launcher"),
        check_platform_launcher_dry_run(model_path=model_path, output_dir=output_dir),
        check_shell_wrapper_entrypoints(),
        check_launcher("run_ddp.sh", "ddp_fallback_launcher"),
        check_ddp_fallback_protocol(),
        check_gpu_count(required_gpu_count),
    ]
    failed_checks = sum(1 for check in checks if check.status == "FAIL")
    return {
        "summary": {
            "status": "FAIL" if failed_checks else "READY_DRY_RUN",
            "failed_checks": failed_checks,
            "total_checks": len(checks),
            "requires_real_gpu_execution": True,
        },
        "checks": [asdict(check) for check in checks],
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# E00 Readiness",
        "",
        f"- status: `{payload['summary']['status']}`",
        f"- failed_checks: `{payload['summary']['failed_checks']}`",
        f"- requires_real_gpu_execution: `{payload['summary']['requires_real_gpu_execution']}`",
        "",
        "| check | status | message |",
        "|---|---|---|",
    ]
    for check in payload["checks"]:
        lines.append(f"| {check['id']} | {check['status']} | {check['message']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an E00 readiness report without running GPU training.")
    parser.add_argument("--json-output", default="reports/e00_readiness.json")
    parser.add_argument("--markdown-output", default="reports/e00_readiness.md")
    parser.add_argument("--require-gpu-count", type=int, default=0)
    parser.add_argument("--model-path", help="Optional local model path or HF id to check instead of the config default.")
    parser.add_argument("--require-runtime-deps", action="store_true", help="Treat missing required Python packages as FAIL instead of WARN.")
    parser.add_argument("--require-package", action="append", default=[], help="Additional Python distribution name that must be installed when --require-runtime-deps is used.")
    parser.add_argument("--output-dir", help="Output directory to check for writability. Defaults to the main config project.output_dir.")
    parser.add_argument("--min-free-gb", type=float, default=0.0, help="Minimum free disk space required in the output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_readiness(
        int(args.require_gpu_count),
        model_path=args.model_path,
        require_runtime_deps=bool(args.require_runtime_deps),
        extra_required_packages=list(args.require_package),
        output_dir=args.output_dir,
        min_free_gb=float(args.min_free_gb),
    )

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.markdown_output:
        md_path = Path(args.markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_markdown(payload), encoding="utf-8")

    if payload["summary"]["status"] == "FAIL":
        for check in payload["checks"]:
            if check["status"] == "FAIL":
                print(f"{check['id']}: {check['message']}", file=sys.stderr)
        raise SystemExit(1)

    print("[e00_readiness] READY_DRY_RUN; real GPU execution still not performed")


if __name__ == "__main__":
    main()
