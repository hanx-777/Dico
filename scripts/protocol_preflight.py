#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.config import load_yaml, validate_known_config_fields


FINAL_COVRA_ALLOCATIONS = {"covra_full", "covra_independent", "covra_module_scalar"}
REFERENCE_COVRA_ALLOCATION = "covra_v05"
FINAL_COVRA_METHODS = {"dico_cd", "dico_cd_da"}
EXPECTED_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
REQUIRED_UNRESOLVED_FIELDS = {
    "model.revision",
    "model.tokenizer_revision",
    "preallocation.rho",
    "preallocation.response_agg_groups",
    "preallocation.r_min_multiplier",
}


@dataclass(frozen=True)
class CheckResult:
    id: str
    status: str
    message: str


def _pass(check_id: str, message: str) -> CheckResult:
    return CheckResult(check_id, "PASS", message)


def _fail(check_id: str, message: str) -> CheckResult:
    return CheckResult(check_id, "FAIL", message)


def _skip(check_id: str, message: str) -> CheckResult:
    return CheckResult(check_id, "SKIP", message)


def _is_final_covra_config(config: dict[str, Any]) -> bool:
    return (
        str(config.get("method")) in FINAL_COVRA_METHODS
        and str(config.get("preallocation", {}).get("allocation_method")) in FINAL_COVRA_ALLOCATIONS
    )


def _is_reference_covra_config(config: dict[str, Any]) -> bool:
    return (
        str(config.get("method")) in FINAL_COVRA_METHODS
        and str(config.get("preallocation", {}).get("allocation_method")) == REFERENCE_COVRA_ALLOCATION
    )


def _is_reference_covra_ablation(config: dict[str, Any]) -> bool:
    config_path = Path(str(config.get("_config_path", "")))
    return "ablations" in config_path.parts or bool(config.get("ablation"))


def check_strict_schema(config: dict[str, Any]) -> CheckResult:
    try:
        validate_known_config_fields(config)
    except ValueError as exc:
        return _fail("strict_schema", str(exc))
    return _pass("strict_schema", "config contains only explicitly known fields")


def check_single_gpu_global_batch_64(config: dict[str, Any]) -> CheckResult:
    training = dict(config.get("training", {}))
    batch_size = int(training.get("batch_size") or 0)
    grad_accum = int(training.get("gradient_accumulation_steps") or 0)
    global_batch = batch_size * grad_accum
    if global_batch != 64:
        return _fail(
            "single_gpu_global_batch_64",
            f"single-GPU protocol requires batch_size*gradient_accumulation_steps=64, got {global_batch}",
        )
    return _pass("single_gpu_global_batch_64", f"single-GPU global batch is {global_batch}")


def _configured_train_samples(config: dict[str, Any]) -> int | None:
    data = dict(config.get("data", {}))
    sources = data.get("train_sources")
    if isinstance(sources, list) and sources:
        total = 0
        for source in sources:
            if not isinstance(source, dict):
                return None
            limit = source.get("limit")
            if limit is None:
                return None
            total += int(limit)
        return total
    train_limit = data.get("train_limit")
    if train_limit is None:
        return None
    return int(train_limit)


def check_max_steps_match_train_volume(config: dict[str, Any]) -> CheckResult:
    name = str(config.get("experiment_name", ""))
    training = dict(config.get("training", {}))
    configured_steps = int(training.get("max_steps") or 0)
    if configured_steps == 1 and "pilot" in name:
        return _skip("max_steps_match_train_volume", "pilot config intentionally uses max_steps=1")
    train_samples = _configured_train_samples(config)
    if train_samples is None:
        return _skip("max_steps_match_train_volume", "train sample count is not statically configured")
    batch_size = int(training.get("batch_size") or 0)
    grad_accum = int(training.get("gradient_accumulation_steps") or 0)
    global_batch = batch_size * grad_accum
    if global_batch <= 0:
        return _fail("max_steps_match_train_volume", f"invalid global batch {global_batch}")
    expected_steps = (
        int(train_samples / global_batch)
        if _is_reference_covra_config(config)
        else int(math.ceil(train_samples / global_batch))
    )
    if configured_steps != expected_steps:
        return _fail(
            "max_steps_match_train_volume",
            (
                f"expected max_steps={expected_steps} from train_samples={train_samples} "
                f"and single-GPU global_batch={global_batch}, got {configured_steps}"
            ),
        )
    return _pass(
        "max_steps_match_train_volume",
        f"max_steps={configured_steps} matches train_samples={train_samples}/global_batch={global_batch}",
    )


def check_covra_candidate_budget(config: dict[str, Any]) -> CheckResult:
    reference_covra = _is_reference_covra_config(config)
    final_covra = _is_final_covra_config(config)
    if not reference_covra and not final_covra:
        return _skip("covra_candidate_budget", "not a CovRA allocation config")
    if reference_covra and _is_reference_covra_ablation(config):
        return _skip("covra_candidate_budget", "reference CovRA ablation intentionally changes candidate controls")
    rank = int(config.get("rank"))
    preallocation = dict(config.get("preallocation", {}))
    top_k_atoms = int(preallocation.get("top_k_atoms") or 0)
    sketch_dim = int(preallocation.get("sketch_dim") or 0)
    r_max = int(rank * float(preallocation.get("r_max_multiplier") or 0.0))
    failures: list[str] = []
    if r_max <= rank:
        failures.append(f"r_max={r_max} must be greater than r_ref/rank={rank}")
    if reference_covra:
        if top_k_atoms != 8:
            failures.append(f"reference CovRA requires top_k_atoms=8, got {top_k_atoms}")
        if sketch_dim != 16:
            failures.append(f"reference CovRA requires sketch_dim=16, got {sketch_dim}")
    else:
        if top_k_atoms < r_max:
            failures.append(f"top_k_atoms={top_k_atoms} must be >= r_max={r_max}")
        if sketch_dim < top_k_atoms:
            failures.append(f"sketch_dim={sketch_dim} must be >= top_k_atoms={top_k_atoms}")
    if failures:
        return _fail("covra_candidate_budget", "; ".join(failures))
    return _pass(
        "covra_candidate_budget",
        f"K={top_k_atoms}, sketch_dim={sketch_dim}, r_max={r_max}, rank={rank}",
    )


def check_covra_calibration_samples(config: dict[str, Any]) -> CheckResult:
    check_id = "covra_calibration_samples"
    reference_covra = _is_reference_covra_config(config)
    final_covra = _is_final_covra_config(config)
    if not reference_covra and not final_covra:
        return _skip(check_id, "not a CovRA allocation config")
    if reference_covra and _is_reference_covra_ablation(config):
        return _skip(check_id, "reference CovRA extension intentionally changes calibration sampling")
    calibration = dict(config.get("calibration", {}))
    enabled = bool(calibration.get("enabled", True))
    num_samples = int(calibration.get("num_samples") or 0)
    if not enabled:
        return _fail(check_id, "CovRA calibration.enabled must be true")
    expected_samples = 256 if reference_covra else 1024
    if num_samples != expected_samples:
        return _fail(
            check_id,
            f"CovRA protocol requires calibration.num_samples={expected_samples}, got {num_samples}",
        )
    return _pass(check_id, f"CovRA calibration uses {expected_samples} training-only samples")


def check_covra_gpu_path(config: dict[str, Any]) -> CheckResult:
    reference_covra = _is_reference_covra_config(config)
    final_covra = _is_final_covra_config(config)
    if not reference_covra and not final_covra:
        return _skip("covra_gpu_path", "not a CovRA allocation config")
    preallocation = dict(config.get("preallocation", {}))
    compute = str(preallocation.get("compute_device"))
    allocation = str(preallocation.get("allocation_device"))
    expected = ("auto", "cpu") if reference_covra else ("cuda", "cuda")
    if (compute, allocation) != expected:
        return _fail(
            "covra_gpu_path",
            f"CovRA protocol requires compute_device={expected[0]} and allocation_device={expected[1]}, got {compute}/{allocation}",
        )
    return _pass("covra_gpu_path", f"CovRA compute/allocation devices are {compute}/{allocation}")


def check_covra_module_scalar_template(config: dict[str, Any]) -> CheckResult:
    preallocation = dict(config.get("preallocation", {}))
    if str(preallocation.get("allocation_method")) != "covra_module_scalar":
        return _skip("covra_module_scalar_template", "not a CovRA-M config")
    rank = int(config.get("rank"))
    r_max = int(rank * float(preallocation.get("r_max_multiplier") or 0.0))
    template = preallocation.get("module_scalar_template")
    failures: list[str] = []
    if not isinstance(template, list):
        failures.append("preallocation.module_scalar_template must be an explicit list")
        values: list[float] = []
    else:
        values = [float(value) for value in template]
        if len(values) < r_max:
            failures.append(
                f"preallocation.module_scalar_template must provide at least r_max={r_max} entries, got {len(values)}"
            )
        if any(value < 0.0 for value in values[:r_max]):
            failures.append("preallocation.module_scalar_template must be non-negative")
        if any(values[idx] < values[idx + 1] for idx in range(min(len(values), r_max) - 1)):
            failures.append("preallocation.module_scalar_template must be non-increasing")
        if sum(values[:r_max]) <= 0.0:
            failures.append("preallocation.module_scalar_template must have positive mass")
    formula = str(preallocation.get("module_scalar_template_formula", ""))
    if formula != "w_j = 1 / j for j=1..r_max":
        failures.append(
            "preallocation.module_scalar_template_formula must be 'w_j = 1 / j for j=1..r_max'"
        )
    normalization = str(preallocation.get("module_scalar_template_normalization", ""))
    if normalization != "sum_to_module_energy":
        failures.append(
            "preallocation.module_scalar_template_normalization must be sum_to_module_energy"
        )
    if failures:
        return _fail("covra_module_scalar_template", "; ".join(failures))
    return _pass(
        "covra_module_scalar_template",
        f"CovRA-M template is explicit, non-increasing, length={len(values)}, normalization=sum_to_module_energy",
    )


def check_budget_policy(config: dict[str, Any]) -> CheckResult:
    budget = dict(config.get("budget", {}))
    failures: list[str] = []
    mode = str(budget.get("mode", ""))
    if mode != "equal_trainable_params":
        failures.append(f"budget.mode must be equal_trainable_params, got {mode!r}")
    target_ratio = float(budget.get("enforce_target_ratio", 0.0) or 0.0)
    if target_ratio != 1.0:
        failures.append(f"budget.enforce_target_ratio must be 1.0, got {target_ratio}")
    min_ratio = float(budget.get("enforce_min_ratio", 0.0) or 0.0)
    if min_ratio < 0.98:
        failures.append(f"budget.enforce_min_ratio must be >= 0.98, got {min_ratio}")
    warning_threshold = budget.get("warning_threshold")
    if warning_threshold is not None and float(warning_threshold) > 0.01:
        failures.append(f"budget.warning_threshold must be <= 0.01 when set, got {warning_threshold}")
    if failures:
        return _fail("budget_policy", "; ".join(failures))
    return _pass(
        "budget_policy",
        "strict equal-trainable-parameter budget policy is locked",
    )


def check_adapter_fp32_base_bf16(config: dict[str, Any]) -> CheckResult:
    model_dtype = str(config.get("model", {}).get("torch_dtype"))
    adapter_dtype = str(config.get("lora", {}).get("adapter_dtype"))
    expected_adapter_dtype = "bfloat16" if _is_reference_covra_config(config) else "float32"
    if model_dtype != "bfloat16" or adapter_dtype != expected_adapter_dtype:
        return _fail(
            "adapter_fp32_base_bf16",
            "expected model.torch_dtype=bfloat16 and "
            f"lora.adapter_dtype={expected_adapter_dtype}, got {model_dtype}/{adapter_dtype}",
        )
    return _pass(
        "adapter_fp32_base_bf16",
        f"frozen base BF16 and adapter {expected_adapter_dtype} are configured",
    )


def check_dropout_zero(config: dict[str, Any]) -> CheckResult:
    dropout = float(config.get("lora", {}).get("dropout", -1.0))
    expected = 0.05 if _is_reference_covra_config(config) else 0.0
    if dropout != expected:
        return _fail("dropout_zero", f"expected lora.dropout={expected}, got {dropout}")
    return _pass("dropout_zero", f"lora.dropout={dropout}")


def check_gradient_checkpointing_off(config: dict[str, Any]) -> CheckResult:
    enabled = bool(config.get("training", {}).get("gradient_checkpointing"))
    expected = _is_reference_covra_config(config)
    if enabled is not expected:
        return _fail(
            "gradient_checkpointing_off",
            f"gradient_checkpointing must be {str(expected).lower()} for this protocol",
        )
    return _pass("gradient_checkpointing_off", f"gradient checkpointing is {str(enabled).lower()}")


def check_target_modules_qkvo(config: dict[str, Any]) -> CheckResult:
    modules = list(config.get("lora", {}).get("target_modules", []))
    if modules != EXPECTED_TARGET_MODULES:
        return _fail("target_modules_qkvo", f"expected {EXPECTED_TARGET_MODULES}, got {modules}")
    return _pass("target_modules_qkvo", "target modules match q/k/v/o projection set")


def check_legacy_fields_isolated(config: dict[str, Any]) -> CheckResult:
    preallocation = dict(config.get("preallocation", {}))
    dico = dict(config.get("dico", {}))
    if _is_reference_covra_config(config):
        failures: list[str] = []
        if float(preallocation.get("beta", -1.0)) != 1.0:
            failures.append("reference CovRA requires preallocation.beta=1.0")
        if not isinstance(dico.get("taxonomy"), dict):
            failures.append("reference CovRA requires dico.taxonomy")
        procurement = dico.get("procurement")
        if not isinstance(procurement, dict) or float(procurement.get("beta", -1.0)) != 0.5:
            failures.append("reference CovRA requires dico.procurement.beta=0.5")
        if failures:
            return _fail("legacy_fields_isolated", "; ".join(failures))
        return _pass(
            "legacy_fields_isolated",
            "reference taxonomy/procurement fields are active and legacy namespace is not used",
        )
    legacy = dict(dico.get("legacy_covra_v05", {}))
    taxonomy = dict(legacy.get("taxonomy", {}))
    failures: list[str] = []
    for field in ("beta", "h", "delta", "permutation", "permutation_count"):
        if field in preallocation:
            failures.append(f"preallocation.{field} must not drive final CovRA")
    for field in ("taxonomy", "procurement"):
        if field in dico:
            failures.append(f"dico.{field} must live under dico.legacy_covra_v05")
    if taxonomy.get("enabled") is not False:
        failures.append("dico.legacy_covra_v05.taxonomy.enabled must be false by default")
    if failures:
        return _fail("legacy_fields_isolated", "; ".join(failures))
    return _pass("legacy_fields_isolated", "legacy taxonomy/procurement fields are isolated and disabled")


def check_unresolved_fields_marked(config: dict[str, Any]) -> CheckResult:
    unresolved = config.get("protocol", {}).get("unresolved_fields", [])
    rows = {str(row.get("field")): row for row in unresolved if isinstance(row, dict)}
    statuses = {str(row.get("status")) for row in rows.values()}
    bad_statuses = statuses - {"provisional", "unresolved"}
    required_fields = (
        {"model.revision", "model.tokenizer_revision"}
        if _is_reference_covra_config(config)
        else REQUIRED_UNRESOLVED_FIELDS
    )
    missing = required_fields - set(rows)
    if missing or bad_statuses:
        details = []
        if missing:
            details.append(f"missing unresolved/provisional fields: {sorted(missing)}")
        if bad_statuses:
            details.append(f"unexpected statuses: {sorted(bad_statuses)}")
        return _fail("unresolved_fields_marked", "; ".join(details))
    return _pass("unresolved_fields_marked", "provisional/unresolved protocol fields are explicit")


def check_evaluation_protocol(config: dict[str, Any]) -> CheckResult:
    evaluation = dict(config.get("evaluation", {}))
    mtbench = dict(evaluation.get("mtbench_local", {}))
    failures: list[str] = []
    humaneval_samples = int(evaluation.get("humaneval_num_samples_per_task", 1) or 1)
    if humaneval_samples != 1:
        failures.append(
            "evaluation.humaneval_num_samples_per_task must be 1 for greedy pass@1 main-table evaluation, "
            f"got {humaneval_samples}"
        )
    answer_extraction = str(evaluation.get("answer_extraction", "strict_then_flexible"))
    if answer_extraction != "strict_then_flexible":
        failures.append(
            "evaluation.answer_extraction must be strict_then_flexible for GSM8K greedy evaluation, "
            f"got {answer_extraction}"
        )
    required_mtbench = {
        "judge_model": mtbench.get("judge_model"),
        "judge_prompt_version": mtbench.get("judge_prompt_version"),
        "conversation_template": mtbench.get("conversation_template"),
        "temperature": mtbench.get("temperature"),
        "seed": mtbench.get("seed"),
        "swap_positions": mtbench.get("swap_positions"),
        "max_retries": mtbench.get("max_retries"),
    }
    missing = [
        f"evaluation.mtbench_local.{key}"
        for key, value in required_mtbench.items()
        if value is None or (isinstance(value, str) and value.strip() == "")
    ]
    if missing:
        failures.append(f"missing MTBench-local judge protocol field(s): {missing}")
    if required_mtbench.get("temperature") is not None and float(required_mtbench["temperature"]) != 0.0:
        failures.append(
            f"evaluation.mtbench_local.temperature must be 0.0 for deterministic local judge, got {required_mtbench['temperature']}"
        )
    if required_mtbench.get("swap_positions") is not None and bool(required_mtbench["swap_positions"]) is not True:
        failures.append("evaluation.mtbench_local.swap_positions must be true to record/mitigate position bias")
    if failures:
        return _fail("evaluation_protocol", "; ".join(failures))
    return _pass("evaluation_protocol", "greedy GSM8K/HumanEval and MTBench-local judge protocol fields are locked")


def check_aligned_baseline_semantics(config: dict[str, Any]) -> CheckResult:
    method = str(config.get("method"))
    lora = dict(config.get("lora", {}))
    training = dict(config.get("training", {}))
    failures: list[str] = []
    if _is_reference_covra_config(config):
        data = dict(config.get("data", {}))
        model = dict(config.get("model", {}))
        evaluation = dict(config.get("evaluation", {}))
        if bool(data.get("shuffle")):
            failures.append("reference CovRA requires data.shuffle=false")
        if int(data.get("dataset_seed", -1)) != 42:
            failures.append("reference CovRA requires data.dataset_seed=42")
        if str(training.get("sample_exposure_policy")) != "fixed_reference_order":
            failures.append("reference CovRA requires training.sample_exposure_policy=fixed_reference_order")
        if training.get("max_grad_norm") is not None:
            failures.append("reference CovRA disables gradient clipping with training.max_grad_norm=null")
        if int(training.get("auto_warmup_steps", -1)) != 0:
            failures.append("reference CovRA requires training.auto_warmup_steps=0")
        if model.get("attn_implementation") is not None:
            failures.append("reference CovRA requires model.attn_implementation=null")
        if int(evaluation.get("batch_size", 0)) != 1:
            failures.append("reference CovRA requires evaluation.batch_size=1")
        if not bool(evaluation.get("compute_loss")):
            failures.append("reference CovRA requires final evaluation.compute_loss=true")
        if not bool(dict(evaluation.get("mid_eval_loss_only", {})).get("enabled", False)):
            failures.append("reference CovRA keeps the reference mid-eval loss-only protocol enabled")
        if failures:
            return _fail("aligned_baseline_semantics", "; ".join(failures))
        return _pass("aligned_baseline_semantics", "reference CovRA semantics match the executed protocol")
    if method == "lora" and str(lora.get("scaling")) != "alpha_over_r":
        failures.append("method=lora must use lora.scaling=alpha_over_r; rsLoRA must use its own method/config")
    if method == "adalora":
        adalora = dict(config.get("adalora", {}))
        if float(training.get("learning_rate", 0.0)) != 5e-4:
            failures.append("AdaLoRA learning_rate must be 5e-4")
        for key in ("beta1", "beta2"):
            if float(adalora.get(key, -1.0)) != 0.85:
                failures.append(f"AdaLoRA {key} must be .85")
    if method in {"gora_public", "gora_bm"}:
        gora = dict(config.get("gora", {}))
        if str(gora.get("official_commit")) != "4037d4d6ba67ff88de87f90b943ff4e3a3649b67":
            failures.append("formal GoRA config must lock the approved official commit")
        if int(config.get("calibration", {}).get("num_samples", 0)) != int(gora.get("gradient_estimation_samples", -1)):
            failures.append("GoRA calibration.num_samples must equal gora.gradient_estimation_samples")
        if method == "gora_public" and bool(gora.get("strict_budget_repair")):
            failures.append("GoRA-public must not enable strict budget repair")
        if method == "gora_bm" and not bool(gora.get("strict_budget_repair")):
            failures.append("GoRA-BM must enable strict budget repair")
        if str(gora.get("gradient_collection")) != "official_weight_grad_hook":
            failures.append("formal GoRA must collect direct target-weight gradients with hooks")
        if str(gora.get("gradient_offload_device")) != "cpu":
            failures.append("formal GoRA gradient offload device must be cpu")
        if not bool(gora.get("clear_gradient_after_offload")):
            failures.append("formal GoRA must clear target gradients after offload")
    data = dict(config.get("data", {}))
    if not bool(data.get("shuffle")) or int(data.get("dataset_seed", -1)) != 42:
        failures.append("formal public training order requires data.shuffle=true and data.dataset_seed=42")
    if str(training.get("sample_exposure_policy")) != "repeat_from_fixed_order_to_max_steps":
        failures.append("training.sample_exposure_policy must record the fixed-order repeat behavior")
    if str(training.get("optimizer_backend")) != "adamw":
        failures.append("formal baseline optimizer_backend must be adamw")
    model = dict(config.get("model", {}))
    if str(model.get("attn_implementation")) != "sdpa":
        failures.append("formal model.attn_implementation must be sdpa")
    if bool(config.get("runtime", {}).get("require_flash_attention_2")):
        failures.append("formal SDPA runtime must not require FlashAttention2")
    evaluation = dict(config.get("evaluation", {}))
    if int(evaluation.get("batch_size", 0)) != 4:
        failures.append("formal final evaluation must use evaluation.batch_size=4")
    if bool(dict(evaluation.get("mid_eval_loss_only", {})).get("enabled", False)):
        failures.append("formal protocol disables training-time GSM8K loss evaluation")
    if bool(evaluation.get("compute_loss", True)):
        failures.append("formal GSM8K protocol evaluates final greedy accuracy only; evaluation.compute_loss must be false")
    if failures:
        return _fail("aligned_baseline_semantics", "; ".join(failures))
    return _pass("aligned_baseline_semantics", "method-specific baseline semantics are aligned")


CHECKS: tuple[Callable[[dict[str, Any]], CheckResult], ...] = (
    check_strict_schema,
    check_single_gpu_global_batch_64,
    check_max_steps_match_train_volume,
    check_covra_candidate_budget,
    check_covra_calibration_samples,
    check_covra_gpu_path,
    check_covra_module_scalar_template,
    check_budget_policy,
    check_adapter_fp32_base_bf16,
    check_dropout_zero,
    check_gradient_checkpointing_off,
    check_target_modules_qkvo,
    check_legacy_fields_isolated,
    check_unresolved_fields_marked,
    check_evaluation_protocol,
    check_aligned_baseline_semantics,
)


def _default_config_paths() -> list[Path]:
    return sorted((ROOT / "configs" / "dico").glob("*.yaml")) + sorted(
        (ROOT / "configs" / "ablations").glob("*.yaml")
    )


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run_preflight(paths: list[Path]) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    total_checks = 0
    failed_checks = 0
    skipped_checks = 0
    for path in paths:
        config = load_yaml(path)
        checks = [check(config) for check in CHECKS]
        total_checks += len(checks)
        failed_checks += sum(1 for check in checks if check.status == "FAIL")
        skipped_checks += sum(1 for check in checks if check.status == "SKIP")
        configs.append(
            {
                "path": _relative_or_absolute(path),
                "status": "FAIL" if any(check.status == "FAIL" for check in checks) else "PASS",
                "checks": [asdict(check) for check in checks],
            }
        )
    return {
        "summary": {
            "status": "FAIL" if failed_checks else "PASS",
            "checked_configs": len(paths),
            "total_checks": total_checks,
            "failed_checks": failed_checks,
            "skipped_checks": skipped_checks,
        },
        "configs": configs,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Protocol Preflight",
        "",
        f"- status: `{payload['summary']['status']}`",
        f"- checked_configs: `{payload['summary']['checked_configs']}`",
        f"- failed_checks: `{payload['summary']['failed_checks']}`",
        "",
        "| config | status | failed_checks |",
        "|---|---|---|",
    ]
    for row in payload["configs"]:
        failed = [check for check in row["checks"] if check["status"] == "FAIL"]
        failed_text = "<br>".join(f"`{check['id']}`: {check['message']}" for check in failed) if failed else "-"
        lines.append(f"| {row['path']} | {row['status']} | {failed_text} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight-check CovRA/GoRA-aligned experiment configs.")
    parser.add_argument("--config", action="append", default=[], help="Config path to check. Defaults to formal configs.")
    parser.add_argument("--json-output", default="reports/protocol_preflight.json")
    parser.add_argument("--markdown-output", default="reports/protocol_preflight.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in args.config] if args.config else _default_config_paths()
    payload = run_preflight(paths)

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.markdown_output:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    if payload["summary"]["status"] == "FAIL":
        for config in payload["configs"]:
            for check in config["checks"]:
                if check["status"] == "FAIL":
                    print(f"{config['path']}: {check['id']}: {check['message']}", file=sys.stderr)
        raise SystemExit(1)

    print(f"[protocol_preflight] checked {payload['summary']['checked_configs']} config(s): PASS")


if __name__ == "__main__":
    main()
