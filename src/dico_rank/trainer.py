from __future__ import annotations

import gc
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from dico_rank.config import save_yaml
from dico_rank.data import SFTCollator, batch_iter, limit_records, load_raw_datasets, tokenize_records
from dico_rank.dynamic_allocation import DynamicRankAllocator
from dico_rank.evaluator import evaluate_gsm8k_accuracy, evaluate_loss
from dico_rank.logging_utils import append_rank_history, init_rank_history, log_eval, log_train
from dico_rank.lora_masked import (
    apply_rank_masks_to_grads,
    inject_masked_lora,
    restore_inactive_parameters,
    trainable_parameter_count,
)
from dico_rank.model_loader import (
    collect_module_dims,
    find_target_linear_modules,
    load_tokenizer_and_model,
    model_device,
    model_input_device,
    select_torch_dtype,
)
from dico_rank.preallocation import (
    MODULE_PROXY_LIMITATION,
    DiCoPreAllocator,
    build_preallocation_cache_context,
    load_preallocation,
)
from dico_rank.rank_budget import BudgetManager, get_uniform_budget
from dico_rank.rank_budget import compute_total_lora_params, module_rank_cost
from dico_rank.utils import ensure_dir, set_seed, write_json


LOGGER = logging.getLogger(__name__)
PREALLOC_METHODS = {"dico_pre", "dico_predynamic"}


def _budget_with_policy_fields(
    budget: dict[str, Any],
    method: str,
    preallocation_eta: float | None,
    generic_repair_applied: bool,
) -> dict[str, Any]:
    target_budget = int(budget.get("target_budget_paramcount", budget.get("target_budget")) or 0)
    actual_budget = int(budget.get("actual_budget_paramcount", budget.get("actual_budget")) or 0)
    budget_ratio = float(
        budget.get(
            "budget_ratio_paramcount",
            budget.get(
                "budget_ratio",
                float(actual_budget / target_budget) if target_budget else (0.0 if actual_budget == 0 else 1.0),
            ),
        )
    )
    payload = {
        **budget,
        "target_budget_paramcount": target_budget,
        "actual_budget_paramcount": actual_budget,
        "budget_ratio_paramcount": budget_ratio,
        "target_budget": target_budget,
        "actual_budget": actual_budget,
        "budget_ratio": budget_ratio,
        "preallocation_eta": preallocation_eta,
        "generic_repair_applied": bool(generic_repair_applied),
    }
    if method in PREALLOC_METHODS:
        eta = float(preallocation_eta if preallocation_eta is not None else 0.98)
        eta_reached = budget_ratio >= eta
        interval_pass = eta_reached and budget_ratio <= 1.0 and not bool(budget.get("over_budget", False))
        payload.update(
            {
                "budget_eta_reached": eta_reached,
                "budget_interval_pass": interval_pass,
            }
        )
    else:
        payload.update(
            {
                "budget_eta_reached": None,
                "budget_interval_pass": not bool(budget.get("over_budget", False))
                and not bool(budget.get("warning")),
            }
        )
    return payload


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_calibration_batches(
    train_records: list[dict[str, Any]],
    collator: SFTCollator,
    input_device: torch.device,
    calibration_cfg: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    calibration_limit = int(calibration_cfg.get("num_samples", min(8, len(train_records))))
    calibration_batch_size = max(1, int(calibration_cfg.get("batch_size", 1)))
    if calibration_limit >= len(train_records):
        selected = list(train_records)
    elif bool(calibration_cfg.get("shuffle", False)):
        rng = random.Random(int(calibration_cfg.get("seed", 42)))
        indices = rng.sample(range(len(train_records)), calibration_limit)
        selected = [train_records[index] for index in indices]
    else:
        selected = train_records[:calibration_limit]
    return [
        _move_batch(collator(selected[start : start + calibration_batch_size]), input_device)
        for start in range(0, len(selected), calibration_batch_size)
    ]


def uniform_allocation(rank: int, module_names: list[str]) -> dict[str, int]:
    return {name: int(rank) for name in module_names}


def _downscale_lora_allocation_to_ratio(
    allocation: dict[str, int],
    module_dims: dict[str, dict[str, int]],
    target_budget: int,
    target_ratio: float,
    min_ratio: float = 0.97,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Lower active ranks until the paramcount ratio falls into [min_ratio, target_ratio].

    The tie-break favors high-cost modules first, then stable module names. This
    keeps LoRA eta baselines budget-fair without changing LoRA weights or masks.
    """

    output = {name: int(rank) for name, rank in allocation.items()}
    lower = int(math.ceil(float(min_ratio) * int(target_budget)))
    upper = int(math.floor(float(target_ratio) * int(target_budget)))
    costs = {name: module_rank_cost(module_dims[name]) for name in output}
    details: list[dict[str, Any]] = []

    def total() -> int:
        return compute_total_lora_params(output, module_dims)

    while total() > upper:
        actual = total()
        candidates = [
            name
            for name, rank in output.items()
            if int(rank) > 0 and actual - costs[name] >= lower
        ]
        if not candidates:
            break
        name = max(candidates, key=lambda item: (costs[item], item))
        before = output[name]
        output[name] = before - 1
        after_budget = total()
        details.append(
            {
                "module_name": name,
                "rank_before": before,
                "rank_after": output[name],
                "rank_cost": costs[name],
                "actual_budget_after": after_budget,
                "budget_ratio_after": float(after_budget / target_budget) if target_budget else 0.0,
            }
        )

    actual = total()
    return output, {
        "lora_baseline_downscaled": bool(details),
        "lora_baseline_target_ratio": float(target_ratio),
        "lora_baseline_min_ratio": float(min_ratio),
        "lora_baseline_actual_ratio": float(actual / target_budget) if target_budget else 0.0,
        "lora_downscale_details": details,
        "lora_downscale_interval_pass": lower <= actual <= upper,
    }


def _evidence_relaxation_summary(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata:
        return None
    module_logs = metadata.get("module_logs") or []
    selected_total = metadata.get("selected_evidence_count_total")
    final_total = metadata.get("final_total_rank")
    beyond_total = metadata.get("rank_beyond_selected_evidence_total")
    modules_with_beyond = metadata.get("modules_with_rank_beyond_selected_evidence") or []
    if selected_total is None or final_total is None or beyond_total is None:
        selected_total = 0
        final_total = 0
        beyond_total = 0
        modules_with_beyond = []
        for row in module_logs:
            selected = int(row.get("selected_evidence_count", row.get("selected_atom_count", 0)) or 0)
            final_rank = int(row.get("final_rank", 0) or 0)
            beyond = int(row.get("rank_beyond_selected_evidence", max(0, final_rank - selected)) or 0)
            selected_total += selected
            final_total += final_rank
            beyond_total += beyond
            if beyond > 0:
                modules_with_beyond.append(row.get("module_name"))
    final_total_int = int(final_total or 0)
    beyond_total_int = int(beyond_total or 0)
    return {
        "selected_evidence_total": int(selected_total or 0),
        "final_rank_total": final_total_int,
        "rank_beyond_evidence_total": beyond_total_int,
        "rank_beyond_evidence_ratio": float(beyond_total_int / final_total_int) if final_total_int else 0.0,
        "modules_with_beyond": len(modules_with_beyond),
        "modules_total": len(module_logs),
    }


def _preallocation_path(config: dict[str, Any], project_root: Path, rank: int) -> Path:
    save_dir = config.get("calibration", {}).get("save_dir", "outputs/preallocations")
    seed = int(config.get("calibration", {}).get("seed", config.get("seed", 42)))
    return _resolve_path(project_root, save_dir) / f"dico_pre_rank{rank}_seed{seed}.json"


def _preallocation_metadata_from_payload(
    payload: dict[str, Any],
    config: dict[str, Any],
    path: Path,
    source: str,
) -> dict[str, Any]:
    atom_mode = payload.get("atom_mode", "module_proxy")
    pre_cfg = config.get("preallocation", {})
    budget_payload = payload.get("budget", {}) or {}
    metadata = {
        "aggregation_mode": payload.get("aggregation_mode", config.get("preallocation", {}).get("aggregation_mode", "weighted_topk")),
        "allocation_method": payload.get("allocation_method", config.get("preallocation", {}).get("allocation_method")),
        "weighted_topk_k": payload.get("weighted_topk_k", config.get("preallocation", {}).get("weighted_topk_k", "auto")),
        "atom_weight_normalization": payload.get(
            "atom_weight_normalization",
            config.get("preallocation", {}).get("atom_weight_normalization", "none"),
        ),
        "profile_norm_mode": payload.get("profile_norm_mode", config.get("preallocation", {}).get("profile_norm_mode")),
        "use_cost_aware_allocation": payload.get(
            "use_cost_aware_allocation",
            config.get("preallocation", {}).get("use_cost_aware_allocation", True),
        ),
        "atom_mode": atom_mode,
        "atom_mode_limitation": payload.get("atom_mode_limitation")
        or (MODULE_PROXY_LIMITATION if atom_mode == "module_proxy" else None),
        "preallocation_source": source,
        "preallocation_path": str(path),
        "module_names": payload.get("module_names"),
        "module_dims": payload.get("module_dims"),
        "cache_context": payload.get("cache_context"),
        "module_logs": payload.get("module_logs", []),
        "budget_error_ratio": payload.get("budget_error_ratio", payload.get("budget", {}).get("budget_error_ratio")),
        "num_atoms": payload.get("num_atoms"),
        "num_selected_atoms": payload.get("num_selected_atoms"),
        "eta": payload.get("eta", pre_cfg.get("eta", 0.98)),
        "allow_rank_beyond_selected_evidence": payload.get(
            "allow_rank_beyond_selected_evidence",
            pre_cfg.get("allow_rank_beyond_selected_evidence", True),
        ),
        "use_soft_tail": payload.get("use_soft_tail", pre_cfg.get("use_soft_tail", True)),
        "target_budget": payload.get("target_budget", budget_payload.get("target_budget")),
        "actual_budget": payload.get("actual_budget", budget_payload.get("actual_budget")),
        "target_budget_paramcount": payload.get(
            "target_budget_paramcount",
            budget_payload.get("target_budget_paramcount", payload.get("target_budget", budget_payload.get("target_budget"))),
        ),
        "target_budget_ranksum": payload.get("target_budget_ranksum", budget_payload.get("target_budget_ranksum")),
        "actual_budget_paramcount": payload.get(
            "actual_budget_paramcount",
            budget_payload.get("actual_budget_paramcount", payload.get("actual_budget", budget_payload.get("actual_budget"))),
        ),
        "actual_budget_ranksum": payload.get("actual_budget_ranksum", budget_payload.get("actual_budget_ranksum")),
        "budget_ratio": payload.get("budget_ratio", budget_payload.get("budget_ratio")),
        "budget_ratio_paramcount": payload.get("budget_ratio_paramcount", budget_payload.get("budget_ratio_paramcount")),
        "budget_ratio_ranksum": payload.get("budget_ratio_ranksum", budget_payload.get("budget_ratio_ranksum")),
        "selected_evidence_count_total": payload.get("selected_evidence_count_total"),
        "final_total_rank": payload.get("final_total_rank"),
        "rank_beyond_selected_evidence_total": payload.get("rank_beyond_selected_evidence_total"),
        "modules_with_rank_beyond_selected_evidence": payload.get("modules_with_rank_beyond_selected_evidence"),
        "cache_incompatible_reasons": payload.get("cache_incompatible_reasons"),
    }
    if metadata["budget_ratio"] is None and metadata["target_budget"]:
        metadata["budget_ratio"] = float((metadata["actual_budget"] or 0) / metadata["target_budget"])
    if metadata["budget_ratio_paramcount"] is None:
        metadata["budget_ratio_paramcount"] = metadata["budget_ratio"]
    return metadata


def _preallocation_cache_incompatible_reasons(
    payload: dict[str, Any],
    config: dict[str, Any],
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
) -> list[str]:
    reasons: list[str] = []
    if "rank_allocation" not in payload:
        reasons.append("missing_rank_allocation")
    rank_payload = payload.get("rank_allocation") or {}
    if "rank_allocation" in payload and set(rank_payload.keys()) != set(module_names):
        reasons.append("module_set_mismatch")
    cached_dims = payload.get("module_dims")
    if cached_dims is not None:
        normalized_cached = {
            name: {"in_dim": int(dims["in_dim"]), "out_dim": int(dims["out_dim"])}
            for name, dims in cached_dims.items()
            if "in_dim" in dims and "out_dim" in dims
        }
        if normalized_cached != module_dims:
            reasons.append("module_dims_mismatch")
    expected_context = build_preallocation_cache_context(config, module_names, module_dims)
    cached_context = payload.get("cache_context")
    if cached_context != expected_context:
        reasons.append("cache_context_mismatch")
        cached_pre = (cached_context or {}).get("preallocation", {}) if isinstance(cached_context, dict) else {}
        expected_pre = expected_context.get("preallocation", {})
        for key, expected_value in expected_pre.items():
            if cached_pre.get(key) != expected_value:
                reasons.append(f"cache_context.preallocation.{key}_mismatch")
    pre_cfg = config.get("preallocation", {})
    expected = {
        "atom_mode": pre_cfg.get("atom_mode", pre_cfg.get("fallback_atom_mode", "module_proxy")),
        "aggregation_mode": pre_cfg.get("aggregation_mode", "weighted_topk"),
        "atom_weight_normalization": pre_cfg.get("atom_weight_normalization", "none"),
        "use_cost_aware_allocation": pre_cfg.get("use_cost_aware_allocation", True),
    }
    for key, value in expected.items():
        payload_value = payload.get(key)
        if payload_value is None and key == "atom_mode":
            payload_value = (payload.get("cache_context") or {}).get("preallocation", {}).get("atom_mode")
        if payload_value != value:
            reasons.append(f"{key}_mismatch")
    if not payload.get("module_logs"):
        reasons.append("missing_module_logs")
    return reasons


def _preallocation_cache_incompatible_reason(
    payload: dict[str, Any],
    config: dict[str, Any],
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
) -> str | None:
    reasons = _preallocation_cache_incompatible_reasons(payload, config, module_names, module_dims)
    return reasons[0] if reasons else None


def _preallocation_cache_is_compatible(
    payload: dict[str, Any],
    config: dict[str, Any],
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
) -> bool:
    return not _preallocation_cache_incompatible_reasons(payload, config, module_names, module_dims)


def load_or_build_preallocation(
    config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
    calibration_batches: list[dict[str, torch.Tensor]],
    target_budget: int,
    project_root: Path,
) -> tuple[dict[str, int], dict[str, Any]]:
    rank = int(config["rank"])
    path = _preallocation_path(config, project_root, rank)
    cache_diagnostics = {
        "cache_hit": path.exists(),
        "cache_compatible": False,
        "cache_incompatible_reason": "cache_missing",
        "cache_incompatible_reasons": ["cache_missing"],
    }
    if path.exists():
        payload = load_preallocation(path)
        incompatible_reasons = _preallocation_cache_incompatible_reasons(payload, config, module_names, module_dims)
        if not incompatible_reasons:
            metadata = _preallocation_metadata_from_payload(payload, config, path, source="cache")
            metadata.update(
                {
                    "cache_hit": True,
                    "cache_compatible": True,
                    "cache_incompatible_reason": None,
                    "cache_incompatible_reasons": [],
                }
            )
            rank_payload = payload["rank_allocation"]
            return {name: int(value) for name, value in rank_payload.items()}, metadata
        cache_diagnostics = {
            "cache_hit": True,
            "cache_compatible": False,
            "cache_incompatible_reason": incompatible_reasons[0],
            "cache_incompatible_reasons": incompatible_reasons,
        }

    allocator = DiCoPreAllocator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        module_names=module_names,
        module_dims=module_dims,
    )
    if config.get("calibration", {}).get("enabled", True):
        allocator.collect_calibration_statistics(calibration_batches)
    result = allocator.allocate(target_budget)
    allocator.save(path, result)
    metadata = _preallocation_metadata_from_payload(result.to_dict(preallocation_path=str(path)), config, path, source="computed")
    metadata.update(cache_diagnostics)
    return result.rank_allocation, metadata


def build_preallocation_cache(config: dict[str, Any]) -> dict[str, Any]:
    """Build or reuse the DiCo preallocation cache without running training.

    This path intentionally stops before LoRA injection, optimizer creation,
    final evaluation, and checkpoint/metrics writing. It is for preparing the
    auditable preallocation cache only.
    """

    project_root = Path(config.get("_project_root", Path.cwd())).resolve()
    set_seed(int(config.get("seed", 42)))
    tokenizer, model = load_tokenizer_and_model(config)
    placement_device = torch.device("cpu") if config.get("model", {}).get("type") == "tiny" else model_device(model)
    if str(placement_device) == "cpu" or not config.get("model", {}).get("device_map"):
        model.to(placement_device)
    input_device = model_input_device(model)

    target_suffixes = config.get("lora", {}).get("target_modules", [])
    target_modules = find_target_linear_modules(model, target_suffixes)
    if not target_modules:
        raise RuntimeError(f"No target linear-like modules matched {target_suffixes}")
    module_names = [name for name, _module in target_modules]
    module_dims = collect_module_dims(target_modules)

    train_raw, _eval_raw = load_raw_datasets(config)
    data_cfg = config.get("data", {})
    train_raw = limit_records(train_raw, data_cfg.get("train_limit"))
    max_length = int(data_cfg.get("max_length", 512))
    train_records = tokenize_records(train_raw, tokenizer, max_length)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or getattr(tokenizer, "eos_token_id", 0) or 0
    collator = SFTCollator(pad_token_id)

    calibration_batches = _build_calibration_batches(
        train_records,
        collator,
        input_device,
        config.get("calibration", {}),
    )

    rank = int(config["rank"])
    budget_cfg = config.get("budget", {})
    budget_info = get_uniform_budget(
        rank,
        module_names,
        module_dims,
        budget_mode=budget_cfg.get("mode", "equal_trainable_params"),
        warning_threshold=float(budget_cfg.get("warning_threshold", 0.01)),
    )
    try:
        pre_cfg = config.get("preallocation", {})
        LOGGER.info(
            "preallocation_start experiment=%s atom_mode=%s num_batches=%d num_modules=%d module_chunk_size=%s",
            config.get("experiment_name", f"{config.get('method')}_r{config.get('rank')}"),
            pre_cfg.get("atom_mode", pre_cfg.get("fallback_atom_mode", "module_proxy")),
            len(calibration_batches),
            len(module_names),
            pre_cfg.get("module_chunk_size", len(module_names)),
        )
        rank_allocation, metadata = load_or_build_preallocation(
            config,
            model,
            tokenizer,
            module_names,
            module_dims,
            calibration_batches,
            budget_info.target_budget,
            project_root,
        )
    finally:
        calibration_batches.clear()
        del calibration_batches
        _release_torch_memory()
    return {
        "rank_allocation": rank_allocation,
        "preallocation": metadata,
        "target_budget": budget_info.target_budget,
        "module_names": module_names,
        "module_dims": module_dims,
    }


def save_masked_lora_state(path: Path, model: torch.nn.Module) -> None:
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name or "rank_mask" in name
    }
    torch.save(state, path)


def train(config: dict[str, Any]) -> dict[str, Any]:
    run_started_at = time.perf_counter()
    project_root = Path(config.get("_project_root", Path.cwd())).resolve()
    experiment_name = config.get("experiment_name", f"{config['method']}_r{config['rank']}")
    output_root = _resolve_path(project_root, config.get("project", {}).get("output_dir", "outputs"))
    output_dir = ensure_dir(output_root / experiment_name)
    save_yaml(output_dir / "config_resolved.yaml", config)
    LOGGER.info("experiment_start experiment=%s output_dir=%s", experiment_name, output_dir)

    set_seed(int(config.get("seed", 42)))
    tokenizer, model = load_tokenizer_and_model(config)
    placement_device = torch.device("cpu") if config.get("model", {}).get("type") == "tiny" else model_device(model)
    if str(placement_device) == "cpu" or not config.get("model", {}).get("device_map"):
        model.to(placement_device)
    input_device = model_input_device(model)

    target_suffixes = config.get("lora", {}).get("target_modules", [])
    target_modules = find_target_linear_modules(model, target_suffixes)
    if not target_modules:
        raise RuntimeError(f"No target linear-like modules matched {target_suffixes}")
    module_names = [name for name, _module in target_modules]
    module_dims = collect_module_dims(target_modules)

    train_raw, eval_raw = load_raw_datasets(config)
    data_cfg = config.get("data", {})
    train_raw = limit_records(train_raw, data_cfg.get("train_limit"))
    eval_raw = limit_records(eval_raw, data_cfg.get("eval_limit"))
    max_length = int(data_cfg.get("max_length", 512))
    train_records = tokenize_records(train_raw, tokenizer, max_length)
    eval_records = tokenize_records(eval_raw, tokenizer, max_length)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or getattr(tokenizer, "eos_token_id", 0) or 0
    collator = SFTCollator(pad_token_id)
    LOGGER.info(
        "data_loaded experiment=%s train_records=%d eval_records=%d max_length=%d",
        experiment_name,
        len(train_records),
        len(eval_records),
        max_length,
    )

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 1))

    rank = int(config["rank"])
    lora_max_mult = float(config.get("lora", {}).get("max_rank_multiplier", 2.0))
    pre_max_mult = float(config.get("preallocation", {}).get("r_max_multiplier", 2.0))
    
    if lora_max_mult < pre_max_mult:
        raise ValueError(f"lora.max_rank_multiplier ({lora_max_mult}) cannot be smaller than preallocation.r_max_multiplier ({pre_max_mult})")
        
    max_rank = int(rank * lora_max_mult)
    budget_cfg = config.get("budget", {})
    budget_info = get_uniform_budget(
        rank,
        module_names,
        module_dims,
        budget_mode=budget_cfg.get("mode", "equal_trainable_params"),
        warning_threshold=float(budget_cfg.get("warning_threshold", 0.01)),
    )
    target_budget = budget_info.target_budget
    method = config["method"]
    preallocation_metadata = None
    preallocation = None
    lora_downscale_metadata: dict[str, Any] = {
        "lora_baseline_downscaled": False,
        "lora_baseline_target_ratio": None,
        "lora_baseline_min_ratio": None,
        "lora_baseline_actual_ratio": None,
        "lora_downscale_details": [],
        "lora_downscale_interval_pass": None,
    }
    if method in {"lora", "dico_dynamic"}:
        initial_allocation = uniform_allocation(rank, module_names)
        enforce_target_ratio = float(budget_cfg.get("enforce_target_ratio", 1.0))
        if method == "lora" and enforce_target_ratio < 1.0:
            initial_allocation, lora_downscale_metadata = _downscale_lora_allocation_to_ratio(
                initial_allocation,
                module_dims,
                target_budget,
                target_ratio=enforce_target_ratio,
                min_ratio=float(budget_cfg.get("enforce_min_ratio", 0.97)),
            )
    elif method in PREALLOC_METHODS:
        calibration_batches = _build_calibration_batches(
            train_records,
            collator,
            input_device,
            config.get("calibration", {}),
        )
        try:
            pre_cfg = config.get("preallocation", {})
            LOGGER.info(
                "preallocation_start experiment=%s atom_mode=%s num_batches=%d num_modules=%d module_chunk_size=%s",
                experiment_name,
                pre_cfg.get("atom_mode", pre_cfg.get("fallback_atom_mode", "module_proxy")),
                len(calibration_batches),
                len(module_names),
                pre_cfg.get("module_chunk_size", len(module_names)),
            )
            initial_allocation, preallocation_metadata = load_or_build_preallocation(
                config,
                model,
                tokenizer,
                module_names,
                module_dims,
                calibration_batches,
                target_budget,
                project_root,
            )
        finally:
            calibration_batches.clear()
            del calibration_batches
            _release_torch_memory()
            LOGGER.info("calibration_memory_released experiment=%s", experiment_name)
        preallocation = dict(initial_allocation)
    else:
        raise ValueError(f"Unsupported method: {method}")

    budget_manager = BudgetManager(
        budget_cfg.get("mode", "equal_trainable_params"),
        module_dims,
        warning_threshold=float(budget_cfg.get("warning_threshold", 0.01)),
    )
    generic_repair_applied = False
    preallocation_eta = float(config.get("preallocation", {}).get("eta", 0.98)) if method in PREALLOC_METHODS else None
    if method in PREALLOC_METHODS:
        budget_payload = budget_manager.describe(initial_allocation, target_budget)
        if bool(budget_payload.get("over_budget", False)):
            raise ValueError(
                "DiCo preallocation exceeds target budget. "
                "This should be fixed inside the DiCo allocator, "
                "not by generic BudgetManager.repair(...)."
            )
        budget_payload = _budget_with_policy_fields(
            budget_payload,
            method=method,
            preallocation_eta=preallocation_eta,
            generic_repair_applied=False,
        )
        if not budget_payload["budget_eta_reached"]:
            LOGGER.warning(
                "dico_preallocation_below_eta experiment=%s actual=%s eta_target=%.1f target=%s budget_ratio=%.6f",
                experiment_name,
                budget_payload["actual_budget"],
                float(preallocation_eta) * target_budget,
                target_budget,
                budget_payload["budget_ratio"],
            )
        if preallocation_metadata is not None:
            preallocation_metadata.update(
                {
                    "target_budget": budget_payload["target_budget"],
                    "actual_budget": budget_payload["actual_budget"],
                    "target_budget_paramcount": budget_payload["target_budget_paramcount"],
                    "actual_budget_paramcount": budget_payload["actual_budget_paramcount"],
                    "budget_ratio_paramcount": budget_payload["budget_ratio_paramcount"],
                    "budget_ratio": budget_payload["budget_ratio"],
                    "preallocation_eta": preallocation_eta,
                    "budget_eta_reached": budget_payload["budget_eta_reached"],
                    "budget_interval_pass": budget_payload["budget_interval_pass"],
                    "generic_repair_applied": False,
                }
            )
    else:
        enforce_target_ratio = float(budget_cfg.get("enforce_target_ratio", 1.0))
        if method == "lora" and enforce_target_ratio < 1.0:
            generic_repair_applied = False
            budget_payload = _budget_with_policy_fields(
                budget_manager.describe(initial_allocation, target_budget),
                method=method,
                preallocation_eta=None,
                generic_repair_applied=False,
            )
        else:
            repaired = budget_manager.repair(
                initial_allocation,
                target_budget,
                r_min=int(config.get("preallocation", {}).get("r_min", 0)),
                r_max=max_rank,
            )
            initial_allocation = repaired.allocation
            generic_repair_applied = True
            budget_payload = _budget_with_policy_fields(
                repaired.budget.to_dict(),
                method=method,
                preallocation_eta=None,
                generic_repair_applied=True,
            )
    budget_payload.update(lora_downscale_metadata)
    write_json(output_dir / "budget.json", budget_payload)
    initial_rank_payload: dict[str, Any] = {
        "rank_allocation": initial_allocation,
        **budget_payload,
    }
    if preallocation_metadata is not None:
        initial_rank_payload.update(
            {
                "module_logs": preallocation_metadata.get("module_logs", []),
                "aggregation_mode": preallocation_metadata.get("aggregation_mode"),
                "atom_weight_normalization": preallocation_metadata.get("atom_weight_normalization"),
                "use_cost_aware_allocation": preallocation_metadata.get("use_cost_aware_allocation"),
                "atom_mode": preallocation_metadata.get("atom_mode"),
                "atom_mode_limitation": preallocation_metadata.get("atom_mode_limitation"),
                "allocation_method": preallocation_metadata.get("allocation_method"),
                "eta": preallocation_metadata.get("eta", preallocation_eta),
                "allow_rank_beyond_selected_evidence": preallocation_metadata.get(
                    "allow_rank_beyond_selected_evidence"
                ),
                "use_soft_tail": preallocation_metadata.get("use_soft_tail"),
                "cache_hit": preallocation_metadata.get("cache_hit"),
                "cache_compatible": preallocation_metadata.get("cache_compatible"),
                "cache_incompatible_reason": preallocation_metadata.get("cache_incompatible_reason"),
                "cache_incompatible_reasons": preallocation_metadata.get("cache_incompatible_reasons"),
                "target_budget_paramcount": preallocation_metadata.get("target_budget_paramcount"),
                "target_budget_ranksum": preallocation_metadata.get("target_budget_ranksum"),
                "actual_budget_paramcount": preallocation_metadata.get("actual_budget_paramcount"),
                "actual_budget_ranksum": preallocation_metadata.get("actual_budget_ranksum"),
                "budget_ratio_paramcount": preallocation_metadata.get("budget_ratio_paramcount"),
                "budget_ratio_ranksum": preallocation_metadata.get("budget_ratio_ranksum"),
                "selected_evidence_count_total": preallocation_metadata.get("selected_evidence_count_total"),
                "final_total_rank": preallocation_metadata.get("final_total_rank"),
                "rank_beyond_selected_evidence_total": preallocation_metadata.get(
                    "rank_beyond_selected_evidence_total"
                ),
                "modules_with_rank_beyond_selected_evidence": preallocation_metadata.get(
                    "modules_with_rank_beyond_selected_evidence"
                ),
            }
        )
    write_json(output_dir / "rank_allocation_initial.json", initial_rank_payload)
    (output_dir / "train_log.jsonl").write_text("", encoding="utf-8")
    (output_dir / "eval_log.jsonl").write_text("", encoding="utf-8")

    masked_modules = inject_masked_lora(
        model,
        initial_allocation,
        max_rank=max_rank,
        alpha=float(config.get("lora", {}).get("alpha", 16)),
        dropout=float(config.get("lora", {}).get("dropout", 0.0)),
        lora_dtype=select_torch_dtype(config.get("model", {}).get("torch_dtype", "bfloat16")),
    )
    LOGGER.info(
        "lora_injected experiment=%s modules=%d max_rank=%d target_budget=%d",
        experiment_name,
        len(masked_modules),
        max_rank,
        target_budget,
    )
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": 0.0}],
        lr=float(training_cfg.get("learning_rate", 2.0e-4)),
    )

    dynamic_allocator = None
    if config.get("dynamic", {}).get("enabled", False):
        (output_dir / "dynamic_adjustments.jsonl").write_text("", encoding="utf-8")
        dynamic_allocator = DynamicRankAllocator(
            masked_lora_modules=masked_modules,
            module_dims=module_dims,
            initial_allocation=initial_allocation,
            target_budget=target_budget,
            config={
                **config.get("dynamic", {}),
                "warning_threshold": float(budget_cfg.get("warning_threshold", 0.01)),
            },
            base_rank=rank,
            preallocation=preallocation,
            budget_manager=budget_manager,
        )

    rank_history_path = output_dir / "rank_history.csv"
    init_rank_history(rank_history_path)
    append_rank_history(
        rank_history_path,
        0,
        initial_allocation,
        max_rank,
        dynamic_allocator.module_scores if dynamic_allocator else {},
        budget_manager,
        target_budget,
        initial_allocation,
        preallocation,
        latest_mid_eval_loss=None,
    )

    max_steps = int(training_cfg.get("max_steps", 1000))
    logging_steps = int(training_cfg.get("logging_steps", 10))
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    learning_rate = float(training_cfg.get("learning_rate", 2.0e-4))
    iterator = batch_iter(train_records, batch_size, collator)
    model.train()
    running_loss = 0.0
    running_loss_observations = 0
    latest_mid_eval_loss: float | None = None
    evaluation_cfg = config.get("evaluation", {})
    mid_eval_cfg_raw = evaluation_cfg.get("mid_eval_loss_only", {"enabled": False})
    if isinstance(mid_eval_cfg_raw, bool):
        mid_eval_cfg = {"enabled": mid_eval_cfg_raw}
    else:
        mid_eval_cfg = dict(mid_eval_cfg_raw or {})
    mid_eval_enabled = bool(mid_eval_cfg.get("enabled", False))
    mid_eval_every = max(1, int(mid_eval_cfg.get("every_n_steps", 50)))
    mid_eval_max_batches = int(mid_eval_cfg.get("max_batches", evaluation_cfg.get("max_batches", 4)))
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    while global_step < max_steps:
        batch = _move_batch(next(iterator), input_device)
        outputs = model(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()
        running_loss += float(loss.detach().cpu().item()) * grad_accum
        running_loss_observations += 1
        micro_step += 1
        del batch, outputs, loss
        if micro_step % grad_accum != 0:
            continue

        if dynamic_allocator is not None:
            dynamic_allocator.update_statistics()
        apply_rank_masks_to_grads(masked_modules)
        optimizer.step()
        restore_inactive_parameters(masked_modules)
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if dynamic_allocator is not None and dynamic_allocator.should_adjust(global_step, max_steps):
            adjustment = dynamic_allocator.adjust_rank(global_step, max_steps)
            log_train(
                output_dir / "dynamic_adjustments.jsonl",
                {
                    "event": "dynamic_adjustment",
                    "elapsed_sec": time.perf_counter() - run_started_at,
                    **adjustment,
                },
            )
            LOGGER.info(
                "dynamic_adjustment experiment=%s step=%d moved=%s budget_error_ratio=%s",
                experiment_name,
                global_step,
                adjustment.get("num_moved"),
                adjustment.get("budget_error_ratio"),
            )
            append_rank_history(
                rank_history_path,
                global_step,
                dynamic_allocator.current_allocation,
                max_rank,
                dynamic_allocator.module_scores,
                budget_manager,
                target_budget,
                initial_allocation,
                preallocation,
                latest_mid_eval_loss=latest_mid_eval_loss,
            )

        if global_step % logging_steps == 0 or global_step == 1:
            elapsed_sec = time.perf_counter() - run_started_at
            avg_loss = running_loss / max(1, running_loss_observations)
            steps_per_sec = global_step / elapsed_sec if elapsed_sec > 0 else 0.0
            log_train(
                output_dir / "train_log.jsonl",
                {
                    "event": "train_step",
                    "step": global_step,
                    "max_steps": max_steps,
                    "loss": avg_loss,
                    "learning_rate": learning_rate,
                    "elapsed_sec": elapsed_sec,
                    "steps_per_sec": steps_per_sec,
                    "micro_step": micro_step,
                    "grad_accumulation_steps": grad_accum,
                },
            )
            LOGGER.info(
                "train_step experiment=%s step=%d/%d loss=%.6f elapsed_sec=%.1f steps_per_sec=%.4f",
                experiment_name,
                global_step,
                max_steps,
                avg_loss,
                elapsed_sec,
                steps_per_sec,
            )
            running_loss = 0.0
            running_loss_observations = 0

        if mid_eval_enabled and global_step < max_steps and global_step % mid_eval_every == 0:
            mid_eval_started_at = time.perf_counter()
            mid_eval = evaluate_loss(
                model,
                eval_records,
                collator,
                batch_size=batch_size,
                device=input_device,
                max_batches=mid_eval_max_batches,
            )
            latest_mid_eval_loss = float(mid_eval["eval_loss"])
            elapsed_sec = time.perf_counter() - run_started_at
            log_train(
                output_dir / "train_log.jsonl",
                {
                    "event": "mid_eval_loss",
                    "step": global_step,
                    "max_steps": max_steps,
                    "elapsed_sec": elapsed_sec,
                    "eval_elapsed_sec": time.perf_counter() - mid_eval_started_at,
                    "eval_loss": latest_mid_eval_loss,
                    "eval_max_batches": mid_eval_max_batches,
                },
            )
            LOGGER.info(
                "mid_eval_loss experiment=%s step=%d/%d eval_loss=%.6f elapsed_sec=%.1f",
                experiment_name,
                global_step,
                max_steps,
                latest_mid_eval_loss,
                elapsed_sec,
            )

    final_allocation = (
        dynamic_allocator.current_allocation if dynamic_allocator is not None else initial_allocation
    )
    LOGGER.info("training_complete experiment=%s steps=%d", experiment_name, global_step)
    write_json(output_dir / "rank_allocation_final.json", final_allocation)
    final_budget = _budget_with_policy_fields(
        budget_manager.describe(final_allocation, target_budget),
        method=method,
        preallocation_eta=preallocation_eta,
        generic_repair_applied=generic_repair_applied,
    )
    final_budget.update(lora_downscale_metadata)
    write_json(output_dir / "budget.json", final_budget)
    save_masked_lora_state(output_dir / "masked_lora_state.pt", model)
    LOGGER.info("final_loss_start experiment=%s", experiment_name)
    final_loss_started_at = time.perf_counter()
    final_eval = evaluate_loss(
        model,
        eval_records,
        collator,
        batch_size=batch_size,
        device=input_device,
        max_batches=int(config.get("evaluation", {}).get("max_batches", 4)),
    )
    LOGGER.info(
        "final_loss_complete experiment=%s eval_loss=%.6f eval_elapsed_sec=%.1f",
        experiment_name,
        final_eval["eval_loss"],
        time.perf_counter() - final_loss_started_at,
    )
    if evaluation_cfg.get("compute_accuracy", True):
        accuracy_samples = evaluation_cfg.get("accuracy_max_samples", data_cfg.get("eval_limit"))
        LOGGER.info(
            "final_accuracy_start experiment=%s max_samples=%s",
            experiment_name,
            accuracy_samples if accuracy_samples is not None else len(eval_records),
        )
        accuracy_started_at = time.perf_counter()

        def log_accuracy_progress(progress: dict[str, Any]) -> None:
            LOGGER.info(
                "final_accuracy_progress experiment=%s eval_total=%d eval_correct=%d eval_accuracy=%.4f",
                experiment_name,
                progress["eval_total"],
                progress["eval_correct"],
                progress["eval_accuracy"],
            )

        accuracy_metrics = evaluate_gsm8k_accuracy(
            model,
            tokenizer,
            eval_records,
            device=input_device,
            max_samples=accuracy_samples,
            max_new_tokens=int(evaluation_cfg.get("generation_max_new_tokens", 256)),
            stop_sequences=evaluation_cfg.get("stop_sequences"),
            extraction_mode=str(evaluation_cfg.get("answer_extraction", "strict_then_flexible")),
            prediction_path=output_dir / "eval_predictions.jsonl",
            progress_callback=log_accuracy_progress,
            progress_interval=int(evaluation_cfg.get("accuracy_logging_steps", 50)),
        )
        final_eval.update(accuracy_metrics)
        LOGGER.info(
            "final_accuracy_complete experiment=%s eval_accuracy=%.4f eval_correct=%s eval_total=%s eval_elapsed_sec=%.1f",
            experiment_name,
            final_eval["eval_accuracy"],
            final_eval["eval_correct"],
            final_eval["eval_total"],
            time.perf_counter() - accuracy_started_at,
        )
    best_eval_loss = final_eval["eval_loss"]
    final_metric_name = str(evaluation_cfg.get("metric", "gsm8k_accuracy"))
    if final_metric_name in {"accuracy", "exact_match", "gsm8k_accuracy"}:
        final_metric = final_eval.get("eval_accuracy", final_eval["eval_loss"])
        best_metric = final_metric
    else:
        final_metric = final_eval["eval_loss"]
        best_metric = best_eval_loss
    log_eval(
        output_dir / "eval_log.jsonl",
        {
            "event": "final_eval",
            "step": max_steps,
            "final": True,
            "elapsed_sec": time.perf_counter() - run_started_at,
            **final_eval,
        },
    )
    metrics = {
        "experiment": experiment_name,
        "method": method,
        "rank": rank,
        "seed": int(config.get("seed", 42)),
        "atom_mode": preallocation_metadata.get("atom_mode") if preallocation_metadata else None,
        "preallocation": preallocation_metadata,
        "evidence_relaxation": _evidence_relaxation_summary(preallocation_metadata),
        "final_eval_loss": final_eval["eval_loss"],
        "best_eval_loss": best_eval_loss,
        "final_eval_accuracy": final_eval.get("eval_accuracy"),
        "final_exact_match": final_eval.get("eval_exact_match"),
        "eval_correct": final_eval.get("eval_correct"),
        "eval_total": final_eval.get("eval_total"),
        "eval_sample_count": final_eval.get("eval_sample_count"),
        "evaluation_protocol": evaluation_cfg.get("protocol", "internal_zero_shot"),
        "evaluation_prompt_style": evaluation_cfg.get("prompt_style", "sft_cot_hash"),
        "answer_extraction": evaluation_cfg.get("answer_extraction", "strict_then_flexible"),
        "final_metric_name": final_metric_name,
        "final_metric": final_metric,
        "best_metric": best_metric,
        "target_budget": final_budget["target_budget"],
        "actual_budget": final_budget["actual_budget"],
        "total_params": final_budget["actual_budget"],
        "budget_ratio": final_budget["budget_ratio"],
        "target_budget_paramcount": final_budget["target_budget_paramcount"],
        "target_budget_ranksum": final_budget.get("target_budget_ranksum"),
        "actual_budget_paramcount": final_budget["actual_budget_paramcount"],
        "actual_budget_ranksum": final_budget.get("actual_budget_ranksum"),
        "budget_ratio_paramcount": final_budget["budget_ratio_paramcount"],
        "budget_ratio_ranksum": final_budget.get("budget_ratio_ranksum"),
        "preallocation_eta": final_budget["preallocation_eta"],
        "budget_eta_reached": final_budget["budget_eta_reached"],
        "budget_interval_pass": final_budget["budget_interval_pass"],
        "generic_repair_applied": final_budget["generic_repair_applied"],
        "budget_error": final_budget["budget_error"],
        "budget_error_ratio": final_budget["budget_error_ratio"],
        "total_active_rank": final_budget["total_active_rank"],
        "trainable_params_physical": trainable_parameter_count(model),
        "lora_baseline_downscaled": final_budget.get("lora_baseline_downscaled"),
        "lora_baseline_target_ratio": final_budget.get("lora_baseline_target_ratio"),
        "lora_baseline_min_ratio": final_budget.get("lora_baseline_min_ratio"),
        "lora_baseline_actual_ratio": final_budget.get("lora_baseline_actual_ratio"),
        "lora_downscale_details": final_budget.get("lora_downscale_details"),
        "lora_downscale_interval_pass": final_budget.get("lora_downscale_interval_pass"),
    }
    write_json(output_dir / "metrics.json", metrics)
    append_rank_history(
        rank_history_path,
        max_steps,
        final_allocation,
        max_rank,
        dynamic_allocator.module_scores if dynamic_allocator else {},
        budget_manager,
        target_budget,
        initial_allocation,
        preallocation,
        latest_mid_eval_loss=latest_mid_eval_loss,
    )
    LOGGER.info(
        "experiment_complete experiment=%s final_metric_name=%s final_metric=%s elapsed_sec=%.1f",
        experiment_name,
        final_metric_name,
        final_metric,
        time.perf_counter() - run_started_at,
    )
    return metrics
