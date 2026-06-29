from __future__ import annotations

import math
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
from dico_rank.utils import ensure_dir, set_seed, write_json


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _build_calibration_batches(
    train_records: list[dict[str, Any]],
    collator: SFTCollator,
    input_device: torch.device,
    calibration_cfg: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    calibration_limit = int(calibration_cfg.get("num_samples", min(8, len(train_records))))
    calibration_batch_size = max(1, int(calibration_cfg.get("batch_size", 1)))
    selected = train_records[:calibration_limit]
    return [
        _move_batch(collator(selected[start : start + calibration_batch_size]), input_device)
        for start in range(0, len(selected), calibration_batch_size)
    ]


def uniform_allocation(rank: int, module_names: list[str]) -> dict[str, int]:
    return {name: int(rank) for name in module_names}


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
    return {
        "aggregation_mode": payload.get("aggregation_mode", config.get("preallocation", {}).get("aggregation_mode", "weighted_topk")),
        "weighted_topk_k": payload.get("weighted_topk_k", config.get("preallocation", {}).get("weighted_topk_k", "auto")),
        "atom_weight_normalization": payload.get(
            "atom_weight_normalization",
            config.get("preallocation", {}).get("atom_weight_normalization", "none"),
        ),
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
    }


def _preallocation_cache_is_compatible(
    payload: dict[str, Any],
    config: dict[str, Any],
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
) -> bool:
    if "rank_allocation" not in payload:
        return False
    rank_payload = payload.get("rank_allocation") or {}
    if set(rank_payload.keys()) != set(module_names):
        return False
    cached_dims = payload.get("module_dims")
    if cached_dims is not None:
        normalized_cached = {
            name: {"in_dim": int(dims["in_dim"]), "out_dim": int(dims["out_dim"])}
            for name, dims in cached_dims.items()
            if "in_dim" in dims and "out_dim" in dims
        }
        if normalized_cached != module_dims:
            return False
    expected_context = build_preallocation_cache_context(config, module_names, module_dims)
    if payload.get("cache_context") != expected_context:
        return False
    pre_cfg = config.get("preallocation", {})
    expected = {
        "aggregation_mode": pre_cfg.get("aggregation_mode", "weighted_topk"),
        "atom_weight_normalization": pre_cfg.get("atom_weight_normalization", "none"),
        "use_cost_aware_allocation": pre_cfg.get("use_cost_aware_allocation", True),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            return False
    return bool(payload.get("module_logs"))


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
    if path.exists():
        payload = load_preallocation(path)
        if _preallocation_cache_is_compatible(payload, config, module_names, module_dims):
            metadata = _preallocation_metadata_from_payload(payload, config, path, source="cache")
            rank_payload = payload["rank_allocation"]
            return {name: int(value) for name, value in rank_payload.items()}, metadata

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
    project_root = Path(config.get("_project_root", Path.cwd())).resolve()
    experiment_name = config.get("experiment_name", f"{config['method']}_r{config['rank']}")
    output_root = _resolve_path(project_root, config.get("project", {}).get("output_dir", "outputs"))
    output_dir = ensure_dir(output_root / experiment_name)
    save_yaml(output_dir / "config_resolved.yaml", config)

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

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 1))
    calibration_batches = _build_calibration_batches(
        train_records,
        collator,
        input_device,
        config.get("calibration", {}),
    )

    rank = int(config["rank"])
    max_rank = int(rank * config.get("lora", {}).get("max_rank_multiplier", 2))
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
    if method in {"lora", "dico_dynamic"}:
        initial_allocation = uniform_allocation(rank, module_names)
    elif method in {"dico_pre", "dico_predynamic"}:
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
        preallocation = dict(initial_allocation)
    else:
        raise ValueError(f"Unsupported method: {method}")

    budget_manager = BudgetManager(
        budget_cfg.get("mode", "equal_trainable_params"),
        module_dims,
        warning_threshold=float(budget_cfg.get("warning_threshold", 0.01)),
    )
    repaired = budget_manager.repair(
        initial_allocation,
        target_budget,
        r_min=int(config.get("preallocation", {}).get("r_min", 0)),
        r_max=max_rank,
    )
    initial_allocation = repaired.allocation
    if method == "dico_predynamic":
        preallocation = dict(initial_allocation)
    write_json(output_dir / "budget.json", repaired.budget.to_dict())
    initial_rank_payload: dict[str, Any] = {
        "rank_allocation": initial_allocation,
        "budget_error_ratio": repaired.budget.budget_error_ratio,
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
            }
        )
    write_json(output_dir / "rank_allocation_initial.json", initial_rank_payload)
    (output_dir / "train_log.jsonl").touch()
    (output_dir / "eval_log.jsonl").touch()

    masked_modules = inject_masked_lora(
        model,
        initial_allocation,
        max_rank=max_rank,
        alpha=float(config.get("lora", {}).get("alpha", 16)),
        dropout=float(config.get("lora", {}).get("dropout", 0.0)),
        lora_dtype=select_torch_dtype(config.get("model", {}).get("torch_dtype", "bfloat16")),
    )
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": 0.0}],
        lr=float(training_cfg.get("learning_rate", 2.0e-4)),
    )

    dynamic_allocator = None
    if config.get("dynamic", {}).get("enabled", False):
        (output_dir / "dynamic_adjustments.jsonl").touch()
        dynamic_allocator = DynamicRankAllocator(
            masked_lora_modules=masked_modules,
            module_dims=module_dims,
            initial_allocation=initial_allocation,
            target_budget=target_budget,
            config={**config.get("dynamic", {}), "warning_threshold": budget_cfg.get("warning_threshold", 0.01)},
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
    )

    max_steps = int(training_cfg.get("max_steps", 1000))
    eval_steps = int(training_cfg.get("eval_steps", 100))
    logging_steps = int(training_cfg.get("logging_steps", 10))
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    iterator = batch_iter(train_records, batch_size, collator)
    model.train()
    best_eval_loss = math.inf
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    while global_step < max_steps:
        batch = _move_batch(next(iterator), input_device)
        outputs = model(**batch)
        loss = outputs.loss / grad_accum
        loss.backward()
        running_loss += float(loss.detach().cpu().item()) * grad_accum
        micro_step += 1
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
            log_train(output_dir / "dynamic_adjustments.jsonl", adjustment)
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
            )

        if global_step % logging_steps == 0 or global_step == 1:
            log_train(
                output_dir / "train_log.jsonl",
                {"step": global_step, "loss": running_loss / max(1, logging_steps)},
            )
            running_loss = 0.0

        if eval_steps > 0 and (global_step % eval_steps == 0 or global_step == max_steps):
            eval_metrics = evaluate_loss(
                model,
                eval_records,
                collator,
                batch_size=batch_size,
                device=input_device,
                max_batches=int(config.get("evaluation", {}).get("max_batches", 4)),
            )
            eval_metrics["step"] = global_step
            log_eval(output_dir / "eval_log.jsonl", eval_metrics)
            best_eval_loss = min(best_eval_loss, eval_metrics["eval_loss"])

    final_allocation = (
        dynamic_allocator.current_allocation if dynamic_allocator is not None else initial_allocation
    )
    write_json(output_dir / "rank_allocation_final.json", final_allocation)
    final_budget = budget_manager.describe(final_allocation, target_budget)
    write_json(output_dir / "budget.json", final_budget)
    save_masked_lora_state(output_dir / "masked_lora_state.pt", model)
    final_eval = evaluate_loss(
        model,
        eval_records,
        collator,
        batch_size=batch_size,
        device=input_device,
        max_batches=int(config.get("evaluation", {}).get("max_batches", 4)),
    )
    evaluation_cfg = config.get("evaluation", {})
    if evaluation_cfg.get("compute_accuracy", True):
        accuracy_samples = evaluation_cfg.get("accuracy_max_samples", data_cfg.get("eval_limit"))
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
        )
        final_eval.update(accuracy_metrics)
    best_eval_loss = min(best_eval_loss, final_eval["eval_loss"])
    final_metric_name = str(evaluation_cfg.get("metric", "gsm8k_accuracy"))
    if final_metric_name in {"accuracy", "exact_match", "gsm8k_accuracy"}:
        final_metric = final_eval.get("eval_accuracy", final_eval["eval_loss"])
        best_metric = final_metric
    else:
        final_metric = final_eval["eval_loss"]
        best_metric = best_eval_loss
    log_eval(output_dir / "eval_log.jsonl", {"step": max_steps, "final": True, **final_eval})
    metrics = {
        "experiment": experiment_name,
        "method": method,
        "rank": rank,
        "atom_mode": preallocation_metadata.get("atom_mode") if preallocation_metadata else None,
        "preallocation": preallocation_metadata,
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
        "budget_error": final_budget["budget_error"],
        "budget_error_ratio": final_budget["budget_error_ratio"],
        "total_active_rank": final_budget["total_active_rank"],
        "trainable_params_physical": trainable_parameter_count(model),
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
    )
    return metrics
