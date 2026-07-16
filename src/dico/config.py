from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


_INJECTED_KEYS = {"_config_path", "_project_root"}


_KNOWN_CONFIG_SCHEMA: dict[str, Any] = {
    "seed": None,
    "experiment_name": None,
    "method": None,
    "rank": None,
    "ablation": {
        "id": None,
        "reference_config": None,
        "mechanism_group": None,
        "single_factor": None,
        "controlled_difference_fields": None,
        "expected_difference": None,
    },
    "project": {"name": None, "output_dir": None},
    "protocol": {
        "framing": None,
        "note": None,
        "reference": None,
        "unresolved_fields": [{"field": None, "status": None, "reason": None}],
    },
    "model": {
        "type": None,
        "name_or_path": None,
        "torch_dtype": None,
        "device_map": None,
        "load_in_8bit": None,
        "load_in_4bit": None,
        "hidden_size": None,
        "vocab_size": None,
        "revision": None,
        "tokenizer_revision": None,
        "attn_implementation": None,
    },
    "data": {
        "source": None,
        "dataset_name": None,
        "dataset_config": None,
        "train_dataset": None,
        "train_path": None,
        "eval_path": None,
        "humaneval_path": None,
        "train_limit": None,
        "eval_limit": None,
        "max_length": None,
        "num_workers": None,
        "shuffle": None,
        "dataset_seed": None,
        "token_cache_dir": None,
        "eval_datasets": None,
        "group_labels": None,
        "train_sources": [{"path": None, "group": None, "limit": None}],
    },
    "training": {
        "max_steps": None,
        "batch_size": None,
        "per_gpu_batch_size": None,
        "gradient_accumulation_steps": None,
        "learning_rate": None,
        "betas": None,
        "eps": None,
        "weight_decay": None,
        "max_grad_norm": None,
        "warmup_ratio": None,
        "auto_warmup_steps": None,
        "auto_warmup_rate": None,
        "lr_decay_ratio": None,
        "logging_steps": None,
        "gradient_checkpointing": None,
        "ddp_timeout_minutes": None,
        "save_steps": None,
        "save_optimizer_state": None,
        "sample_exposure_policy": None,
        "optimizer_backend": None,
    },
    "lora": {
        "injection": None,
        "target_modules": None,
        "alpha": None,
        "alpha_ref": None,
        "r_ref": None,
        "dropout": None,
        "bias": None,
        "scaling": None,
        "max_rank_multiplier": None,
        "adapter_dtype": None,
    },
    "budget": {
        "mode": None,
        "warning_threshold": None,
        "enforce_min_ratio": None,
        "enforce_target_ratio": None,
    },
    "calibration": {
        "enabled": None,
        "num_samples": None,
        "batch_size": None,
        "seed": None,
        "save_dir": None,
        "shuffle": None,
        "group_sampling": None,
    },
    "preallocation": {
        "atom_mode": None,
        "fallback_atom_mode": None,
        "allocation_method": None,
        "top_k_atoms": None,
        "sketch_dim": None,
        "sketch_seed": None,
        "sketch_dtype": None,
        "sketch_oversample": None,
        "sketch_block_mode": None,
        "response_agg_groups": None,
        "lambda_cov": None,
        "rho": None,
        "sign_split": None,
        "type_scaling": None,
        "log_compression": None,
        "solver": None,
        "module_scalar_template": None,
        "module_scalar_template_formula": None,
        "module_scalar_template_normalization": None,
        "subspace_init": None,
        "rank_override": None,
        "answer_only": None,
        "profile_norm_mode": None,
        "beta": None,
        "eta": None,
        "r_min_multiplier": None,
        "r_max_multiplier": None,
        "module_chunk_size": None,
        "compute_device": None,
        "allocation_device": None,
        "verify_cpu_reference": None,
        "progress_logging_steps": None,
        "allow_rank_beyond_selected_evidence": None,
        "evidence_selection": {"max_selected_atoms": None},
    },
    "dico": {
        "version": None,
        "init": {"mode": None, "zero_B": None},
        "split": {"enabled": None, "mode": None, "physical_merge": None},
        "profile": {"domain": None, "eps": None},
        "taxonomy": {
            "enabled": None,
            "alpha": None,
            "permutation_count": None,
            "fdr": {"enabled": None, "scope": None, "method": None},
        },
        "pseudo_group": {
            "enabled": None,
            "min_k": None,
            "max_k": None,
            "val_fraction": None,
        },
        "coverage": {
            "objective": None,
            "residual_space": None,
            "eps": None,
            "window_h": None,
            "relative_stop_delta": None,
            "kappa_calibration": None,
        },
        "procurement": {
            "mode": None,
            "beta": None,
            "reserve_queue": None,
            "relaxation_fallback": None,
        },
        "legacy_covra_v05": {
            "note": None,
            "taxonomy": {
                "enabled": None,
                "alpha": None,
                "permutation_count": None,
                "fdr": {"enabled": None, "scope": None, "method": None},
            },
            "procurement": {
                "mode": None,
                "beta": None,
                "reserve_queue": None,
                "relaxation_fallback": None,
            },
        },
    },
    "gora_bw": {
        "enabled": None,
        "r_ref": None,
        "module_chunk_size": None,
        "compute_device": None,
        "progress_logging_steps": None,
    },
    "gora": {
        "official_commit": None,
        "gradient_estimation_samples": None,
        "aggregation": None,
        "rounding": None,
        "r_ref": None,
        "r_min": None,
        "r_max": None,
        "rank_stabilize": None,
        "dynamic_scaling": None,
        "scale_by_lr": None,
        "init_lr": None,
        "b_lr_multiplier": None,
        "strict_budget_repair": None,
        "compute_device": None,
        "module_chunk_size": None,
        "progress_logging_steps": None,
        "gradient_collection": None,
        "gradient_offload_device": None,
        "gradient_accumulation_dtype": None,
        "clear_gradient_after_offload": None,
    },
    "runtime": {
        "require_flash_attention_2": None,
        "protocol_scope": None,
    },
    "adalora": {
        "init_rank": None,
        "target_rank": None,
        "tinit": None,
        "tfinal": None,
        "ti": None,
        "tf": None,
        "update_interval": None,
        "deltaT": None,
        "delta_t": None,
        "beta1": None,
        "beta2": None,
        "orth_reg_weight": None,
    },
    "evaluation": {
        "protocol": None,
        "prompt_style": None,
        "metric": None,
        "compute_loss": None,
        "compute_accuracy": None,
        "accuracy_during_training": None,
        "accuracy_max_samples": None,
        "accuracy_logging_steps": None,
        "generation_max_new_tokens": None,
        "batch_size": None,
        "stop_sequences": None,
        "answer_extraction": None,
        "max_batches": None,
        "eval_datasets": None,
        "humaneval_max_samples": None,
        "humaneval_max_new_tokens": None,
        "humaneval_timeout_seconds": None,
        "humaneval_num_samples_per_task": None,
        "mid_eval_loss_only": {"enabled": None, "every_n_steps": None, "max_batches": None},
        "mtbench_local": {
            "enabled": None,
            "judge_model": None,
            "judge_prompt_version": None,
            "conversation_template": None,
            "temperature": None,
            "seed": None,
            "swap_positions": None,
            "max_retries": None,
        },
    },
}


def _find_unknown_config_fields(value: Any, schema: Any, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return []
    if not isinstance(schema, dict):
        return []
    unknown: list[str] = []
    for key, child in value.items():
        if prefix == "" and key in _INJECTED_KEYS:
            continue
        child_path = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            unknown.append(child_path)
            continue
        child_schema = schema[key]
        if isinstance(child_schema, dict):
            unknown.extend(_find_unknown_config_fields(child, child_schema, child_path))
        elif isinstance(child_schema, list):
            item_schema = child_schema[0] if child_schema else None
            if isinstance(child, list) and isinstance(item_schema, dict):
                for index, item in enumerate(child):
                    unknown.extend(_find_unknown_config_fields(item, item_schema, f"{child_path}[{index}]"))
    return unknown


def validate_known_config_fields(config: dict[str, Any]) -> None:
    unknown = _find_unknown_config_fields(config, _KNOWN_CONFIG_SCHEMA)
    if unknown:
        formatted = ", ".join(unknown)
        raise ValueError(
            "Unknown config field(s): "
            f"{formatted}. Add the field to the explicit schema if it is intentionally consumed."
        )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inherits = data.pop("inherits", None)
    if inherits:
        parent = (path.parent / inherits).resolve()
        data = deep_merge(load_yaml(parent), data)
    resolved = path.resolve()
    config_root = next((parent for parent in resolved.parents if parent.name == "configs"), resolved.parent)
    data["_config_path"] = str(resolved)
    data["_project_root"] = str(config_root.parent)
    return data


def parse_override_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        path, raw_value = item.split("=", 1)
        target = result
        parts = path.split(".")
        for key in parts[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise ValueError(f"Override path crosses non-dict value: {path}")
        target[parts[-1]] = parse_override_value(raw_value)
    return result


def save_yaml(path: Path | str, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
