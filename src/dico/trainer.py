from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch

try:
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs, broadcast_object_list
    _ACCELERATE_AVAILABLE = True
except ImportError:
    InitProcessGroupKwargs = None
    _ACCELERATE_AVAILABLE = False

from dico.adalora import AdaLoRAConfig, AdaLoRAController, inject_adalora
from dico.atom_svd import _run_backward_and_stream
from dico.config import save_yaml
from dico.data import (
    SFTCollator,
    batch_iter,
    dataset_hash,
    format_prompt,
    limit_records,
    load_humaneval_records,
    load_raw_datasets,
    order_records,
    stable_record_hash,
    tokenize_records,
    tokenize_records_cached,
)
from dico.evaluator import DEFAULT_GSM8K_STOP_SEQUENCES, evaluate_gsm8k_accuracy, evaluate_humaneval_pass_at_1, evaluate_loss
from dico.logging_utils import append_rank_history, init_rank_history, log_eval, log_train
from dico.lora_masked import (
    apply_rank_masks_to_grads,
    inject_masked_lora,
    restore_inactive_parameters,
    trainable_parameter_count,
)
from dico.lora_checkpoint import save_lora_state
from dico.lora_scaling import compute_covra_module_alpha
from dico.lora_static import inject_static_lora
from dico.model_loader import (
    collect_module_dims,
    find_target_linear_modules,
    load_tokenizer_and_model,
    model_device,
    model_input_device,
    select_torch_dtype,
)
from dico.preallocation import (
    MODULE_PROXY_LIMITATION,
    DiCoPreAllocator,
    build_preallocation_cache_context,
    load_direction_bank,
    load_preallocation,
)
from dico.diagnostics import compute_diagnostics
from dico.init import build_direction_anchored_init
from dico.gora_bw import allocate_gora_bw
from dico.gora import (
    OFFICIAL_GORA_COMMIT,
    allocate_gora_ranks,
    collect_average_weight_gradients,
    compute_gora_importance,
    gora_pseudoinverse_init,
    scale_gora_b_initialization,
    strict_budget_repair,
)
from dico.rank_budget import BudgetManager, get_uniform_budget
from dico.rank_budget import compute_total_lora_params, module_rank_cost
from dico.utils import ensure_dir, set_seed, write_json
from dico.stage_metrics import StageMetricsRecorder


LOGGER = logging.getLogger(__name__)
PREALLOC_METHODS = {"dico_cd", "dico_cd_da"}
WINDOW_ALLOC_METHODS = PREALLOC_METHODS | {"gora_bw", "gora_bm"}
FORMAL_GORA_METHODS = {"gora_public", "gora_bm"}


def build_cosine_schedule_with_warmup_and_floor(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_decay_ratio: float = 0.0,
    num_cycles: float = 0.5,
    auto_warmup_steps: int = 0,
    auto_warmup_rate: float = 0.05,
    decay_over_post_warmup_steps: bool = False,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then cosine decay from peak LR down to `lr_decay_ratio * peak_lr`
    (GoRA protocol uses 0.1, i.e. never decays all the way to 0). `lr_decay_ratio=0.0`
    reproduces transformers.get_cosine_schedule_with_warmup's default behavior."""

    def lr_lambda(current_step: int) -> float:
        if auto_warmup_steps > 0 and current_step < auto_warmup_steps:
            return min(float(auto_warmup_rate), float(current_step) / float(max(1, num_warmup_steps)))
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # The executed CovRA reference decays across the post-warmup interval and
        # therefore reaches the configured floor on the final optimizer step.
        # Keep the historical GoRA denominator as the default for baselines.
        decay_steps = (
            num_training_steps - num_warmup_steps
            if decay_over_post_warmup_steps
            else num_training_steps
        )
        progress = float(current_step - num_warmup_steps) / float(max(1, decay_steps))
        progress = min(1.0, progress)
        cosine_decay = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        return lr_decay_ratio + (1.0 - lr_decay_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def official_warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    """GoRA script scheduler converts a ratio with int(total*ratio)+1."""
    return int(int(total_steps) * float(warmup_ratio)) + 1


def is_reference_covra_config(config: Mapping[str, Any]) -> bool:
    return (
        str(config.get("method")) in {"dico_cd", "dico_cd_da"}
        and str(config.get("preallocation", {}).get("allocation_method")) == "covra_v05"
    )


def training_warmup_steps(config: Mapping[str, Any]) -> int:
    training_cfg = config.get("training", {})
    total_steps = int(training_cfg.get("max_steps", 1000))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.03))
    if is_reference_covra_config(config):
        return max(1, int(total_steps * warmup_ratio))
    return official_warmup_steps(total_steps, warmup_ratio)


def build_method_optimizer(
    model: torch.nn.Module,
    *,
    method: str,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    gora_b_lr_multiplier: float = 16.0,
) -> torch.optim.AdamW:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    common = {"weight_decay": float(weight_decay), "betas": tuple(betas), "eps": float(eps)}
    if method in {"gora_public", "gora_bm"}:
        b_params = [parameter for name, parameter in trainable if name.endswith("lora_B")]
        other_params = [parameter for name, parameter in trainable if not name.endswith("lora_B")]
        return torch.optim.AdamW(
            [
                {"params": other_params, "lr": float(learning_rate), **common, "group_name": "gora_A"},
                {"params": b_params, "lr": float(learning_rate) * float(gora_b_lr_multiplier), **common, "group_name": "gora_B"},
            ]
        )
    return torch.optim.AdamW(
        [{"params": [parameter for _name, parameter in trainable], "lr": float(learning_rate), **common, "group_name": "adapter"}]
    )


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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _git_commit() -> str | None:
    return _git_output(["rev-parse", "HEAD"])


def _source_control_manifest() -> dict[str, Any]:
    status = _git_output(["status", "--porcelain"])
    return {
        "repo_root": str(_repo_root()),
        "git_commit": _git_commit(),
        "git_branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(status),
        "uncommitted_change_count": 0 if status is None else len(status.splitlines()),
        "git_status_porcelain_sha256": None if status is None else hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _dependency_versions_manifest() -> dict[str, str | None]:
    packages = [
        "python",
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "numpy",
        "scipy",
        "sklearn",
        "pandas",
        "vllm",
    ]
    versions: dict[str, str | None] = {}
    for package in packages:
        if package == "python":
            versions[package] = sys.version.split()[0]
            continue
        if package == "torch":
            versions[package] = torch.__version__
            continue
        dist_name = "scikit-learn" if package == "sklearn" else package
        try:
            versions[package] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _runtime_environment_manifest(num_processes: int, local_rank: int) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_names: list[str] = []
    cuda_current_device = None
    cuda_current_device_name = None
    if cuda_available:
        for index in range(cuda_device_count):
            try:
                cuda_device_names.append(str(torch.cuda.get_device_name(index)))
            except RuntimeError:
                cuda_device_names.append("<unavailable>")
        try:
            cuda_current_device = int(torch.cuda.current_device())
            cuda_current_device_name = str(torch.cuda.get_device_name(cuda_current_device))
        except RuntimeError:
            cuda_current_device = None
            cuda_current_device_name = None
    source_control = _source_control_manifest()
    command = {
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
    }
    return {
        "git_commit": source_control.get("git_commit"),
        "command_line": command["argv"],
        "source_control": source_control,
        "command": command,
        "python": sys.version.split()[0],
        "dependency_versions": _dependency_versions_manifest(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": cuda_device_names,
        "cuda_current_device": cuda_current_device,
        "cuda_current_device_name": cuda_current_device_name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "local_rank": int(local_rank),
        "world_size": int(num_processes),
    }


def _calibration_selection_manifest(selected: list[dict[str, Any]]) -> dict[str, Any]:
    sample_hashes = [str(record.get("sample_hash", "")) for record in selected]
    digest = hashlib.sha256()
    for value in sample_hashes:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return {
        "num_selected_samples": len(selected),
        "sample_ids": [record.get("sample_id") for record in selected],
        "sample_hashes": sample_hashes,
        "sample_indices": [record.get("sample_index") for record in selected],
        "selection_hash": digest.hexdigest(),
    }


def _preallocation_timing_manifest(metadata: dict[str, Any] | None, wall_sec: float) -> dict[str, float]:
    metadata = metadata or {}
    calibration_sec = sum(
        float(metadata.get(key, 0.0) or 0.0)
        for key in ("sketch_pass_sec", "basis_pass_sec", "profile_pass_sec", "gradient_collection_sec")
    )
    has_explicit_allocation_timing = metadata.get("rank_allocation_sec") is not None
    allocation_sec = (
        float(metadata.get("rank_allocation_sec", 0.0) or 0.0)
        if has_explicit_allocation_timing
        else max(0.0, float(wall_sec) - calibration_sec)
    )
    initialization_sec = float(metadata.get("pseudoinverse_init_sec", 0.0) or 0.0)
    attributed_sec = calibration_sec + allocation_sec + initialization_sec
    return {
        "calibration_sec": calibration_sec,
        "allocation_sec": allocation_sec,
        "initialization_sec": initialization_sec,
        "unattributed_sec": max(0.0, float(wall_sec) - attributed_sec),
    }


def _optimizer_manifest(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    base_optimizer = getattr(optimizer, "optimizer", optimizer)
    param_groups = []
    for index, group in enumerate(base_optimizer.param_groups):
        params = list(group.get("params", []))
        param_groups.append(
            {
                "index": index,
                "lr": float(group.get("lr", 0.0)),
                "initial_lr": float(group.get("initial_lr", group.get("lr", 0.0))),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "num_tensors": len(params),
                "num_params": int(sum(param.numel() for param in params)),
                "group_name": group.get("group_name"),
            }
        )
    first_group = base_optimizer.param_groups[0] if base_optimizer.param_groups else {}
    betas = first_group.get("betas", (None, None))
    return {
        "name": type(base_optimizer).__name__,
        "betas": [float(betas[0]), float(betas[1])] if betas[0] is not None else None,
        "eps": float(first_group.get("eps", 0.0)) if first_group else None,
        "param_groups": param_groups,
    }


def _optimizer_state_estimate_manifest(
    raw_model: torch.nn.Module,
    optimizer_manifest: dict[str, Any],
) -> dict[str, Any]:
    optimizer_name = optimizer_manifest.get("name")
    trainable_params = int(trainable_parameter_count(raw_model))
    adam_like = optimizer_name in {"Adam", "AdamW", "FusedAdam"}
    state_tensors_per_param = 2 if adam_like else None
    state_dtype = "float32" if adam_like else None
    state_bytes_per_param = 8 if adam_like else None
    estimated_state_bytes = trainable_params * state_bytes_per_param if state_bytes_per_param is not None else None
    return {
        "optimizer": optimizer_name,
        "trainable_params": trainable_params,
        "state_tensors_per_param": state_tensors_per_param,
        "state_dtype": state_dtype,
        "state_bytes_per_param": state_bytes_per_param,
        "estimated_state_bytes": estimated_state_bytes,
        "includes_master_params": False,
        "note": (
            "Adam-family estimate covers first and second moment tensors for trainable adapter "
            "parameters only; parameter storage is reported separately."
        ),
    }


def _revision_manifest(value: Any) -> dict[str, str]:
    if value is None or str(value).strip() == "":
        return {"revision": "UNRESOLVED", "status": "UNRESOLVED"}
    return {"revision": str(value), "status": "LOCKED"}


def _data_loading_manifest(
    *,
    total_train_records: int,
    per_rank_train_records: int,
    num_processes: int,
    local_rank: int,
    batch_size: int,
    grad_accum: int,
    max_steps: int,
) -> dict[str, Any]:
    dataloader_length_batches = math.ceil(per_rank_train_records / max(1, batch_size))
    global_batch_size = int(batch_size) * int(max(1, num_processes)) * int(grad_accum)
    sample_exposures = int(max_steps) * global_batch_size
    return {
        "sampler_type": "manual_stride_shard" if int(num_processes) > 1 else "single_process_full_dataset",
        "distributed_sampler_used": False,
        "shard_strategy": "records[local_rank::world_size]" if int(num_processes) > 1 else "none",
        "drop_last": False,
        "infinite_repeating_iterator": True,
        "last_accumulation_behavior": "full_accumulation_from_repeating_iterator",
        "total_train_records": int(total_train_records),
        "per_rank_train_records": int(per_rank_train_records),
        "local_rank": int(local_rank),
        "world_size": int(num_processes),
        "batch_size_per_process": int(batch_size),
        "gradient_accumulation_steps": int(grad_accum),
        "samples_per_optimizer_step_per_rank": int(batch_size) * int(grad_accum),
        "global_batch_size": global_batch_size,
        "configured_optimizer_steps": int(max_steps),
        "sample_exposures": sample_exposures,
        "unique_samples": int(total_train_records),
        "repeated_exposures": max(0, sample_exposures - int(total_train_records)),
        "sample_exposure_policy": "repeat_from_fixed_order_to_max_steps",
        "dataloader_length_batches": int(dataloader_length_batches),
        "optimizer_steps_source": "training_loop_global_step",
    }


def _adapter_dtype_name(model: torch.nn.Module) -> str | None:
    for name, value in model.state_dict().items():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            return str(value.dtype).replace("torch.", "")
    return None


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _model_directory_fingerprint(value: Any) -> dict[str, Any] | None:
    path = Path(str(value)).expanduser()
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = str(child.relative_to(path))
        size = int(child.stat().st_size)
        digest.update(relative.encode("utf-8"))
        digest.update(str(size).encode("ascii"))
        # Hash small metadata/tokenizer files; weight shards are represented by
        # stable relative name and size to avoid rereading many GB at startup.
        if size <= 10 * 1024 * 1024:
            child_hash = _file_sha256(child)
            if child_hash:
                digest.update(child_hash.encode("ascii"))
        file_count += 1
        total_size += size
    return {
        "path": str(path.resolve()),
        "fingerprint_sha256": digest.hexdigest(),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "weight_content_hash_policy": "filename_and_size; files<=10MiB include content sha256",
    }


def _jsonl_num_rows(path: Path) -> int | None:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return None


def _evaluation_artifacts_manifest(output_dir: Path) -> dict[str, dict[str, Any]]:
    specs = {
        "gsm8k_predictions": {
            "filename": "eval_predictions.jsonl",
            "required_fields": [
                "question",
                "gold_answer",
                "raw_prediction",
                "prediction",
                "pred_final",
                "gold_final",
                "metric",
                "score",
            ],
        },
        "humaneval_predictions": {
            "filename": "humaneval_predictions.jsonl",
            "required_fields": [
                "task_id",
                "raw_completion",
                "completion",
                "correct",
                "metric",
                "score",
                "pass_at_k_estimator",
            ],
        },
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        path = output_dir / str(spec["filename"])
        if not path.exists():
            continue
        artifacts[name] = {
            "path": str(path),
            "sha256": _file_sha256(path),
            "num_rows": _jsonl_num_rows(path),
            "format": "jsonl",
            "required_fields": list(spec["required_fields"]),
        }
    return artifacts


def _checkpoint_artifacts_manifest(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / "masked_lora_state.pt"
    if not path.exists():
        return {}
    return {
        "adapter_checkpoint": {
            "path": str(path),
            "sha256": _file_sha256(path),
            "size_bytes": int(path.stat().st_size),
            "format": "torch_state_dict",
            "contains_base_model_weights": False,
            "checkpoint_selection_rule": "final_checkpoint_only",
        }
    }


def _method_artifacts_manifest(output_dir: Path) -> dict[str, dict[str, Any]]:
    specs = {
        "budget": "budget.json",
        "rank_dict": "rank_dict.json",
        "rank_allocation_initial": "rank_allocation_initial.json",
        "rank_allocation_final": "rank_allocation_final.json",
        "init_summary": "init_summary.json",
        "diagnostics": "diagnostics.json",
        "physical_utility": "physical_utility.json",
        "normalization_stats": "normalization_stats.json",
        "quota_stats": "quota_stats.json",
        "kappa_calibration": "kappa_calibration.json",
        "taxonomy_stats": "taxonomy_stats.json",
        "coverage_trace": "coverage_trace.json",
        "procurement_trace": "procurement_trace.json",
        "adalora_schedule": "adalora_schedule.json",
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, filename in specs.items():
        path = output_dir / filename
        if not path.exists():
            continue
        artifacts[name] = {
            "path": str(path),
            "sha256": _file_sha256(path),
            "size_bytes": int(path.stat().st_size),
            "format": "json",
        }
    return artifacts


def _run_artifacts_manifest(output_dir: Path) -> dict[str, dict[str, Any]]:
    specs = {
        "train_log": ("train_log.jsonl", "jsonl"),
        "eval_log": ("eval_log.jsonl", "jsonl"),
        "stage_metrics": ("stage_metrics.jsonl", "jsonl"),
        "metrics": ("metrics.json", "json"),
        "evaluation_protocol": ("evaluation_protocol.json", "json"),
        "run_summary": ("run_summary.md", "markdown"),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for name, (filename, file_format) in specs.items():
        path = output_dir / filename
        if not path.exists():
            continue
        record: dict[str, Any] = {
            "path": str(path),
            "sha256": _file_sha256(path),
            "size_bytes": int(path.stat().st_size),
            "format": file_format,
        }
        if file_format == "jsonl":
            record["num_rows"] = _jsonl_num_rows(path)
        artifacts[name] = record
    return artifacts


def _module_budget_manifest(
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
    initial_allocation: dict[str, int],
    final_allocation: dict[str, int],
) -> dict[str, Any]:
    modules: dict[str, dict[str, int]] = {}
    total_initial_params = 0
    total_final_params = 0
    for name in module_names:
        dims = module_dims[name]
        in_dim = int(dims.get("in_dim", dims.get("d_in", dims.get("in_features", 0))))
        out_dim = int(dims.get("out_dim", dims.get("d_out", dims.get("out_features", 0))))
        rank_cost = int(module_rank_cost(dims))
        initial_rank = int(initial_allocation.get(name, 0))
        final_rank = int(final_allocation.get(name, initial_rank))
        initial_params = initial_rank * rank_cost
        final_params = final_rank * rank_cost
        total_initial_params += initial_params
        total_final_params += final_params
        modules[name] = {
            "in_dim": in_dim,
            "out_dim": out_dim,
            "rank_cost": rank_cost,
            "initial_rank": initial_rank,
            "final_rank": final_rank,
            "initial_params": initial_params,
            "final_params": final_params,
        }
    return {
        "cost_formula": "rank * (d_in + d_out)",
        "modules": modules,
        "total_initial_params": int(total_initial_params),
        "total_final_params": int(total_final_params),
    }


def _write_run_manifest_and_summary(
    output_dir: Path,
    *,
    config: dict[str, Any],
    experiment_name: str,
    method: str,
    rank: int,
    num_processes: int,
    local_rank: int,
    batch_size: int,
    grad_accum: int,
    max_steps: int,
    global_step: int,
    warmup_steps: int,
    learning_rate: float,
    lr_decay_ratio: float,
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
    initial_allocation: dict[str, int],
    final_allocation: dict[str, int],
    final_budget: dict[str, Any],
    metrics: dict[str, Any],
    raw_model: Any,
    run_elapsed_sec: float,
    data_manifest: dict[str, Any],
    calibration_manifest: dict[str, Any],
    timing_manifest: dict[str, Any],
    optimizer_manifest: dict[str, Any],
    data_loading_manifest: dict[str, Any],
) -> None:
    lora_cfg = dict(config.get("lora", {}))
    training_cfg = dict(config.get("training", {}))
    model_cfg = dict(config.get("model", {}))
    calibration_cfg = dict(config.get("calibration", {}))
    pre_cfg = dict(config.get("preallocation", {}))
    global_batch_size = int(batch_size) * int(max(1, num_processes)) * int(grad_accum)
    cuda_peak_memory = None
    if torch.cuda.is_available():
        try:
            cuda_peak_memory = int(torch.cuda.max_memory_allocated())
        except RuntimeError:
            cuda_peak_memory = None
    resolved_config_path = output_dir / "config_resolved.yaml"
    base_seed = int(config.get("seed", 42))
    calibration_seed = int(calibration_cfg.get("seed", base_seed))
    sketch_seed_raw = pre_cfg.get("sketch_seed", base_seed)
    preallocation_sketch_seed = None if sketch_seed_raw is None else int(sketch_seed_raw)
    method_artifacts = _method_artifacts_manifest(output_dir)
    evaluation_artifacts = _evaluation_artifacts_manifest(output_dir)
    checkpoint_artifacts = _checkpoint_artifacts_manifest(output_dir)
    module_budget = _module_budget_manifest(module_names, module_dims, initial_allocation, final_allocation)
    optimizer_state_estimate = _optimizer_state_estimate_manifest(raw_model, optimizer_manifest)
    model_revision = _revision_manifest(model_cfg.get("revision"))
    tokenizer_revision = _revision_manifest(model_cfg.get("tokenizer_revision"))
    parameter_counts = {
        "requires_grad": int(trainable_parameter_count(raw_model)),
        "active_final": int(
            final_budget.get(
                "adalora_final_active_params",
                final_budget.get("actual_budget_paramcount", final_budget.get("actual_budget")),
            )
        ),
        "active_peak": int(
            final_budget.get(
                "adalora_peak_active_params",
                final_budget.get("actual_budget_paramcount", final_budget.get("actual_budget")),
            )
        ),
    }
    parameter_metrics = {
        "requires_grad_params": parameter_counts["requires_grad"],
        "peak_active_params": parameter_counts["active_peak"],
        "final_active_params": parameter_counts["active_final"],
        "budget_target": int(final_budget.get("target_budget_paramcount", final_budget.get("target_budget"))),
        "budget_actual": int(final_budget.get("actual_budget_paramcount", final_budget.get("actual_budget"))),
        "budget_error": int(final_budget.get("budget_error")),
    }
    manifest = {
        **_runtime_environment_manifest(num_processes, local_rank),
        "protocol_scope": str(config.get("runtime", {}).get("protocol_scope", "formal_e01")),
        "experiment_name": experiment_name,
        "method": method,
        "method_protocol": {
            "source": (
                "https://github.com/hhnqqq/MyTransformers"
                if method in FORMAL_GORA_METHODS
                else (
                    "https://github.com/QingruZhang/AdaLoRA"
                    if method == "adalora"
                    else "in-repository aligned trainer"
                )
            ),
            "official_commit": (
                OFFICIAL_GORA_COMMIT
                if method in FORMAL_GORA_METHODS
                else ("d10f5ebee16c478fa2f41a44a237b38e8c9b0338" if method == "adalora" else None)
            ),
            "effective_scaling": lora_cfg.get("scaling"),
            "protocol_conflict_resolution": (
                {
                    "warmup": "official script .03 (paper table .3 not used)",
                    "gradient_estimation": "1024 global training samples mapped from script N=32",
                    "initialization": "scale_by_lr=true, init_lr=.05",
                }
                if method in FORMAL_GORA_METHODS
                else None
            ),
        },
        "rank": int(rank),
        "seed": base_seed,
        "calibration_seed": calibration_seed,
        "seeds": {
            "base_seed": base_seed,
            "model_and_lora_init_seed": base_seed,
            "training_rng_seed": base_seed + int(local_rank),
            "calibration_seed": calibration_seed,
            "preallocation_sketch_seed": preallocation_sketch_seed,
            "local_rank": int(local_rank),
        },
        "config": {
            "resolved_config_path": str(resolved_config_path),
            "resolved_config_sha256": _file_sha256(resolved_config_path),
        },
        "model": {
            "name_or_path": model_cfg.get("name_or_path"),
            "type": model_cfg.get("type"),
            "model_torch_dtype": str(model_cfg.get("torch_dtype")),
            "tokenizer_revision": tokenizer_revision["revision"],
            "model_revision": model_revision["revision"],
            "tokenizer_revision_status": tokenizer_revision["status"],
            "model_revision_status": model_revision["status"],
            "attn_implementation_configured": model_cfg.get("attn_implementation"),
            "attn_implementation_effective": getattr(getattr(raw_model, "config", None), "_attn_implementation", None),
            "local_directory_fingerprint": _model_directory_fingerprint(model_cfg.get("name_or_path")),
        },
        "data": {
            "dataset_name": config.get("data", {}).get("dataset_name"),
            "train_path": config.get("data", {}).get("train_path"),
            "eval_path": config.get("data", {}).get("eval_path"),
            "train_limit": config.get("data", {}).get("train_limit"),
            "eval_limit": config.get("data", {}).get("eval_limit"),
            "max_length": config.get("data", {}).get("max_length"),
            **data_manifest,
        },
        "data_loading": data_loading_manifest,
        "calibration": {
            "num_samples_configured": calibration_cfg.get("num_samples"),
            "batch_size": calibration_cfg.get("batch_size"),
            "shuffle": bool(calibration_cfg.get("shuffle", False)),
            "group_sampling": calibration_cfg.get("group_sampling"),
            **calibration_manifest,
        },
        "training": {
            "batch_size_per_process": int(batch_size),
            "gradient_accumulation_steps": int(grad_accum),
            "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", False)),
            "learning_rate": float(learning_rate),
            "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
            "max_grad_norm": (
                None
                if training_cfg.get("max_grad_norm") is None
                else float(training_cfg.get("max_grad_norm"))
            ),
            "lr_decay_ratio": float(lr_decay_ratio),
            "optimizer_backend": training_cfg.get("optimizer_backend", "adamw"),
            "sample_exposure_policy": training_cfg.get(
                "sample_exposure_policy", "repeat_from_fixed_order_to_max_steps"
            ),
        },
        "optimizer": optimizer_manifest,
        "optimizer_state_estimate": optimizer_state_estimate,
        "scheduler": {
            "name": "cosine_with_warmup_and_floor",
            "warmup_steps": int(warmup_steps),
            "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
            "lr_decay_ratio": float(lr_decay_ratio),
            "num_training_steps": int(max_steps),
            "optimizer_steps_source": "training_loop_global_step",
            "auto_warmup_steps": int(training_cfg.get("auto_warmup_steps", 10)),
            "auto_warmup_rate": float(training_cfg.get("auto_warmup_rate", 0.05)),
        },
        "world_size": int(num_processes),
        "global_batch_size": global_batch_size,
        "optimizer_steps": int(global_step),
        "configured_max_steps": int(max_steps),
        "warmup_steps": int(warmup_steps),
        "precision": {
            "model_torch_dtype": str(model_cfg.get("torch_dtype")),
            "adapter_dtype": _adapter_dtype_name(raw_model) or str(lora_cfg.get("adapter_dtype", "float32")),
        },
        "lora": {
            "target_modules": list(lora_cfg.get("target_modules", [])),
            "dropout": float(lora_cfg.get("dropout", 0.0)),
            "alpha": lora_cfg.get("alpha"),
            "injection": lora_cfg.get("injection"),
            "scaling": lora_cfg.get("scaling"),
            "effective_scaling_values": sorted(
                {
                    float(module.scaling)
                    for module in raw_model.modules()
                    if hasattr(module, "scaling") and hasattr(module, "lora_A")
                }
            ),
        },
        "evaluation": {
            "batch_size": int(config.get("evaluation", {}).get("batch_size", 4)),
            "decoding": "greedy",
            "padding_side": "left",
            "final_checkpoint_only": True,
        },
        "preallocation": {
            "allocation_method": pre_cfg.get("allocation_method"),
            "top_k_atoms": pre_cfg.get("top_k_atoms"),
            "sketch_dim": pre_cfg.get("sketch_dim"),
            "rho": pre_cfg.get("rho"),
            "solver": pre_cfg.get("solver"),
        },
        "gora": (
            {
                "official_commit": config.get("gora", {}).get("official_commit"),
                "gradient_collection": config.get("gora", {}).get("gradient_collection"),
                "gradient_offload_device": config.get("gora", {}).get("gradient_offload_device"),
                "gradient_accumulation_dtype": config.get("gora", {}).get("gradient_accumulation_dtype"),
                "clear_gradient_after_offload": config.get("gora", {}).get("clear_gradient_after_offload"),
                "b_lr_multiplier": config.get("gora", {}).get("b_lr_multiplier"),
                "strict_budget_repair": config.get("gora", {}).get("strict_budget_repair"),
            }
            if method in FORMAL_GORA_METHODS
            else None
        ),
        "target_modules_resolved": list(module_names),
        "module_budget": module_budget,
        "rank_allocation_initial": initial_allocation,
        "rank_allocation_final": final_allocation,
        "budget": final_budget,
        "metrics": metrics,
        "parameter_metrics": parameter_metrics,
        "method_artifacts": method_artifacts,
        "run_artifacts": {},
        "evaluation_artifacts": evaluation_artifacts,
        "checkpoint_artifacts": checkpoint_artifacts,
        "parameter_counts": parameter_counts,
        "timing": {
            "run_elapsed_sec": float(run_elapsed_sec),
            **timing_manifest,
        },
        "hardware": {
            "cuda_peak_memory_allocated_bytes": cuda_peak_memory,
            "flash_attention_2_required": bool(config.get("runtime", {}).get("require_flash_attention_2", False)),
            "flash_attention_2_configured": model_cfg.get("attn_implementation") == "flash_attention_2",
        },
    }

    lines = [
        f"# Run Summary: {experiment_name}",
        "",
        f"- method: `{method}`",
        f"- rank: `{rank}`",
        f"- world_size: `{num_processes}`",
        f"- global_batch_size: `{global_batch_size}`",
        f"- optimizer_steps: `{global_step}`",
        f"- scheduler: `{manifest['scheduler']['name']}`",
        f"- warmup_steps: `{manifest['scheduler']['warmup_steps']}`",
        f"- warmup_ratio: `{manifest['scheduler']['warmup_ratio']}`",
        f"- lr_decay_ratio: `{manifest['scheduler']['lr_decay_ratio']}`",
        f"- resolved_config_sha256: `{manifest['config']['resolved_config_sha256']}`",
        f"- git_commit: `{manifest['source_control']['git_commit']}`",
        f"- git_dirty: `{manifest['source_control']['git_dirty']}`",
        f"- command_cwd: `{manifest['command']['cwd']}`",
        f"- dependency_versions: `torch={manifest['dependency_versions']['torch']}, transformers={manifest['dependency_versions']['transformers']}, accelerate={manifest['dependency_versions']['accelerate']}`",
        f"- base_seed: `{manifest['seeds']['base_seed']}`",
        f"- calibration_seed: `{manifest['seeds']['calibration_seed']}`",
        f"- preallocation_sketch_seed: `{manifest['seeds']['preallocation_sketch_seed']}`",
        f"- target_budget: `{final_budget.get('target_budget')}`",
        f"- actual_budget: `{final_budget.get('actual_budget')}`",
        f"- budget_error: `{final_budget.get('budget_error')}`",
        f"- requires_grad: `{manifest['parameter_counts']['requires_grad']}`",
        f"- optimizer_state_estimate: `estimated_state_bytes={optimizer_state_estimate['estimated_state_bytes']}`",
        f"- active_final: `{manifest['parameter_counts']['active_final']}`",
        f"- active_peak: `{manifest['parameter_counts']['active_peak']}`",
        f"- module_budget: `total_final_params={module_budget['total_final_params']}`",
        f"- method_artifacts: `{', '.join(method_artifacts) if method_artifacts else 'none'}`",
        f"- run_artifacts: `train_log, eval_log, metrics, evaluation_protocol, run_summary`",
        f"- checkpoint_artifacts: `{', '.join(checkpoint_artifacts) if checkpoint_artifacts else 'none'}`",
        f"- evaluation_artifacts: `{', '.join(evaluation_artifacts) if evaluation_artifacts else 'none'}`",
        f"- training_sec: `{manifest['timing'].get('training_sec')}`",
        f"- train_tokens_per_sec: `{manifest['timing'].get('train_tokens_per_sec')}`",
        f"- final_metric: `{metrics.get('final_metric')}`",
        f"- gpu_test_executed: `{torch.cuda.is_available()}`",
        "",
        "This summary is generated from the same payload as `run_manifest.json`.",
        "",
    ]
    (output_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")
    manifest["run_artifacts"] = _run_artifacts_manifest(output_dir)
    write_json(output_dir / "run_manifest.json", manifest)


def _evaluation_protocol_manifest(config: dict[str, Any]) -> dict[str, Any]:
    evaluation_cfg = dict(config.get("evaluation", {}))
    mtbench_cfg = dict(evaluation_cfg.get("mtbench_local", {}))
    return {
        "checkpoint_selection": {
            "rule": "final_checkpoint_only",
            "uses_validation_metric_for_selection": False,
            "uses_test_metric_for_selection": False,
            "max_evaluations_per_checkpoint": 1,
            "artifact_note": (
                "The trainer evaluates the final in-memory adapter state once; "
                "no validation/test metric is used to choose among checkpoints."
            ),
        },
        "gsm8k": {
            "metric": "exact_match",
            "decoding": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": int(evaluation_cfg.get("generation_max_new_tokens", 256)),
            "stop_sequences": evaluation_cfg.get("stop_sequences", DEFAULT_GSM8K_STOP_SEQUENCES),
            "answer_extraction": evaluation_cfg.get("answer_extraction", "strict_then_flexible"),
        },
        "humaneval": {
            "metric": "pass@1",
            "decoding": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": int(evaluation_cfg.get("humaneval_max_new_tokens", 256)),
            "pass_at_k_estimator": "official_unbiased",
            "num_samples_per_task": int(evaluation_cfg.get("humaneval_num_samples_per_task", 1)),
        },
        "mtbench_local": {
            "status": "CONFIGURED_NOT_EXECUTED",
            "enabled": bool(mtbench_cfg.get("enabled", False)),
            "judge_model": mtbench_cfg.get("judge_model"),
            "judge_prompt_version": mtbench_cfg.get("judge_prompt_version"),
            "conversation_template": mtbench_cfg.get("conversation_template"),
            "temperature": float(mtbench_cfg.get("temperature", 0.0)),
            "seed": int(mtbench_cfg.get("seed", 0)),
            "swap_positions": bool(mtbench_cfg.get("swap_positions", True)),
            "max_retries": int(mtbench_cfg.get("max_retries", 2)),
            "artifact_note": (
                "MTBench-local judge execution is external to the trainer path; use "
                "scripts/mtbench_local_judge.py on archived answer sets to produce "
                "mtbench_local_judgments.jsonl and mtbench_local_metrics.json."
            ),
        },
    }


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _tokenize_with_optional_cache(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_value = config.get("data", {}).get("token_cache_dir")
    if not cache_value:
        return tokenize_records(records, tokenizer, max_length), {
            "cache_enabled": False,
            "cache_hit": False,
            "sample_order_hash": dataset_hash(records),
        }
    project_root = Path(config.get("_project_root", Path.cwd())).resolve()
    cache_dir = _resolve_path(project_root, cache_value)
    tokenized, metadata = tokenize_records_cached(records, tokenizer, max_length, cache_dir)
    return tokenized, {"cache_enabled": True, **metadata}


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _stratified_group_sample(
    train_records: list[dict[str, Any]], calibration_limit: int, seed: int
) -> list[dict[str, Any]]:
    """calibration.group_sampling=balanced (§6.5 mixed math+code): draw
    calibration_limit // num_groups samples from each group present in
    train_records (tagged "group" by data.py::tokenize_records, itself sourced from
    load_multi_source_train_records's "_group" tag on the pre-tokenization record --
    `train_records` here are already-tokenized SFT examples, so the field is "group",
    not "_group"), instead of a plain random/prefix sample from the concatenated set --
    avoids an accidental math/code skew if the source files differ in size after
    limiting. Deterministic given `seed`, independent of `num_processes` (calibration
    only ever runs on rank-0 and gets broadcast).
    """
    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in train_records:
        by_group.setdefault(str(record.get("group", "unlabeled")), []).append(record)
    groups = sorted(by_group)
    if len(groups) < 2:
        rng = random.Random(seed)
        indices = rng.sample(range(len(train_records)), min(calibration_limit, len(train_records)))
        return [train_records[index] for index in indices]
    per_group = max(1, calibration_limit // len(groups))
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for group in groups:
        pool = by_group[group]
        count = min(per_group, len(pool))
        indices = rng.sample(range(len(pool)), count)
        selected.extend(pool[index] for index in indices)
    rng.shuffle(selected)
    return selected[:calibration_limit]


def _build_calibration_batches(
    train_records: list[dict[str, Any]],
    collator: SFTCollator,
    input_device: torch.device,
    calibration_cfg: dict[str, Any],
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Returns (batches, selected) -- `selected` is the raw record list in the exact
    same order as `batches`, so callers can derive real per-sample calibration group
    labels (record["_group"]) for the allocator without re-deriving the sampling logic.
    """
    calibration_limit = int(calibration_cfg.get("num_samples", min(8, len(train_records))))
    calibration_batch_size = max(1, int(calibration_cfg.get("batch_size", 1)))
    if calibration_limit >= len(train_records):
        selected = list(train_records)
    elif str(calibration_cfg.get("group_sampling", "")) == "balanced":
        selected = _stratified_group_sample(train_records, calibration_limit, int(calibration_cfg.get("seed", 42)))
    elif bool(calibration_cfg.get("shuffle", False)):
        rng = random.Random(int(calibration_cfg.get("seed", 42)))
        indices = rng.sample(range(len(train_records)), calibration_limit)
        selected = [train_records[index] for index in indices]
    else:
        selected = train_records[:calibration_limit]
    batches = [
        _move_batch(collator(selected[start : start + calibration_batch_size]), input_device)
        for start in range(0, len(selected), calibration_batch_size)
    ]
    return batches, selected


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
    return _resolve_path(project_root, save_dir) / f"dico_v03_rank{rank}_seed{seed}.json"


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
        "direction_bank_path": payload.get("direction_bank_path"),
        "taxonomy_stats": payload.get("taxonomy_stats"),
        "coverage_trace": payload.get("coverage_trace"),
        "procurement_trace": payload.get("procurement_trace"),
        "kappa_calibration": payload.get("kappa_calibration"),
        "physical_utility": payload.get("physical_utility"),
        "normalized_utility": payload.get("normalized_utility"),
        "normalization_stats": payload.get("normalization_stats"),
        "module_quota": payload.get("module_quota"),
        "procurement_beta": payload.get("procurement_beta"),
        "r_min": payload.get("r_min"),
        "reserve_filled_ratio": payload.get("reserve_filled_ratio"),
        "balanced_fill_ratio": payload.get("balanced_fill_ratio"),
        "zero_rank_module_ratio": payload.get("zero_rank_module_ratio"),
        "budget_gap_ratio": payload.get("budget_gap_ratio"),
        "procurement_warning": payload.get("procurement_warning"),
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

    original_requires_grad = {parameter: parameter.requires_grad for parameter in model.parameters()}
    try:
        # CovRA needs activation/output gradients, not gradients for 8B frozen
        # base weights.  Input embeddings are configured to require gradients by
        # the model loader, so the activation graph remains available while this
        # removes the largest avoidable calibration-memory allocation.
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
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
    finally:
        for parameter, requires_grad in original_requires_grad.items():
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
    allocator.save(path, result)
    metadata = _preallocation_metadata_from_payload(result.to_dict(preallocation_path=str(path)), config, path, source="computed")
    metadata["base_parameters_frozen_during_calibration"] = True
    metadata.update(cache_diagnostics)
    return result.rank_allocation, metadata


def build_gora_bw_allocation(
    config: dict[str, Any],
    model: torch.nn.Module,
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
    calibration_batches: list[dict[str, torch.Tensor]],
) -> tuple[dict[str, int], dict[str, Any]]:
    modules = dict(model.named_modules())
    compute_device = torch.device(str(config.get("gora_bw", {}).get("compute_device", "cpu")))
    grad_states = {
        name: torch.zeros(
            int(module_dims[name]["out_dim"]),
            int(module_dims[name]["in_dim"]),
            dtype=torch.float32,
            device=compute_device,
        )
        for name in module_names
    }
    answer_only = bool(config.get("preallocation", {}).get("answer_only", True))
    module_chunk_size = int(config.get("gora_bw", {}).get("module_chunk_size", len(module_names)))

    for batch_index, batch in enumerate(calibration_batches, start=1):
        def update_gradient(name: str, _sample_idx: int, a_tokens: torch.Tensor, g_tokens: torch.Tensor) -> None:
            if a_tokens.numel() == 0:
                return
            grad_states[name] += g_tokens.T @ a_tokens

        _run_backward_and_stream(
            model,
            modules,
            module_names,
            batch,
            answer_only,
            compute_device,
            update_gradient,
            module_chunk_size=module_chunk_size,
            pass_name="gora_bw_grad_pass",
            batch_index=batch_index,
            total_batches=len(calibration_batches),
            progress_logging_steps=int(config.get("gora_bw", {}).get("progress_logging_steps", 1)),
        )

    weights = {name: modules[name].weight.detach().float().to(device="cpu") for name in module_names}
    grads = {name: grad_states[name].detach().float().to(device="cpu") for name in module_names}
    result = allocate_gora_bw(
        weights,
        grads,
        module_dims,
        r_ref=int(config.get("gora_bw", {}).get("r_ref", config.get("rank", 8))),
        eta=float(config.get("budget", {}).get("enforce_min_ratio", config.get("preallocation", {}).get("eta", 0.98))),
    )
    metadata = {
        "allocation_method": "gora_bw",
        "module_logs": [
            {
                "module_name": name,
                "advantage": result.advantages.get(name, 0.0),
                "final_rank": result.rank_dict[name],
                "rank_cost": module_rank_cost(module_dims[name]),
            }
            for name in module_names
        ],
        "procurement_trace": result.trace,
        "target_budget": result.target_budget,
        "actual_budget": result.realized_params,
        "target_budget_paramcount": result.target_budget,
        "actual_budget_paramcount": result.realized_params,
        "budget_ratio_paramcount": float(result.realized_params / result.target_budget) if result.target_budget else 0.0,
        "eta": float(config.get("preallocation", {}).get("eta", 0.98)),
    }
    return result.rank_dict, metadata


def build_gora_allocation(
    config: dict[str, Any],
    model: torch.nn.Module,
    module_names: list[str],
    module_dims: dict[str, dict[str, int]],
    calibration_batches: list[dict[str, torch.Tensor]],
    target_budget: int,
) -> tuple[dict[str, int], dict[str, Any], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Run the formal GoRA calibration/allocation/initialization path.

    The implementation follows the locked public code's direct weight-gradient
    path: one full backward per calibration batch, with no activation-gradient
    reconstruction or answer-only mask.
    """
    modules = dict(model.named_modules())
    gora_cfg = dict(config.get("gora", {}))
    collection = str(gora_cfg.get("gradient_collection", "official_weight_grad_hook"))
    if collection != "official_weight_grad_hook":
        raise ValueError(
            "Formal GoRA requires gora.gradient_collection=official_weight_grad_hook; "
            "use the legacy gora_bw method for activation-gradient reconstruction."
        )
    gradient_collection_started_at = time.perf_counter()
    average_gradients, gradient_metadata = collect_average_weight_gradients(
        model,
        module_names,
        calibration_batches,
        offload_device=str(gora_cfg.get("gradient_offload_device", "cpu")),
        accumulation_dtype=str(gora_cfg.get("gradient_accumulation_dtype", "float32")),
    )
    importances: dict[str, float] = {}
    for name in module_names:
        weight = modules[name].weight.detach()
        gradient = average_gradients[name].to(device=weight.device, dtype=torch.float32)
        importances[name] = float(compute_gora_importance(weight, gradient).item())
        del gradient
    gradient_collection_sec = time.perf_counter() - gradient_collection_started_at
    rank_allocation_started_at = time.perf_counter()
    allocation = allocate_gora_ranks(
        importances,
        module_dims,
        r_ref=int(gora_cfg.get("r_ref", config.get("rank", 8))),
        r_min=int(gora_cfg.get("r_min", 4)),
        r_max=int(gora_cfg.get("r_max", 32)),
        rounding=str(gora_cfg.get("rounding", "moderate")),
    )
    if str(config.get("method")) == "gora_bm":
        allocation = strict_budget_repair(
            allocation,
            importances,
            module_dims,
            target_budget,
            r_min=int(gora_cfg.get("r_min", 4)),
            r_max=int(gora_cfg.get("r_max", 32)),
        )
    rank_allocation_sec = time.perf_counter() - rank_allocation_started_at
    init_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    alpha = float(config.get("lora", {}).get("alpha", 16.0))
    init_lr = float(gora_cfg.get("init_lr", 0.05))
    pseudoinverse_started_at = time.perf_counter()
    for name in module_names:
        rank = int(allocation[name])
        in_features = int(module_dims[name]["in_dim"])
        module_device = modules[name].weight.device
        A = torch.empty(rank, in_features, dtype=torch.float32, device=module_device)
        torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        gradient = average_gradients.pop(name).to(device=module_device, dtype=torch.float32)
        B = gora_pseudoinverse_init(gradient, A)
        if bool(gora_cfg.get("scale_by_lr", True)):
            B = scale_gora_b_initialization(
                B,
                rank=rank,
                in_features=in_features,
                alpha=alpha,
                init_lr=init_lr,
            )
        init_tensors[name] = (A, B)
        del gradient
    pseudoinverse_init_sec = time.perf_counter() - pseudoinverse_started_at
    actual_budget = compute_total_lora_params(allocation, module_dims)
    metadata = {
        "allocation_method": str(config.get("method")),
        "source": "GoRA official public code",
        "official_commit": OFFICIAL_GORA_COMMIT,
        "gradient_estimation_samples": int(config.get("calibration", {}).get("num_samples", 0)),
        "observed_calibration_samples": int(gradient_metadata["observed_samples"]),
        "gradient_estimation_steps": int(gradient_metadata["backward_passes"]),
        "gradient_collection_sec": float(gradient_collection_sec),
        "rank_allocation_sec": float(rank_allocation_sec),
        "pseudoinverse_init_sec": float(pseudoinverse_init_sec),
        **gradient_metadata,
        "importance_formula": "mean(abs(W * G_avg))",
        "aggregation": str(gora_cfg.get("aggregation", "union_mean")),
        "rounding": str(gora_cfg.get("rounding", "moderate")),
        "strict_budget_repair": str(config.get("method")) == "gora_bm",
        "target_budget_paramcount": int(target_budget),
        "actual_budget_paramcount": int(actual_budget),
        "budget_ratio_paramcount": float(actual_budget / target_budget) if target_budget else 0.0,
        "module_logs": [
            {
                "module_name": name,
                "importance": importances[name],
                "final_rank": allocation[name],
                "rank_cost": module_rank_cost(module_dims[name]),
            }
            for name in module_names
        ],
    }
    return allocation, metadata, init_tensors


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
    train_raw = order_records(
        train_raw,
        shuffle=bool(data_cfg.get("shuffle", False)),
        dataset_seed=int(data_cfg.get("dataset_seed", 42)),
    )
    max_length = int(data_cfg.get("max_length", 512))
    train_records, _token_cache_metadata = _tokenize_with_optional_cache(
        config, train_raw, tokenizer, max_length
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or getattr(tokenizer, "eos_token_id", 0) or 0
    collator = SFTCollator(pad_token_id)

    calibration_batches, calibration_selected = _build_calibration_batches(
        train_records,
        collator,
        input_device,
        config.get("calibration", {}),
    )
    # §6.5 mixed math+code: real per-sample task-group labels (record["group"], set by
    # data.py::tokenize_records from load_multi_source_train_records's "_group" tag),
    # in the exact same order as calibration_batches, so the allocator's group-split
    # path (candidates.py) can use real labels instead of falling back to pseudo-groups.
    # build_sft_example always sets "group" (default "math" for every single-source
    # config, unchanged behavior), so only treat it as a real signal when the sampled
    # calibration set actually spans >1 distinct group -- otherwise every ordinary
    # single-source run would spuriously look "configured" with one degenerate group
    # instead of falling through to real pseudo-group construction as before.
    calibration_group_labels = [row.get("group") for row in calibration_selected]
    if len(set(calibration_group_labels)) > 1:
        config.setdefault("data", {})["group_labels"] = calibration_group_labels

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
        calibration_selected.clear()
        del calibration_batches, calibration_selected
        _release_torch_memory()
    return {
        "rank_allocation": rank_allocation,
        "preallocation": metadata,
        "target_budget": budget_info.target_budget,
        "module_names": module_names,
        "module_dims": module_dims,
    }


def save_masked_lora_state(path: Path, model: torch.nn.Module) -> None:
    save_lora_state(path, model)


def train(config: dict[str, Any]) -> dict[str, Any]:
    # -- Distributed setup (Accelerate DDP or single-process fallback) --
    if _ACCELERATE_AVAILABLE and not config.get("_disable_accelerate", False):
        # Final accuracy eval (GSM8K/HumanEval generation) runs on rank-0 only and can take
        # far longer than NCCL's default 10-minute collective timeout (e.g. ~49min for the
        # full 1319-example GSM8K test set at ~2.2s/sample) -- other ranks sit idle at the
        # `wait_for_everyone()` barrier after training and get killed by the watchdog before
        # rank-0 finishes. Raise the process-group timeout to cover that, not just NCCL's
        # collective-op default which is tuned for uniform per-step work across ranks.
        ddp_timeout_minutes = float(config.get("training", {}).get("ddp_timeout_minutes", 180))
        accelerator = Accelerator(
            mixed_precision="no",  # model already loaded in bfloat16; no additional mixed-precision cast
            gradient_accumulation_steps=max(1, int(config.get("training", {}).get("gradient_accumulation_steps", 1))),
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(minutes=ddp_timeout_minutes))],
        )
    else:
        accelerator = None
    is_main = (accelerator is None) or accelerator.is_main_process
    num_processes = 1 if accelerator is None else accelerator.num_processes
    local_rank = 0 if accelerator is None else accelerator.local_process_index

    run_started_at = time.perf_counter()
    timing_manifest: dict[str, Any] = {
        "calibration_sec": 0.0,
        "allocation_sec": 0.0,
        "initialization_sec": 0.0,
        "training_sec": 0.0,
        "train_tokens": 0,
        "train_tokens_per_sec": 0.0,
    }
    project_root = Path(config.get("_project_root", Path.cwd())).resolve()
    experiment_name = config.get("experiment_name", f"{config['method']}_r{config['rank']}")
    output_root = _resolve_path(project_root, config.get("project", {}).get("output_dir", "outputs"))
    output_dir = ensure_dir(output_root / experiment_name)
    if is_main and (output_dir / "run_manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite completed run: {output_dir}. Use a new experiment/output name."
        )
    stage_metrics = StageMetricsRecorder(output_dir / "stage_metrics.jsonl", enabled=is_main)
    if is_main:
        (output_dir / "stage_metrics.jsonl").write_text("", encoding="utf-8")
    if is_main:
        save_yaml(output_dir / "config_resolved.yaml", config)
        write_json(output_dir / "evaluation_protocol.json", _evaluation_protocol_manifest(config))
    LOGGER.info("experiment_start experiment=%s output_dir=%s rank=%d/%d", experiment_name, output_dir, local_rank, num_processes)

    # seed: use the same base seed (no rank offset) through model load + LoRA injection so
    # every rank independently derives IDENTICAL initial LoRA weights in the legacy
    # (non direction-anchored) init path, which draws from the global torch RNG. The
    # rank-offset seed is applied later (right before the training loop) so that
    # rank-local stochastic ops (e.g. dropout) can still decorrelate across replicas.
    base_seed = int(config.get("seed", 42))
    set_seed(base_seed)

    # model.device_map="auto" does single-process model-parallel sharding across every
    # visible GPU, which conflicts with DDP (each of num_processes ranks would try to shard
    # the SAME model across all GPUs simultaneously). Under multi-process DDP, ignore it and
    # give each rank a full model replica on its own assigned device instead.
    ddp_multi_process = accelerator is not None and num_processes > 1
    model_load_config = config
    if ddp_multi_process and config.get("model", {}).get("device_map"):
        LOGGER.warning(
            "model.device_map=%r is incompatible with %d-process DDP and will be ignored; "
            "each rank loads a full model replica onto its own device (%s) instead.",
            config["model"]["device_map"], num_processes, accelerator.device,
        )
        model_load_config = dict(config)
        model_load_config["model"] = {k: v for k, v in config.get("model", {}).items() if k != "device_map"}
    model_load_stage = stage_metrics.begin("model_load")
    tokenizer, model = load_tokenizer_and_model(model_load_config)
    stage_metrics.end(
        model_load_stage,
        details={"model": str(config.get("model", {}).get("name_or_path"))},
    )

    if ddp_multi_process:
        # Do NOT manually move model to GPU here. accelerator.prepare() will handle
        # device placement (model.to(accelerator.device) + DDP wrapping) atomically.
        # Manual .to() before prepare() causes device_ids mismatch → NCCL ALLGATHER hang.
        placement_device = torch.device("cpu") if config.get("model", {}).get("type") == "tiny" else torch.device("cpu")
        input_device = torch.device("cpu")  # updated after accelerator.prepare()
    else:
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

    tokenization_stage = stage_metrics.begin("data_loading_and_tokenization")
    train_raw, eval_raw = load_raw_datasets(config)
    data_cfg = config.get("data", {})
    train_raw = limit_records(train_raw, data_cfg.get("train_limit"))
    train_raw = order_records(
        train_raw,
        shuffle=bool(data_cfg.get("shuffle", False)),
        dataset_seed=int(data_cfg.get("dataset_seed", 42)),
    )
    eval_raw = limit_records(eval_raw, data_cfg.get("eval_limit"))
    data_manifest = {
        "train_count": len(train_raw),
        "eval_count": len(eval_raw),
        "train_hash": dataset_hash(train_raw),
        "eval_hash": dataset_hash(eval_raw),
        "train_unique_count": len({stable_record_hash(record) for record in train_raw}),
        "shuffle": bool(data_cfg.get("shuffle", False)),
        "dataset_seed": int(data_cfg.get("dataset_seed", 42)),
        "train_membership_hash": hashlib.sha256(
            "\n".join(sorted(stable_record_hash(record) for record in train_raw)).encode("utf-8")
        ).hexdigest(),
        "prompt_template_sha256": hashlib.sha256(
            format_prompt("__QUESTION__", group="math").encode("utf-8")
        ).hexdigest(),
        "prompt_example": format_prompt(str(train_raw[0].get("question", "")), group="math")
        if train_raw
        else None,
    }
    max_length = int(data_cfg.get("max_length", 512))
    all_train_records, train_token_cache = _tokenize_with_optional_cache(
        config, train_raw, tokenizer, max_length
    )
    eval_records, eval_token_cache = _tokenize_with_optional_cache(
        config, eval_raw, tokenizer, max_length
    )
    data_manifest["token_cache"] = {
        "train": train_token_cache,
        "eval": eval_token_cache,
    }
    stage_metrics.end(
        tokenization_stage,
        details={
            "train_records": len(all_train_records),
            "eval_records": len(eval_records),
            "train_cache_hit": bool(train_token_cache.get("cache_hit")),
            "eval_cache_hit": bool(eval_token_cache.get("cache_hit")),
        },
    )
    # DDP per-rank data shard: each process takes a non-overlapping slice.
    # Together the slices cover the full dataset (GoRA's dp_rank::num_dp_ranks pattern).
    train_records = all_train_records[local_rank::num_processes]
    if not train_records:
        raise RuntimeError(
            f"Rank {local_rank}: no training records assigned (total={len(all_train_records)}, "
            f"num_processes={num_processes}). Ensure train_limit > num_processes."
        )
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or getattr(tokenizer, "eos_token_id", 0) or 0
    collator = SFTCollator(pad_token_id)
    LOGGER.info(
        "data_loaded experiment=%s rank=%d/%d train_records=%d eval_records=%d max_length=%d",
        experiment_name,
        local_rank,
        num_processes,
        len(train_records),
        len(eval_records),
        max_length,
    )

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 1))

    rank = int(config["rank"])
    method = config["method"]
    lora_max_mult = float(config.get("lora", {}).get("max_rank_multiplier", 2.0))
    pre_max_mult = float(config.get("preallocation", {}).get("r_max_multiplier", 2.0))

    if method in PREALLOC_METHODS and lora_max_mult < pre_max_mult:
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
    preallocation_metadata = None
    preallocation = None
    adalora_controller: AdaLoRAController | None = None
    adalora_config: AdaLoRAConfig | None = None
    adalora_target_allocation: dict[str, int] | None = None
    method_init_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    calibration_manifest: dict[str, Any] = {
        "num_selected_samples": 0,
        "sample_ids": [],
        "sample_hashes": [],
        "sample_indices": [],
        "selection_hash": hashlib.sha256(b"").hexdigest(),
    }
    lora_downscale_metadata: dict[str, Any] = {
        "lora_baseline_downscaled": False,
        "lora_baseline_target_ratio": None,
        "lora_baseline_min_ratio": None,
        "lora_baseline_actual_ratio": None,
        "lora_downscale_details": [],
        "lora_downscale_interval_pass": None,
    }
    allocation_stage = stage_metrics.begin("calibration_and_rank_allocation")
    if method in {"lora", "rs_lora"}:
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
    elif method == "adalora":
        adalora_cfg = dict(config.get("adalora", {}))
        adalora_config = AdaLoRAConfig(
            init_rank=int(adalora_cfg.get("init_rank", math.ceil(rank * 1.5))),
            target_rank=int(adalora_cfg.get("target_rank", rank)),
            tinit=int(adalora_cfg.get("tinit", adalora_cfg.get("ti", 150))),
            tfinal=int(adalora_cfg.get("tfinal", adalora_cfg.get("tf", 900))),
            delta_t=int(adalora_cfg.get("deltaT", adalora_cfg.get("delta_t", adalora_cfg.get("update_interval", 1)))),
            total_steps=int(training_cfg.get("max_steps", 1000)),
            beta1=float(adalora_cfg.get("beta1", 0.85)),
            beta2=float(adalora_cfg.get("beta2", 0.85)),
            orth_reg_weight=float(adalora_cfg.get("orth_reg_weight", 0.5)),
        )
        if adalora_config.init_rank > max_rank:
            raise ValueError(
                f"AdaLoRA init_rank ({adalora_config.init_rank}) exceeds lora max_rank ({max_rank}); "
                "increase lora.max_rank_multiplier or lower adalora.init_rank."
            )
        initial_allocation = uniform_allocation(adalora_config.init_rank, module_names)
        adalora_target_allocation = uniform_allocation(adalora_config.target_rank, module_names)
        preallocation_metadata = {
            "allocation_method": "adalora",
            "adalora": adalora_config.to_dict(),
            "target_budget_paramcount": int(target_budget),
            "actual_budget_paramcount": compute_total_lora_params(adalora_target_allocation, module_dims),
            "peak_budget_paramcount": compute_total_lora_params(initial_allocation, module_dims),
            "artifact_note": "Formal AdaLoRA A/E/B implementation with global budget pruning.",
        }
    elif method in PREALLOC_METHODS:
        # Preallocation runs only on rank-0 (requires backward pass; must precede DDP wrapping).
        # Result is broadcast to all other ranks.  Mirrors GoRA: prepare_lora() before init_distributed_model().
        if is_main:
            # In DDP mode the model is still on CPU at this point (it will be moved to GPU
            # by accelerator.prepare() later). For calibration we temporarily move it to
            # GPU-0 to avoid extremely slow CPU forward/backward passes, then move it back.
            if ddp_multi_process:
                _calib_device = accelerator.device
                model.to(_calib_device)
                _calib_input_device = _calib_device
            else:
                _calib_input_device = input_device
            # Calibration must sample from the full dataset, not this rank's DDP shard,
            # so rank allocation is invariant to num_processes (methodology requires
            # calibration/SVD statistics to be training-scale-independent).
            calibration_batches, calibration_selected = _build_calibration_batches(
                all_train_records,
                collator,
                _calib_input_device,
                config.get("calibration", {}),
            )
            # §6.5 mixed math+code: see the analogous block in compute_preallocation()
            # above for why the `len(set(...)) > 1` guard is required (build_sft_example
            # always tags a "group", default "math", for every single-source config).
            calibration_group_labels = [row.get("group") for row in calibration_selected]
            if len(set(calibration_group_labels)) > 1:
                config.setdefault("data", {})["group_labels"] = calibration_group_labels
            calibration_manifest = _calibration_selection_manifest(calibration_selected)
            preallocation_started_at = time.perf_counter()
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
                _alloc, _meta = load_or_build_preallocation(
                    config,
                    model,
                    tokenizer,
                    module_names,
                    module_dims,
                    calibration_batches,
                    target_budget,
                    project_root,
                )
                timing_manifest.update(
                    _preallocation_timing_manifest(_meta, time.perf_counter() - preallocation_started_at)
                )
            finally:
                calibration_batches.clear()
                calibration_selected.clear()
                del calibration_batches, calibration_selected
                _release_torch_memory()
                LOGGER.info("calibration_memory_released experiment=%s", experiment_name)
            # Move model back to CPU so accelerator.prepare() can place it correctly
            if ddp_multi_process:
                model.to("cpu")
                _release_torch_memory()
        else:
            _alloc, _meta = None, None
        if accelerator is not None:
            [_alloc, _meta] = broadcast_object_list([_alloc, _meta])
        initial_allocation, preallocation_metadata = _alloc, _meta
        preallocation = dict(initial_allocation)
    elif method == "gora_bw":
        if is_main:
            # In DDP mode, temporarily move model to GPU for calibration.
            if ddp_multi_process:
                _calib_device = accelerator.device
                model.to(_calib_device)
                _calib_input_device = _calib_device
            else:
                _calib_input_device = input_device
            # Calibration must sample from the full dataset, not this rank's DDP shard,
            # so rank allocation is invariant to num_processes.
            # GoRA-BW is a separate baseline protocol -- deliberately not wired into the
            # §6.5 group_labels mechanism above (`_selected` intentionally unused).
            calibration_batches, _calibration_selected = _build_calibration_batches(
                all_train_records,
                collator,
                _calib_input_device,
                config.get("calibration", {}),
            )
            calibration_manifest = _calibration_selection_manifest(_calibration_selected)
            preallocation_started_at = time.perf_counter()
            try:
                _alloc, _meta = build_gora_bw_allocation(
                    config,
                    model,
                    module_names,
                    module_dims,
                    calibration_batches,
                )
                timing_manifest.update(
                    _preallocation_timing_manifest(_meta, time.perf_counter() - preallocation_started_at)
                )
            finally:
                calibration_batches.clear()
                del calibration_batches, _calibration_selected
                _release_torch_memory()
                LOGGER.info("gora_bw_calibration_memory_released experiment=%s", experiment_name)
            # Move model back to CPU so accelerator.prepare() can place it correctly
            if ddp_multi_process:
                model.to("cpu")
                _release_torch_memory()
        else:
            _alloc, _meta = None, None
        if accelerator is not None:
            [_alloc, _meta] = broadcast_object_list([_alloc, _meta])
        initial_allocation, preallocation_metadata = _alloc, _meta
        preallocation = dict(initial_allocation)
    elif method in FORMAL_GORA_METHODS:
        if is_main:
            if ddp_multi_process:
                _calib_device = accelerator.device
                model.to(_calib_device)
                _calib_input_device = _calib_device
            else:
                _calib_input_device = input_device
            calibration_batches, calibration_selected = _build_calibration_batches(
                all_train_records,
                collator,
                _calib_input_device,
                config.get("calibration", {}),
            )
            calibration_manifest = _calibration_selection_manifest(calibration_selected)
            preallocation_started_at = time.perf_counter()
            try:
                _alloc, _meta, _method_init = build_gora_allocation(
                    config,
                    model,
                    module_names,
                    module_dims,
                    calibration_batches,
                    target_budget,
                )
                timing_manifest.update(
                    _preallocation_timing_manifest(_meta, time.perf_counter() - preallocation_started_at)
                )
            finally:
                calibration_batches.clear()
                calibration_selected.clear()
                del calibration_batches, calibration_selected
                _release_torch_memory()
                LOGGER.info("gora_calibration_memory_released experiment=%s", experiment_name)
            if ddp_multi_process:
                model.to("cpu")
                _release_torch_memory()
        else:
            _alloc, _meta, _method_init = None, None, None
        if accelerator is not None:
            [_alloc, _meta, _method_init] = broadcast_object_list([_alloc, _meta, _method_init])
        initial_allocation, preallocation_metadata = _alloc, _meta
        method_init_tensors = _method_init or {}
        preallocation = dict(initial_allocation)
    else:
        raise ValueError(f"Unsupported method: {method}")
    stage_metrics.end(
        allocation_stage,
        details={
            "method": str(method),
            "calibration_samples": int(calibration_manifest.get("num_selected_samples", 0)),
            "backward_passes": (preallocation_metadata or {}).get("backward_passes"),
        },
    )
    if preallocation_metadata:
        for key, stage_name in (
            ("sketch_pass_sec", "calibration_sketch_pass"),
            ("basis_pass_sec", "candidate_svd_and_direction_recovery"),
            ("profile_pass_sec", "response_profile_construction"),
            ("gradient_collection_sec", "gora_direct_weight_gradient_collection"),
            ("pseudoinverse_init_sec", "gora_pseudoinverse_initialization"),
        ):
            if preallocation_metadata.get(key) is not None:
                stage_metrics.record_completed(
                    stage_name,
                    float(preallocation_metadata.get(key, 0.0) or 0.0),
                    details={"method": str(method)},
                )
        if float(timing_manifest.get("allocation_sec", 0.0) or 0.0) > 0:
            stage_metrics.record_completed(
                "rank_allocation",
                float(timing_manifest["allocation_sec"]),
                details={"method": str(method)},
            )

    budget_manager = BudgetManager(
        budget_cfg.get("mode", "equal_trainable_params"),
        module_dims,
        warning_threshold=float(budget_cfg.get("warning_threshold", 0.01)),
    )
    generic_repair_applied = False
    preallocation_eta = float(config.get("preallocation", {}).get("eta", 0.98)) if method in WINDOW_ALLOC_METHODS else None
    if method in WINDOW_ALLOC_METHODS:
        budget_payload = budget_manager.describe(initial_allocation, target_budget)
        if bool(budget_payload.get("over_budget", False)):
            raise ValueError(
                "Windowed allocation exceeds target budget. "
                "This should be fixed inside the method allocator, "
                "not by generic BudgetManager.repair(...)."
            )
        budget_payload = _budget_with_policy_fields(
            budget_payload,
            method=method,
            preallocation_eta=preallocation_eta,
            generic_repair_applied=False,
        )
        if method in PREALLOC_METHODS | {"gora_bw"} and not budget_payload["budget_eta_reached"]:
            LOGGER.warning(
                "window_allocation_below_eta experiment=%s actual=%s eta_target=%.1f target=%s budget_ratio=%.6f",
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
    elif method == "gora_public":
        # Method-faithful GoRA intentionally reports its realized budget instead of
        # silently repairing it to uniform LoRA's exact parameter count.
        budget_payload = _budget_with_policy_fields(
            budget_manager.describe(initial_allocation, target_budget),
            method=method,
            preallocation_eta=None,
            generic_repair_applied=False,
        )
    else:
        enforce_target_ratio = float(budget_cfg.get("enforce_target_ratio", 1.0))
        if method == "adalora":
            if adalora_target_allocation is None:
                raise RuntimeError("AdaLoRA target allocation was not initialized")
            peak_params = compute_total_lora_params(initial_allocation, module_dims)
            final_target_params = compute_total_lora_params(adalora_target_allocation, module_dims)
            budget_payload = _budget_with_policy_fields(
                budget_manager.describe(adalora_target_allocation, target_budget),
                method=method,
                preallocation_eta=None,
                generic_repair_applied=False,
            )
            budget_payload.update(
                {
                    "adalora_init_rank": int(adalora_config.init_rank if adalora_config else 0),
                    "adalora_target_rank": int(adalora_config.target_rank if adalora_config else rank),
                    "adalora_peak_active_params": int(peak_params),
                    "adalora_final_active_params": int(final_target_params),
                    "adalora_peak_allocation": initial_allocation,
                    "adalora_target_allocation": adalora_target_allocation,
                    "generic_repair_applied": False,
                }
            )
        elif method == "lora" and enforce_target_ratio < 1.0:
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
    if is_main:
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
    if is_main:
        write_json(output_dir / "rank_allocation_initial.json", initial_rank_payload)
        write_json(output_dir / "rank_dict.json", initial_allocation)
        taxonomy_stats = (preallocation_metadata or {}).get("taxonomy_stats")
        coverage_trace = (preallocation_metadata or {}).get("coverage_trace")
        procurement_trace = (preallocation_metadata or {}).get("procurement_trace")
        if taxonomy_stats is not None:
            write_json(output_dir / "taxonomy_stats.json", taxonomy_stats)
        if coverage_trace is not None:
            write_json(output_dir / "coverage_trace.json", coverage_trace)
        if procurement_trace is not None:
            write_json(output_dir / "procurement_trace.json", procurement_trace)

    init_stage = stage_metrics.begin("adapter_initialization")
    init_started_at = time.perf_counter()
    dico_init_cfg = dict(config.get("dico", {}).get("init", {}))
    init_mode = str(dico_init_cfg.get("mode", "legacy"))
    init_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    init_tensors.update(method_init_tensors)
    init_summary_payload: dict[str, Any] = {
        "mode": init_mode,
        "delta_w_zero": None,
        "artifact_note": "Direction-anchored tensors are emitted when the v0.3 DA-init path supplies init_tensors.",
    }
    if init_mode == "direction_anchored":
        direction_bank_path = (preallocation_metadata or {}).get("direction_bank_path")
        if direction_bank_path:
            direction_bank = load_direction_bank(direction_bank_path)
            zero_B = bool(dico_init_cfg.get("zero_B", True))
            init_seed = int(config.get("seed", 42))
            delta_w_zero_by_module: dict[str, bool] = {}
            init_summaries_by_module: dict[str, dict[str, object]] = {}
            for module_name, module_rank in initial_allocation.items():
                module_rank = int(module_rank)
                if module_rank <= 0:
                    continue
                dims = module_dims[module_name]
                result = build_direction_anchored_init(
                    direction_bank.get(module_name, []),
                    rank=module_rank,
                    in_dim=int(dims["in_dim"]),
                    out_dim=int(dims["out_dim"]),
                    seed=init_seed,
                    zero_B=zero_B,
                )
                init_tensors[module_name] = (result.A, result.B)
                delta_w_zero_by_module[module_name] = bool(result.summary.get("delta_w_zero", False))
                init_summaries_by_module[module_name] = dict(result.summary)
            init_summary_payload["delta_w_zero"] = (
                all(delta_w_zero_by_module.values()) if delta_w_zero_by_module else None
            )
            init_summary_payload["delta_w_zero_by_module"] = delta_w_zero_by_module
            init_summary_payload["direction_bank_path"] = direction_bank_path
        else:
            init_summary_payload["artifact_note"] = (
                "dico.init.mode=direction_anchored but no direction_bank_path was found in preallocation "
                "metadata; falling back to default LoRA initialization."
            )
            init_summaries_by_module = {}
    else:
        init_summaries_by_module = {}
    if is_main:
        write_json(output_dir / "init_summary.json", init_summary_payload)
    direction_init_sec = time.perf_counter() - init_started_at

    # v0.5 4.6.7节 + Appendix B: diagnostics and previously-missing output artifacts.
    metadata = preallocation_metadata or {}
    diagnostics_payload = compute_diagnostics(
        initial_allocation,
        module_dims,
        r_max=max_rank,
        target_budget=target_budget,
        balanced_fill_ratio=float(metadata.get("balanced_fill_ratio", 0.0) or 0.0),
        init_summaries=init_summaries_by_module,
        module_quota=metadata.get("module_quota") or {},
        r_min=int(metadata.get("r_min", 0) or 0),
    )
    diagnostics_payload["budget_gap_ratio"] = metadata.get("budget_gap_ratio", 0.0)
    diagnostics_payload["procurement_warning"] = metadata.get("procurement_warning")
    if is_main:
        write_json(output_dir / "diagnostics.json", diagnostics_payload)
        write_json(output_dir / "physical_utility.json", metadata.get("physical_utility", {}))
        write_json(output_dir / "normalization_stats.json", metadata.get("normalization_stats", {}))
        write_json(output_dir / "quota_stats.json", metadata.get("module_quota", {}))
        write_json(output_dir / "kappa_calibration.json", metadata.get("kappa_calibration", {}))
        write_json(
            output_dir / "seeds.json",
            {
                "seed": int(config.get("seed", 42)),
                "calibration_seed": int(config.get("calibration", {}).get("seed", config.get("seed", 42))),
            },
        )
        save_yaml(output_dir / "resolved_config.yaml", config)

        (output_dir / "train_log.jsonl").write_text("", encoding="utf-8")
        (output_dir / "eval_log.jsonl").write_text("", encoding="utf-8")

    lora_cfg = config.get("lora", {})
    lora_injection = str(lora_cfg.get("injection", "masked"))
    lora_scaling = str(lora_cfg.get("scaling", "alpha_over_sqrt_r"))
    adapter_dtype = select_torch_dtype(lora_cfg.get("adapter_dtype", "float32"))
    # 3.1節: CovRA fixes alpha_m/r_m = alpha_ref/r_ref across all modules (a scaling
    # ratio independent of rank_dict's heterogeneous r_m), so comparisons against
    # uniform-rank baselines aren't confounded by per-module effective-learning-rate
    # drift. Scoped to the CovRA method+allocation+static-injection combo only --
    # GoRA-BW and plain `lora` keep their own existing single-global-alpha convention
    # (separate baseline protocols, not touched by this).
    covra_fixed_scaling_ratio = (
        lora_injection == "static"
        and str(config.get("method")) in {"dico_cd", "dico_cd_da"}
        and str(config.get("preallocation", {}).get("allocation_method"))
        in {"covra_full", "covra_independent", "covra_module_scalar", "covra_v05"}
    )
    if method == "adalora":
        lora_injection_started_at = time.perf_counter()
        masked_modules = {}
        adalora_modules = inject_adalora(
            model,
            initial_allocation,
            alpha=float(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            lora_dtype=adapter_dtype,
        )
        injected_module_count = len(adalora_modules)
        if adalora_config is None:
            raise RuntimeError("AdaLoRA config was not initialized")
        adalora_controller = AdaLoRAController(adalora_modules, adalora_config)
    elif lora_injection == "static":
        lora_injection_started_at = time.perf_counter()
        masked_modules = {}
        if covra_fixed_scaling_ratio:
            alpha_ref = float(lora_cfg.get("alpha_ref", lora_cfg.get("alpha", 16)))
            r_ref = float(lora_cfg.get("r_ref", config.get("rank", 8)))
            per_module_alpha = compute_covra_module_alpha(initial_allocation, alpha_ref, r_ref)
            lora_scaling = "alpha_over_r"
            alpha_arg: float | dict[str, float] = per_module_alpha
            if is_main:
                write_json(
                    output_dir / "lora_scaling.json",
                    {
                        "alpha_ref": alpha_ref,
                        "r_ref": r_ref,
                        "scaling_ratio": alpha_ref / r_ref,
                        "alpha_by_module": per_module_alpha,
                    },
                )
        else:
            alpha_arg = float(lora_cfg.get("alpha", 16))
        static_modules = inject_static_lora(
            model,
            initial_allocation,
            alpha=alpha_arg,
            dropout=float(lora_cfg.get("dropout", 0.0)),
            scaling=lora_scaling,
            lora_dtype=adapter_dtype,
            init_tensors=init_tensors or None,
        )
        injected_module_count = len(static_modules)
    else:
        lora_injection_started_at = time.perf_counter()
        masked_modules = inject_masked_lora(
            model,
            initial_allocation,
            max_rank=max_rank,
            alpha=float(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.0)),
            lora_dtype=adapter_dtype,
            scaling=lora_scaling,
        )
        injected_module_count = len(masked_modules)
        if method == "adalora":
            if adalora_config is None:
                raise RuntimeError("AdaLoRA config was not initialized")
            adalora_controller = AdaLoRAController(masked_modules, adalora_config)
    timing_manifest["initialization_sec"] = float(direction_init_sec + (time.perf_counter() - lora_injection_started_at))
    stage_metrics.end(
        init_stage,
        details={"method": str(method), "injected_modules": int(injected_module_count)},
    )
    LOGGER.info(
        "lora_injected experiment=%s injection=%s scaling=%s modules=%d max_rank=%d target_budget=%d",
        experiment_name,
        lora_injection,
        lora_scaling,
        injected_module_count,
        max_rank,
        target_budget,
    )
    # Now that model load + LoRA init are done identically on every rank, decorrelate the
    # global RNG per rank for training-time stochastic ops (e.g. dropout).
    set_seed(base_seed + local_rank)
    optimizer = build_method_optimizer(
        model,
        method=method,
        learning_rate=float(training_cfg.get("learning_rate", 2.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        betas=tuple(training_cfg.get("betas", (0.9, 0.999))),
        eps=float(training_cfg.get("eps", 1e-8)),
        gora_b_lr_multiplier=float(config.get("gora", {}).get("b_lr_multiplier", 16.0)),
    )
    max_steps = int(training_cfg.get("max_steps", 1000))
    warmup_steps = training_warmup_steps(config)
    lr_decay_ratio = float(training_cfg.get("lr_decay_ratio", 0.0))
    scheduler = build_cosine_schedule_with_warmup_and_floor(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
        lr_decay_ratio=lr_decay_ratio,
        num_cycles=0.5,
        auto_warmup_steps=(
            0 if is_reference_covra_config(config) else int(training_cfg.get("auto_warmup_steps", 10))
        ),
        auto_warmup_rate=float(training_cfg.get("auto_warmup_rate", 0.05)),
        decay_over_post_warmup_steps=is_reference_covra_config(config),
    )
    LOGGER.info(
        "scheduler_created experiment=%s warmup_steps=%d max_steps=%d lr_decay_ratio=%.3f",
        experiment_name, warmup_steps, max_steps, lr_decay_ratio,
    )

    # DDP wrapping: accelerator.prepare makes model + optimizer distributed-aware.
    # Must happen AFTER LoRA injection so DDP hooks see the LoRA parameters.
    # In multi-process mode the model is still on CPU here; prepare() moves it to
    # the correct GPU and wraps it with DistributedDataParallel atomically.
    if accelerator is not None:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        # After prepare, input device is the local CUDA device
        input_device = accelerator.device

    rank_history_path = output_dir / "rank_history.csv"
    if is_main:
        init_rank_history(rank_history_path)
        append_rank_history(
            rank_history_path,
            0,
            initial_allocation,
            max_rank,
            {},
            budget_manager,
            target_budget,
            initial_allocation,
            preallocation,
            latest_mid_eval_loss=None,
        )
    if accelerator is not None:
        accelerator.wait_for_everyone()    # resolve per-GPU batch size for DDP (falls back to batch_size for single-process)
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    per_gpu_batch = training_cfg.get("per_gpu_batch_size")
    if per_gpu_batch is not None and num_processes > 1:
        batch_size = int(per_gpu_batch)
    elif num_processes > 1:
        # No explicit per_gpu_batch_size under multi-process DDP: auto-scale down so the
        # effective batch size (per_gpu_batch_size * num_processes * grad_accum) matches the
        # single-GPU protocol instead of silently multiplying it by num_processes.
        original_batch_size = batch_size
        batch_size = max(1, round(original_batch_size / num_processes))
        LOGGER.warning(
            "per_gpu_batch_size not set under %d-process DDP; auto-scaling per-GPU batch_size "
            "%d -> %d to preserve effective batch size (%d x %d gpus x %d grad_accum = %d, "
            "vs single-GPU effective %d). Set training.per_gpu_batch_size explicitly to override.",
            num_processes,
            original_batch_size,
            batch_size,
            batch_size,
            num_processes,
            grad_accum,
            batch_size * num_processes * grad_accum,
            original_batch_size * grad_accum,
        )
    data_loading_manifest = _data_loading_manifest(
        total_train_records=len(all_train_records),
        per_rank_train_records=len(train_records),
        num_processes=num_processes,
        local_rank=local_rank,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
    )

    logging_steps = int(training_cfg.get("logging_steps", 10))
    learning_rate = float(training_cfg.get("learning_rate", 2.0e-4))
    max_grad_norm_raw = training_cfg.get("max_grad_norm", None)
    max_grad_norm = None if max_grad_norm_raw is None else float(max_grad_norm_raw)
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
    train_tokens_local = 0
    training_stage = stage_metrics.begin("training")
    training_started_at = time.perf_counter()
    while global_step < max_steps:
        batch = _move_batch(next(iterator), input_device)
        if "attention_mask" in batch:
            train_tokens_local += int(batch["attention_mask"].detach().sum().cpu().item())
        elif "input_ids" in batch:
            train_tokens_local += int(batch["input_ids"].numel())
        outputs = model(**batch)
        objective = outputs.loss
        if adalora_controller is not None:
            objective = objective + adalora_controller.orthogonal_regularization()
        loss = objective / grad_accum
        # DDP-aware backward: Accelerator handles gradient sync across processes
        if accelerator is not None:
            accelerator.backward(loss)
        else:
            loss.backward()
        running_loss += float(loss.detach().cpu().item()) * grad_accum
        running_loss_observations += 1
        micro_step += 1
        del batch, outputs, loss
        if micro_step % grad_accum != 0:
            continue

        # apply_rank_masks runs AFTER DDP all-reduce (gradient sync already done)
        apply_rank_masks_to_grads(masked_modules)
        grad_norm_before_clip: float | None = None
        # AdaLoRA importance must observe the raw accumulated gradients.  Calling
        # it after clipping changes the baseline's allocation trajectory.
        if adalora_controller is not None:
            adalora_controller.update_importance()
        if max_grad_norm is not None and max_grad_norm > 0:
            params_to_clip = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if params_to_clip:
                clipped_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
                if isinstance(clipped_norm, torch.Tensor):
                    grad_norm_before_clip = float(clipped_norm.detach().cpu().item())
                else:
                    grad_norm_before_clip = float(clipped_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        restore_inactive_parameters(masked_modules)
        adalora_event = adalora_controller.step(global_step + 1) if adalora_controller is not None else None
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        if adalora_event is not None and is_main:
            log_train(
                output_dir / "train_log.jsonl",
                {
                    "event": "adalora_rank_update",
                    "step": global_step,
                    **adalora_event,
                },
            )

        if global_step % logging_steps == 0 or global_step == 1:
            elapsed_sec = time.perf_counter() - run_started_at
            avg_loss = running_loss / max(1, running_loss_observations)
            steps_per_sec = global_step / elapsed_sec if elapsed_sec > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else learning_rate
            if is_main:
                log_train(
                    output_dir / "train_log.jsonl",
                    {
                        "event": "train_step",
                        "step": global_step,
                        "max_steps": max_steps,
                        "loss": avg_loss,
                        "learning_rate": current_lr,
                        "elapsed_sec": elapsed_sec,
                        "steps_per_sec": steps_per_sec,
                        "micro_step": micro_step,
                        "grad_accumulation_steps": grad_accum,
                        "num_processes": num_processes,
                        "max_grad_norm": max_grad_norm,
                        "grad_norm_before_clip": grad_norm_before_clip,
                    },
                )
            LOGGER.info(
                "train_step experiment=%s step=%d/%d loss=%.6f lr=%.2e elapsed_sec=%.1f steps_per_sec=%.4f",
                experiment_name,
                global_step,
                max_steps,
                avg_loss,
                current_lr,
                elapsed_sec,
                steps_per_sec,
            )
            running_loss = 0.0
            running_loss_observations = 0

        if mid_eval_enabled and global_step < max_steps and global_step % mid_eval_every == 0:
            # Eval only on rank-0; unwrap to avoid DDP generate hang
            if is_main:
                eval_model = accelerator.unwrap_model(model) if accelerator is not None else model
                mid_eval_started_at = time.perf_counter()
                mid_eval = evaluate_loss(
                    eval_model,
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
            if accelerator is not None:
                accelerator.wait_for_everyone()

    final_allocation = (
        adalora_controller.current_allocation()
        if adalora_controller is not None
        else initial_allocation
    )
    training_sec = time.perf_counter() - training_started_at
    train_tokens_global = int(train_tokens_local * max(1, num_processes))
    timing_manifest["training_sec"] = float(training_sec)
    timing_manifest["train_tokens"] = train_tokens_global
    timing_manifest["train_tokens_per_sec"] = (
        float(train_tokens_global) / float(training_sec) if training_sec > 0 else 0.0
    )
    stage_metrics.end(
        training_stage,
        details={"optimizer_steps": int(global_step), "train_tokens": int(train_tokens_global)},
    )
    LOGGER.info("training_complete experiment=%s steps=%d rank=%d", experiment_name, global_step, local_rank)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    # Unwrap model before evaluation and checkpoint saving (removes DDP wrapper)
    raw_model = accelerator.unwrap_model(model) if accelerator is not None else model

    if is_main:
        checkpoint_stage = stage_metrics.begin("checkpoint_save")
        write_json(output_dir / "rank_allocation_final.json", final_allocation)
        final_budget = _budget_with_policy_fields(
            budget_manager.describe(final_allocation, target_budget),
            method=method,
            preallocation_eta=preallocation_eta,
            generic_repair_applied=generic_repair_applied,
        )
        final_budget.update(lora_downscale_metadata)
        if adalora_controller is not None:
            peak_allocation = adalora_controller.peak_allocation()
            # AdaLoRA physically trains A/E/B, so each active rank includes one
            # additional singular-value scalar beyond ordinary LoRA's A+B cost.
            peak_params = compute_total_lora_params(peak_allocation, module_dims) + sum(peak_allocation.values())
            final_params = compute_total_lora_params(final_allocation, module_dims) + sum(final_allocation.values())
            final_budget.update(
                {
                    "adalora_init_rank": int(adalora_controller.config.init_rank),
                    "adalora_target_rank": int(adalora_controller.config.target_rank),
                    "adalora_peak_active_params": int(peak_params),
                    "adalora_final_active_params": int(final_params),
                    "adalora_peak_allocation": peak_allocation,
                    "adalora_final_allocation": final_allocation,
                }
            )
            write_json(output_dir / "adalora_schedule.json", adalora_controller.summary())
        write_json(output_dir / "budget.json", final_budget)
        save_masked_lora_state(output_dir / "masked_lora_state.pt", raw_model)
        stage_metrics.end(
            checkpoint_stage,
            details={"checkpoint": str(output_dir / "masked_lora_state.pt")},
        )
        evaluation_stage = stage_metrics.begin("evaluation")
        if bool(config.get("evaluation", {}).get("compute_loss", True)):
            LOGGER.info("final_loss_start experiment=%s", experiment_name)
            final_loss_started_at = time.perf_counter()
            final_eval = evaluate_loss(
                raw_model,
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
        else:
            final_eval = {"eval_loss": None}
    else:
        final_budget = {}
        final_eval = {"eval_loss": float("nan")}
    if is_main and evaluation_cfg.get("compute_accuracy", True):
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
            raw_model,
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
            batch_size=int(evaluation_cfg.get("batch_size", 4)),
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

        # 6.5节: mixed math+code experiments also evaluate HumanEval and report
        # mean + worst-group score across the two task groups.
        if "humaneval" in list(data_cfg.get("eval_datasets", []) or []):
            LOGGER.info("humaneval_eval_start experiment=%s", experiment_name)
            humaneval_started_at = time.perf_counter()
            humaneval_records = load_humaneval_records(config)

            def log_humaneval_progress(progress: dict[str, Any]) -> None:
                LOGGER.info(
                    "humaneval_eval_progress experiment=%s eval_total=%d eval_correct=%d eval_accuracy=%.4f",
                    experiment_name,
                    progress["eval_total"],
                    progress["eval_correct"],
                    progress["eval_accuracy"],
                )

            humaneval_metrics = evaluate_humaneval_pass_at_1(
                raw_model,
                tokenizer,
                humaneval_records,
                device=input_device,
                max_samples=evaluation_cfg.get("humaneval_max_samples"),
                max_new_tokens=int(evaluation_cfg.get("humaneval_max_new_tokens", 256)),
                timeout_seconds=float(evaluation_cfg.get("humaneval_timeout_seconds", 5.0)),
                prediction_path=output_dir / "humaneval_predictions.jsonl",
                progress_callback=log_humaneval_progress,
                progress_interval=int(evaluation_cfg.get("accuracy_logging_steps", 50)),
            )
            final_eval["eval_humaneval_pass_at_1"] = humaneval_metrics["eval_pass_at_1"]
            final_eval["eval_humaneval_correct"] = humaneval_metrics["eval_correct"]
            final_eval["eval_humaneval_total"] = humaneval_metrics["eval_total"]
            group_scores = {
                "gsm8k": float(final_eval["eval_accuracy"]),
                "humaneval": float(humaneval_metrics["eval_pass_at_1"]),
            }
            final_eval["group_scores"] = group_scores
            final_eval["eval_mean_group_score"] = sum(group_scores.values()) / len(group_scores)
            worst_group_name = min(group_scores, key=lambda name: group_scores[name])
            final_eval["eval_worst_group_score"] = group_scores[worst_group_name]
            final_eval["eval_worst_group_name"] = worst_group_name
            LOGGER.info(
                "humaneval_eval_complete experiment=%s pass_at_1=%.4f elapsed_sec=%.1f",
                experiment_name,
                humaneval_metrics["eval_pass_at_1"],
                time.perf_counter() - humaneval_started_at,
            )
    if is_main:
        stage_metrics.end(
            evaluation_stage,
            details={"metric": str(evaluation_cfg.get("metric", "gsm8k_accuracy"))},
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
            "num_processes": num_processes,
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
            "target_budget": final_budget.get("target_budget"),
            "actual_budget": final_budget.get("actual_budget"),
            "total_params": final_budget.get("actual_budget"),
            "budget_ratio": final_budget.get("budget_ratio"),
            "target_budget_paramcount": final_budget.get("target_budget_paramcount"),
            "target_budget_ranksum": final_budget.get("target_budget_ranksum"),
            "actual_budget_paramcount": final_budget.get("actual_budget_paramcount"),
            "actual_budget_ranksum": final_budget.get("actual_budget_ranksum"),
            "budget_ratio_paramcount": final_budget.get("budget_ratio_paramcount"),
            "budget_ratio_ranksum": final_budget.get("budget_ratio_ranksum"),
            "preallocation_eta": final_budget.get("preallocation_eta"),
            "budget_eta_reached": final_budget.get("budget_eta_reached"),
            "budget_interval_pass": final_budget.get("budget_interval_pass"),
            "generic_repair_applied": final_budget.get("generic_repair_applied"),
            "budget_error": final_budget.get("budget_error"),
            "budget_error_ratio": final_budget.get("budget_error_ratio"),
            "total_active_rank": final_budget.get("total_active_rank"),
            "trainable_params_physical": trainable_parameter_count(raw_model),
            "lora_baseline_downscaled": final_budget.get("lora_baseline_downscaled"),
            "lora_baseline_target_ratio": final_budget.get("lora_baseline_target_ratio"),
            "lora_baseline_min_ratio": final_budget.get("lora_baseline_min_ratio"),
            "lora_baseline_actual_ratio": final_budget.get("lora_baseline_actual_ratio"),
            "lora_downscale_details": final_budget.get("lora_downscale_details"),
            "lora_downscale_interval_pass": final_budget.get("lora_downscale_interval_pass"),
        }
        write_json(output_dir / "metrics.json", metrics)
        _write_run_manifest_and_summary(
            output_dir,
            config=config,
            experiment_name=experiment_name,
            method=method,
            rank=rank,
            num_processes=num_processes,
            local_rank=local_rank,
            batch_size=batch_size,
            grad_accum=grad_accum,
            max_steps=max_steps,
            global_step=global_step,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            lr_decay_ratio=lr_decay_ratio,
            module_names=module_names,
            module_dims=module_dims,
            initial_allocation=initial_allocation,
            final_allocation=final_allocation,
            final_budget=final_budget,
            metrics=metrics,
            raw_model=raw_model,
            run_elapsed_sec=time.perf_counter() - run_started_at,
            data_manifest=data_manifest,
            calibration_manifest=calibration_manifest,
            timing_manifest=timing_manifest,
            optimizer_manifest=_optimizer_manifest(optimizer),
            data_loading_manifest=data_loading_manifest,
        )
        append_rank_history(
            rank_history_path,
            max_steps,
            final_allocation,
            max_rank,
            {},
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
    else:
        metrics = {"experiment": experiment_name, "rank": rank, "is_main": False}

    if accelerator is not None:
        accelerator.wait_for_everyone()
    return metrics
