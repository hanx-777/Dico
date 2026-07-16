from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any, Mapping

import torch
from torch import nn

from dico.rank_budget import compute_total_lora_params, module_rank_cost


OFFICIAL_GORA_COMMIT = "4037d4d6ba67ff88de87f90b943ff4e3a3649b67"


def _gradient_accumulation_dtype(value: str | torch.dtype) -> torch.dtype | None:
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).lower()
    if normalized in {"native", "same"}:
        return None
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported GoRA gradient accumulation dtype: {value!r}")
    return aliases[normalized]


def collect_average_weight_gradients(
    model: nn.Module,
    module_names: Sequence[str],
    calibration_batches: Iterable[Mapping[str, Any]],
    *,
    offload_device: str | torch.device = "cpu",
    accumulation_dtype: str | torch.dtype = "float32",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Collect GoRA's direct target-weight gradients with one backward per batch.

    The official implementation temporarily enables gradients only for target
    base weights and offloads each accumulated gradient from its parameter hook.
    This deliberately does not reconstruct weight gradients from activations or
    apply an answer-token mask.
    """

    named_modules = dict(model.named_modules())
    targets: dict[str, nn.Parameter] = {}
    for name in module_names:
        module = named_modules.get(name)
        if module is None or not hasattr(module, "weight"):
            raise KeyError(f"GoRA target module has no weight: {name}")
        weight = getattr(module, "weight")
        if not isinstance(weight, nn.Parameter):
            raise TypeError(f"GoRA target weight is not a Parameter: {name}")
        targets[name] = weight

    target_dtype = _gradient_accumulation_dtype(accumulation_dtype)
    offload_device = torch.device(offload_device)
    sums: dict[str, torch.Tensor] = {}
    backward_passes = 0
    observed_samples = 0
    original_requires_grad = {parameter: parameter.requires_grad for parameter in model.parameters()}
    was_training = model.training
    handles: list[Any] = []
    hook_kind = "post_accumulate"

    def accumulate(name: str, gradient: torch.Tensor) -> None:
        detached = gradient.detach()
        dtype = target_dtype or detached.dtype
        offloaded = detached.to(device=offload_device, dtype=dtype)
        if name not in sums:
            sums[name] = offloaded.clone()
        else:
            sums[name].add_(offloaded)

    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in targets.values():
            parameter.requires_grad_(True)

        supports_post_hook = all(hasattr(parameter, "register_post_accumulate_grad_hook") for parameter in targets.values())
        if supports_post_hook:
            for name, parameter in targets.items():
                def post_hook(param: nn.Parameter, *, _name: str = name) -> None:
                    if param.grad is not None:
                        accumulate(_name, param.grad)
                        param.grad = None

                handles.append(parameter.register_post_accumulate_grad_hook(post_hook))
        else:  # pragma: no cover - only for older supported PyTorch releases
            hook_kind = "tensor_hook"
            for name, parameter in targets.items():
                def tensor_hook(gradient: torch.Tensor, *, _name: str = name) -> torch.Tensor:
                    accumulate(_name, gradient)
                    return torch.zeros_like(gradient)

                handles.append(parameter.register_hook(tensor_hook))

        model.eval()
        model_device = next(model.parameters()).device
        for batch in calibration_batches:
            model.zero_grad(set_to_none=True)
            moved = {
                key: value.to(model_device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            outputs = model(**moved)
            if not hasattr(outputs, "loss") or outputs.loss is None:
                raise ValueError("GoRA calibration batch did not produce a loss")
            outputs.loss.backward()
            backward_passes += 1
            input_ids = moved.get("input_ids")
            observed_samples += int(input_ids.shape[0]) if torch.is_tensor(input_ids) and input_ids.ndim else 1
            for parameter in targets.values():
                parameter.grad = None

        if backward_passes == 0:
            raise ValueError("GoRA calibration requires at least one batch")
        missing = [name for name in targets if name not in sums]
        if missing:
            raise RuntimeError(f"GoRA did not observe gradients for target modules: {missing}")
        averages = {name: value / float(backward_passes) for name, value in sums.items()}
        metadata = {
            "gradient_collection": "official_weight_grad_hook",
            "gradient_offload_device": str(offload_device),
            "gradient_accumulation_dtype": str(target_dtype or "native"),
            "clear_gradient_after_offload": True,
            "hook_kind": hook_kind,
            "backward_passes": backward_passes,
            "observed_samples": observed_samples,
            "answer_only": False,
        }
        return averages, metadata
    finally:
        for handle in handles:
            handle.remove()
        for parameter, requires_grad in original_requires_grad.items():
            parameter.requires_grad_(requires_grad)
            parameter.grad = None
        model.train(was_training)


def compute_gora_importance(weight: torch.Tensor, average_gradient: torch.Tensor) -> torch.Tensor:
    """GoRA reference importance: mean(abs(W elementwise-multiplied by G_avg))."""
    if weight.shape != average_gradient.shape:
        raise ValueError("GoRA weight and average gradient shapes must match")
    return torch.mean(torch.abs(weight.detach().float() * average_gradient.detach().float()))


def allocate_gora_ranks(
    importance: Mapping[str, float],
    module_dims: Mapping[str, Mapping[str, int]],
    r_ref: int,
    r_min: int,
    r_max: int,
    rounding: str = "moderate",
) -> dict[str, int]:
    if rounding != "moderate":
        raise ValueError("Only GoRA moderate rounding is supported by the aligned baseline")
    names = list(module_dims)
    total_budget = sum(module_rank_cost(module_dims[name]) * int(r_ref) for name in names)
    positive = {name: max(0.0, float(importance.get(name, 0.0))) for name in names}
    normalizer = sum(positive.values())
    if normalizer <= 0:
        positive = {name: 1.0 for name in names}
        normalizer = float(len(names))
    allocation: dict[str, int] = {}
    for name in names:
        smooth_trainable = round(total_budget * positive[name] / normalizer)
        rank = round(smooth_trainable // module_rank_cost(module_dims[name]))
        allocation[name] = max(int(r_min), min(int(r_max), int(rank)))
    return allocation


def strict_budget_repair(
    allocation: Mapping[str, int],
    importance: Mapping[str, float],
    module_dims: Mapping[str, Mapping[str, int]],
    target_budget: int,
    r_min: int,
    r_max: int,
) -> dict[str, int]:
    names = list(module_dims)
    base = {name: int(r_min) for name in names}
    base_cost = compute_total_lora_params(base, module_dims)
    if base_cost > int(target_budget):
        raise ValueError("GoRA-BM cannot satisfy strict budget without violating r_min")
    costs = [module_rank_cost(module_dims[name]) for name in names]
    gcd = 0
    for cost in costs:
        gcd = math.gcd(gcd, int(cost))
    capacity = (int(target_budget) - base_cost) // max(1, gcd)
    # state -> (utility, per-module additional ranks). A large preservation
    # bonus makes this a repair of the public allocation, while the DP still
    # guarantees the closest feasible budget and exact equality when possible.
    states: dict[int, float] = {0: 0.0}
    parent_stages: list[dict[int, tuple[int, int]]] = []
    for module_index, name in enumerate(names):
        unit_cost = module_rank_cost(module_dims[name]) // max(1, gcd)
        next_states: dict[int, float] = {}
        parents: dict[int, tuple[int, int]] = {}
        for used, score in states.items():
            for increment in range(0, int(r_max) - int(r_min) + 1):
                new_used = used + increment * unit_cost
                if new_used > capacity:
                    break
                preserved = min(increment, max(0, int(allocation.get(name, r_min)) - int(r_min)))
                utility = score + increment * float(importance.get(name, 0.0)) + preserved * 1_000_000.0
                if new_used not in next_states or utility > next_states[new_used]:
                    next_states[new_used] = utility
                    parents[new_used] = (used, increment)
        states = next_states
        parent_stages.append(parents)
    best_used = max(states)
    increments = [0 for _ in names]
    cursor = best_used
    for module_index in range(len(names) - 1, -1, -1):
        previous, increment = parent_stages[module_index][cursor]
        increments[module_index] = increment
        cursor = previous
    return {name: int(r_min) + int(increments[index]) for index, name in enumerate(names)}


def gora_pseudoinverse_init(
    average_gradient: torch.Tensor,
    lora_A: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    gram = lora_A @ lora_A.T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return average_gradient.to(dtype=lora_A.dtype, device=lora_A.device) @ lora_A.T @ torch.linalg.pinv(gram + eps * identity)


def scale_gora_b_initialization(
    lora_B: torch.Tensor,
    *,
    rank: int,
    in_features: int,
    alpha: float,
    init_lr: float = 0.05,
) -> torch.Tensor:
    # Locked official code semantics for scale_by_lr + rank stabilization.
    scale_rank = math.sqrt(float(rank))
    stable_gamma = (float(init_lr) / math.sqrt(float(rank) / float(in_features))) * scale_rank
    return lora_B * (stable_gamma / float(alpha))
