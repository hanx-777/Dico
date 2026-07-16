from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from dico.lora_masked import _get_parent_module, _module_in_features, _module_out_features


@dataclass(frozen=True)
class AdaLoRAConfig:
    init_rank: int
    target_rank: int
    tinit: int
    tfinal: int
    delta_t: int = 1
    total_steps: int = 1000
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5
    # Backwards-compatible config spelling used by the previous in-repo baseline.
    update_interval: int | None = None

    def __post_init__(self) -> None:
        if self.init_rank <= 0 or self.target_rank <= 0:
            raise ValueError("AdaLoRA ranks must be positive")
        if self.target_rank > self.init_rank:
            raise ValueError("AdaLoRA target_rank cannot exceed init_rank")
        if self.tinit < 0 or self.tfinal < 0:
            raise ValueError("AdaLoRA tinit/tfinal must be non-negative")
        if self.total_steps <= 0:
            raise ValueError("AdaLoRA total_steps must be positive")
        interval = self.delta_t if self.update_interval is None else self.update_interval
        if interval <= 0:
            raise ValueError("AdaLoRA deltaT must be positive")
        object.__setattr__(self, "delta_t", int(interval))
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdaLoRA EMA beta values must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["deltaT"] = payload["delta_t"]
        return payload


class AdaLoRALinear(nn.Module):
    """AdaLoRA SVD adapter with trainable A/E/B triplets.

    E starts at zero, so injection preserves the frozen model output even though
    both direction matrices use the AdaLoRA Normal(0, .02) initialization.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        lora_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("AdaLoRALinear rank must be positive")
        self.base_layer = base_layer
        for parameter in base_layer.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.in_features = _module_in_features(base_layer)
        self.out_features = _module_out_features(base_layer)
        dtype = lora_dtype or (base_layer.weight.dtype if torch.is_floating_point(base_layer.weight) else torch.float32)
        device = base_layer.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features, device=device, dtype=dtype))
        self.lora_E = nn.Parameter(torch.zeros(self.rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank, device=device, dtype=dtype))
        self.register_buffer("rank_mask", torch.ones(self.rank, device=device, dtype=dtype), persistent=True)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    @property
    def active_rank(self) -> int:
        return int(torch.count_nonzero(self.rank_mask).item())

    def set_rank_mask(self, mask: torch.Tensor) -> None:
        if mask.numel() != self.rank:
            raise ValueError("AdaLoRA rank mask has the wrong size")
        normalized = mask.to(device=self.rank_mask.device, dtype=self.rank_mask.dtype)
        self.rank_mask.copy_(normalized)
        # Official AdaLoRA prunes by zeroing E.  The mask is retained only for
        # allocation reporting/checkpoint compatibility; multiplying E by a
        # permanent mask in forward would prevent a pruned component from ever
        # receiving a gradient again.
        with torch.no_grad():
            self.lora_E.masked_fill_(normalized <= 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        adapter_input = self.dropout(x).to(device=self.lora_A.device, dtype=self.lora_A.dtype)
        hidden = F.linear(adapter_input, self.lora_A) * self.lora_E
        delta = F.linear(hidden, self.lora_B)
        return result + delta.to(device=result.device, dtype=result.dtype) * self.scaling


def inject_adalora(
    model: nn.Module,
    rank_allocation: Mapping[str, int],
    alpha: float = 16.0,
    dropout: float = 0.0,
    lora_dtype: torch.dtype | None = torch.float32,
) -> dict[str, AdaLoRALinear]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    wrapped: dict[str, AdaLoRALinear] = {}
    for module_name, rank in rank_allocation.items():
        parent, attribute = _get_parent_module(model, module_name)
        layer = AdaLoRALinear(
            getattr(parent, attribute),
            rank=int(rank),
            alpha=alpha,
            dropout=dropout,
            lora_dtype=lora_dtype,
        )
        setattr(parent, attribute, layer)
        wrapped[module_name] = layer
    return wrapped


class AdaLoRAController:
    """Global AdaLoRA budget scheduler using EMA parameter-gradient importance."""

    def __init__(self, modules: Mapping[str, nn.Module], config: AdaLoRAConfig):
        self.modules = dict(modules)
        self.config = config
        self.events: list[dict[str, Any]] = []
        self.exp_avg_ipt: dict[str, torch.Tensor] = {}
        self.exp_avg_unc: dict[str, torch.Tensor] = {}

    def phase_at_step(self, step: int) -> str:
        if step <= self.config.tinit:
            return "initial_warmup"
        if step > max(self.config.tinit, self.config.total_steps - self.config.tfinal):
            return "final_finetune"
        return "budget_decrease"

    def total_rank_at_step(self, step: int) -> int:
        module_count = len(self.modules)
        if module_count == 0:
            return 0
        initial = module_count * self.config.init_rank
        target = module_count * self.config.target_rank
        if step <= self.config.tinit:
            return initial
        final_start = max(self.config.tinit, self.config.total_steps - self.config.tfinal)
        if step > final_start:
            return target
        span = max(1, final_start - self.config.tinit)
        progress = (step - self.config.tinit) / span
        multiplier = (1.0 - progress) ** 3
        return int(target + (initial - target) * multiplier)

    # Compatibility helper for old callers/tests; the actual policy is global.
    def target_rank_at_step(self, step: int) -> int:
        if not self.modules:
            return self.config.target_rank
        return int(round(self.total_rank_at_step(step) / len(self.modules)))

    def update_importance(self) -> None:
        for module_name, module in self.modules.items():
            if not isinstance(module, AdaLoRALinear):
                continue
            for suffix, parameter in (("A", module.lora_A), ("E", module.lora_E), ("B", module.lora_B)):
                if parameter.grad is None:
                    continue
                key = f"{module_name}.{suffix}"
                importance = (parameter.detach() * parameter.grad.detach()).abs()
                previous = self.exp_avg_ipt.get(key, torch.zeros_like(importance))
                average = self.config.beta1 * previous + (1.0 - self.config.beta1) * importance
                uncertainty_now = (importance - average).abs()
                previous_unc = self.exp_avg_unc.get(key, torch.zeros_like(importance))
                uncertainty = self.config.beta2 * previous_unc + (1.0 - self.config.beta2) * uncertainty_now
                self.exp_avg_ipt[key] = average
                self.exp_avg_unc[key] = uncertainty

    def _rank_scores(self, name: str, module: nn.Module) -> torch.Tensor:
        if isinstance(module, AdaLoRALinear):
            def score(suffix: str, parameter: torch.Tensor) -> torch.Tensor:
                key = f"{name}.{suffix}"
                avg = self.exp_avg_ipt.get(key, torch.zeros_like(parameter))
                unc = self.exp_avg_unc.get(key, torch.zeros_like(parameter))
                return avg * unc
            a = score("A", module.lora_A).mean(dim=1)
            e = score("E", module.lora_E)
            b = score("B", module.lora_B).mean(dim=0)
            return a + e + b
        # Legacy masked module compatibility; no longer used by the formal config.
        return module.channel_scores().detach()

    def step(self, global_step: int) -> dict[str, Any] | None:
        step = int(global_step)
        if step <= self.config.tinit:
            return None
        final_start = max(self.config.tinit, self.config.total_steps - self.config.tfinal)
        if step <= final_start and (step - self.config.tinit) % self.config.delta_t != 0:
            return None
        if self.modules and all(not isinstance(module, AdaLoRALinear) for module in self.modules.values()):
            # Compatibility for checkpoints/tests created by the retired masked-rank
            # baseline. Formal AdaLoRA configs never enter this branch.
            if step >= self.config.tfinal:
                per_module_target = self.config.target_rank
            else:
                span = max(1, self.config.tfinal - self.config.tinit)
                progress = (step - self.config.tinit) / span
                per_module_target = int(round(self.config.init_rank - progress * (self.config.init_rank - self.config.target_rank)))
            module_events: dict[str, Any] = {}
            for name, module in self.modules.items():
                scores = module.channel_scores().detach()
                kept_indices = torch.topk(scores, k=per_module_target, largest=True).indices
                mask = torch.zeros(module.max_rank, device=module.rank_mask.device)
                mask[kept_indices] = 1.0
                previous = module.get_active_rank()
                module.set_rank_mask(mask)
                module_events[name] = {"previous_rank": previous, "new_rank": module.get_active_rank()}
            event = {
                "step": step,
                "phase": "legacy_compatibility",
                "target_total_rank": per_module_target * len(self.modules),
                "target_active_rank": per_module_target,
                "modules": module_events,
            }
            self.events.append(event)
            return event
        desired = self.total_rank_at_step(step)
        names = list(self.modules)
        scores = [self._rank_scores(name, self.modules[name]).detach() for name in names]
        total_components = sum(int(score.numel()) for score in scores)
        desired = max(0, min(desired, total_components))

        # All singular values compete under one global budget.  The threshold is
        # computed on the accelerator and mirrors the official kthvalue policy.
        if scores:
            device = scores[0].device
            dtype = scores[0].dtype
            flat_scores = torch.cat([score.to(device=device, dtype=dtype).reshape(-1) for score in scores])
            prune_count = total_components - desired
            if prune_count <= 0:
                flat_mask = torch.ones_like(flat_scores)
            elif desired <= 0:
                flat_mask = torch.zeros_like(flat_scores)
            else:
                threshold = torch.kthvalue(flat_scores, k=prune_count).values
                flat_mask = (flat_scores > threshold).to(dtype=flat_scores.dtype)
        else:
            flat_mask = torch.empty(0)
        module_events: dict[str, Any] = {}
        offset = 0
        for name, module, score in zip(names, (self.modules[name] for name in names), scores):
            rank = module.rank if isinstance(module, AdaLoRALinear) else module.max_rank
            device = module.rank_mask.device
            mask = flat_mask[offset:offset + int(score.numel())].to(device=device)
            offset += int(score.numel())
            previous = module.active_rank if isinstance(module, AdaLoRALinear) else module.get_active_rank()
            module.set_rank_mask(mask)
            current = module.active_rank if isinstance(module, AdaLoRALinear) else module.get_active_rank()
            module_events[name] = {"previous_rank": previous, "new_rank": current}
        event = {
            "step": step,
            "phase": self.phase_at_step(step),
            "target_total_rank": desired,
            "target_active_rank": self.target_rank_at_step(step),
            "modules": module_events,
        }
        self.events.append(event)
        return event

    def orthogonal_regularization(self) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for module in self.modules.values():
            if not isinstance(module, AdaLoRALinear):
                continue
            eye = torch.eye(module.rank, device=module.lora_A.device, dtype=module.lora_A.dtype)
            losses.append(torch.linalg.matrix_norm(module.lora_A @ module.lora_A.T - eye))
            losses.append(torch.linalg.matrix_norm(module.lora_B.T @ module.lora_B - eye))
        if not losses:
            return torch.tensor(0.0)
        return sum(losses) / len(losses) * self.config.orth_reg_weight

    def current_allocation(self) -> dict[str, int]:
        return {
            name: (module.active_rank if isinstance(module, AdaLoRALinear) else module.get_active_rank())
            for name, module in self.modules.items()
        }

    def peak_allocation(self) -> dict[str, int]:
        return {name: self.config.init_rank for name in self.modules}

    def summary(self) -> dict[str, Any]:
        return {**self.config.to_dict(), "events": self.events}
