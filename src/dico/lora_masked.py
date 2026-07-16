from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from dico.lora_scaling import lora_scale


def _module_in_features(module: nn.Module) -> int:
    if not hasattr(module, "in_features"):
        raise TypeError("base_layer must expose in_features")
    return int(getattr(module, "in_features"))


def _module_out_features(module: nn.Module) -> int:
    if not hasattr(module, "out_features"):
        raise TypeError("base_layer must expose out_features")
    return int(getattr(module, "out_features"))


class MaskedLoRALinear(nn.Module):
    """LoRA linear layer with fixed max_rank capacity and runtime rank masks.

    Newly activated rank channels are pre-initialized at LoRA injection time and
    remain frozen while inactive. When activated, they start from their original
    initialized values.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        max_rank: int,
        active_rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        scaling: str = "alpha_over_max_rank",
    ):
        super().__init__()
        if max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if not 0 <= active_rank <= max_rank:
            raise ValueError("active_rank must be between 0 and max_rank")
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.max_rank = int(max_rank)
        self.alpha = float(alpha)
        self.scaling_mode = str(scaling)
        self.scaling = lora_scale(alpha, self.max_rank, self.scaling_mode)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.in_features = _module_in_features(base_layer)
        self.out_features = _module_out_features(base_layer)
        self.lora_A = nn.Parameter(torch.empty(max_rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, max_rank))
        self.register_buffer("rank_mask", torch.zeros(max_rank, dtype=torch.float32))
        self._inactive_snapshot: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.reset_parameters()
        self.set_active_rank(active_rank)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.rank_mask.to(dtype=self.lora_A.dtype, device=self.lora_A.device)
        active_A = self.lora_A * mask[:, None]
        active_B = self.lora_B * mask[None, :]
        result = self.base_layer(x)
        lora_input = self.dropout(x).to(dtype=self.lora_A.dtype, device=self.lora_A.device)
        delta = F.linear(F.linear(lora_input, active_A), active_B)
        delta = delta.to(dtype=result.dtype, device=result.device)
        return result + delta * self.scaling

    def get_rank_mask(self) -> torch.Tensor:
        return self.rank_mask.detach().clone()

    def set_rank_mask(self, mask: torch.Tensor | Iterable[float]) -> None:
        value = torch.as_tensor(mask, dtype=torch.float32, device=self.rank_mask.device)
        if value.numel() != self.max_rank:
            raise ValueError(f"rank mask must have {self.max_rank} entries")
        value = (value.reshape(self.max_rank) > 0).to(dtype=torch.float32)
        self.rank_mask.copy_(value)

    def get_active_rank(self) -> int:
        return int(self.rank_mask.sum().item())

    def set_active_rank(self, rank: int) -> None:
        if not 0 <= int(rank) <= self.max_rank:
            raise ValueError("active rank must be between 0 and max_rank")
        mask = torch.zeros(self.max_rank, dtype=torch.float32, device=self.rank_mask.device)
        if int(rank) > 0:
            mask[: int(rank)] = 1.0
        self.rank_mask.copy_(mask)

    def open_channel(self, channel: int) -> None:
        self.rank_mask[int(channel)] = 1.0

    def close_channel(self, channel: int) -> None:
        self.rank_mask[int(channel)] = 0.0

    def _save_inactive_snapshot(self) -> None:
        inactive = self.rank_mask.to(device=self.lora_A.device) <= 0
        inactive_idx = torch.nonzero(inactive, as_tuple=False).flatten()
        self._inactive_snapshot = (
            inactive_idx.detach().clone(),
            self.lora_A.detach().index_select(0, inactive_idx).clone(),
            self.lora_B.detach().index_select(1, inactive_idx).clone(),
        )

    def apply_rank_mask_to_grads(self) -> None:
        self._save_inactive_snapshot()
        mask = self.rank_mask.to(dtype=self.lora_A.dtype, device=self.lora_A.device)
        if self.lora_A.grad is not None:
            self.lora_A.grad.mul_(mask[:, None])
        if self.lora_B.grad is not None:
            self.lora_B.grad.mul_(mask[None, :])

    def zero_inactive_grads(self) -> None:
        self.apply_rank_mask_to_grads()

    def restore_inactive_parameters(self) -> None:
        if self._inactive_snapshot is None:
            return
        inactive_idx, inactive_a, inactive_b = self._inactive_snapshot
        inactive_idx = inactive_idx.to(device=self.lora_A.device)
        with torch.no_grad():
            self.lora_A.index_copy_(0, inactive_idx, inactive_a.to(device=self.lora_A.device, dtype=self.lora_A.dtype))
            self.lora_B.index_copy_(1, inactive_idx, inactive_b.to(device=self.lora_B.device, dtype=self.lora_B.dtype))
        self._inactive_snapshot = None

    def channel_scores(
        self,
        grad_weight: float = 0.5,
        update_weight: float = 0.5,
    ) -> torch.Tensor:
        grad_a = (
            torch.linalg.norm(self.lora_A.grad.detach(), dim=1)
            if self.lora_A.grad is not None
            else torch.zeros(self.max_rank, device=self.lora_A.device)
        )
        grad_b = (
            torch.linalg.norm(self.lora_B.grad.detach(), dim=0)
            if self.lora_B.grad is not None
            else torch.zeros(self.max_rank, device=self.lora_A.device)
        )
        update = torch.linalg.norm(self.lora_A.detach(), dim=1) * torch.linalg.norm(
            self.lora_B.detach(), dim=0
        )
        return float(grad_weight) * (grad_a + grad_b) + float(update_weight) * update


def iter_masked_lora_modules(model: nn.Module) -> dict[str, MaskedLoRALinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, MaskedLoRALinear)}


def apply_rank_masks_to_grads(modules: Iterable[MaskedLoRALinear] | dict[str, MaskedLoRALinear]) -> None:
    values = modules.values() if isinstance(modules, dict) else modules
    for module in values:
        module.apply_rank_mask_to_grads()


def restore_inactive_parameters(modules: Iterable[MaskedLoRALinear] | dict[str, MaskedLoRALinear]) -> None:
    values = modules.values() if isinstance(modules, dict) else modules
    for module in values:
        module.restore_inactive_parameters()


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def inject_masked_lora(
    model: nn.Module,
    rank_allocation: dict[str, int],
    max_rank: int,
    alpha: float = 16.0,
    dropout: float = 0.0,
    lora_dtype: torch.dtype | None = None,
    scaling: str = "alpha_over_max_rank",
) -> dict[str, MaskedLoRALinear]:
    for param in model.parameters():
        param.requires_grad = False
    wrapped: dict[str, MaskedLoRALinear] = {}
    for module_name, rank in rank_allocation.items():
        parent, attr = _get_parent_module(model, module_name)
        base = getattr(parent, attr)
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError(f"Target module {module_name} is not a supported linear-like module")
        base_device = base.weight.device
        base_dtype = base.weight.dtype if torch.is_floating_point(base.weight) else None
        target_dtype = lora_dtype or base_dtype or torch.bfloat16
        layer = MaskedLoRALinear(
            base,
            max_rank=int(max_rank),
            active_rank=int(rank),
            alpha=alpha,
            dropout=dropout,
            scaling=scaling,
        )
        with torch.no_grad():
            layer.lora_A.data = layer.lora_A.data.to(device=base_device, dtype=target_dtype)
            layer.lora_B.data = layer.lora_B.data.to(device=base_device, dtype=target_dtype)
            layer.rank_mask.data = layer.rank_mask.data.to(device=base_device)
        setattr(parent, attr, layer)
        wrapped[module_name] = layer
    return wrapped


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
