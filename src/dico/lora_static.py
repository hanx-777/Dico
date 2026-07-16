from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F

from dico.lora_scaling import lora_scale
from dico.lora_masked import _get_parent_module, _module_in_features, _module_out_features


class StaticLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        scaling: str = "alpha_over_sqrt_r",
        init_A: torch.Tensor | None = None,
        init_B: torch.Tensor | None = None,
    ):
        super().__init__()
        if int(rank) <= 0:
            raise ValueError("StaticLoRALinear requires rank > 0")
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling_mode = str(scaling)
        self.scaling = lora_scale(alpha, self.rank, self.scaling_mode)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.in_features = _module_in_features(base_layer)
        self.out_features = _module_out_features(base_layer)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))
        self.reset_parameters()
        with torch.no_grad():
            if init_A is not None:
                self.lora_A.copy_(init_A.to(dtype=self.lora_A.dtype, device=self.lora_A.device))
            if init_B is not None:
                self.lora_B.copy_(init_B.to(dtype=self.lora_B.dtype, device=self.lora_B.device))

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return getattr(self.base_layer, "bias", None)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Standard LoRA initialization: the adapter is an exact no-op at step 0.
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        lora_input = self.dropout(x).to(dtype=self.lora_A.dtype, device=self.lora_A.device)
        delta = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        return result + delta.to(dtype=result.dtype, device=result.device) * self.scaling


def inject_static_lora(
    model: nn.Module,
    rank_allocation: Mapping[str, int],
    alpha: float | Mapping[str, float] = 16.0,
    dropout: float = 0.0,
    scaling: str = "alpha_over_sqrt_r",
    lora_dtype: torch.dtype | None = None,
    init_tensors: Mapping[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, StaticLoRALinear]:
    """`alpha` may be a single float shared by every module (legacy behavior) or a
    per-module mapping -- the latter is how CovRA's fixed alpha_m/r_m scaling ratio
    (3.1節: alpha_m = r_m * alpha_ref/r_ref) is threaded in, see
    lora_scaling.compute_covra_module_alpha.
    """
    for param in model.parameters():
        param.requires_grad = False
    wrapped: dict[str, StaticLoRALinear] = {}
    for module_name, rank in rank_allocation.items():
        if int(rank) <= 0:
            continue
        parent, attr = _get_parent_module(model, module_name)
        base = getattr(parent, attr)
        base_device = base.weight.device
        base_dtype = base.weight.dtype if torch.is_floating_point(base.weight) else None
        target_dtype = lora_dtype or base_dtype or torch.bfloat16
        init_A = None
        init_B = None
        if init_tensors and module_name in init_tensors:
            init_A, init_B = init_tensors[module_name]
        module_alpha = float(alpha[module_name]) if isinstance(alpha, Mapping) else float(alpha)
        layer = StaticLoRALinear(
            base,
            rank=int(rank),
            alpha=module_alpha,
            dropout=dropout,
            scaling=scaling,
            init_A=init_A,
            init_B=init_B,
        )
        with torch.no_grad():
            layer.lora_A.data = layer.lora_A.data.to(device=base_device, dtype=target_dtype)
            layer.lora_B.data = layer.lora_B.data.to(device=base_device, dtype=target_dtype)
        setattr(parent, attr, layer)
        wrapped[module_name] = layer
    return wrapped


def iter_static_lora_modules(model: nn.Module) -> dict[str, StaticLoRALinear]:
    return {name: module for name, module in model.named_modules() if isinstance(module, StaticLoRALinear)}
