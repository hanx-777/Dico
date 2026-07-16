from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from dico.lora_static import StaticLoRALinear
from dico.trainer import build_method_optimizer, official_warmup_steps, training_warmup_steps


def test_standard_lora_scaling_and_zero_b_preserve_base_output():
    torch.manual_seed(7)
    base = nn.Linear(5, 4, bias=False)
    layer = StaticLoRALinear(copy.deepcopy(base), rank=8, alpha=16, scaling="alpha_over_r")
    x = torch.randn(3, 5)

    assert layer.scaling == pytest.approx(2.0)
    assert torch.count_nonzero(layer.lora_B).item() == 0
    assert torch.equal(layer(x), base(x))


def test_gora_optimizer_keeps_b_to_a_lr_ratio_at_16():
    model = nn.Sequential(StaticLoRALinear(nn.Linear(4, 3, bias=False), rank=2, alpha=4, scaling="alpha_over_r"))
    optimizer = build_method_optimizer(
        model,
        method="gora_public",
        learning_rate=5e-5,
        weight_decay=5e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        gora_b_lr_multiplier=16.0,
    )
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"] == pytest.approx(16.0)


def test_official_warmup_rounds_with_plus_one():
    assert official_warmup_steps(1563, 0.03) == 47


def test_reference_covra_warmup_uses_floor_product_without_plus_one():
    config = {
        "method": "dico_cd_da",
        "preallocation": {"allocation_method": "covra_v05"},
        "training": {"max_steps": 1562, "warmup_ratio": 0.03},
    }

    assert training_warmup_steps(config) == 46


@pytest.mark.parametrize(
    ("method", "allocation_method"),
    [
        ("lora", None),
        ("dico_cd_da", "covra_full"),
        ("gora_public", "gora_public"),
    ],
)
def test_non_reference_methods_keep_plus_one_warmup(method: str, allocation_method: str | None):
    config = {
        "method": method,
        "preallocation": {"allocation_method": allocation_method},
        "training": {"max_steps": 1563, "warmup_ratio": 0.03},
    }

    assert training_warmup_steps(config) == 47
