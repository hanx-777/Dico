import copy

import torch
from torch import nn

from dico.lora_masked import MaskedLoRALinear
from dico.lora_scaling import lora_scale
from dico.lora_static import StaticLoRALinear, inject_static_lora


def test_static_lora_matches_legacy_masked_lora_when_rank_equals_max_rank():
    torch.manual_seed(0)
    base = nn.Linear(4, 3, bias=False)
    static = StaticLoRALinear(copy.deepcopy(base), rank=2, alpha=6.0, dropout=0.0, scaling="alpha_over_r")
    legacy = MaskedLoRALinear(copy.deepcopy(base), max_rank=2, active_rank=2, alpha=6.0, dropout=0.0, scaling="alpha_over_r")
    with torch.no_grad():
        static.lora_A.copy_(torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.5, 0.6, -0.7, 0.8]]))
        static.lora_B.copy_(torch.tensor([[0.2, -0.1], [0.3, 0.4], [-0.5, 0.7]]))
        legacy.lora_A.copy_(static.lora_A)
        legacy.lora_B.copy_(static.lora_B)
    x_static = torch.randn(5, 4, requires_grad=True)
    x_legacy = x_static.detach().clone().requires_grad_(True)

    y_static = static(x_static)
    y_legacy = legacy(x_legacy)
    assert torch.allclose(y_static, y_legacy, atol=1e-6)

    y_static.sum().backward()
    y_legacy.sum().backward()
    assert torch.allclose(x_static.grad, x_legacy.grad, atol=1e-6)
    assert torch.allclose(static.lora_A.grad, legacy.lora_A.grad, atol=1e-6)
    assert torch.allclose(static.lora_B.grad, legacy.lora_B.grad, atol=1e-6)
    assert sum(p.numel() for p in static.parameters() if p.requires_grad) == sum(
        p.numel() for p in legacy.parameters() if p.requires_grad
    )


def test_static_rank_zero_does_not_wrap_module():
    model = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Linear(3, 2, bias=False))

    wrapped = inject_static_lora(
        model,
        {"0": 0, "1": 2},
        alpha=8.0,
        dropout=0.0,
        scaling="alpha_over_sqrt_r",
    )

    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], StaticLoRALinear)
    assert list(wrapped) == ["1"]
    assert model[1].scaling == lora_scale(8.0, 2, "alpha_over_sqrt_r")


def test_inject_static_lora_accepts_per_module_alpha_mapping():
    # 3.1節: CovRA's fixed alpha_m/r_m ratio is threaded in as a per-module mapping,
    # not a single shared float.
    model = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Linear(3, 2, bias=False))
    rank_allocation = {"0": 3, "1": 5}
    alpha_by_module = {"0": 6.0, "1": 10.0}  # alpha_m = r_m * 2.0 for both -> ratio 2.0

    wrapped = inject_static_lora(
        model,
        rank_allocation,
        alpha=alpha_by_module,
        dropout=0.0,
        scaling="alpha_over_r",
    )

    assert wrapped["0"].alpha == 6.0
    assert wrapped["1"].alpha == 10.0
    for module_name, rank in rank_allocation.items():
        assert wrapped[module_name].scaling == alpha_by_module[module_name] / rank
