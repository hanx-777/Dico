from __future__ import annotations

import torch
from torch import nn

from dico.adalora import AdaLoRAConfig, AdaLoRAController, AdaLoRALinear, inject_adalora
from dico.rank_budget import compute_total_lora_params


def test_adalora_triplet_zero_e_preserves_output():
    torch.manual_seed(4)
    base = nn.Linear(4, 3, bias=False)
    x = torch.randn(2, 4)
    expected = base(x).detach()
    layer = AdaLoRALinear(base, rank=3, alpha=6, dropout=0.0, lora_dtype=torch.float32)

    assert layer.lora_A.shape == (3, 4)
    assert layer.lora_E.shape == (3,)
    assert layer.lora_B.shape == (3, 3)
    assert torch.count_nonzero(layer.lora_E).item() == 0
    assert torch.equal(layer(x), expected)


def test_adalora_cubic_schedule_uses_tfinal_as_final_warmup():
    cfg = AdaLoRAConfig(init_rank=12, target_rank=8, tinit=150, tfinal=900, delta_t=1, total_steps=1563)
    controller = AdaLoRAController({}, cfg)
    assert controller.total_rank_at_step(149) == 0
    # no modules means zero total budget, but phase boundaries remain auditable
    assert controller.phase_at_step(150) == "initial_warmup"
    assert controller.phase_at_step(151) == "budget_decrease"
    assert controller.phase_at_step(663) == "budget_decrease"
    assert controller.phase_at_step(664) == "final_finetune"


def test_adalora_global_budget_prunes_across_modules_and_has_orthogonal_loss():
    model = nn.Sequential(nn.Linear(4, 3, bias=False), nn.Linear(3, 2, bias=False))
    modules = inject_adalora(model, {"0": 2, "1": 2}, alpha=4, dropout=0.0, lora_dtype=torch.float32)
    cfg = AdaLoRAConfig(init_rank=2, target_rank=1, tinit=0, tfinal=0, delta_t=1, total_steps=1)
    controller = AdaLoRAController(modules, cfg)
    for module in modules.values():
        for parameter in (module.lora_A, module.lora_E, module.lora_B):
            parameter.grad = torch.ones_like(parameter)
    controller.update_importance()
    event = controller.step(1)

    assert event is not None
    assert sum(module.active_rank for module in modules.values()) == 2
    assert controller.orthogonal_regularization().ndim == 0


def test_adalora_has_no_per_module_target_rank_floor():
    model = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
    modules = inject_adalora(model, {"0": 2, "1": 2}, alpha=4, dropout=0.0, lora_dtype=torch.float32)
    controller = AdaLoRAController(
        modules,
        AdaLoRAConfig(init_rank=2, target_rank=1, tinit=0, tfinal=0, delta_t=1, total_steps=1),
    )
    # Make both globally best components belong to module 0.  Official AdaLoRA
    # permits module 1 to receive rank zero; it does not enforce target_r per layer.
    controller.exp_avg_ipt = {
        "0.A": torch.ones_like(modules["0"].lora_A),
        "0.E": torch.ones_like(modules["0"].lora_E),
        "0.B": torch.ones_like(modules["0"].lora_B),
        "1.A": torch.full_like(modules["1"].lora_A, 1e-3),
        "1.E": torch.full_like(modules["1"].lora_E, 1e-3),
        "1.B": torch.full_like(modules["1"].lora_B, 1e-3),
    }
    controller.exp_avg_unc = {key: torch.ones_like(value) for key, value in controller.exp_avg_ipt.items()}

    controller.step(1)

    assert modules["0"].active_rank == 2
    assert modules["1"].active_rank == 0


def test_pruned_adalora_singular_value_can_receive_gradient_again():
    torch.manual_seed(7)
    layer = AdaLoRALinear(nn.Linear(4, 3, bias=False), rank=2, alpha=4, lora_dtype=torch.float32)
    layer.set_rank_mask(torch.tensor([1.0, 0.0]))
    layer.zero_grad(set_to_none=True)

    layer(torch.randn(2, 4)).sum().backward()

    assert layer.lora_E.grad is not None
    assert layer.lora_E.grad[1].abs().item() > 0


def test_adalora_orthogonal_regularization_uses_unsquared_frobenius_norm():
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    modules = inject_adalora(model, {"0": 1}, alpha=1, dropout=0.0, lora_dtype=torch.float32)
    layer = modules["0"]
    with torch.no_grad():
        layer.lora_A.fill_(2.0)
        layer.lora_B.fill_(3.0)
    controller = AdaLoRAController(
        modules,
        AdaLoRAConfig(init_rank=1, target_rank=1, tinit=0, tfinal=0, total_steps=1, orth_reg_weight=0.5),
    )
    expected = 0.5 * (
        torch.linalg.matrix_norm(layer.lora_A @ layer.lora_A.T - torch.eye(1))
        + torch.linalg.matrix_norm(layer.lora_B.T @ layer.lora_B - torch.eye(1))
    ) / 2
    assert torch.allclose(controller.orthogonal_regularization(), expected)


def test_llama3_adalora_physical_peak_and_final_active_parameter_counts():
    dims = {}
    for layer in range(32):
        dims[f"layers.{layer}.q_proj"] = {"in_dim": 4096, "out_dim": 4096}
        dims[f"layers.{layer}.o_proj"] = {"in_dim": 4096, "out_dim": 4096}
        dims[f"layers.{layer}.k_proj"] = {"in_dim": 4096, "out_dim": 1024}
        dims[f"layers.{layer}.v_proj"] = {"in_dim": 4096, "out_dim": 1024}
    peak_allocation = {name: 12 for name in dims}
    final_allocation = {name: 8 for name in dims}

    peak = compute_total_lora_params(peak_allocation, dims) + sum(peak_allocation.values())
    final = compute_total_lora_params(final_allocation, dims) + sum(final_allocation.values())

    assert peak == 10_225_152
    assert final == 6_816_768
