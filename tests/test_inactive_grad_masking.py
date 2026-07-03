import torch
from torch import nn

from dico_rank.lora_masked import MaskedLoRALinear, apply_rank_masks_to_grads


def test_inactive_channel_backward_gradient_is_zero():
    torch.manual_seed(0)
    layer = MaskedLoRALinear(nn.Linear(4, 3, bias=False), max_rank=4, active_rank=2, alpha=4.0)
    x = torch.randn(6, 4)

    loss = layer(x).pow(2).mean()
    loss.backward()
    layer.apply_rank_mask_to_grads()

    assert torch.count_nonzero(layer.lora_A.grad[2:]).item() == 0
    assert torch.count_nonzero(layer.lora_B.grad[:, 2:]).item() == 0


def test_inactive_channel_params_do_not_change_after_optimizer_step_with_weight_decay():
    torch.manual_seed(0)
    layer = MaskedLoRALinear(nn.Linear(4, 3, bias=False), max_rank=4, active_rank=2, alpha=4.0)
    inactive_a_before = layer.lora_A.detach()[2:].clone()
    inactive_b_before = layer.lora_B.detach()[:, 2:].clone()

    optimizer = torch.optim.AdamW(
        [{"params": [layer.lora_A, layer.lora_B], "weight_decay": 0.1}],
        lr=1e-2,
    )
    x = torch.randn(6, 4)
    loss = layer(x).pow(2).mean()
    loss.backward()
    apply_rank_masks_to_grads([layer])
    optimizer.step()
    layer.restore_inactive_parameters()

    assert torch.allclose(layer.lora_A.detach()[2:], inactive_a_before)
    assert torch.allclose(layer.lora_B.detach()[:, 2:], inactive_b_before)


def test_inactive_channels_stay_bitwise_fixed_across_adamw_steps():
    torch.manual_seed(0)
    layer = MaskedLoRALinear(nn.Linear(4, 3, bias=False), max_rank=4, active_rank=2, alpha=4.0)
    inactive_a_before = layer.lora_A.detach()[2:].clone()
    inactive_b_before = layer.lora_B.detach()[:, 2:].clone()

    optimizer = torch.optim.AdamW(
        [{"params": [layer.lora_A, layer.lora_B], "weight_decay": 0.1}],
        lr=1e-2,
    )
    for step in range(10):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(step)
        x = torch.randn(6, 4)
        loss = layer(x).pow(2).mean()
        loss.backward()
        apply_rank_masks_to_grads([layer])
        optimizer.step()
        layer.restore_inactive_parameters()

    assert torch.equal(layer.lora_A.detach()[2:], inactive_a_before)
    assert torch.equal(layer.lora_B.detach()[:, 2:], inactive_b_before)

    state_a = optimizer.state[layer.lora_A]
    state_b = optimizer.state[layer.lora_B]
    assert torch.count_nonzero(state_a["exp_avg"][2:]).item() == 0
    assert torch.count_nonzero(state_b["exp_avg"][:, 2:]).item() == 0
