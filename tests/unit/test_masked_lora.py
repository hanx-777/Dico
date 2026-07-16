import torch
from torch import nn

from dico.lora_masked import MaskedLoRALinear


def test_active_rank_changes_forward_and_query():
    torch.manual_seed(0)
    base = nn.Linear(3, 2, bias=False)
    layer = MaskedLoRALinear(base, max_rank=4, active_rank=1, alpha=4.0)
    x = torch.randn(5, 3)

    assert layer.get_active_rank() == 1
    y_rank1 = layer(x)

    layer.set_active_rank(3)

    assert layer.get_active_rank() == 3
    y_rank3 = layer(x)
    assert y_rank3.shape == y_rank1.shape
    assert not torch.allclose(y_rank1, y_rank3)


def test_zero_mask_matches_base_output():
    torch.manual_seed(0)
    base = nn.Linear(3, 2, bias=False)
    layer = MaskedLoRALinear(base, max_rank=4, active_rank=2, alpha=4.0)
    x = torch.randn(2, 3)

    layer.set_rank_mask(torch.zeros(4))

    assert layer.get_active_rank() == 0
    assert torch.allclose(layer(x), base(x))


def test_open_close_channel_updates_mask():
    base = nn.Linear(3, 2, bias=False)
    layer = MaskedLoRALinear(base, max_rank=4, active_rank=0, alpha=4.0)

    layer.open_channel(2)
    layer.open_channel(0)
    layer.close_channel(2)

    assert layer.get_rank_mask().tolist() == [1.0, 0.0, 0.0, 0.0]
    assert layer.get_active_rank() == 1
