import torch
from torch import nn

from dico.lora_checkpoint import load_lora_state, save_lora_state
from dico.lora_masked import MaskedLoRALinear, inject_masked_lora
from dico.lora_static import StaticLoRALinear, inject_static_lora


def test_static_lora_checkpoint_round_trips_adapter_weights(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3, bias=False))
    inject_static_lora(model, {"0": 2}, alpha=4.0, dropout=0.0, scaling="alpha_over_r")
    assert isinstance(model[0], StaticLoRALinear)
    with torch.no_grad():
        model[0].lora_A.copy_(torch.tensor([[0.1, -0.2, 0.3, -0.4], [0.5, 0.6, -0.7, 0.8]]))
        model[0].lora_B.copy_(torch.tensor([[0.2, -0.1], [0.3, 0.4], [-0.5, 0.7]]))
    x = torch.randn(5, 4)
    expected = model(x)

    path = tmp_path / "adapter.pt"
    save_lora_state(path, model)
    with torch.no_grad():
        model[0].lora_A.zero_()
        model[0].lora_B.zero_()
    assert not torch.allclose(model(x), expected)

    report = load_lora_state(path, model)

    assert report["loaded_keys"] == ["0.lora_A", "0.lora_B"]
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert torch.allclose(model(x), expected)


def test_masked_lora_checkpoint_restores_rank_mask(tmp_path):
    model = nn.Sequential(nn.Linear(4, 3, bias=False))
    inject_masked_lora(model, {"0": 1}, max_rank=3, alpha=4.0, dropout=0.0)
    assert isinstance(model[0], MaskedLoRALinear)
    model[0].set_rank_mask(torch.tensor([1.0, 0.0, 1.0]))
    path = tmp_path / "masked_adapter.pt"
    save_lora_state(path, model)

    model[0].set_rank_mask(torch.zeros(3))
    assert model[0].get_active_rank() == 0

    report = load_lora_state(path, model)

    assert "0.rank_mask" in report["loaded_keys"]
    assert model[0].get_rank_mask().tolist() == [1.0, 0.0, 1.0]
