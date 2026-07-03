import torch
from torch import nn

from dico_rank.lora_masked import MaskedLoRALinear, inject_masked_lora
from dico_rank.model_loader import collect_module_dims, find_target_linear_modules, is_linear_like_module


class LinearLikeModule(nn.Module):
    def __init__(self, in_features=4, out_features=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = None

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class ModelWithLinearLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].q_proj = LinearLikeModule()

    def forward(self, x):
        return self.layers[0].q_proj(x)


def test_linear_like_modules_are_discoverable_for_quantized_loading():
    model = ModelWithLinearLike()

    found = find_target_linear_modules(model, ["q_proj"])
    dims = collect_module_dims(found)

    assert is_linear_like_module(model.layers[0].q_proj)
    assert [name for name, _module in found] == ["layers.0.q_proj"]
    assert dims["layers.0.q_proj"] == {"in_dim": 4, "out_dim": 3}


def test_inject_masked_lora_wraps_linear_like_module():
    model = ModelWithLinearLike()

    wrapped = inject_masked_lora(
        model,
        {"layers.0.q_proj": 2},
        max_rank=4,
        alpha=4.0,
        dropout=0.0,
        lora_dtype=torch.bfloat16,
    )

    assert isinstance(model.layers[0].q_proj, MaskedLoRALinear)
    assert wrapped["layers.0.q_proj"].lora_A.dtype == torch.bfloat16
    assert wrapped["layers.0.q_proj"].get_active_rank() == 2
