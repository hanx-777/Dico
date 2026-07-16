import torch
from torch import nn

from dico.adalora import AdaLoRAController, AdaLoRAConfig
from dico.lora_masked import inject_masked_lora


def test_adalora_controller_prunes_to_target_rank_using_channel_scores():
    model = nn.Sequential(nn.Linear(4, 3, bias=False))
    modules = inject_masked_lora(
        model,
        {"0": 2},
        max_rank=2,
        alpha=4.0,
        dropout=0.0,
        lora_dtype=torch.float32,
    )
    layer = modules["0"]
    with torch.no_grad():
        layer.lora_A[0].fill_(10.0)
        layer.lora_B[:, 0].fill_(10.0)
        layer.lora_A[1].fill_(0.01)
        layer.lora_B[:, 1].fill_(0.01)

    controller = AdaLoRAController(
        modules,
        AdaLoRAConfig(init_rank=2, target_rank=1, tinit=0, tfinal=1, update_interval=1),
    )
    event = controller.step(global_step=1)

    assert event is not None
    assert event["target_active_rank"] == 1
    assert layer.get_active_rank() == 1
    assert layer.get_rank_mask().tolist() == [1.0, 0.0]
    assert controller.current_allocation() == {"0": 1}
    assert controller.peak_allocation() == {"0": 2}
