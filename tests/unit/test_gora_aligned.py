from __future__ import annotations

import torch
from torch import nn

from dico.gora import (
    allocate_gora_ranks,
    collect_average_weight_gradients,
    compute_gora_importance,
    gora_pseudoinverse_init,
    strict_budget_repair,
)
from dico.rank_budget import compute_total_lora_params


def test_gora_importance_matches_reference_formula():
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    avg_grad = torch.tensor([[2.0, 1.0], [-1.0, 0.5]])
    assert compute_gora_importance(weight, avg_grad).item() == torch.mean(torch.abs(weight * avg_grad)).item()


def test_gora_pseudoinverse_initialization_matches_reference_formula():
    grad = torch.tensor([[1.0, 2.0, 3.0], [2.0, -1.0, 0.0]])
    A = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    B = gora_pseudoinverse_init(grad, A, eps=1e-8)
    expected = grad @ A.T @ torch.linalg.pinv(A @ A.T + 1e-8 * torch.eye(2))
    assert torch.allclose(B, expected)


def test_gora_public_is_not_strictly_repaired_but_bm_never_exceeds_budget():
    dims = {"a": {"in_dim": 4, "out_dim": 4}, "b": {"in_dim": 4, "out_dim": 8}}
    importance = {"a": 0.9, "b": 0.1}
    allocation = allocate_gora_ranks(importance, dims, r_ref=8, r_min=4, r_max=32, rounding="moderate")
    target = compute_total_lora_params({"a": 8, "b": 8}, dims)
    repaired = strict_budget_repair(allocation, importance, dims, target, r_min=4, r_max=32)
    assert compute_total_lora_params(repaired, dims) == target


class _TwoTargetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(3, 3, bias=False)
        self.second = nn.Linear(3, 2, bias=False)
        self.untargeted = nn.Linear(2, 1, bias=False)
        self.backward_calls = 0

    def forward(self, input_ids, labels=None, attention_mask=None):
        del labels, attention_mask
        hidden = self.second(torch.tanh(self.first(input_ids.float())))
        output = self.untargeted(hidden)
        loss = output.square().mean()
        loss.register_hook(lambda grad: self._record_backward(grad))
        return type("Output", (), {"loss": loss})()

    def _record_backward(self, grad):
        self.backward_calls += 1
        return grad


def test_gora_collects_direct_weight_gradients_once_per_batch_without_untargeted_grads():
    torch.manual_seed(3)
    model = _TwoTargetModel()
    batches = [
        {"input_ids": torch.randn(2, 3), "labels": torch.zeros(2, 3, dtype=torch.long)},
        {"input_ids": torch.randn(2, 3), "labels": torch.zeros(2, 3, dtype=torch.long)},
    ]

    # Independent direct autograd reference.
    reference = {"first": torch.zeros_like(model.first.weight), "second": torch.zeros_like(model.second.weight)}
    for batch in batches:
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        grads = torch.autograd.grad(output.loss, [model.first.weight, model.second.weight])
        for name, grad in zip(reference, grads):
            reference[name] += grad.detach()
    reference = {name: value / len(batches) for name, value in reference.items()}
    model.backward_calls = 0

    gradients, metadata = collect_average_weight_gradients(
        model,
        ["first", "second"],
        batches,
        offload_device="cpu",
        accumulation_dtype="float32",
    )

    assert model.backward_calls == len(batches)
    assert metadata["backward_passes"] == len(batches)
    assert metadata["answer_only"] is False
    assert model.untargeted.weight.grad is None
    for name in reference:
        assert torch.allclose(gradients[name], reference[name], atol=1e-6, rtol=1e-5)
