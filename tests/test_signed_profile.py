import torch
from torch import nn

from dico_rank.atom_svd import (
    direction_alignment,
    gradient_conflict,
    normalize_signed_profiles,
    _run_backward_and_collect,
    sample_response_norm_from_token_factors,
    signed_projection_from_token_factors,
)


def test_signed_projection_matches_explicit_response_matrix():
    activations = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    gradients = torch.tensor([[0.5, 1.0, -1.0], [2.0, -0.5, 0.25]])
    u = torch.tensor([1.0, -2.0, 0.5])
    v = torch.tensor([0.25, -1.5])

    explicit_response = sum(torch.outer(g, a) for g, a in zip(gradients, activations))
    explicit = u @ explicit_response @ v
    streaming = signed_projection_from_token_factors(activations, gradients, u, v)

    assert torch.allclose(streaming, explicit)


def test_gradient_conflict_edges():
    assert gradient_conflict(torch.tensor([1.0, 2.0, 3.0])) == 0.0
    assert gradient_conflict(torch.tensor([-1.0, -2.0, -3.0])) == 0.0
    assert gradient_conflict(torch.tensor([1.0, -1.0, 2.0, -2.0])) == 1.0


def test_direction_alignment_uses_signed_net_over_absolute_mass():
    aligned = direction_alignment(torch.tensor([1.0, 2.0, 3.0]))
    conflicting = direction_alignment(torch.tensor([2.0, -1.0, -1.0]))

    assert aligned == 1.0
    assert conflicting == 0.0


def test_profile_normalization_exact_small_centers_and_normalizes():
    alpha = torch.tensor(
        [
            [2.0, 4.0],
            [4.0, 8.0],
            [8.0, 2.0],
        ]
    )
    denominators = torch.tensor([2.0, 4.0, 2.0])

    normalized = normalize_signed_profiles(alpha, denominators, mode="exact_small")

    assert torch.allclose(normalized.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(torch.linalg.norm(normalized, dim=0), torch.ones(2), atol=1e-6)


def test_sample_response_norm_exact_small_matches_outer_product_sum():
    activations = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    gradients = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    explicit_response = torch.outer(gradients[0], activations[0]) + torch.outer(gradients[1], activations[1])

    norm = sample_response_norm_from_token_factors(activations, gradients, mode="exact_small")

    assert torch.allclose(norm, torch.linalg.norm(explicit_response))


class TinyTokenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 3, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask, labels
        x = torch.nn.functional.one_hot(input_ids, num_classes=2).float()
        y = self.proj(x)
        return type("Output", (), {"loss": y.sum(), "logits": y})()


def test_backward_collect_applies_answer_mask_before_returning_token_factors():
    model = TinyTokenLinear()
    modules = dict(model.named_modules())
    batch = {
        "input_ids": torch.tensor([[0, 1, 0, 1]]),
        "labels": torch.tensor([[-100, -100, 5, 5]]),
    }

    answer_only = _run_backward_and_collect(
        model,
        modules,
        ["proj"],
        batch,
        answer_only=True,
        module_chunk_size=1,
    )
    full = _run_backward_and_collect(
        model,
        modules,
        ["proj"],
        batch,
        answer_only=False,
        module_chunk_size=1,
    )

    answer_a, answer_g = answer_only["proj"].token_slices(0)
    full_a, full_g = full["proj"].token_slices(0)
    assert answer_a.shape == (2, 2)
    assert answer_g.shape == (2, 3)
    assert full_a.shape == (4, 2)
    assert full_g.shape == (4, 3)
