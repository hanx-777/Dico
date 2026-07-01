import torch

from dico_rank.atom_svd import (
    gradient_conflict,
    normalize_signed_profiles,
    sample_response_norm_from_token_factors,
    signed_projection_from_token_factors,
)


def test_signed_projection_matches_explicit_response_matrix():
    activations = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    gradients = torch.tensor([[0.5, 1.0, -1.0], [2.0, -0.5, 0.25]])
    u = torch.tensor([1.0, -2.0, 0.5])
    v = torch.tensor([0.25, -1.5])

    explicit_response = sum(torch.outer(g, a) for g, a in zip(gradients, activations)) / 2.0
    explicit = u @ explicit_response @ v
    streaming = signed_projection_from_token_factors(activations, gradients, u, v)

    assert torch.allclose(streaming, explicit)


def test_gradient_conflict_edges():
    assert gradient_conflict(torch.tensor([1.0, 2.0, 3.0])) == 0.0
    assert gradient_conflict(torch.tensor([-1.0, -2.0, -3.0])) == 0.0
    assert gradient_conflict(torch.tensor([1.0, -1.0, 2.0, -2.0])) == 1.0


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


def test_sample_response_norm_exact_small_matches_outer_product_average():
    activations = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    gradients = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    explicit_response = (torch.outer(gradients[0], activations[0]) + torch.outer(gradients[1], activations[1])) / 2.0

    norm = sample_response_norm_from_token_factors(activations, gradients, mode="exact_small")

    assert torch.allclose(norm, torch.linalg.norm(explicit_response))
