import torch

from dico.profiles import (
    compute_sketch_signed_profile,
    recover_full_input_direction,
)


def test_signed_profile_uses_sketch_domain_v_tilde_not_full_v():
    gradients = torch.tensor([[[2.0, -1.0], [1.0, 3.0]]])
    activations = torch.tensor([[[1.0, 4.0, -2.0], [0.5, -1.0, 3.0]]])
    omega = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]])
    u = torch.tensor([0.25, -0.5])
    v_tilde = torch.tensor([2.0, -1.0])
    misleading_full_v = torch.tensor([100.0, -50.0, 25.0])

    profile = compute_sketch_signed_profile(
        gradients,
        activations,
        u,
        v_tilde,
        omega,
        full_v=misleading_full_v,
    )

    left = gradients[0] @ u
    right = (activations[0] @ omega) @ v_tilde
    expected = torch.sum(left * right)
    full_v_wrong = torch.sum(left * (activations[0] @ misleading_full_v))
    assert torch.allclose(profile, torch.tensor([expected]))
    assert not torch.allclose(profile, torch.tensor([full_v_wrong]))


def test_recovered_full_direction_is_normalized_and_separate_from_profile():
    gradients = torch.tensor([[[2.0, -1.0], [1.0, 3.0]]])
    activations = torch.tensor([[[1.0, 4.0, -2.0], [0.5, -1.0, 3.0]]])
    u = torch.tensor([0.25, -0.5])

    direction = recover_full_input_direction(gradients, activations, u)

    expected = ((gradients[0] @ u).reshape(-1, 1) * activations[0]).sum(dim=0)
    expected = expected / torch.linalg.norm(expected)
    assert torch.allclose(direction, expected, atol=1e-6)
    assert torch.allclose(torch.linalg.norm(direction), torch.tensor(1.0), atol=1e-6)
