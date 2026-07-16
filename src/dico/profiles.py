from __future__ import annotations

import torch


def _ensure_batched_tokens(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().float()
    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError("expected tensor with shape [batch, tokens, dim] or [tokens, dim]")
    return value


def compute_sketch_signed_profile(
    gradients: torch.Tensor,
    activations: torch.Tensor,
    u: torch.Tensor,
    v_tilde: torch.Tensor,
    omega: torch.Tensor,
    full_v: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute CovRA v0.5 signed profiles in sketch geometry (4.3节).

    ``full_v`` is accepted only to make accidental dependency visible in tests;
    it is intentionally ignored because full-dimensional directions are reserved
    for direction-anchored initialization.
    """

    del full_v
    gradients = _ensure_batched_tokens(gradients)
    activations = _ensure_batched_tokens(activations)
    if gradients.shape[:2] != activations.shape[:2]:
        raise ValueError("gradients and activations must share batch/token axes")
    u = u.detach().float()
    v_tilde = v_tilde.detach().float()
    omega = omega.detach().float()
    left = torch.matmul(gradients, u)
    sketch_activations = torch.matmul(activations, omega)
    right = torch.matmul(sketch_activations, v_tilde)
    return torch.sum(left * right, dim=1)


def recover_full_input_direction(
    gradients: torch.Tensor,
    activations: torch.Tensor,
    u: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    gradients = _ensure_batched_tokens(gradients)
    activations = _ensure_batched_tokens(activations)
    u = u.detach().float()
    weights = torch.matmul(gradients, u).unsqueeze(-1)
    z = torch.sum(weights * activations, dim=(0, 1))
    norm = torch.linalg.norm(z)
    if float(norm.item()) <= float(eps):
        return torch.zeros_like(z)
    return z / norm
