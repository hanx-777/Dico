from __future__ import annotations

from dataclasses import dataclass

import torch

from dico.profiles import compute_sketch_signed_profile, recover_full_input_direction


@dataclass(frozen=True)
class SketchSvd:
    u: torch.Tensor
    singular_values: torch.Tensor
    v_tilde: torch.Tensor


@dataclass(frozen=True)
class DirectionProfile:
    profile: torch.Tensor
    full_v: torch.Tensor


def accumulate_response_sketch(
    y_state: torch.Tensor,
    gradients: torch.Tensor,
    activations: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    gradients = gradients.detach().float()
    activations = activations.detach().float()
    omega = omega.detach().float()
    if gradients.ndim == 3:
        gradients = gradients.reshape(-1, gradients.shape[-1])
    if activations.ndim == 3:
        activations = activations.reshape(-1, activations.shape[-1])
    return y_state.float() + gradients.T @ (activations @ omega)


def sketch_svd(y_state: torch.Tensor, top_k: int) -> SketchSvd:
    u, singular_values, vh = torch.linalg.svd(y_state.float(), full_matrices=False)
    k = min(int(top_k), int(singular_values.numel()))
    return SketchSvd(u=u[:, :k], singular_values=singular_values[:k], v_tilde=vh[:k, :].T)


def second_pass_direction_profile(
    gradients: torch.Tensor,
    activations: torch.Tensor,
    u: torch.Tensor,
    v_tilde: torch.Tensor,
    omega: torch.Tensor,
) -> DirectionProfile:
    return DirectionProfile(
        profile=compute_sketch_signed_profile(gradients, activations, u, v_tilde, omega),
        full_v=recover_full_input_direction(gradients, activations, u),
    )
