from __future__ import annotations

import torch


def make_random_projection(
    in_dim: int,
    sketch_dim: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    omega = torch.randn(int(in_dim), int(sketch_dim), generator=generator, dtype=dtype)
    omega = omega / max(float(sketch_dim) ** 0.5, 1.0)
    return omega.to(device)


def orthonormal_basis(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    q, r = torch.linalg.qr(matrix, mode="reduced")
    if r.numel() == 0:
        return q
    keep = torch.abs(torch.diagonal(r)) > eps
    if bool(keep.any()):
        return q[:, keep]
    return q[:, : min(matrix.shape)]


def append_gram_schmidt(
    basis: torch.Tensor,
    vector: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    vector = vector.float()
    if basis.numel() and basis.shape[1] > 0:
        vector = vector - basis @ (basis.T @ vector)
    norm = torch.linalg.norm(vector)
    if float(norm) <= eps:
        return basis
    new_col = (vector / norm).reshape(-1, 1)
    if basis.numel() == 0:
        return new_col
    return torch.cat([basis, new_col], dim=1)
