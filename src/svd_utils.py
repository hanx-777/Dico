from typing import Tuple

import torch


def randomized_svd_topk(
    matrix: torch.Tensor,
    k: int,
    oversample: int = 8,
    n_iter: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2:
        raise ValueError("matrix must be rank 2")
    if k <= 0:
        raise ValueError("k must be positive")
    rows, cols = matrix.shape
    rank = min(k, rows, cols)
    q_dim = min(rank + oversample, rows, cols)
    work = matrix.detach().to(dtype=torch.float32, device="cpu")
    omega = torch.randn(cols, q_dim, dtype=torch.float32)
    y = work @ omega
    for _ in range(n_iter):
        y = work @ (work.T @ y)
    q, _ = torch.linalg.qr(y, mode="reduced")
    b = q.T @ work
    u_hat, s, vh = torch.linalg.svd(b, full_matrices=False)
    u = q @ u_hat
    return u[:, :rank].contiguous(), s[:rank].contiguous(), vh[:rank, :].T.contiguous()
