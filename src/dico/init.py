from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class InitResult:
    A: torch.Tensor
    B: torch.Tensor
    summary: dict[str, object]


def _normalized_residual(vector: torch.Tensor, basis: list[torch.Tensor], eps: float = 1.0e-12) -> torch.Tensor | None:
    residual = vector.detach().float().clone()
    for row in basis:
        residual = residual - row * torch.dot(residual, row)
    norm = torch.linalg.norm(residual)
    if float(norm.item()) <= float(eps):
        return None
    return residual / norm


def build_direction_anchored_init(
    directions: Sequence[Mapping[str, object]],
    rank: int,
    in_dim: int,
    out_dim: int,
    seed: int = 42,
    zero_B: bool = True,
) -> InitResult:
    ordered = sorted(directions, key=lambda row: float(row.get("utility", 0.0)), reverse=True)
    rows: list[torch.Tensor] = []
    certified_rows = 0
    for row in ordered:
        if len(rows) >= int(rank):
            break
        value = row.get("v")
        if value is None:
            continue
        residual = _normalized_residual(torch.as_tensor(value, dtype=torch.float32).reshape(int(in_dim)), rows)
        if residual is None:
            continue
        rows.append(residual)
        if row.get("source", "certified") != "relaxation":
            certified_rows += 1
    # 3.5節: "一个方向单元至多贡献一个 rank" -- r_m' <= r_m is checked explicitly here
    # ("实现中以断言检查"), not left as an implicit consequence of the loop guard above.
    assert certified_rows <= int(rank), (
        f"consumed {certified_rows} certified/reserve directions but only {rank} rank slots available"
    )
    generator = torch.Generator().manual_seed(int(seed))
    relaxation_rows = 0
    while len(rows) < int(rank):
        # 3.5节: 标准 Kaiming 随机向量 for the relaxation fallback rows (any
        # isotropic distribution is equivalent after _normalized_residual's
        # normalization, but this matches the doc's literal wording and
        # StaticLoRALinear.reset_parameters's own convention).
        candidate = torch.empty(1, int(in_dim))
        torch.nn.init.kaiming_uniform_(candidate, a=math.sqrt(5), generator=generator)
        candidate = candidate.squeeze(0)
        residual = _normalized_residual(candidate, rows)
        if residual is None:
            continue
        rows.append(residual)
        relaxation_rows += 1
    A = torch.stack(rows, dim=0) if rows else torch.empty(0, int(in_dim))
    if zero_B:
        B = torch.zeros(int(out_dim), int(rank), dtype=torch.float32)
    else:
        b_generator = torch.Generator().manual_seed(int(seed) + 1)
        B = torch.empty(int(out_dim), int(rank)).normal_(mean=0.0, std=1e-3, generator=b_generator)
    return InitResult(
        A=A,
        B=B,
        summary={
            "rank": int(rank),
            "in_dim": int(in_dim),
            "out_dim": int(out_dim),
            "certified_rows": certified_rows,
            "relaxation_rows": relaxation_rows,
            "delta_w_zero": bool(torch.count_nonzero(B @ A).item() == 0),
        },
    )
