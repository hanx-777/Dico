from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class ResponseBlock:
    module_name: str
    candidate_index: int
    matrix: torch.Tensor
    rank_cost: int
    split: bool
    positive_energy_ratio: float
    negative_energy_ratio: float


@dataclass(frozen=True)
class UtilityCurve:
    selected_indices: list[int]
    marginal_gains: list[float]
    cumulative_utility: list[float]
    trace: list[dict[str, object]]


def _as_response_vector(response: torch.Tensor) -> torch.Tensor:
    tensor = response.detach().float().reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("response vector must contain at least one sample")
    if not torch.isfinite(tensor).all():
        raise ValueError("response vector contains NaN or Inf")
    return tensor


def build_response_block(
    module_name: str,
    candidate_index: int,
    response: torch.Tensor,
    rho: float,
    sign_split: bool = True,
) -> ResponseBlock:
    """Represent one physical candidate as a response block.

    If both positive and negative parts carry at least ``rho`` of the response
    energy, the block has two columns but still consumes exactly one rank.
    """

    vector = _as_response_vector(response)
    positive = torch.clamp(vector, min=0.0)
    negative = torch.clamp(-vector, min=0.0)
    total_energy = float(torch.sum(vector * vector).item())
    if total_energy <= 0.0:
        positive_ratio = 0.0
        negative_ratio = 0.0
    else:
        positive_ratio = float(torch.sum(positive * positive).item() / total_energy)
        negative_ratio = float(torch.sum(negative * negative).item() / total_energy)

    should_split = bool(sign_split) and min(positive_ratio, negative_ratio) >= float(rho)
    matrix = torch.stack([positive, negative], dim=1) if should_split else vector.reshape(-1, 1)
    return ResponseBlock(
        module_name=str(module_name),
        candidate_index=int(candidate_index),
        matrix=matrix,
        rank_cost=1,
        split=should_split,
        positive_energy_ratio=positive_ratio,
        negative_energy_ratio=negative_ratio,
    )


def _orthonormal_basis(columns: torch.Tensor, eps: float) -> torch.Tensor:
    if columns.numel() == 0 or columns.shape[1] == 0:
        return columns.new_zeros((columns.shape[0], 0))
    q, r = torch.linalg.qr(columns, mode="reduced")
    diag = torch.abs(torch.diagonal(r))
    keep = diag > float(eps)
    if not bool(keep.any()):
        return columns.new_zeros((columns.shape[0], 0))
    return q[:, keep]


def _residual_block(block: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.shape[1] == 0:
        return block
    return block - basis @ (basis.T @ block)


def _block_gain(block: torch.Tensor, basis: torch.Tensor, eps: float) -> tuple[float, bool]:
    residual = _residual_block(block, basis)
    gain = float(torch.sum(residual * residual).item() / max(int(block.shape[0]), 1))
    return gain, gain <= float(eps)


def greedy_conditional_coverage(
    blocks: Sequence[ResponseBlock],
    r_max: int,
    eps: float = 1.0e-10,
) -> UtilityCurve:
    """Greedily build a module-local conditional marginal utility curve."""

    if int(r_max) < 0:
        raise ValueError("r_max must be non-negative")
    if not blocks:
        return UtilityCurve([], [], [0.0], [])

    remaining = list(blocks)
    selected_indices: list[int] = []
    marginal_gains: list[float] = []
    trace: list[dict[str, object]] = []
    basis = remaining[0].matrix.new_zeros((remaining[0].matrix.shape[0], 0))
    selected_columns = basis
    limit = min(int(r_max), len(remaining))

    for step in range(limit):
        scored: list[tuple[float, int, bool]] = []
        for idx, block in enumerate(remaining):
            gain, near_zero = _block_gain(block.matrix, basis, eps)
            scored.append((gain, idx, near_zero))
        best_gain, best_pos, near_zero = max(
            scored,
            key=lambda item: (item[0], -remaining[item[1]].candidate_index),
        )
        chosen = remaining.pop(best_pos)
        selected_indices.append(chosen.candidate_index)
        marginal_gains.append(0.0 if abs(best_gain) <= float(eps) else float(best_gain))
        selected_columns = torch.cat([selected_columns, chosen.matrix], dim=1)
        basis = _orthonormal_basis(selected_columns, eps)
        trace.append(
            {
                "step": step + 1,
                "candidate_index": chosen.candidate_index,
                "gain_before_selection": float(best_gain),
                "selected_indices": list(selected_indices),
                "near_zero_residual": bool(near_zero),
                "basis_rank": int(basis.shape[1]),
            }
        )

    return UtilityCurve(
        selected_indices=selected_indices,
        marginal_gains=marginal_gains,
        cumulative_utility=_cumulative(marginal_gains),
        trace=trace,
    )


def independent_utility_curve(blocks: Sequence[ResponseBlock], r_max: int) -> UtilityCurve:
    gains = [
        float(torch.sum(block.matrix * block.matrix).item() / max(int(block.matrix.shape[0]), 1))
        for block in blocks
    ]
    order = sorted(range(len(blocks)), key=lambda idx: (gains[idx], -blocks[idx].candidate_index), reverse=True)
    order = order[: int(r_max)]
    selected_indices = [blocks[idx].candidate_index for idx in order]
    marginal_gains = [gains[idx] for idx in order]
    trace = [
        {
            "step": step,
            "candidate_index": candidate_index,
            "fixed_independent_gain": gain,
        }
        for step, (candidate_index, gain) in enumerate(zip(selected_indices, marginal_gains), start=1)
    ]
    return UtilityCurve(selected_indices, marginal_gains, _cumulative(marginal_gains), trace)


def module_scalar_utility_curve(module_energy: float, r_max: int, template: Sequence[float]) -> list[float]:
    if int(r_max) < 0:
        raise ValueError("r_max must be non-negative")
    if len(template) < int(r_max):
        raise ValueError("module scalar template must provide at least r_max entries")
    weights = [max(0.0, float(value)) for value in template[: int(r_max)]]
    if any(weights[i] < weights[i + 1] for i in range(len(weights) - 1)):
        raise ValueError("module scalar template must be non-increasing")
    total = sum(weights)
    if total <= 0:
        raise ValueError("module scalar template must have positive mass")
    gains = [float(module_energy) * weight / total for weight in weights]
    return _cumulative(gains)


def build_type_scaled_utility_curves(
    cumulative_curves: Mapping[str, Sequence[float]],
    module_types: Mapping[str, str],
    type_scaling: bool = True,
    log_compression: bool = True,
    eps: float = 1.0e-12,
) -> dict[str, list[float]]:
    marginal_by_module = {
        name: _marginal_from_cumulative(curve)
        for name, curve in cumulative_curves.items()
    }
    type_medians: dict[str, float] = {}
    for module_type in sorted(set(module_types.values())):
        positives = [
            value
            for name, values in marginal_by_module.items()
            if module_types[name] == module_type
            for value in values
            if value > 0.0 and math.isfinite(value)
        ]
        if positives:
            type_medians[module_type] = float(torch.median(torch.tensor(positives)).item())
        else:
            type_medians[module_type] = 1.0
        if type_medians[module_type] <= float(eps):
            type_medians[module_type] = 1.0

    result: dict[str, list[float]] = {}
    for name, gains in marginal_by_module.items():
        module_type = module_types[name]
        scaled_gains = []
        for gain in gains:
            value = max(0.0, float(gain))
            if type_scaling:
                value = value / (type_medians[module_type] + float(eps))
            if log_compression:
                value = math.log1p(value)
            scaled_gains.append(value)
        result[name] = _cumulative(scaled_gains)
    return result


def _cumulative(gains: Sequence[float]) -> list[float]:
    out = [0.0]
    total = 0.0
    for gain in gains:
        total += max(0.0, float(gain))
        out.append(total)
    return out


def _marginal_from_cumulative(curve: Sequence[float]) -> list[float]:
    values = [float(value) for value in curve]
    if not values:
        return []
    if values[0] != 0.0:
        values = [0.0, *values]
    return [values[idx] - values[idx - 1] for idx in range(1, len(values))]
