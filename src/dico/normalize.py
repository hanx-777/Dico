from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class NormalizationStats:
    median_by_type: dict[str, float] = field(default_factory=dict)
    mad_by_type: dict[str, float] = field(default_factory=dict)


def _median_mad(values: list[float], eps: float) -> tuple[float, float]:
    median = statistics.median(values)
    mad = statistics.median([abs(value - median) for value in values]) + float(eps)
    return median, mad


def _stable_softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def compute_normalized_utility(
    joint_utilities: Mapping[str, float],
    type_of: Mapping[str, str],
    eps: float = 1.0e-6,
) -> tuple[dict[str, float], NormalizationStats]:
    """3.4.1节: log compression, then type-wise median/MAD z-scoring, softplus.

    Returns (w_bar_p per physical direction id, NormalizationStats for
    normalization_stats.json). This is the single source of "which physical
    directions are comparable to which" -- callers must supply type_of for
    every key in joint_utilities.
    """
    physical_ids = list(joint_utilities.keys())
    log_utility = {p: math.log(float(joint_utilities[p]) + float(eps)) for p in physical_ids}

    by_type: dict[str, list[str]] = {}
    for p in physical_ids:
        by_type.setdefault(type_of[p], []).append(p)

    median_by_type: dict[str, float] = {}
    mad_by_type: dict[str, float] = {}
    for type_name, ids in by_type.items():
        median, mad = _median_mad([log_utility[p] for p in ids], eps)
        median_by_type[type_name] = median
        mad_by_type[type_name] = mad

    w_bar: dict[str, float] = {}
    for p in physical_ids:
        type_name = type_of[p]
        z_p = (log_utility[p] - median_by_type[type_name]) / mad_by_type[type_name]
        w_bar[p] = _stable_softplus(z_p)

    stats = NormalizationStats(median_by_type=median_by_type, mad_by_type=mad_by_type)
    return w_bar, stats


def apply_normalized_utility(
    raw_utilities: Mapping[str, float],
    type_of: Mapping[str, str],
    stats: NormalizationStats,
    eps: float = 1.0e-6,
) -> dict[str, float]:
    """Apply an already-fitted (median_by_type, mad_by_type) to a *different*
    pool (e.g. the reserve queue) without re-estimating scale from that pool --
    3.4.2节: "标准化时借用主方向的类型级统计量...不在后备队列内部重新估计". A type
    that never appeared in the fitted pool falls back to (median=0.0, mad=1.0)
    (no re-centering/scaling) rather than raising -- callers should treat this as
    a diagnostic-worthy edge case, not an error.
    """
    w_bar: dict[str, float] = {}
    for p, value in raw_utilities.items():
        log_u = math.log(float(value) + float(eps))
        type_name = type_of[p]
        median = stats.median_by_type.get(type_name, 0.0)
        mad = stats.mad_by_type.get(type_name, 1.0)
        w_bar[p] = _stable_softplus((log_u - median) / mad)
    return w_bar
