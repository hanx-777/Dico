from __future__ import annotations

import math
from typing import Mapping


def lora_scale(alpha: float, rank: int, mode: str = "alpha_over_sqrt_r") -> float:
    if int(rank) <= 0:
        return 0.0
    if mode == "alpha_over_sqrt_r":
        return float(alpha) / math.sqrt(float(rank))
    if mode in {"alpha_over_r", "alpha_over_rank"}:
        return float(alpha) / float(rank)
    if mode == "alpha_over_max_rank":
        return float(alpha) / float(rank)
    raise ValueError(f"Unsupported LoRA scaling mode: {mode}")


def compute_covra_module_alpha(
    rank_allocation: Mapping[str, int], alpha_ref: float, r_ref: float,
) -> dict[str, float]:
    """3.1節: alpha_m = r_m * alpha_ref / r_ref, so alpha_m/r_m == alpha_ref/r_ref for
    every module -- a fixed effective LoRA scaling ratio, independent of CovRA's
    heterogeneous r_m allocation. Paired with `scaling="alpha_over_r"`
    (`StaticLoRALinear`'s `alpha/rank` mode), this makes each module's realized
    `.scaling` come out to exactly `alpha_ref/r_ref`, so comparisons against
    uniform-rank baselines aren't confounded by per-module effective-learning-rate
    drift (3.1節's stated reason for fixing this ratio at all).
    """
    return {name: float(rank) * float(alpha_ref) / float(r_ref) for name, rank in rank_allocation.items()}
