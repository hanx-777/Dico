from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from scipy.stats import ks_2samp

from dico.candidates import VirtualCandidate
from dico.coverage import compute_kappa
from dico.path_utils import extract_layer_index


@dataclass(frozen=True)
class KappaCalibrationResult:
    module_type: str
    observed_mean_abs_kappa: float
    null_mean_abs_kappa: float
    ks_statistic: float
    ks_pvalue: float
    num_pairs: int
    fallback_h0: bool


def kappa_calibration_diagnostic(
    candidates: Sequence[VirtualCandidate],
    module_type: str,
    seed: int = 42,
    max_pairs: int = 2000,
    indistinguishable_alpha: float = 0.1,
) -> KappaCalibrationResult:
    """3.3.3節 kappa 校准诊断: compares the empirical cross-layer |kappa| distribution
    for one module type against a random-unit-vector null via a two-sample KS test. If
    the two distributions are statistically indistinguishable (ks p-value above
    `indistinguishable_alpha`, i.e. we fail to reject "same distribution as random"),
    that module type's cross-layer kappa signal isn't trustworthy and coverage
    selection should fall back to h=0 (layer-local-only competition) for it.

    Deduplicates by physical_direction_id first (kappa is a per-direction-unit
    quantity; sign/group splits of the same atom would otherwise double-count pairs)
    and only considers pairs whose modules resolve to *different* layers (same-layer
    pairs are already within any reasonable window and aren't what this diagnostic is
    deciding). Falls back to h0 when there are too few cross-layer pairs to say
    anything (num_pairs < 2), since that's not evidence the signal is trustworthy.
    """
    by_unit: dict[str, VirtualCandidate] = {}
    for candidate in candidates:
        by_unit.setdefault(candidate.physical_direction_id, candidate)
    units = list(by_unit.values())

    observed: list[float] = []
    for i in range(len(units)):
        if len(observed) >= max_pairs:
            break
        for j in range(i + 1, len(units)):
            a, b = units[i], units[j]
            layer_a = extract_layer_index(a.module_name)
            layer_b = extract_layer_index(b.module_name)
            if layer_a is None or layer_b is None or layer_a == layer_b:
                continue
            observed.append(abs(compute_kappa(a, b)))
            if len(observed) >= max_pairs:
                break

    if len(observed) < 2:
        return KappaCalibrationResult(
            module_type=module_type,
            observed_mean_abs_kappa=0.0,
            null_mean_abs_kappa=0.0,
            ks_statistic=0.0,
            ks_pvalue=1.0,
            num_pairs=len(observed),
            fallback_h0=True,
        )

    dim = 8
    for unit in units:
        if unit.u is not None:
            dim = max(1, int(unit.u.numel()))
            break

    generator = torch.Generator().manual_seed(int(seed))

    def _random_unit(size: int) -> torch.Tensor:
        vector = torch.randn(size, generator=generator)
        return vector / torch.linalg.norm(vector).clamp_min(1e-12)

    null_samples: list[float] = []
    for _ in range(len(observed)):
        u_dot = float(torch.dot(_random_unit(dim), _random_unit(dim)).item())
        v_dot = float(torch.dot(_random_unit(dim), _random_unit(dim)).item())
        null_samples.append(abs(u_dot * v_dot))

    ks_result = ks_2samp(observed, null_samples)
    fallback = bool(ks_result.pvalue > float(indistinguishable_alpha))
    return KappaCalibrationResult(
        module_type=module_type,
        observed_mean_abs_kappa=float(sum(observed) / len(observed)),
        null_mean_abs_kappa=float(sum(null_samples) / len(null_samples)),
        ks_statistic=float(ks_result.statistic),
        ks_pvalue=float(ks_result.pvalue),
        num_pairs=len(observed),
        fallback_h0=fallback,
    )
