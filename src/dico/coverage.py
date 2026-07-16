from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from dico.candidates import VirtualCandidate
from dico.path_utils import extract_layer_index


@dataclass(frozen=True)
class CoverageResult:
    selected: list[VirtualCandidate]
    trace: list[dict[str, object]]


def compute_group_coverage(candidates: Sequence[VirtualCandidate], groups: Sequence[str]) -> dict[str, float]:
    labels = sorted(set(str(group) for group in groups))
    coverage = {label: 0.0 for label in labels}
    for label in labels:
        indices = torch.tensor([str(group) == label for group in groups], dtype=torch.bool)
        denom = max(int(indices.sum().item()), 1)
        total = torch.tensor(0.0)
        for candidate in candidates:
            profile = candidate.profile.detach().float()
            total = total + torch.sum(torch.abs(profile[indices]) ** 2)
        coverage[label] = float((total / denom).item())
    return coverage


def nsw_objective(coverage: dict[str, float], eps: float = 1.0e-6) -> float:
    return sum(math.log(float(eps) + max(0.0, float(value))) for value in coverage.values())


def compute_kappa(a: VirtualCandidate, b: VirtualCandidate) -> float:
    """3.3.3節: kappa(q',q*) = (u_q'.u_q*)(v_tilde_q'.v_tilde_q*), the sketch-domain
    rank-one-atom correlation coefficient used to scale residual deduction between two
    direction candidates. Falls back to 0.0 (no deduction) when either candidate lacks
    u/v_tilde -- e.g. hand-built test fixtures that predate the hybrid sketch pipeline,
    or the fallback SVD path for an all-zero response matrix.
    """
    if a.u is None or a.v_tilde is None or b.u is None or b.v_tilde is None:
        return 0.0
    return float(torch.dot(a.u.float(), b.u.float()) * torch.dot(a.v_tilde.float(), b.v_tilde.float()))


def _within_layer_window(module_a: str, module_b: str, window_h: int | None) -> bool:
    """4.5.1节: local-window residual orthogonalization -- candidates only compete
    against same-type peers within |layer(m)-layer(m')| <= h (default h=2). When
    window_h is None, or either module name has no resolvable layer index (e.g.
    synthetic test fixtures), falls back to the unrestricted legacy behavior
    rather than raising, since layer-window suppression is a refinement on top
    of module-type isolation, not a hard requirement for it to work at all.
    """
    if window_h is None:
        return True
    layer_a = extract_layer_index(module_a)
    layer_b = extract_layer_index(module_b)
    if layer_a is None or layer_b is None:
        return True
    return abs(layer_a - layer_b) <= int(window_h)


def greedy_group_fair_coverage(
    candidates: Sequence[VirtualCandidate],
    groups: Sequence[str],
    max_selected: int | None = None,
    eps: float = 1.0e-6,
    relative_stop_delta: float = 1.0e-3,
    window_h: int | None = 2,
) -> CoverageResult:
    remaining = list(candidates)
    selected: list[VirtualCandidate] = []
    trace: list[dict[str, object]] = []
    limit = len(remaining) if max_selected is None else int(max_selected)
    current_cov = compute_group_coverage(selected, groups)
    current_obj = nsw_objective(current_cov, eps=eps)

    # 3.3.3節: persistent per-candidate residual state, updated by a single
    # kappa-scaled subtraction each round (rho_q' <- rho_q' - kappa(q',q*)*rho_q*)
    # rather than a full Gram-Schmidt recompute against the whole selected basis
    # every round -- this is what the doc's update rule literally specifies, and is
    # strictly cheaper (O(1) update per candidate per round instead of O(|window|)).
    # Keyed by object identity: VirtualCandidate isn't hashable/frozen, and
    # virtual_candidate_id isn't guaranteed unique across pathological test fixtures.
    residuals: dict[int, torch.Tensor] = {
        id(candidate): candidate.profile.detach().float().clone() for candidate in remaining
    }

    while remaining and len(selected) < limit:
        best_idx = -1
        best_gain = -float("inf")
        best_after: dict[str, float] | None = None
        for idx, candidate in enumerate(remaining):
            trial = VirtualCandidate(
                virtual_candidate_id=candidate.virtual_candidate_id,
                physical_direction_id=candidate.physical_direction_id,
                module_name=candidate.module_name,
                atom_index=candidate.atom_index,
                profile=residuals[id(candidate)],
                split_type=candidate.split_type,
                cost=candidate.cost,
                utility=candidate.utility,
                raw_energy=candidate.raw_energy,
                full_v=candidate.full_v,
            )
            after = compute_group_coverage([*selected, trial], groups)
            gain = nsw_objective(after, eps=eps) - current_obj
            key = (gain, candidate.utility, -candidate.cost, candidate.virtual_candidate_id)
            if best_idx < 0 or key > (
                best_gain,
                remaining[best_idx].utility,
                -remaining[best_idx].cost,
                remaining[best_idx].virtual_candidate_id,
            ):
                best_idx = idx
                best_gain = gain
                best_after = after
        if best_idx < 0 or best_after is None:
            break
        # 3.3.2节终止条件: max_q ΔF(q|S)/|T| < δ, normalized by the number of task groups.
        num_groups = max(1, len(set(groups)))
        if trace and (best_gain / num_groups) < float(relative_stop_delta):
            break
        chosen = remaining.pop(best_idx)
        chosen_residual = residuals.pop(id(chosen))
        chosen.profile = chosen_residual
        chosen.certified_gain = max(0.0, float(best_gain))
        selected.append(chosen)
        trace.append(
            {
                "step": len(selected),
                "module": chosen.module_name,
                "atom": chosen.atom_index,
                "candidate": chosen.split_type,
                "physical_direction": chosen.physical_direction_id,
                "gain": chosen.certified_gain,
                "group_cov_before": current_cov,
                "group_cov_after": best_after,
            }
        )
        current_cov = best_after
        current_obj = nsw_objective(current_cov, eps=eps)

        # 3.3.3節: deduct the chosen candidate's residual from every remaining
        # candidate in its local competition window, EXCEPT candidates sharing its
        # own physical direction unit (same-unit exclusion guards against degenerate
        # self-subtraction -- one direction unit's own sign/group splits must never
        # cancel each other out).
        for candidate in remaining:
            if candidate.physical_direction_id == chosen.physical_direction_id:
                continue
            if not _within_layer_window(candidate.module_name, chosen.module_name, window_h):
                continue
            kappa = compute_kappa(candidate, chosen)
            if kappa == 0.0:
                continue
            residuals[id(candidate)] = residuals[id(candidate)] - kappa * chosen_residual

    return CoverageResult(selected=selected, trace=trace)
