from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from dico.candidates import VirtualCandidate
from dico.coverage import compute_group_coverage, nsw_objective


def compute_physical_joint_utility(
    certified: Sequence[VirtualCandidate],
    groups: Sequence[str],
    eps: float = 1.0e-6,
) -> dict[str, float]:
    """3.4节: recompute a joint physical utility per physical direction, post-certification.

    A physical direction p may own several certified virtual candidates Q_p (its
    sign-split or task-group-split pieces). Summing each candidate's own
    certified_gain double-counts shared structure across those pieces, so this
    instead recomputes w_p^joint = F(S_-p ∪ Q_p) - F(S_-p), where S_-p is the
    certified set with all of p's own candidates removed and F is the same
    Nash-welfare coverage objective used during certification (coverage.nsw_objective).
    When Q_p has only one candidate this reduces to its ordinary marginal gain.

    Evaluated on each candidate's INITIAL (pre-deduction) profile, not the
    coverage-mutated `.profile` left behind by greedy_group_fair_coverage's
    selection-order-dependent residual bookkeeping -- otherwise w_p^joint would
    depend on the order candidates happened to be selected in, which the doc
    explicitly requires it not to (falls back to `.profile` when `.initial_profile`
    is unset, e.g. hand-built test fixtures that predate this field).
    """
    pristine = [
        replace(c, profile=c.initial_profile if c.initial_profile is not None else c.profile)
        for c in certified
    ]
    grouped: dict[str, list[VirtualCandidate]] = {}
    for candidate in pristine:
        grouped.setdefault(candidate.physical_direction_id, []).append(candidate)

    joint_utility: dict[str, float] = {}
    for physical_id, q_p in grouped.items():
        s_minus_p = [c for c in pristine if c.physical_direction_id != physical_id]
        f_minus_p = nsw_objective(compute_group_coverage(s_minus_p, groups), eps=eps)
        f_full = nsw_objective(compute_group_coverage(s_minus_p + q_p, groups), eps=eps)
        joint_utility[physical_id] = max(0.0, f_full - f_minus_p)
    return joint_utility
