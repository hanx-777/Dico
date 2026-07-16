import torch

from dico.candidates import VirtualCandidate
from dico.coverage import (
    compute_group_coverage,
    compute_kappa,
    greedy_group_fair_coverage,
    nsw_objective,
)

_DIRECTION_DIM = 16
_direction_registry: dict[str, int] = {}


def _direction_vectors(direction_id: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically maps a direction_id to a one-hot (u, v_tilde) pair: distinct
    ids get orthogonal vectors (kappa=0), identical ids get identical vectors
    (kappa=1) -- a controllable primitive for exercising compute_kappa's geometry.
    """
    if direction_id not in _direction_registry:
        _direction_registry[direction_id] = len(_direction_registry)
    idx = _direction_registry[direction_id]
    assert idx < _DIRECTION_DIM, "increase _DIRECTION_DIM if more distinct test directions are needed"
    vec = torch.zeros(_DIRECTION_DIM)
    vec[idx] = 1.0
    return vec, vec.clone()


def _candidate(
    candidate_id: str,
    profile: list[float],
    module_name: str = "layers.0.q_proj",
    direction_id: str | None = None,
    utility: float = 0.0,
) -> VirtualCandidate:
    if direction_id is not None:
        u, v_tilde = _direction_vectors(direction_id)
    else:
        u, v_tilde = None, None
    return VirtualCandidate(
        virtual_candidate_id=candidate_id,
        physical_direction_id=candidate_id,
        module_name=module_name,
        atom_index=0,
        profile=torch.tensor(profile, dtype=torch.float32),
        split_type="positive",
        cost=4,
        utility=utility,
        u=u,
        v_tilde=v_tilde,
    )


def test_group_coverage_uses_squared_nonnegative_group_normalized_profiles():
    selected = [_candidate("a", [2.0, -2.0, 1.0, -1.0])]
    groups = ["math", "math", "code", "code"]

    coverage = compute_group_coverage(selected, groups)

    assert coverage == {"code": 1.0, "math": 4.0}
    assert nsw_objective(coverage) > 0.0


def test_compute_kappa_matches_u_v_tilde_dot_product_formula():
    # 3.3.3節: kappa(q',q*) = (u_q'.u_q*)(v_tilde_q'.v_tilde_q*).
    a = _candidate("a", [1.0, 0.0], direction_id="shared")
    b = _candidate("b", [0.0, 1.0], direction_id="shared")
    c = _candidate("c", [1.0, 1.0], direction_id="other")

    assert compute_kappa(a, b) == 1.0  # identical direction -> full correlation
    assert compute_kappa(a, c) == 0.0  # orthogonal direction -> no correlation


def test_compute_kappa_falls_back_to_zero_without_u_v_tilde():
    a = _candidate("a", [1.0, 0.0])  # no direction_id -> u/v_tilde unset
    b = _candidate("b", [1.0, 0.0])

    assert compute_kappa(a, b) == 0.0


def test_greedy_coverage_selects_worst_group_helpful_candidate_first():
    candidates = [
        _candidate("math_only", [3.0, 3.0, 0.0, 0.0]),
        _candidate("balanced", [1.0, 1.0, 2.0, 2.0]),
    ]
    groups = ["math", "math", "code", "code"]

    result = greedy_group_fair_coverage(candidates, groups, max_selected=1, eps=1e-3)

    assert [candidate.virtual_candidate_id for candidate in result.selected] == ["balanced"]
    assert result.trace[0]["group_cov_after"]["code"] > 0.0


def test_termination_condition_normalizes_by_group_count_and_stops_on_zero_residual_gain():
    # "collinear_dup" shares both its profile AND its (u, v_tilde) direction with
    # "primary" (kappa=1), so once "primary" is certified, kappa-based deduction
    # cancels "collinear_dup"'s residual to exactly zero -- the marginal NSW gain
    # is ~0, which must clear the 3.3.2節 max_q ΔF(q|S)/|T| < δ termination
    # criterion and stop the greedy loop.
    candidates = [
        _candidate("primary", [3.0, 3.0, 0.0, 0.0], direction_id="dup"),
        _candidate("collinear_dup", [3.0, 3.0, 0.0, 0.0], direction_id="dup"),
    ]
    groups = ["math", "math", "code", "code"]

    result = greedy_group_fair_coverage(
        candidates, groups, max_selected=2, eps=1e-3, relative_stop_delta=1e-3
    )

    assert [candidate.virtual_candidate_id for candidate in result.selected] == ["primary"]


def test_termination_condition_disabled_when_delta_is_zero():
    candidates = [
        _candidate("primary", [3.0, 3.0, 0.0, 0.0], direction_id="dup2"),
        _candidate("collinear_dup", [3.0, 3.0, 0.0, 0.0], direction_id="dup2"),
    ]
    groups = ["math", "math", "code", "code"]

    result = greedy_group_fair_coverage(
        candidates, groups, max_selected=2, eps=1e-3, relative_stop_delta=0.0
    )

    assert len(result.selected) == 2


def _sum_coverage_greedy_selection_order(
    candidates: list[VirtualCandidate], groups: list[str], max_selected: int
) -> list[str]:
    """Reference greedy selection using plain summed coverage (no Nash-welfare log
    transform) and the SAME persistent kappa-based residual deduction mechanism as
    greedy_group_fair_coverage, so the two are comparable step-for-step.
    """
    remaining = list(candidates)
    selected: list[VirtualCandidate] = []
    current_obj = sum(compute_group_coverage([], groups).values())
    order: list[str] = []
    residuals = {id(c): c.profile.detach().float().clone() for c in remaining}
    while remaining and len(selected) < max_selected:
        best_idx = -1
        best_gain = -float("inf")
        best_after = 0.0
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
            after = sum(compute_group_coverage([*selected, trial], groups).values())
            gain = after - current_obj
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
        chosen = remaining.pop(best_idx)
        chosen_residual = residuals.pop(id(chosen))
        chosen.profile = chosen_residual
        selected.append(chosen)
        order.append(chosen.virtual_candidate_id)
        current_obj = best_after
        for candidate in remaining:
            kappa = compute_kappa(candidate, chosen)
            if kappa == 0.0:
                continue
            residuals[id(candidate)] = residuals[id(candidate)] - kappa * chosen_residual
    return order


def _profile_fixture() -> list[list[float]]:
    return [[5.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def test_single_group_nash_welfare_selection_matches_sum_coverage_selection():
    # 3.3.2節 Proposition 1: with |T|=1, the Nash-welfare-coverage greedy
    # selection sequence must match the plain summed-coverage greedy sequence,
    # step for step, since log(eps+x) is monotonic in x for a single group.
    # Each candidate gets its own distinct direction_id -- their profiles already
    # have disjoint support, so no cross-candidate deduction should occur either
    # way, and this isolates the test from kappa mechanics entirely.
    # Independent candidate lists are built for each run since both greedy
    # loops mutate candidate.profile in place as they select.
    ids = ["a", "b", "c", "d"]
    groups = ["single"] * 4

    nsw_candidates = [
        _candidate(i, p, direction_id=f"single_{i}") for i, p in zip(ids, _profile_fixture())
    ]
    nsw_order = [
        c.virtual_candidate_id
        for c in greedy_group_fair_coverage(nsw_candidates, groups, max_selected=4, eps=1e-6).selected
    ]

    sum_candidates = [
        _candidate(i, p, direction_id=f"single_{i}") for i, p in zip(ids, _profile_fixture())
    ]
    sum_order = _sum_coverage_greedy_selection_order(sum_candidates, groups, max_selected=4)

    assert nsw_order == sum_order == ["a", "b", "c", "d"]


def test_local_window_h2_exempts_distant_layer_from_orthogonalization():
    # 3.3.3節: residual deduction is restricted to |layer(m)-layer(m')| <= h
    # (default h=2), not applied globally within a module type. All three
    # candidates share the same (u, v_tilde) direction (kappa=1) and an
    # identical profile, so full deduction exactly cancels a candidate within
    # the window of an already-selected one. "layer_1" sits within the window
    # of "layer_0" and gets fully cancelled (near-zero gain, never certified).
    # "layer_10" is the same direction but 10 layers away (outside the window),
    # so it is NOT deducted against "layer_0" and should still be certified.
    # "layer_0" gets a higher utility so round-1's 3-way profile tie resolves to
    # it deterministically (all three start with an identical profile, by design,
    # so full kappa=1 cancellation is exact).
    candidates = [
        _candidate("layer_0", [10.0, 10.0, 0.0, 0.0], module_name="layers.0.q_proj", direction_id="win", utility=1.0),
        _candidate("layer_1", [10.0, 10.0, 0.0, 0.0], module_name="layers.1.q_proj", direction_id="win"),
        _candidate("layer_10", [10.0, 10.0, 0.0, 0.0], module_name="layers.10.q_proj", direction_id="win"),
    ]
    groups = ["math", "math", "code", "code"]

    result = greedy_group_fair_coverage(candidates, groups, max_selected=3, eps=1e-3, window_h=2)

    assert [candidate.virtual_candidate_id for candidate in result.selected] == ["layer_0", "layer_10"]


def test_unbounded_window_falls_back_to_legacy_global_orthogonalization():
    # With window_h=None the local-window restriction is disabled entirely, so
    # "layer_10" is deducted against "layer_0" just like "layer_1" is --
    # reproducing the pre-3.3.3 global-within-type behavior.
    candidates = [
        _candidate("layer_0", [10.0, 10.0, 0.0, 0.0], module_name="layers.0.q_proj", direction_id="win2", utility=1.0),
        _candidate("layer_1", [10.0, 10.0, 0.0, 0.0], module_name="layers.1.q_proj", direction_id="win2"),
        _candidate("layer_10", [10.0, 10.0, 0.0, 0.0], module_name="layers.10.q_proj", direction_id="win2"),
    ]
    groups = ["math", "math", "code", "code"]

    result = greedy_group_fair_coverage(candidates, groups, max_selected=3, eps=1e-3, window_h=None)

    assert [candidate.virtual_candidate_id for candidate in result.selected] == ["layer_0"]


def test_same_physical_direction_unit_is_excluded_from_deduction():
    # 3.3.3節: same-unit exclusion -- a physical direction's own sign/group splits
    # must never deduct against each other, even if they'd otherwise correlate
    # (kappa=1) and fall within the competition window. Both candidates below
    # share physical_direction_id "p" (as if they were the two sign-splits of the
    # same atom); after one is certified, the other's residual must be untouched.
    a = VirtualCandidate(
        virtual_candidate_id="p/positive",
        physical_direction_id="p",
        module_name="layers.0.q_proj",
        atom_index=0,
        profile=torch.tensor([5.0, 5.0, 0.0, 0.0]),
        split_type="positive",
        cost=4,
        u=torch.tensor([1.0, 0.0]),
        v_tilde=torch.tensor([1.0, 0.0]),
    )
    b = VirtualCandidate(
        virtual_candidate_id="p/negative",
        physical_direction_id="p",
        module_name="layers.0.q_proj",
        atom_index=0,
        profile=torch.tensor([0.0, 0.0, 5.0, 5.0]),
        split_type="negative",
        cost=4,
        u=torch.tensor([1.0, 0.0]),
        v_tilde=torch.tensor([1.0, 0.0]),
    )
    groups = ["math", "math", "code", "code"]

    result = greedy_group_fair_coverage([a, b], groups, max_selected=2, eps=1e-3)

    # Both get certified with their full, undiminished coverage contribution --
    # if same-unit exclusion were missing, "b" would be deducted by kappa=1
    # against "a" and its coverage/selection would be corrupted.
    assert {c.virtual_candidate_id for c in result.selected} == {"p/positive", "p/negative"}
    selected_by_id = {c.virtual_candidate_id: c for c in result.selected}
    assert torch.allclose(selected_by_id["p/negative"].profile, torch.tensor([0.0, 0.0, 5.0, 5.0]))
