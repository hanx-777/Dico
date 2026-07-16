import math
import statistics

from dico.normalize import NormalizationStats, apply_normalized_utility, compute_normalized_utility


def _expected_softplus(z: float) -> float:
    return math.log1p(math.exp(z))


def test_type_wise_whitening_matches_hand_computed_zscore():
    # v0.6.2 3.4.1节: pure type-wise median/MAD whitening, no module-wise term.
    joint_utilities = {"p1": 1.0, "p2": 2.0, "p3": 3.0, "p4": 5.0}
    type_of = {p: "q_proj" for p in joint_utilities}
    eps = 1.0e-6

    w_bar, stats = compute_normalized_utility(joint_utilities, type_of, eps=eps)

    log_u = {p: math.log(v + eps) for p, v in joint_utilities.items()}
    type_values = list(log_u.values())
    type_median = statistics.median(type_values)
    type_mad = statistics.median([abs(v - type_median) for v in type_values]) + eps

    for p in joint_utilities:
        z_type = (log_u[p] - type_median) / type_mad
        expected = _expected_softplus(z_type)
        assert abs(w_bar[p] - expected) < 1e-9

    assert stats.median_by_type["q_proj"] == type_median
    assert stats.mad_by_type["q_proj"] == type_mad


def test_identical_utilities_do_not_divide_by_zero():
    joint_utilities = {"p1": 2.0, "p2": 2.0, "p3": 2.0}
    type_of = {p: "q_proj" for p in joint_utilities}

    w_bar, stats = compute_normalized_utility(joint_utilities, type_of, eps=1.0e-6)

    assert all(math.isfinite(value) for value in w_bar.values())
    # MAD of identical values is 0 before the +eps floor; z-scores should collapse
    # to ~0 (softplus(0) = log(2)), not blow up.
    for value in w_bar.values():
        assert abs(value - math.log(2.0)) < 1e-3


def test_higher_joint_utility_yields_higher_normalized_utility_holding_population_fixed():
    type_of = {"p1": "q_proj", "p2": "q_proj", "p3": "q_proj"}
    joint_utilities = {"p1": 1.0, "p2": 2.0, "p3": 3.0}

    w_bar, _ = compute_normalized_utility(joint_utilities, type_of)

    assert w_bar["p1"] < w_bar["p2"] < w_bar["p3"]


def test_apply_normalized_utility_reuses_fitted_stats_without_recomputing():
    certified = {"p1": 1.0, "p2": 2.0, "p3": 3.0}
    type_of_certified = {p: "q_proj" for p in certified}
    _, stats = compute_normalized_utility(certified, type_of_certified)

    # A disjoint reserve pool, including an extreme outlier -- applying the
    # certified-only stats must not let this outlier perturb the fitted
    # (median, mad), unlike re-running compute_normalized_utility on the union.
    reserve = {"r1": 0.5, "r2": 1000.0}
    type_of_reserve = {p: "q_proj" for p in reserve}

    w_bar_reserve = apply_normalized_utility(reserve, type_of_reserve, stats, eps=1.0e-6)

    for pid, value in reserve.items():
        log_u = math.log(value + 1.0e-6)
        z = (log_u - stats.median_by_type["q_proj"]) / stats.mad_by_type["q_proj"]
        assert abs(w_bar_reserve[pid] - _expected_softplus(z)) < 1e-9


def test_apply_normalized_utility_falls_back_for_unseen_type():
    stats = NormalizationStats(median_by_type={"q_proj": 1.0}, mad_by_type={"q_proj": 0.5})
    raw = {"p1": 4.0}
    type_of = {"p1": "v_proj"}  # never appeared in the fitted pool

    w_bar = apply_normalized_utility(raw, type_of, stats, eps=1.0e-6)

    log_u = math.log(4.0 + 1.0e-6)
    assert abs(w_bar["p1"] - _expected_softplus(log_u)) < 1e-9  # median=0, mad=1 fallback
