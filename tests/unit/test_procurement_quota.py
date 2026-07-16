from dico.procurement import compute_module_quota


def test_quota_shares_sum_to_approximately_one():
    normalized_utility = {"m1/0": 2.0, "m1/1": 1.0, "m2/0": 3.0}
    physical_module_of = {"m1/0": "m1", "m1/1": "m1", "m2/0": "m2"}
    module_names = ["m1", "m2"]
    costs = {"m1": 4, "m2": 8}

    quota = compute_module_quota(
        normalized_utility, physical_module_of, module_names, costs, r_min=2, budget_remaining=100.0
    )

    # r_bar_m = r_min + s_m * B_rem / c_m, so recovering s_m from quota should sum to ~1.
    implied_share_sum = sum((quota[name] - 2) * costs[name] for name in module_names) / 100.0
    assert abs(implied_share_sum - 1.0) < 1e-9


def test_module_with_zero_certified_candidates_gets_nonzero_floor_quota_not_divide_by_zero():
    normalized_utility = {"m1/0": 5.0}
    physical_module_of = {"m1/0": "m1"}
    module_names = ["m1", "m2"]  # m2 has no certified candidates at all
    costs = {"m1": 4, "m2": 4}

    quota = compute_module_quota(
        normalized_utility, physical_module_of, module_names, costs, r_min=0, budget_remaining=50.0
    )

    assert quota["m2"] >= 0.0
    assert quota["m2"] < quota["m1"]


def test_quota_is_a_reference_point_not_enforced_as_a_hard_rank_cap():
    # compute_module_quota itself has no notion of "current rank" -- it only
    # produces a reference point r_bar_m. Nothing here clips or bounds it by
    # r_max or any observed allocation, confirming callers are responsible for
    # treating it as soft pressure (see test_soft_quota_is_pressure_not_a_hard_cap
    # in test_dico_procurement_v03.py for the end-to-end procurement behavior).
    normalized_utility = {"m1/0": 100.0}
    physical_module_of = {"m1/0": "m1"}
    module_names = ["m1"]
    costs = {"m1": 4}

    quota = compute_module_quota(
        normalized_utility, physical_module_of, module_names, costs, r_min=0, budget_remaining=8.0
    )

    # With a single module taking the entire remaining budget share, r_bar_m
    # can legitimately exceed what a naive per-module budget split would give.
    assert quota["m1"] == 2.0
