from dico.diagnostics import compute_diagnostics, gini


def test_gini_of_perfectly_equal_values_is_zero():
    assert gini([5.0, 5.0, 5.0]) == 0.0


def test_gini_of_empty_or_all_zero_is_zero():
    assert gini([]) == 0.0
    assert gini([0.0, 0.0]) == 0.0


def test_gini_matches_hand_computed_value_for_unequal_distribution():
    # sorted [2,6,32], n=3, total=40, weighted_sum=1*2+2*6+3*32=110
    # gini = (2*110 - 4*40) / (3*40) = 60/120 = 0.5
    assert abs(gini([2, 6, 32]) - 0.5) < 1e-9


def test_compute_diagnostics_matches_hand_computed_values_for_all_ten_metrics():
    module_dims = {
        "layers.0.q_proj": {"in_dim": 2, "out_dim": 2},
        "layers.1.q_proj": {"in_dim": 2, "out_dim": 2},
        "layers.0.k_proj": {"in_dim": 4, "out_dim": 4},
    }
    rank_dict = {"layers.0.q_proj": 2, "layers.1.q_proj": 6, "layers.0.k_proj": 32}
    init_summaries = {
        "layers.0.q_proj": {"certified_rows": 2, "relaxation_rows": 0},
        "layers.1.q_proj": {"certified_rows": 4, "relaxation_rows": 2},
        "layers.0.k_proj": {"certified_rows": 20, "relaxation_rows": 12},
    }

    result = compute_diagnostics(
        rank_dict,
        module_dims,
        r_max=32,
        target_budget=300,
        balanced_fill_ratio=0.1,
        init_summaries=init_summaries,
    )

    assert abs(result["rank_gini"] - 0.5) < 1e-9
    assert abs(result["param_share_gini"] - (496.0 / 864.0)) < 1e-9
    assert abs(result["cap_hit_ratio"] - (1.0 / 3.0)) < 1e-9
    assert result["zero_rank_ratio"] == 0.0
    assert abs(result["type_budget_share"]["q_proj"] - (32.0 / 288.0)) < 1e-9
    assert abs(result["type_budget_share"]["k_proj"] - (256.0 / 288.0)) < 1e-9
    assert result["top10_module_share"] == 1.0
    assert abs(result["mean_abs_adjacent_rank_diff"] - 4.0) < 1e-9
    assert abs(result["anchored_rank_ratio"] - 0.65) < 1e-9
    assert result["balanced_fill_ratio"] == 0.1
    assert abs(result["budget_realized_ratio"] - (288.0 / 300.0)) < 1e-9


def test_compute_diagnostics_returns_all_ten_required_keys():
    module_dims = {"m1": {"in_dim": 2, "out_dim": 2}}
    rank_dict = {"m1": 4}

    result = compute_diagnostics(
        rank_dict, module_dims, r_max=8, target_budget=16, balanced_fill_ratio=0.0, init_summaries={}
    )

    expected_keys = {
        "rank_gini",
        "param_share_gini",
        "cap_hit_ratio",
        "zero_rank_ratio",
        "type_budget_share",
        "top10_module_share",
        "mean_abs_adjacent_rank_diff",
        "anchored_rank_ratio",
        "balanced_fill_ratio",
        "budget_realized_ratio",
    }
    assert expected_keys <= set(result.keys())


def test_compute_diagnostics_handles_no_layer_resolvable_modules_without_crashing():
    module_dims = {"cheap": {"in_dim": 2, "out_dim": 2}, "expensive": {"in_dim": 4, "out_dim": 4}}
    rank_dict = {"cheap": 1, "expensive": 3}

    result = compute_diagnostics(
        rank_dict, module_dims, r_max=8, target_budget=100, balanced_fill_ratio=0.0, init_summaries={}
    )

    assert result["mean_abs_adjacent_rank_diff"] == 0.0
    assert result["anchored_rank_ratio"] == 0.0


def test_qds_and_spearman_are_none_without_module_quota():
    # Non-CovRA allocation paths never populate module_quota.
    module_dims = {"m1": {"in_dim": 2, "out_dim": 2}}
    rank_dict = {"m1": 4}

    result = compute_diagnostics(
        rank_dict, module_dims, r_max=8, target_budget=16, balanced_fill_ratio=0.0, init_summaries={},
    )

    assert result["qds"] is None
    assert result["spearman_corr_r_rbar"] is None


def test_qds_hand_computed_for_a_small_fixture():
    # 3.4.3節: QDS = (1/B*) * sum_m |r_m - clip(round(r_bar_m), r_min, r_max)| * c_m.
    module_dims = {"m1": {"in_dim": 2, "out_dim": 2}, "m2": {"in_dim": 4, "out_dim": 4}}
    rank_dict = {"m1": 5, "m2": 2}
    module_quota = {"m1": 3.4, "m2": 2.6}  # round -> m1:3, m2:3
    # costs: m1 = 2+2=4, m2 = 4+4=8
    # clip(round(3.4), r_min=1, r_max=8) = 3; clip(round(2.6), 1, 8) = 3
    # QDS = (|5-3|*4 + |2-3|*8) / target_budget=100 = (8+8)/100 = 0.16

    result = compute_diagnostics(
        rank_dict, module_dims, r_max=8, target_budget=100, balanced_fill_ratio=0.0, init_summaries={},
        module_quota=module_quota, r_min=1,
    )

    assert abs(result["qds"] - 0.16) < 1e-9


def test_spearman_perfect_correlation_and_anticorrelation():
    module_dims = {name: {"in_dim": 2, "out_dim": 2} for name in ("m1", "m2", "m3")}

    positive = compute_diagnostics(
        {"m1": 1, "m2": 2, "m3": 3}, module_dims, r_max=8, target_budget=100, balanced_fill_ratio=0.0,
        init_summaries={}, module_quota={"m1": 1.0, "m2": 2.0, "m3": 3.0}, r_min=0,
    )
    assert abs(positive["spearman_corr_r_rbar"] - 1.0) < 1e-9

    negative = compute_diagnostics(
        {"m1": 1, "m2": 2, "m3": 3}, module_dims, r_max=8, target_budget=100, balanced_fill_ratio=0.0,
        init_summaries={}, module_quota={"m1": 3.0, "m2": 2.0, "m3": 1.0}, r_min=0,
    )
    assert abs(negative["spearman_corr_r_rbar"] - (-1.0)) < 1e-9


def test_spearman_is_none_for_degenerate_constant_quota():
    module_dims = {name: {"in_dim": 2, "out_dim": 2} for name in ("m1", "m2", "m3")}

    result = compute_diagnostics(
        {"m1": 1, "m2": 2, "m3": 3}, module_dims, r_max=8, target_budget=100, balanced_fill_ratio=0.0,
        init_summaries={}, module_quota={"m1": 5.0, "m2": 5.0, "m3": 5.0}, r_min=0,
    )

    assert result["spearman_corr_r_rbar"] is None
    assert result["qds"] is not None  # QDS itself doesn't require variance to be defined
