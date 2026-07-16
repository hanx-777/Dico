from dico.rank_budget import (
    BudgetManager,
    compute_lora_params_for_module,
    compute_total_lora_params,
    get_uniform_budget,
    solve_rank_dp,
    repair_allocation_to_budget,
)


def test_uniform_budget_uses_module_dimensions():
    dims = {"q": {"in_dim": 4, "out_dim": 4}, "up": {"in_dim": 4, "out_dim": 8}}

    assert compute_lora_params_for_module(2, 4, 8) == 24
    assert get_uniform_budget(2, ["q", "up"], dims).target_budget == 40


def test_repair_allocation_respects_budget_and_bounds():
    dims = {"cheap": {"in_dim": 2, "out_dim": 2}, "expensive": {"in_dim": 10, "out_dim": 10}}
    allocation = {"cheap": 8, "expensive": 8}

    result = repair_allocation_to_budget(
        allocation,
        target_budget=28,
        module_dims=dims,
        r_min=0,
        r_max=8,
    )

    assert result.budget.actual_budget <= result.budget.target_budget
    assert result.allocation["cheap"] == 7
    assert result.allocation["expensive"] == 0
    assert result.budget.budget_error == 0


def test_repair_lookahead_avoids_greedy_largest_cost_trap():
    dims = {"six": {"in_dim": 3, "out_dim": 3}, "ten": {"in_dim": 5, "out_dim": 5}}

    result = repair_allocation_to_budget(
        {"six": 0, "ten": 0},
        target_budget=12,
        module_dims=dims,
        r_min=0,
        r_max=2,
    )

    assert result.allocation == {"six": 2, "ten": 0}
    assert result.budget.actual_budget == 12
    assert result.budget.budget_error_ratio == 0.0


def test_repair_finds_global_budget_optimum_for_multimodule_counterexample():
    dims = {
        "m0": {"in_dim": 2, "out_dim": 0},
        "m1": {"in_dim": 2, "out_dim": 0},
        "m2": {"in_dim": 2, "out_dim": 0},
        "m3": {"in_dim": 3, "out_dim": 0},
    }

    result = repair_allocation_to_budget(
        {"m0": 0, "m1": 0, "m2": 0, "m3": 0},
        target_budget=15,
        module_dims=dims,
        r_min=0,
        r_max=4,
    )

    assert result.budget.actual_budget == 15
    assert result.budget.budget_error == 0
    assert result.budget.budget_error_ratio == 0.0


def test_repair_warns_when_minimum_rank_bounds_exceed_budget():
    dims = {"m": {"in_dim": 100, "out_dim": 100}}

    result = repair_allocation_to_budget(
        {"m": 1},
        target_budget=50,
        module_dims=dims,
        r_min=1,
        r_max=2,
    )

    assert result.budget.actual_budget > result.budget.target_budget
    assert result.budget.over_budget is True
    assert result.budget.warning


def test_budget_manager_describes_warning_for_large_error():
    dims = {"m": {"in_dim": 100, "out_dim": 100}}
    manager = BudgetManager("equal_trainable_params", dims, warning_threshold=0.01)

    info = manager.describe({"m": 0}, target_budget=50)

    assert info["budget_error_ratio"] == -1.0
    assert "exceeds 1%" in info["warning"]


def test_solve_rank_dp_matches_bruteforce_optimum_with_real_parameter_costs():
    dims = {
        "cheap": {"in_dim": 2, "out_dim": 2},
        "expensive": {"in_dim": 6, "out_dim": 6},
    }
    utility_curves = {
        "cheap": {0: 0.0, 1: 3.0, 2: 5.0, 3: 5.5},
        "expensive": {0: 0.0, 1: 8.0, 2: 8.1, 3: 8.2},
    }

    result = solve_rank_dp(
        utility_curves,
        dims,
        target_budget=20,
        r_min=0,
        r_max=3,
        eta=1.0,
    )

    assert result.allocation == {"cheap": 2, "expensive": 1}
    assert result.budget.actual_budget == 20
    assert result.budget.over_budget is False
    assert result.diagnostics["solver"] == "dp"
    assert result.diagnostics["budget_lower_bound"] == 20


def test_solve_rank_dp_reports_infeasible_lower_bound_without_exceeding_target_when_possible():
    dims = {
        "a": {"in_dim": 2, "out_dim": 2},
        "b": {"in_dim": 5, "out_dim": 5},
    }
    utility_curves = {
        "a": {0: 0.0, 1: 10.0, 2: 11.0},
        "b": {0: 0.0, 1: 9.0, 2: 9.5},
    }

    result = solve_rank_dp(
        utility_curves,
        dims,
        target_budget=13,
        r_min=0,
        r_max=2,
        eta=1.0,
    )

    assert result.budget.actual_budget == 10
    assert result.budget.actual_budget <= result.budget.target_budget
    assert result.diagnostics["budget_lower_bound_feasible"] is False
    assert result.diagnostics["fallback"] == "best_under_budget"


def test_solve_rank_dp_fails_when_minimum_rank_budget_exceeds_target():
    dims = {
        "a": {"in_dim": 4, "out_dim": 4},
        "b": {"in_dim": 5, "out_dim": 5},
    }

    result = solve_rank_dp(
        {"a": {1: 1.0}, "b": {1: 1.0}},
        dims,
        target_budget=12,
        r_min=1,
        r_max=1,
    )

    assert result.allocation == {"a": 1, "b": 1}
    assert result.budget.over_budget is True
    assert result.diagnostics["fallback"] == "minimum_rank_exceeds_target"
