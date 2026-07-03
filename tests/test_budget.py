from dico_rank.rank_budget import (
    BudgetManager,
    compute_lora_params_for_module,
    compute_total_lora_params,
    get_uniform_budget,
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
