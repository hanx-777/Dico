from dico.rank_budget import repair_allocation_to_budget


def test_repair_minimizes_feasible_error_with_unequal_dims():
    dims = {
        "q": {"in_dim": 4, "out_dim": 4},
        "up": {"in_dim": 4, "out_dim": 12},
    }

    result = repair_allocation_to_budget(
        {"q": 0, "up": 4},
        target_budget=40,
        module_dims=dims,
        r_min=0,
        r_max=4,
    )

    assert result.budget.actual_budget == 40
    assert result.budget.budget_error_ratio == 0.0
