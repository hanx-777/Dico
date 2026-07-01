from dico_rank.dynamic_allocation import DynamicRankAllocator
from dico_rank.rank_budget import repair_allocation_to_budget


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


def test_dynamic_adjustment_repairs_to_active_param_budget():
    dims = {
        "q": {"in_dim": 4, "out_dim": 4},
        "up": {"in_dim": 4, "out_dim": 12},
    }
    allocator = DynamicRankAllocator(
        masked_lora_modules={},
        module_dims=dims,
        initial_allocation={"q": 3, "up": 1},
        target_budget=40,
        config={
            "enabled": True,
            "update_ratios": [0.2],
            "move_ratio": 0.5,
            "score_smoothing": 1.0e-6,
            "r_min": 0,
            "r_max_multiplier": 2,
        },
        base_rank=2,
    )
    allocator.module_scores = {"q": 0.01, "up": 100.0}

    log = allocator.adjust_rank(global_step=2, total_steps=10)

    assert log["budget_after"] <= log["target_budget"]
    assert log["budget_error_ratio"] <= 0.01
