from dico_rank.dynamic_allocation import DynamicRankAllocator


def make_allocator():
    dims = {
        "a": {"in_dim": 4, "out_dim": 4},
        "b": {"in_dim": 4, "out_dim": 4},
        "c": {"in_dim": 4, "out_dim": 4},
    }
    allocator = DynamicRankAllocator(
        masked_lora_modules={},
        module_dims=dims,
        initial_allocation={"a": 2, "b": 2, "c": 2},
        target_budget=48,
        config={
            "enabled": True,
            "update_ratios": [0.2, 0.4, 0.6],
            "freeze_after_ratio": 0.6,
            "move_ratio": 0.2,
            "score_smoothing": 1.0e-6,
            "r_min": 0,
            "r_max_multiplier": 2,
        },
        base_rank=2,
    )
    allocator.module_scores = {"a": 100.0, "b": 1.0, "c": 1.0}
    return allocator


def test_should_adjust_once_at_ratios_only():
    allocator = make_allocator()

    assert not allocator.should_adjust(global_step=1, total_steps=10)
    assert allocator.should_adjust(global_step=2, total_steps=10)
    assert not allocator.should_adjust(global_step=2, total_steps=10)
    assert allocator.should_adjust(global_step=4, total_steps=10)
    assert allocator.should_adjust(global_step=6, total_steps=10)
    assert not allocator.should_adjust(global_step=7, total_steps=10)


def test_move_ratio_limits_num_moved_and_bounds_hold():
    allocator = make_allocator()

    log = allocator.adjust_rank(global_step=2, total_steps=10)

    assert log["num_moved"] <= 1
    assert log["rank_distance_this_adjustment"] <= 2 * log["move_budget"]
    assert all(0 <= rank <= 4 for rank in allocator.current_allocation.values())
    assert log["budget_after"] <= log["target_budget"]


def test_dynamic_repair_does_not_fill_budget_beyond_move_limit():
    dims = {
        "a": {"in_dim": 10, "out_dim": 0},
        "b": {"in_dim": 10, "out_dim": 0},
        "cheap_receiver": {"in_dim": 1, "out_dim": 0},
    }
    allocator = DynamicRankAllocator(
        masked_lora_modules={},
        module_dims=dims,
        initial_allocation={"a": 1, "b": 1, "cheap_receiver": 0},
        target_budget=25,
        config={
            "enabled": True,
            "move_ratio": 0.1,
            "r_min": 0,
            "r_max_multiplier": 10,
            "score_smoothing": 1.0e-6,
        },
        base_rank=1,
    )
    allocator.module_scores = {"a": 0.01, "b": 0.01, "cheap_receiver": 100.0}

    log = allocator.adjust_rank(global_step=1, total_steps=10)

    assert log["move_budget"] == 1
    assert log["rank_distance_this_adjustment"] <= 2
    assert log["num_moved"] <= 1
    assert log["budget_after"] <= log["target_budget"]
