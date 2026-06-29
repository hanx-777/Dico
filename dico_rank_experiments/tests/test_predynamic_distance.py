from dico_rank.dynamic_allocation import DynamicRankAllocator


def test_predynamic_records_distance_from_preallocation():
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
            "update_ratios": [0.2],
            "move_ratio": 0.10,
            "score_smoothing": 1.0e-6,
            "r_min": 0,
            "r_max_multiplier": 2,
        },
        base_rank=2,
        preallocation={"a": 2, "b": 2, "c": 2},
    )
    allocator.module_scores = {"a": 100.0, "b": 1.0, "c": 1.0}

    log = allocator.adjust_rank(global_step=2, total_steps=10)

    assert "rank_distance_from_preallocation" in log
    assert log["rank_distance_from_preallocation"] <= 2
