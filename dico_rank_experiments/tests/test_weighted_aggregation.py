from dico_rank.preallocation import DiCoPreAllocator, MODULE_PROXY_LIMITATION
from dico_rank.rank_budget import allocate_by_weighted_utility


def _allocator(aggregation_mode="weighted_topk", normalization="none"):
    return DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config={
            "rank": 3,
            "preallocation": {
                "aggregation_mode": aggregation_mode,
                "weighted_topk_k": "auto",
                "atom_weight_normalization": normalization,
                "fallback_atom_mode": "module_proxy",
                "r_max_multiplier": 2,
                "use_cost_aware_allocation": True,
            },
        },
        module_names=["A", "B"],
        module_dims={
            "A": {"in_dim": 5, "out_dim": 5},
            "B": {"in_dim": 5, "out_dim": 5},
        },
    )


def test_strong_atom_beats_multiple_weak_atoms_with_no_normalization():
    allocator = _allocator(aggregation_mode="weighted_topk", normalization="none")
    atoms = allocator.compute_atom_utilities(
        [
            {"module_name": "A", "atom_id": 0, "importance": 10.0, "redundancy": 0.0},
            {"module_name": "B", "atom_id": 0, "importance": 2.0, "redundancy": 0.0},
            {"module_name": "B", "atom_id": 1, "importance": 2.0, "redundancy": 0.0},
            {"module_name": "B", "atom_id": 2, "importance": 2.0, "redundancy": 0.0},
        ]
    )
    atoms = allocator.normalize_atom_utilities(atoms, mode="none")

    weighted = allocator.aggregate_module_utilities(atoms, aggregation_mode="weighted_topk")

    assert weighted["A"] > weighted["B"]


def test_count_and_weighted_topk_behave_differently():
    allocator = _allocator(aggregation_mode="weighted_topk", normalization="none")
    records = [
        {"module_name": "A", "atom_id": 0, "importance": 10.0, "redundancy": 0.0},
        {"module_name": "B", "atom_id": 0, "importance": 2.0, "redundancy": 0.0},
        {"module_name": "B", "atom_id": 1, "importance": 2.0, "redundancy": 0.0},
        {"module_name": "B", "atom_id": 2, "importance": 2.0, "redundancy": 0.0},
    ]

    weighted = allocator.aggregate_module_utilities(
        allocator.normalize_atom_utilities(allocator.compute_atom_utilities(records), mode="none"),
        aggregation_mode="weighted_topk",
    )
    count = allocator.aggregate_module_utilities(
        allocator.normalize_atom_utilities(allocator.compute_atom_utilities(records), mode="none"),
        aggregation_mode="count",
    )

    assert weighted["A"] > weighted["B"]
    assert count["B"] > count["A"]


def test_cost_aware_allocation_favors_lower_cost_equal_utility_module():
    result = allocate_by_weighted_utility(
        module_utilities={"A": 10.0, "B": 10.0},
        module_dims={
            "A": {"in_dim": 50, "out_dim": 50},
            "B": {"in_dim": 500, "out_dim": 500},
        },
        total_rank_budget=4,
        target_budget=400,
        r_min=0,
        r_max=4,
        use_cost_aware=True,
    )

    assert result.allocation["A"] > result.allocation["B"]
    assert result.budget.actual_budget <= result.budget.target_budget


def test_greedy_budget_fill_reduces_budget_error_without_exceeding_target():
    result = allocate_by_weighted_utility(
        module_utilities={"cheap": 10.0, "expensive": 1.0},
        module_dims={
            "cheap": {"in_dim": 3, "out_dim": 4},
            "expensive": {"in_dim": 20, "out_dim": 20},
        },
        total_rank_budget=1,
        target_budget=21,
        r_min=0,
        r_max=3,
        use_cost_aware=True,
    )

    assert result.allocation["cheap"] == 3
    assert result.budget.actual_budget == 21
    assert result.budget.budget_error_ratio == 0.0


def test_preallocation_logs_weighted_metadata_fields():
    allocator = DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config={
            "rank": 2,
            "preallocation": {
                "aggregation_mode": "weighted_topk",
                "weighted_topk_k": "auto",
                "atom_weight_normalization": "none",
                "fallback_atom_mode": "module_proxy",
                "use_cost_aware_allocation": True,
            },
        },
        module_names=["A", "B"],
        module_dims={
            "A": {"in_dim": 4, "out_dim": 4},
            "B": {"in_dim": 4, "out_dim": 4},
        },
        module_scores={"A": 10.0, "B": 1.0},
    )

    result = allocator.allocate(rank_budget=32)
    payload = result.to_dict(preallocation_path="outputs/preallocations/mock.json")

    assert payload["aggregation_mode"] == "weighted_topk"
    assert payload["atom_weight_normalization"] == "none"
    assert payload["use_cost_aware_allocation"] is True
    assert payload["atom_mode"] == "module_proxy"
    assert payload["atom_mode_limitation"] == MODULE_PROXY_LIMITATION
    assert {"utility", "aggregation_mode", "atom_weight_normalization", "atom_mode"} <= set(result.atom_logs[0])
    assert {
        "module_utility",
        "rank_cost",
        "cost_aware_score",
        "continuous_rank",
        "final_rank",
    } <= set(result.module_logs[0])
    assert "budget_error_ratio" in payload
