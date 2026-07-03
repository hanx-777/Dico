from dico_rank.preallocation import DiCoPreAllocator
from dico_rank.rank_budget import compute_total_lora_params


def _allocate(seed: int):
    module_names = [f"m{i}" for i in range(8)]
    module_dims = {name: {"in_dim": i + 1, "out_dim": 0} for i, name in enumerate(module_names)}
    config = {
        "method": "dico_pre",
        "rank": 4,
        "seed": seed,
        "preallocation": {
            "allocation_method": "random_at_budget",
            "sketch_seed": seed,
            "r_min": 0,
            "r_max_multiplier": 2,
        },
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
    }
    allocator = DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config=config,
        module_names=module_names,
        module_dims=module_dims,
    )
    result = allocator.allocate(rank_budget=4 * sum(i + 1 for i in range(8)))
    return result, module_dims


def test_random_at_budget_is_seeded_and_within_param_budget():
    first, dims = _allocate(42)
    second, _dims = _allocate(42)
    third, _dims = _allocate(43)
    target = 4 * sum(i + 1 for i in range(8))

    assert first.rank_allocation == second.rank_allocation
    assert compute_total_lora_params(first.rank_allocation, dims) <= target
    assert first.allocation_method == "random_at_budget"
    assert first.rank_allocation != third.rank_allocation
