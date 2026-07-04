import pytest

from dico_rank.rank_budget import allocate_by_rank_allocator, compute_total_lora_params


def _dims():
    return {
        "model.layers.0.self_attn.q_proj": {"in_dim": 2, "out_dim": 2},
        "model.layers.1.self_attn.q_proj": {"in_dim": 2, "out_dim": 2},
        "model.layers.2.self_attn.q_proj": {"in_dim": 2, "out_dim": 2},
        "model.layers.0.mlp.up_proj": {"in_dim": 4, "out_dim": 4},
    }


def _atoms():
    return [
        {
            "module_name": "model.layers.0.self_attn.q_proj",
            "atom_index": 0,
            "utility": 9.0,
            "coverage_gain": 9.0,
            "align": 1.0,
            "selected": True,
            "pi": [1.0, 0.0, 0.0],
        },
        {
            "module_name": "model.layers.0.self_attn.q_proj",
            "atom_index": 1,
            "utility": 8.0,
            "coverage_gain": 8.0,
            "align": 1.0,
            "selected": True,
            "pi": [0.99, 0.01, 0.0],
        },
        {
            "module_name": "model.layers.1.self_attn.q_proj",
            "atom_index": 0,
            "utility": 4.0,
            "coverage_gain": 4.0,
            "align": 1.0,
            "selected": True,
            "pi": [0.0, 1.0, 0.0],
        },
        {
            "module_name": "model.layers.2.self_attn.q_proj",
            "atom_index": 0,
            "utility": 3.0,
            "coverage_gain": 3.0,
            "align": 1.0,
            "selected": True,
            "pi": [0.0, 0.0, 1.0],
        },
        {
            "module_name": "model.layers.0.mlp.up_proj",
            "atom_index": 0,
            "utility": 5.0,
            "coverage_gain": 5.0,
            "align": 1.0,
            "selected": True,
            "pi": [0.0, 1.0, 1.0],
        },
    ]


@pytest.mark.parametrize("atom_to_rank", ["marginal_curve", "prototype_bundle", "soft_slot"])
@pytest.mark.parametrize("smoothing", ["budget_guardrails", "layer_diffusion", "concentration_penalty"])
def test_all_rank_allocator_combinations_are_budget_safe(atom_to_rank, smoothing):
    dims = _dims()
    result = allocate_by_rank_allocator(
        atom_logs=_atoms(),
        module_dims=dims,
        target_budget=24,
        eta=0.75,
        r_min=0,
        r_max=4,
        config={
            "atom_to_rank": atom_to_rank,
            "smoothing": smoothing,
            "cost_beta": 0.5,
            "utility": {"type_normalization": "none"},
            "budget_guardrails": {"max_rank_per_module": 3},
            "concentration_penalty": {"lambda": 1.0},
        },
    )

    assert result.budget.actual_budget <= 24
    assert compute_total_lora_params(result.allocation, dims) == result.budget.actual_budget
    assert all(isinstance(rank, int) for rank in result.allocation.values())
    assert all(0 <= rank <= 4 for rank in result.allocation.values())
    assert result.diagnostics["atom_to_rank"] == atom_to_rank
    assert result.diagnostics["smoothing"] == smoothing


def test_legacy_atom_purchase_matches_direct_density_fixture():
    dims = {
        "cheap": {"in_dim": 2, "out_dim": 2},
        "expensive": {"in_dim": 8, "out_dim": 8},
    }
    atoms = [
        {"module_name": "cheap", "atom_index": 0, "utility": 4.0, "selected": True},
        {"module_name": "expensive", "atom_index": 0, "utility": 8.0, "selected": True},
    ]

    result = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=16,
        eta=1.0,
        r_min=0,
        r_max=4,
        config={"atom_to_rank": "legacy_atom_purchase", "smoothing": "none", "cost_beta": 1.0},
    )

    assert result.allocation == {"cheap": 4, "expensive": 0}
    assert result.module_logs[0]["evidence_relaxation_rank"] == 3


def test_legacy_atom_purchase_uses_raw_utility_despite_new_defaults():
    dims = {
        "model.layers.0.self_attn.a_proj": {"in_dim": 2, "out_dim": 2},
        "model.layers.0.mlp.z_proj": {"in_dim": 2, "out_dim": 2},
    }
    atoms = [
        {
            "module_name": "model.layers.0.self_attn.a_proj",
            "atom_index": 0,
            "utility": 10.0,
            "selected": True,
        },
        {
            "module_name": "model.layers.0.mlp.z_proj",
            "atom_index": 0,
            "utility": 1.0,
            "selected": True,
        },
    ]

    result = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=4,
        eta=0.0,
        r_min=0,
        r_max=1,
        config={"atom_to_rank": "legacy_atom_purchase", "smoothing": "none", "cost_beta": 0.0},
    )

    assert result.allocation == {
        "model.layers.0.self_attn.a_proj": 1,
        "model.layers.0.mlp.z_proj": 0,
    }


def test_marginal_curve_differs_from_legacy_on_many_weak_atoms():
    dims = {"many": {"in_dim": 2, "out_dim": 2}, "strong": {"in_dim": 2, "out_dim": 2}}
    atoms = [
        {"module_name": "many", "atom_index": idx, "utility": 1.0, "selected": True}
        for idx in range(4)
    ] + [{"module_name": "strong", "atom_index": 0, "utility": 3.0, "selected": True}]

    legacy = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=16,
        eta=0.0,
        r_min=0,
        r_max=4,
        config={"atom_to_rank": "legacy_atom_purchase", "smoothing": "none", "cost_beta": 0.0},
    )
    marginal = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=16,
        eta=0.0,
        r_min=0,
        r_max=4,
        config={
            "atom_to_rank": "marginal_curve",
            "smoothing": "none",
            "cost_beta": 0.0,
            "utility": {"type_normalization": "none"},
            "marginal_curve": {"decay": "sqrt"},
        },
    )

    assert legacy.allocation != marginal.allocation
    assert marginal.allocation["strong"] > legacy.allocation["strong"]


def test_prototype_bundle_merges_similar_response_profiles():
    dims = {"m": {"in_dim": 2, "out_dim": 2}}
    atoms = [
        {"module_name": "m", "atom_index": 0, "utility": 4.0, "selected": True, "pi": [1.0, 0.0]},
        {"module_name": "m", "atom_index": 1, "utility": 3.0, "selected": True, "pi": [0.99, 0.01]},
        {"module_name": "m", "atom_index": 2, "utility": 2.0, "selected": True, "pi": [0.0, 1.0]},
    ]

    result = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=12,
        eta=0.0,
        r_min=0,
        r_max=3,
        config={
            "atom_to_rank": "prototype_bundle",
            "smoothing": "none",
            "utility": {"type_normalization": "none"},
            "prototype_bundle": {"similarity_threshold": 0.8, "residual_weight": 0.25},
        },
    )

    assert result.module_logs[0]["bundle_count"] == 2


def test_soft_slot_respects_slot_precedence():
    dims = {"m": {"in_dim": 2, "out_dim": 2}}
    atoms = [{"module_name": "m", "atom_index": idx, "utility": 1.0, "selected": True} for idx in range(3)]

    result = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=12,
        eta=0.0,
        r_min=0,
        r_max=3,
        config={
            "atom_to_rank": "soft_slot",
            "smoothing": "none",
            "utility": {"type_normalization": "none"},
            "soft_slot": {"temperature": 1.0, "slot_decay": 0.15},
        },
    )

    assert result.allocation["m"] == 3
    assert result.module_logs[0]["purchased_slots"] == [1, 2, 3]


def test_layer_diffusion_reduces_isolated_layer_spike():
    dims = {
        "model.layers.0.self_attn.q_proj": {"in_dim": 2, "out_dim": 2},
        "model.layers.1.self_attn.q_proj": {"in_dim": 2, "out_dim": 2},
        "model.layers.2.self_attn.q_proj": {"in_dim": 2, "out_dim": 2},
    }
    atoms = [
        {
            "module_name": "model.layers.1.self_attn.q_proj",
            "atom_index": idx,
            "utility": 10.0,
            "selected": True,
        }
        for idx in range(3)
    ]

    unsmoothed = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=12,
        eta=0.0,
        r_min=0,
        r_max=3,
        config={"atom_to_rank": "marginal_curve", "smoothing": "none", "utility": {"type_normalization": "none"}},
    )
    diffused = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=12,
        eta=0.0,
        r_min=0,
        r_max=3,
        config={
            "atom_to_rank": "marginal_curve",
            "smoothing": "layer_diffusion",
            "utility": {"type_normalization": "none"},
            "layer_diffusion": {"kernel": [0.25, 0.5, 0.25]},
        },
    )

    assert diffused.allocation["model.layers.1.self_attn.q_proj"] < unsmoothed.allocation["model.layers.1.self_attn.q_proj"]
    assert diffused.diagnostics["layer_total_variation"] < unsmoothed.diagnostics["layer_total_variation"]


def test_concentration_penalty_lowers_hhi_against_no_penalty():
    dims = _dims()
    atoms = [
        {"module_name": "model.layers.0.self_attn.q_proj", "atom_index": idx, "utility": 10.0, "selected": True}
        for idx in range(5)
    ] + [
        {"module_name": "model.layers.1.self_attn.q_proj", "atom_index": 0, "utility": 9.0, "selected": True},
        {"module_name": "model.layers.2.self_attn.q_proj", "atom_index": 0, "utility": 8.0, "selected": True},
    ]

    base = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=20,
        eta=0.0,
        r_min=0,
        r_max=5,
        config={"atom_to_rank": "marginal_curve", "smoothing": "none", "utility": {"type_normalization": "none"}},
    )
    penalized = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=20,
        eta=0.0,
        r_min=0,
        r_max=5,
        config={
            "atom_to_rank": "marginal_curve",
            "smoothing": "concentration_penalty",
            "utility": {"type_normalization": "none"},
            "concentration_penalty": {"lambda": 10.0},
        },
    )

    assert penalized.diagnostics["hhi"] < base.diagnostics["hhi"]


def test_guardrails_warn_when_eta_is_infeasible():
    dims = {"m": {"in_dim": 10, "out_dim": 0}}
    atoms = [{"module_name": "m", "atom_index": 0, "utility": 5.0, "selected": True}]

    result = allocate_by_rank_allocator(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=30,
        eta=1.0,
        r_min=0,
        r_max=3,
        config={
            "atom_to_rank": "marginal_curve",
            "smoothing": "budget_guardrails",
            "utility": {"type_normalization": "none"},
            "budget_guardrails": {"max_rank_per_module": 1},
        },
    )

    assert result.budget.actual_budget == 10
    assert any("eta target" in warning for warning in result.diagnostics["warnings"])
