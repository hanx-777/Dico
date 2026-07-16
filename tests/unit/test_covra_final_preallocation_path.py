import torch

from dico.atom_svd import SvdAtomRecord
from dico.preallocation import DiCoPreAllocator, load_direction_bank


def _atom(module_name: str, atom_index: int, profile: list[float]) -> SvdAtomRecord:
    return SvdAtomRecord(
        module_name=module_name,
        atom_index=atom_index,
        cost=4,
        singular_value=1.0,
        spectral_ratio=1.0,
        profile=torch.tensor(profile, dtype=torch.float32),
        conflict=0.0,
        coverage=0.0,
        lambda_cov=1.0,
        utility=0.0,
        module_importance=1.0,
        v=torch.ones(4),
    )


def _allocator(method: str, tmp_path=None) -> DiCoPreAllocator:
    config = {
        "rank": 1,
        "method": "dico_cd_da",
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
        "preallocation": {
            "allocation_method": method,
            "top_k_atoms": 2,
            "r_min_multiplier": 0.0,
            "r_max_multiplier": 2.0,
            "rho": 1.0,
            "type_scaling": False,
            "log_compression": False,
        },
    }
    if tmp_path is not None:
        config["_project_root"] = str(tmp_path)
        config["calibration"] = {"save_dir": str(tmp_path / "preallocations")}
    return DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config=config,
        module_names=["layers.0.q_proj", "layers.1.q_proj"],
        module_dims={
            "layers.0.q_proj": {"in_dim": 2, "out_dim": 2},
            "layers.1.q_proj": {"in_dim": 2, "out_dim": 2},
        },
    )


def test_covra_full_preallocation_path_uses_final_core_not_legacy_procurement():
    atoms = [
        _atom("layers.0.q_proj", 0, [1.0, 0.0]),
        _atom("layers.0.q_proj", 1, [1.0, 0.0]),
        _atom("layers.1.q_proj", 0, [0.0, 1.0]),
        _atom("layers.1.q_proj", 1, [0.0, 0.5]),
    ]

    result = _allocator("covra_full")._allocate_final_covra_from_svd_atoms(
        atoms,
        rank_budget=8,
        base_diagnostics={},
    )

    assert result.diagnostics["allocation_method"] == "covra_full"
    assert result.diagnostics["solver"] == "dp"
    assert "taxonomy_stats" not in result.diagnostics
    assert "procurement_trace" not in result.diagnostics
    assert result.budget.actual_budget <= result.budget.target_budget
    assert sum(result.allocation.values()) == 2


def test_svd_dispatch_keeps_reference_and_experimental_covra_paths_isolated(monkeypatch):
    reference = _allocator("covra_v05")
    legacy_reference_alias = _allocator("dico_v03")
    experimental = _allocator("covra_full")
    calls: list[str] = []

    monkeypatch.setattr(
        reference,
        "_allocate_v03_from_svd_atoms",
        lambda atoms, rank_budget, diagnostics: calls.append("reference") or "reference-result",
    )
    monkeypatch.setattr(
        reference,
        "_allocate_final_covra_from_svd_atoms",
        lambda atoms, rank_budget, diagnostics: calls.append("experimental") or "experimental-result",
    )
    monkeypatch.setattr(
        legacy_reference_alias,
        "_allocate_v03_from_svd_atoms",
        lambda atoms, rank_budget, diagnostics: calls.append("legacy-reference") or "legacy-reference-result",
    )
    monkeypatch.setattr(
        experimental,
        "_allocate_v03_from_svd_atoms",
        lambda atoms, rank_budget, diagnostics: calls.append("reference") or "reference-result",
    )
    monkeypatch.setattr(
        experimental,
        "_allocate_final_covra_from_svd_atoms",
        lambda atoms, rank_budget, diagnostics: calls.append("experimental") or "experimental-result",
    )

    assert reference._allocate_from_svd_atoms([], 8, {}) == "reference-result"
    assert legacy_reference_alias._allocate_from_svd_atoms([], 8, {}) == "legacy-reference-result"
    assert experimental._allocate_from_svd_atoms([], 8, {}) == "experimental-result"
    assert calls == ["reference", "legacy-reference", "experimental"]


def test_reference_covra_reports_effective_procurement_beta_from_dico_config(tmp_path):
    allocator = _allocator("covra_v05", tmp_path)
    allocator.pre_cfg["beta"] = 1.0
    allocator.config["data"] = {"group_labels": ["a", "b"]}
    allocator.config["dico"] = {
        "taxonomy": {"alpha": 0.05, "permutation_count": 8},
        "pseudo_group": {"enabled": False},
        "split": {"mode": "sign"},
        "coverage": {"kappa_calibration": {"enabled": False}, "relative_stop_delta": 0.0},
        "procurement": {"beta": 0.5, "reserve_queue": True, "relaxation_fallback": True},
    }
    atoms = [
        _atom("layers.0.q_proj", 0, [1.0, 0.0]),
        _atom("layers.0.q_proj", 1, [0.5, 0.0]),
        _atom("layers.1.q_proj", 0, [0.0, 1.0]),
        _atom("layers.1.q_proj", 1, [0.0, 0.5]),
    ]

    result = allocator._allocate_v03_from_svd_atoms(atoms, rank_budget=8, base_diagnostics={})

    assert result.diagnostics["procurement_beta"] == 0.5


def test_covra_final_path_emits_direction_bank_for_selected_physical_ranks(tmp_path):
    atoms = [
        _atom("layers.0.q_proj", 0, [1.0, 0.0]),
        _atom("layers.0.q_proj", 1, [0.2, 0.0]),
        _atom("layers.1.q_proj", 0, [0.0, 1.0]),
        _atom("layers.1.q_proj", 1, [0.0, 0.2]),
    ]

    result = _allocator("covra_full", tmp_path)._allocate_final_covra_from_svd_atoms(
        atoms,
        rank_budget=8,
        base_diagnostics={},
    )

    direction_bank_path = result.diagnostics["direction_bank_path"]
    bank = load_direction_bank(direction_bank_path)

    assert set(bank) == set(result.allocation)
    for module_name, rows in bank.items():
        assert len(rows) == result.allocation[module_name]
        assert all(row["source"] == "certified" for row in rows)
        assert all(isinstance(row["v"], torch.Tensor) for row in rows)
        assert all(row["v"].shape == (4,) for row in rows)


def test_covra_independent_and_module_scalar_use_distinct_registered_methods():
    atoms = [
        _atom("layers.0.q_proj", 0, [1.0, 0.0]),
        _atom("layers.0.q_proj", 1, [1.0, 0.0]),
        _atom("layers.1.q_proj", 0, [0.0, 1.0]),
        _atom("layers.1.q_proj", 1, [0.0, 0.5]),
    ]

    independent = _allocator("covra_independent")._allocate_final_covra_from_svd_atoms(atoms, 8, {})
    module_scalar = _allocator("covra_module_scalar")._allocate_final_covra_from_svd_atoms(atoms, 8, {})

    assert independent.diagnostics["allocation_method"] == "covra_independent"
    assert module_scalar.diagnostics["allocation_method"] == "covra_module_scalar"
    assert independent.diagnostics["utility_builder"] == "independent"
    assert module_scalar.diagnostics["utility_builder"] == "module_scalar"


def test_covra_module_scalar_shares_covra_i_initialization_order(tmp_path):
    atoms = [
        _atom("layers.0.q_proj", 0, [0.1, 0.0]),
        _atom("layers.0.q_proj", 1, [5.0, 0.0]),
        _atom("layers.1.q_proj", 0, [0.0, 0.1]),
        _atom("layers.1.q_proj", 1, [0.0, 4.0]),
    ]

    independent = _allocator("covra_independent", tmp_path / "independent")._allocate_final_covra_from_svd_atoms(
        atoms,
        rank_budget=8,
        base_diagnostics={},
    )
    module_scalar = _allocator("covra_module_scalar", tmp_path / "module_scalar")._allocate_final_covra_from_svd_atoms(
        atoms,
        rank_budget=8,
        base_diagnostics={},
    )

    assert module_scalar.diagnostics["utility_builder"] == "module_scalar"
    assert module_scalar.diagnostics["initialization_selection_builder"] == "independent"
    assert module_scalar.diagnostics["selected_atom_indices"] == independent.diagnostics["selected_atom_indices"]

    independent_bank = load_direction_bank(independent.diagnostics["direction_bank_path"])
    module_scalar_bank = load_direction_bank(module_scalar.diagnostics["direction_bank_path"])
    assert {
        name: [row["atom_index"] for row in rows]
        for name, rows in module_scalar_bank.items()
    } == {
        name: [row["atom_index"] for row in rows]
        for name, rows in independent_bank.items()
    }
    assert {
        name: [row["utility"] for row in rows]
        for name, rows in module_scalar_bank.items()
    } == {
        name: [row["utility"] for row in rows]
        for name, rows in independent_bank.items()
    }


def test_covra_module_scalar_diagnostics_record_effective_rank_template():
    atoms = [
        _atom("layers.0.q_proj", 0, [1.0, 0.0]),
        _atom("layers.0.q_proj", 1, [0.5, 0.0]),
        _atom("layers.1.q_proj", 0, [0.0, 1.0]),
        _atom("layers.1.q_proj", 1, [0.0, 0.5]),
    ]

    result = _allocator("covra_module_scalar")._allocate_final_covra_from_svd_atoms(
        atoms,
        rank_budget=8,
        base_diagnostics={},
    )

    assert result.diagnostics["module_scalar_template_formula"] == "w_j = 1 / j for j=1..r_max"
    assert result.diagnostics["module_scalar_template_normalization"] == "sum_to_module_energy"
    assert result.diagnostics["module_scalar_template"] == [1.0, 0.5]


def test_covra_proportional_rounding_solver_is_registered_ablation():
    atoms = [
        _atom("layers.0.q_proj", 0, [1.0, 0.0]),
        _atom("layers.0.q_proj", 1, [0.5, 0.0]),
        _atom("layers.1.q_proj", 0, [0.0, 1.0]),
        _atom("layers.1.q_proj", 1, [0.0, 0.5]),
    ]
    allocator = _allocator("covra_full")
    allocator.pre_cfg["solver"] = "proportional_rounding"

    result = allocator._allocate_final_covra_from_svd_atoms(atoms, 8, {})

    assert result.diagnostics["solver"] == "proportional_rounding"
    assert result.budget.actual_budget <= result.budget.target_budget
    assert sum(result.allocation.values()) == 2


def test_uniform_rank_covra_init_override_keeps_covra_directions_but_forces_uniform_rank(tmp_path):
    atoms = [
        _atom("layers.0.q_proj", 0, [4.0, 0.0]),
        _atom("layers.0.q_proj", 1, [3.0, 0.0]),
        _atom("layers.1.q_proj", 0, [0.0, 0.1]),
        _atom("layers.1.q_proj", 1, [0.0, 0.1]),
    ]
    allocator = _allocator("covra_full", tmp_path)
    allocator.pre_cfg["rank_override"] = "uniform_ref"

    result = allocator._allocate_final_covra_from_svd_atoms(atoms, rank_budget=16, base_diagnostics={})

    assert result.allocation == {"layers.0.q_proj": 1, "layers.1.q_proj": 1}
    assert result.diagnostics["rank_override"] == "uniform_ref"
    assert result.diagnostics["allocation_before_rank_override"] != result.allocation
    bank = load_direction_bank(result.diagnostics["direction_bank_path"])
    assert set(bank) == set(result.allocation)
    assert all(len(rows) == 1 for rows in bank.values())
