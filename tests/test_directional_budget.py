from dico_rank.rank_budget import allocate_by_directional_evidence


def test_directional_budget_purchases_atoms_by_density_without_exceeding_target():
    dims = {
        "cheap": {"in_dim": 2, "out_dim": 2},
        "expensive": {"in_dim": 8, "out_dim": 8},
    }
    atoms = [
        {"module_name": "cheap", "atom_index": 0, "utility": 4.0, "selected": True},
        {"module_name": "expensive", "atom_index": 0, "utility": 8.0, "selected": True},
    ]

    result = allocate_by_directional_evidence(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=16,
        eta=1.0,
        r_min=0,
        r_max=4,
        beta=1.0,
        allow_rank_beyond_selected_evidence=True,
    )

    assert result.allocation == {"cheap": 4, "expensive": 0}
    assert result.budget.actual_budget == 16
    assert result.module_logs[0]["rank_beyond_selected_evidence"] == 3
    assert result.module_logs[0]["evidence_relaxation_rank"] == 3


def test_directional_budget_records_warning_when_eta_cannot_be_reached():
    dims = {"m": {"in_dim": 10, "out_dim": 0}}
    atoms = [{"module_name": "m", "atom_index": 0, "utility": 1.0, "selected": True}]

    result = allocate_by_directional_evidence(
        atom_logs=atoms,
        module_dims=dims,
        target_budget=25,
        eta=0.9,
        r_min=0,
        r_max=1,
        beta=1.0,
        allow_rank_beyond_selected_evidence=False,
    )

    assert result.allocation == {"m": 1}
    assert result.budget.actual_budget == 10
    assert "prevented reaching eta target" in result.budget.warning
