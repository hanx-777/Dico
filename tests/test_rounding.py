from dico_rank.rank_budget import allocate_by_evidence_aware_utility, compute_total_lora_params


def test_evidence_aware_rounding_does_not_exceed_selected_counts():
    dims = {
        "A": {"in_dim": 5, "out_dim": 5},
        "B": {"in_dim": 5, "out_dim": 5},
    }

    result = allocate_by_evidence_aware_utility(
        module_utilities={"A": 100.0, "B": 10.0},
        module_dims=dims,
        selected_atom_utilities={"A": [100.0], "B": [10.0, 9.0, 8.0, 7.0]},
        target_budget=40,
        eta=1.0,
        lambda_next=1.0,
        r_min=0,
        r_max=4,
        allow_rank_beyond_selected_evidence=False,
    )

    assert result.allocation["A"] <= 1
    assert result.allocation["B"] <= 4
    assert compute_total_lora_params(result.allocation, dims) <= 40
    assert compute_total_lora_params(result.allocation, dims) == 40


def test_evidence_aware_rounding_warns_instead_of_exceeding_evidence():
    dims = {
        "A": {"in_dim": 5, "out_dim": 5},
        "B": {"in_dim": 5, "out_dim": 5},
    }

    result = allocate_by_evidence_aware_utility(
        module_utilities={"A": 100.0, "B": 0.0},
        module_dims=dims,
        selected_atom_utilities={"A": [100.0], "B": []},
        target_budget=40,
        eta=1.0,
        lambda_next=1.0,
        r_min=0,
        r_max=4,
        allow_rank_beyond_selected_evidence=False,
    )

    assert result.allocation == {"A": 1, "B": 0}
    assert result.budget.actual_budget == 10
    assert "selected evidence" in (result.budget.warning or "")


def test_evidence_aware_budget_fair_relaxation_reaches_eta_without_exceeding_target():
    dims = {
        "A": {"in_dim": 5, "out_dim": 5},
        "B": {"in_dim": 5, "out_dim": 5},
    }

    result = allocate_by_evidence_aware_utility(
        module_utilities={"A": 100.0, "B": 10.0},
        module_dims=dims,
        selected_atom_utilities={"A": [100.0], "B": []},
        target_budget=40,
        eta=0.98,
        lambda_next=1.0,
        r_min=0,
        r_max=4,
        allow_rank_beyond_selected_evidence=True,
    )

    assert result.budget.actual_budget >= int(0.98 * 40)
    assert result.budget.actual_budget <= 40
    assert result.allocation["A"] > 1
    assert any(row["rank_beyond_selected_evidence"] > 0 for row in result.module_logs)


def test_evidence_aware_allocation_respects_r_min():
    dims = {
        "A": {"in_dim": 5, "out_dim": 5},
        "B": {"in_dim": 5, "out_dim": 5},
    }

    result = allocate_by_evidence_aware_utility(
        module_utilities={"A": 100.0, "B": 1.0},
        module_dims=dims,
        selected_atom_utilities={"A": [100.0], "B": [1.0]},
        target_budget=40,
        eta=0.98,
        lambda_next=1.0,
        r_min=2,
        r_max=4,
        allow_rank_beyond_selected_evidence=True,
    )

    assert result.allocation["A"] >= 2
    assert result.allocation["B"] >= 2
    assert result.budget.actual_budget <= 40
