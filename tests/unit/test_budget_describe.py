from dico.rank_budget import BudgetManager, get_uniform_budget


def test_budget_describe_reports_paramcount_and_ranksum_units():
    dims = {
        "q": {"in_dim": 4, "out_dim": 4},
        "up": {"in_dim": 4, "out_dim": 12},
    }
    manager = BudgetManager("equal_trainable_params", dims, warning_threshold=0.01)

    info = manager.describe({"q": 1, "up": 2}, target_budget=48)

    assert info["target_budget_paramcount"] == 48
    assert info["actual_budget_paramcount"] == 40
    assert info["target_budget_ranksum"] == 4
    assert info["actual_budget_ranksum"] == 3
    assert info["budget_ratio_paramcount"] == 40 / 48
    assert info["budget_ratio_ranksum"] == 3 / 4
    assert info["target_budget"] == info["target_budget_paramcount"]
    assert info["actual_budget"] == info["actual_budget_paramcount"]
    assert info["budget_ratio"] == info["budget_ratio_paramcount"]
    assert info["budget_error"] == -8
    assert info["budget_error_ratio"] == -8 / 48
    assert "paramcount" in info["budget_units_note"]


def test_uniform_budget_reports_target_rank_sum():
    dims = {
        "q": {"in_dim": 4, "out_dim": 4},
        "up": {"in_dim": 4, "out_dim": 12},
    }

    info = get_uniform_budget(2, ["q", "up"], dims).to_dict()

    assert info["target_budget_paramcount"] == 48
    assert info["actual_budget_paramcount"] == 48
    assert info["target_budget_ranksum"] == 4
    assert info["actual_budget_ranksum"] == 4
    assert info["budget_ratio_paramcount"] == 1.0
    assert info["budget_ratio_ranksum"] == 1.0
