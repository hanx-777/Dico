import torch

from dico.gora_bw import compute_gora_importance, allocate_gora_bw


def test_gora_importance_is_mean_abs_weight_times_gradient():
    weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    grad = torch.tensor([[0.5, 1.0], [-2.0, 0.25]])

    assert compute_gora_importance(weight, grad) == torch.mean(torch.abs(weight * grad)).item()


def test_gora_bw_repairs_rank_dict_into_budget_window():
    weights = {
        "a": torch.ones(2, 2),
        "b": torch.ones(8, 8) * 0.1,
        "c": torch.ones(2, 8) * 0.5,
    }
    grads = {
        "a": torch.ones(2, 2) * 4.0,
        "b": torch.ones(8, 8) * 0.1,
        "c": torch.ones(2, 8) * 1.0,
    }
    dims = {
        "a": {"in_dim": 2, "out_dim": 2},
        "b": {"in_dim": 8, "out_dim": 8},
        "c": {"in_dim": 2, "out_dim": 8},
    }

    result = allocate_gora_bw(weights, grads, dims, r_ref=8, eta=0.98)

    assert result.r_min == 4
    assert result.r_max == 32
    assert result.realized_params <= result.target_budget
    assert result.realized_params >= int(0.98 * result.target_budget)
    assert all(4 <= rank <= 32 for rank in result.rank_dict.values())
