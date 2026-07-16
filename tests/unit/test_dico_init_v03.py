import torch

from dico.init import build_direction_anchored_init


def test_direction_anchored_init_orders_by_utility_and_zeroes_initial_update():
    directions = [
        {"v": torch.tensor([1.0, 1.0, 0.0]), "utility": 1.0, "source": "certified"},
        {"v": torch.tensor([1.0, 0.0, 0.0]), "utility": 3.0, "source": "certified"},
    ]

    init = build_direction_anchored_init(directions, rank=2, in_dim=3, out_dim=4, seed=123)

    assert init.A.shape == (2, 3)
    assert init.B.shape == (4, 2)
    assert torch.allclose(init.B, torch.zeros_like(init.B))
    assert torch.allclose(init.B @ init.A, torch.zeros(4, 3))
    gram = init.A @ init.A.T
    assert torch.allclose(gram, torch.eye(2), atol=1e-6)
    assert torch.allclose(init.A[0], torch.tensor([1.0, 0.0, 0.0]))


def test_relaxation_rows_are_orthogonalized_against_certified_rows():
    directions = [{"v": torch.tensor([1.0, 0.0, 0.0]), "utility": 1.0, "source": "certified"}]

    init = build_direction_anchored_init(directions, rank=2, in_dim=3, out_dim=2, seed=5)

    assert torch.allclose(init.A @ init.A.T, torch.eye(2), atol=1e-5)
    assert init.summary["relaxation_rows"] == 1


def test_fewer_certified_directions_than_rank_yields_anchored_rank_ratio_below_one():
    # Appendix B checklist item 8: if K_atom < r_m (fewer candidate directions
    # than the purchased rank), anchored_rank_ratio must come out strictly < 1.
    directions = [{"v": torch.tensor([1.0, 0.0, 0.0]), "utility": 1.0, "source": "certified"}]

    init = build_direction_anchored_init(directions, rank=3, in_dim=3, out_dim=2, seed=5)

    certified = init.summary["certified_rows"]
    relaxation = init.summary["relaxation_rows"]
    anchored_rank_ratio = certified / (certified + relaxation)
    assert certified == 1
    assert relaxation == 2
    assert anchored_rank_ratio < 1.0
