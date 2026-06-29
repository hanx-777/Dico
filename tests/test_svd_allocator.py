import torch

from src.dico_allocator import allocate_dico_lite
from src.module_coverage_allocator import allocate_module_coverage
from src.svd_utils import randomized_svd_topk
from src.uniform_allocator import allocate_uniform


def test_randomized_svd_topk_shapes_and_reconstruction_signal():
    torch.manual_seed(0)
    left = torch.randn(6, 2)
    right = torch.randn(2, 5)
    matrix = left @ right

    u, s, v = randomized_svd_topk(matrix, k=2, oversample=2, n_iter=1)

    assert u.shape == (6, 2)
    assert s.shape == (2,)
    assert v.shape == (5, 2)
    assert torch.isfinite(u).all()
    recon = u @ torch.diag(s) @ v.T
    assert torch.linalg.norm(matrix - recon) / torch.linalg.norm(matrix) < 0.35


def test_allocators_respect_budget_and_dico_prefix_constraint():
    module_names = ["layers.0.q_proj", "layers.0.v_proj"]
    module_dims = {
        "layers.0.q_proj": {"d_in": 4, "d_out": 4, "cost": 8},
        "layers.0.v_proj": {"d_in": 4, "d_out": 4, "cost": 8},
    }
    profiles = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
    )
    importance = {"layers.0.q_proj": 1.0, "layers.0.v_proj": 1.0}
    rho = {
        "layers.0.q_proj": torch.tensor([0.9, 0.1]),
        "layers.0.v_proj": torch.tensor([0.8, 0.2]),
    }

    dico = allocate_dico_lite(
        module_names=module_names,
        module_dims=module_dims,
        normalized_profiles=profiles,
        importance=importance,
        rho=rho,
        avg_rank=1,
    )
    assert sum(dico.rank_pattern.values()) <= 2
    assert all(step["atom_index"] == 0 for step in dico.allocation_steps[:2])
    assert dico.used_budget <= dico.total_budget

    module_profiles = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    module_alloc = allocate_module_coverage(module_names, module_dims, module_profiles, avg_rank=1)
    assert module_alloc.used_budget <= module_alloc.total_budget

    uniform = allocate_uniform(module_names, avg_rank=1)
    assert uniform == {"layers.0.q_proj": 1, "layers.0.v_proj": 1}
