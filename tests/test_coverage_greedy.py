import math

import torch

from dico_rank.atom_svd import (
    SvdAtomRecord,
    aggregate_selected_module_utilities,
    coverage_residual,
    select_coverage_evidence,
)


def _atom(module: str, idx: int, profile: torch.Tensor, utility: float = 1.0) -> SvdAtomRecord:
    return SvdAtomRecord(
        module_name=module,
        atom_index=idx,
        cost=10,
        singular_value=1.0,
        spectral_ratio=1.0,
        profile=profile / torch.linalg.norm(profile),
        conflict=0.0,
        coverage=0.0,
        lambda_cov=1.0,
        utility=utility,
        module_importance=1.0,
    )


def test_coverage_residual_empty_same_and_orthogonal():
    e1 = torch.tensor([1.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 1.0, 0.0])
    basis = torch.empty(3, 0)

    assert coverage_residual(e1, basis) == 1.0
    basis = e1[:, None]
    assert coverage_residual(e1, basis) < 1e-6
    assert abs(coverage_residual(e2, basis) - 1.0) < 1e-6


def test_prefix_stop_conditions_do_not_select_all_atoms():
    atoms = [
        _atom("A", 0, torch.tensor([1.0, 0.0, 0.0]), utility=10.0),
        _atom("A", 1, torch.tensor([0.0, 1.0, 0.0]), utility=9.0),
        _atom("B", 0, torch.tensor([0.0, 0.0, 1.0]), utility=8.0),
        _atom("B", 1, torch.tensor([1.0, 1.0, 0.0]), utility=7.0),
    ]

    selected = select_coverage_evidence(
        atoms,
        max_selected_atoms=2,
        epsilon_cov=0.05,
        sparse_stop_by_coverage=False,
    )

    assert len(selected) == 2
    assert sum(atom.selected for atom in atoms) == 2
    selected_by_module = {}
    for atom in selected:
        selected_by_module.setdefault(atom.module_name, []).append(atom.atom_index)
    for indices in selected_by_module.values():
        assert indices == list(range(len(indices)))


def test_weighted_log_aggregates_selected_evidence_only():
    atoms = [
        _atom("A", 0, torch.tensor([1.0, 0.0]), utility=3.0),
        _atom("A", 1, torch.tensor([0.0, 1.0]), utility=100.0),
        _atom("B", 0, torch.tensor([1.0, 1.0]), utility=2.0),
    ]
    atoms[0].selected = True
    atoms[1].selected = False
    atoms[2].selected = True

    utilities = aggregate_selected_module_utilities(atoms, ["A", "B"], aggregation_mode="weighted_log")

    assert utilities["A"] == math.log1p(3.0)
    assert utilities["B"] == math.log1p(2.0)
