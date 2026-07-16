import torch

from dico.covra_core import (
    build_response_block,
    build_type_scaled_utility_curves,
    greedy_conditional_coverage,
    independent_utility_curve,
    module_scalar_utility_curve,
)


def test_sign_split_response_block_charges_one_physical_rank():
    response = torch.tensor([2.0, -3.0, 0.0, 1.0])

    block = build_response_block(module_name="m", candidate_index=7, response=response, rho=0.2)

    assert block.split is True
    assert block.rank_cost == 1
    assert block.matrix.shape == (4, 2)
    assert torch.allclose(block.matrix[:, 0], torch.tensor([2.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(block.matrix[:, 1], torch.tensor([0.0, 3.0, -0.0, 0.0]))
    assert 0.0 < block.positive_energy_ratio < 1.0
    assert 0.0 < block.negative_energy_ratio < 1.0


def test_no_sign_split_keeps_single_signed_column():
    response = torch.tensor([2.0, -3.0, 0.0, 1.0])

    block = build_response_block(module_name="m", candidate_index=1, response=response, rho=0.9)

    assert block.split is False
    assert block.rank_cost == 1
    assert block.matrix.shape == (4, 1)
    assert torch.allclose(block.matrix[:, 0], response)


def test_conditional_coverage_discount_overlapping_candidate_but_independent_does_not():
    blocks = [
        build_response_block("m", 0, torch.tensor([1.0, 0.0, 0.0]), rho=1.0),
        build_response_block("m", 1, torch.tensor([1.0, 0.0, 0.0]), rho=1.0),
        build_response_block("m", 2, torch.tensor([0.0, 1.0, 0.0]), rho=1.0),
    ]

    conditional = greedy_conditional_coverage(blocks, r_max=3)
    independent = independent_utility_curve(blocks, r_max=3)

    assert conditional.selected_indices == [0, 2, 1]
    assert conditional.marginal_gains[0] > 0
    assert conditional.marginal_gains[1] > 0
    assert conditional.marginal_gains[2] == 0
    assert independent.marginal_gains[0] == independent.marginal_gains[1]
    assert independent.selected_indices[:2] == [0, 1]


def test_type_scaling_and_log_compression_are_independent_switches():
    curves = {
        "q0": [0.0, 10.0, 14.0],
        "q1": [0.0, 2.0, 3.0],
        "v0": [0.0, 100.0, 150.0],
    }
    module_types = {"q0": "q_proj", "q1": "q_proj", "v0": "v_proj"}

    scaled_log = build_type_scaled_utility_curves(
        curves,
        module_types,
        type_scaling=True,
        log_compression=True,
    )
    unscaled_linear = build_type_scaled_utility_curves(
        curves,
        module_types,
        type_scaling=False,
        log_compression=False,
    )

    assert unscaled_linear["v0"][1] == 100.0
    assert scaled_log["v0"][1] < unscaled_linear["v0"][1]
    assert scaled_log["q0"][2] > scaled_log["q0"][1]


def test_module_scalar_curve_uses_fixed_nonincreasing_template():
    curve = module_scalar_utility_curve(module_energy=12.0, r_max=3, template=[0.5, 0.3, 0.2])

    assert curve == [0.0, 6.0, 9.6, 12.0]
