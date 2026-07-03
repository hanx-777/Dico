from dico_rank.rank_budget import compute_total_lora_params
from dico_rank.trainer import _downscale_lora_allocation_to_ratio


def test_lora_eta_downscale_prefers_high_cost_modules_within_interval():
    module_dims = {
        "cheap": {"in_dim": 1, "out_dim": 0},
        "expensive": {"in_dim": 3, "out_dim": 0},
    }
    allocation = {"cheap": 100, "expensive": 100}
    target_budget = compute_total_lora_params(allocation, module_dims)

    downscaled, metadata = _downscale_lora_allocation_to_ratio(
        allocation,
        module_dims,
        target_budget,
        target_ratio=0.98,
        min_ratio=0.97,
    )

    actual = compute_total_lora_params(downscaled, module_dims)
    assert 0.97 <= actual / target_budget <= 0.98
    assert downscaled["cheap"] == 100
    assert downscaled["expensive"] < 100
    assert metadata["lora_baseline_downscaled"] is True
    assert metadata["lora_downscale_interval_pass"] is True
