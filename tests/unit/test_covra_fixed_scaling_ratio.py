import json
from pathlib import Path

from dico.lora_scaling import compute_covra_module_alpha
from dico.trainer import train


def test_compute_covra_module_alpha_hand_computed_ratio():
    rank_allocation = {"m1": 2, "m2": 6, "m3": 10}

    alpha_by_module = compute_covra_module_alpha(rank_allocation, alpha_ref=16.0, r_ref=8.0)

    for name, rank in rank_allocation.items():
        assert alpha_by_module[name] == rank * 16.0 / 8.0
        assert alpha_by_module[name] / rank == 16.0 / 8.0  # fixed ratio, independent of rank


def _tiny_config(tmp_path: Path, method: str, allocation_method: str, experiment_name: str) -> dict:
    return {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": experiment_name,
        "method": method,
        "rank": 4,
        "project": {"output_dir": str(tmp_path / "outputs")},
        "model": {"type": "tiny", "name_or_path": "tiny", "hidden_size": 16, "vocab_size": 64, "torch_dtype": "float32"},
        "data": {
            "source": "tiny",
            "train_path": "tiny",
            "eval_path": "tiny",
            "max_length": 16,
            "train_limit": 8,
            "eval_limit": 2,
        },
        "training": {"max_steps": 1, "batch_size": 1, "gradient_accumulation_steps": 1},
        "lora": {
            "injection": "static",
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "max_rank_multiplier": 4,
        },
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
        "calibration": {
            "enabled": True,
            "num_samples": 8,
            "batch_size": 1,
            "seed": 42,
            "save_dir": str(tmp_path / f"preallocations_{experiment_name}"),
        },
        "preallocation": {
            "atom_mode": "svd",
            "allocation_method": allocation_method,
            "top_k_atoms": 16,
            "sketch_dim": 16,
            "sketch_seed": 42,
            "answer_only": False,
            "profile_norm_mode": "streaming_estimate",
            "rho": 0.05,
            "sign_split": True,
            "type_scaling": True,
            "log_compression": True,
            "solver": "dp",
            "eta": 0.5,
            "r_min_multiplier": 0.25,
            "r_max_multiplier": 4.0,
        },
        "dico": {
            "version": "cd_da",
            "init": {"mode": "direction_anchored", "zero_B": True},
        },
    }


def test_covra_run_writes_lora_scaling_json_with_constant_ratio_across_modules(tmp_path: Path):
    config = _tiny_config(tmp_path, method="dico_cd_da", allocation_method="covra_full", experiment_name="covra_fixed_ratio")

    train(config)

    output_dir = tmp_path / "outputs" / "covra_fixed_ratio"
    rank_dict = json.loads((output_dir / "rank_dict.json").read_text())
    assert rank_dict

    lora_scaling = json.loads((output_dir / "lora_scaling.json").read_text())
    expected_ratio = lora_scaling["alpha_ref"] / lora_scaling["r_ref"]
    assert lora_scaling["scaling_ratio"] == expected_ratio

    alpha_by_module = lora_scaling["alpha_by_module"]
    assert alpha_by_module
    for module_name, rank in rank_dict.items():
        if int(rank) <= 0:
            continue
        assert module_name in alpha_by_module
        assert abs(alpha_by_module[module_name] / rank - expected_ratio) < 1e-9


def test_gora_bw_run_does_not_use_fixed_scaling_ratio(tmp_path: Path):
    """Regression guard for the CovRA-only scope decision: a method=gora_bw run must
    keep using the old single-global-alpha convention untouched, not lora_scaling.json."""
    config = _tiny_config(tmp_path, method="gora_bw", allocation_method="gora_bw", experiment_name="gora_bw_untouched")

    train(config)

    output_dir = tmp_path / "outputs" / "gora_bw_untouched"
    assert not (output_dir / "lora_scaling.json").exists()
