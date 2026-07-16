import json
from pathlib import Path

import pytest

from dico.trainer import train


def _config(tmp_path: Path, method: str):
    return {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": f"tiny_{method}",
        "method": method,
        "rank": 1,
        "project": {"output_dir": str(tmp_path / "outputs")},
        "model": {"type": "tiny", "name_or_path": "tiny", "hidden_size": 8, "vocab_size": 64, "torch_dtype": "float32"},
        "data": {"source": "tiny", "train_path": "tiny", "eval_path": "tiny", "max_length": 16, "train_limit": 2, "eval_limit": 1},
        "training": {
            "max_steps": 1, "batch_size": 1, "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5, "weight_decay": 5e-4, "warmup_ratio": 0.03,
            "lr_decay_ratio": 0.1, "betas": [0.9, 0.999], "eps": 1e-8,
        },
        "lora": {
            "injection": "static", "alpha": 2, "dropout": 0.0,
            "scaling": "alpha_over_sqrt_r", "adapter_dtype": "float32",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "max_rank_multiplier": 2,
        },
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
        "calibration": {"enabled": True, "num_samples": 2, "batch_size": 1, "seed": 42, "shuffle": False},
        "preallocation": {"answer_only": True, "eta": 0.0, "r_max_multiplier": 2.0},
        "gora": {
            "official_commit": "4037d4d6ba67ff88de87f90b943ff4e3a3649b67",
            "r_ref": 1, "r_min": 1, "r_max": 2, "rounding": "moderate",
            "aggregation": "union_mean", "scale_by_lr": True, "init_lr": 0.05,
            "b_lr_multiplier": 16, "compute_device": "cpu", "module_chunk_size": 4,
            "strict_budget_repair": method == "gora_bm",
        },
        "dico": {"version": method, "init": {"mode": "gora_pseudoinverse"}},
        "evaluation": {"compute_accuracy": False, "max_batches": 1, "batch_size": 1},
    }


@pytest.mark.parametrize("method", ["gora_public", "gora_bm"])
def test_formal_gora_runs_through_training_entrypoint(tmp_path: Path, method: str):
    train(_config(tmp_path, method))
    output = tmp_path / "outputs" / f"tiny_{method}"
    manifest = json.loads((output / "run_manifest.json").read_text())
    budget = json.loads((output / "budget.json").read_text())
    assert manifest["method"] == method
    assert manifest["calibration"]["num_selected_samples"] == 2
    assert manifest["optimizer"]["param_groups"][1]["lr"] / manifest["optimizer"]["param_groups"][0]["lr"] == pytest.approx(16)
    if method == "gora_bm":
        assert budget["actual_budget_paramcount"] <= budget["target_budget_paramcount"]
