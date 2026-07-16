import json
from pathlib import Path

from dico.trainer import train


def _tiny_adalora_config(tmp_path: Path) -> dict:
    return {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": "tiny_adalora",
        "method": "adalora",
        "rank": 1,
        "project": {"output_dir": str(tmp_path / "outputs")},
        "model": {"type": "tiny", "name_or_path": "tiny", "hidden_size": 8, "vocab_size": 64, "torch_dtype": "float32"},
        "data": {
            "source": "tiny",
            "train_path": "tiny",
            "eval_path": "tiny",
            "max_length": 16,
            "train_limit": 2,
            "eval_limit": 2,
        },
        "training": {
            "max_steps": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.03,
            "lr_decay_ratio": 0.1,
            "weight_decay": 5e-4,
            "max_grad_norm": 1.0,
        },
        "lora": {
            "injection": "masked",
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "max_rank_multiplier": 2,
            "adapter_dtype": "float32",
        },
        "adalora": {
            "init_rank": 2,
            "target_rank": 1,
            "tinit": 0,
            "tfinal": 1,
            "update_interval": 1,
        },
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
        "calibration": {"enabled": False, "num_samples": 0, "batch_size": 1, "seed": 42},
        "preallocation": {"r_max_multiplier": 2.0, "eta": 0.98},
        "dico": {"version": "adalora", "init": {"mode": "kaiming_zero_B"}},
        "evaluation": {"compute_accuracy": False, "max_batches": 1},
    }


def test_adalora_trainer_records_peak_and_final_parameter_budgets(tmp_path: Path):
    train(_tiny_adalora_config(tmp_path))

    output_dir = tmp_path / "outputs" / "tiny_adalora"
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    budget = json.loads((output_dir / "budget.json").read_text(encoding="utf-8"))
    schedule = json.loads((output_dir / "adalora_schedule.json").read_text(encoding="utf-8"))

    assert manifest["method"] == "adalora"
    assert manifest["rank_allocation_initial"]
    # Official AdaLoRA applies one global budget.  Individual modules may end
    # above or below target_rank as long as the global final rank is exact.
    assert sum(manifest["rank_allocation_final"].values()) == len(manifest["rank_allocation_final"])
    assert manifest["parameter_counts"]["active_peak"] > manifest["parameter_counts"]["active_final"]
    assert budget["adalora_peak_active_params"] == manifest["parameter_counts"]["active_peak"]
    assert budget["adalora_final_active_params"] == manifest["parameter_counts"]["active_final"]
    assert schedule["init_rank"] == 2
    assert schedule["target_rank"] == 1
    assert schedule["events"]
