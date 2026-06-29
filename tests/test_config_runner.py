import json
from pathlib import Path

from src.run_experiment_config import build_run_args, load_experiment_config


def test_build_run_args_merges_common_experiment_and_run_overrides(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "common": {
                    "model_name_or_path": "/models/qwen",
                    "dataset_name": "openai/gsm8k",
                    "dataset_config": "main",
                    "finetune_mode": "full",
                    "load_in_4bit": False,
                    "gradient_checkpointing": True,
                    "seed": 42,
                },
                "experiments": {
                    "smoke": {
                        "defaults": {
                            "calibration_size": 4,
                            "train_limit": 8,
                            "eval_limit": 8,
                            "max_length": 256,
                        },
                        "runs": [
                            {
                                "method": "dico",
                                "output_dir": "outputs/smoke_dico",
                                "top_k_atoms": 2,
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_experiment_config(config_path)
    runs = build_run_args(cfg, "smoke")

    assert len(runs) == 1
    run = runs[0]
    assert run["model_name_or_path"] == "/models/qwen"
    assert run["dataset_name"] == "openai/gsm8k"
    assert run["finetune_mode"] == "full"
    assert run["load_in_4bit"] is False
    assert run["gradient_checkpointing"] is True
    assert run["calibration_size"] == 4
    assert run["method"] == "dico"
    assert run["top_k_atoms"] == 2
    assert run["output_dir"] == "outputs/smoke_dico"
