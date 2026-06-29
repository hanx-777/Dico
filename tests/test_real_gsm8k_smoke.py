from pathlib import Path

import pytest

from src.run_experiment_config import build_run_args, load_experiment_config, run_experiment


def test_default_config_uses_real_gsm8k_full_finetune_and_no_quantization():
    config = load_experiment_config(Path("configs/mvp_gsm8k.json"))
    config = {
        **config,
        "common": {
            **config["common"],
            "model_name_or_path": "/models/qwen",
        },
    }
    smoke_runs = build_run_args(config, "smoke")
    mvp_runs = build_run_args(config, "mvp")

    assert len(smoke_runs) == 1
    assert smoke_runs[0]["dataset_name"] == "openai/gsm8k"
    assert smoke_runs[0]["dataset_config"] == "main"
    assert smoke_runs[0]["finetune_mode"] == "full"
    assert smoke_runs[0]["load_in_4bit"] is False
    assert smoke_runs[0]["load_in_8bit"] is False
    assert smoke_runs[0]["use_chat_template"] is False
    assert smoke_runs[0]["enable_thinking"] is False
    assert [run["method"] for run in mvp_runs] == ["uniform", "module_coverage", "dico"]


@pytest.mark.integration
def test_real_gsm8k_smoke_runs_from_config_when_model_path_is_set(tmp_path: Path):
    config = load_experiment_config(Path("configs/mvp_gsm8k.json"))
    model_path = config["common"].get("model_name_or_path")
    if model_path == "/path/to/Qwen3-8B":
        pytest.skip("Edit configs/mvp_gsm8k.json common.model_name_or_path to run the real smoke test.")

    config_path = tmp_path / "real_smoke_config.json"
    config["experiments"]["smoke"]["runs"][0]["output_dir"] = str(tmp_path / "real_smoke")
    import json

    config_path.write_text(json.dumps(config), encoding="utf-8")
    run_experiment(config_path, "smoke", dry_run=False)

    assert (tmp_path / "real_smoke" / "config.json").exists()
    assert (tmp_path / "real_smoke" / "calibration_pass1.pt").exists()
    assert (tmp_path / "real_smoke" / "full_finetune_verification.json").exists()
    assert (tmp_path / "real_smoke" / "eval_results.json").exists()
