import json
from pathlib import Path

import torch

from dico_rank.config import load_yaml
import dico_rank.trainer as trainer_module
from dico_rank.trainer import train


ROOT = Path(__file__).resolve().parents[1]


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_train_skips_in_loop_eval_and_logs_timestamped_final_eval(tmp_path: Path):
    config = load_yaml(ROOT / "configs" / "debug" / "tiny_lora.yaml")
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    config["training"]["max_steps"] = 2
    config["training"]["logging_steps"] = 1
    config["training"]["eval_steps"] = 1
    config["evaluation"]["accuracy_max_samples"] = 1
    config["evaluation"]["generation_max_new_tokens"] = 4

    metrics = train(config)

    output_dir = tmp_path / "outputs" / "tiny_lora"
    eval_rows = _jsonl(output_dir / "eval_log.jsonl")
    assert len(eval_rows) == 1
    assert eval_rows[0]["final"] is True
    assert eval_rows[0]["event"] == "final_eval"
    assert eval_rows[0]["step"] == 2
    assert "timestamp" in eval_rows[0]
    assert "elapsed_sec" in eval_rows[0]
    assert "eval_loss" in eval_rows[0]
    assert "eval_accuracy" in eval_rows[0]

    train_rows = _jsonl(output_dir / "train_log.jsonl")
    assert len(train_rows) == 2
    for row in train_rows:
        assert row["event"] == "train_step"
        assert "timestamp" in row
        assert "elapsed_sec" in row
        assert "steps_per_sec" in row
        assert "max_steps" in row
        assert "micro_step" in row
        assert "grad_accumulation_steps" in row

    metrics_on_disk = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["final_eval_loss"] == metrics_on_disk["final_eval_loss"]
    assert metrics_on_disk["final_eval_accuracy"] is not None


def test_lora_training_does_not_build_calibration_batches(monkeypatch, tmp_path: Path):
    config = load_yaml(ROOT / "configs" / "debug" / "tiny_lora.yaml")
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    config["training"]["max_steps"] = 1
    config["evaluation"]["accuracy_max_samples"] = 1
    config["evaluation"]["generation_max_new_tokens"] = 4

    def fail_build_calibration_batches(*_args, **_kwargs):
        raise AssertionError("LoRA training should not build calibration batches")

    monkeypatch.setattr(trainer_module, "_build_calibration_batches", fail_build_calibration_batches)

    train(config)


def test_pre_training_releases_calibration_batches(monkeypatch, tmp_path: Path):
    config = load_yaml(ROOT / "configs" / "debug" / "tiny_lora.yaml")
    config["experiment_name"] = "tiny_dico_pre"
    config["method"] = "dico_pre"
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    config["training"]["max_steps"] = 1
    config["evaluation"]["accuracy_max_samples"] = 1
    config["evaluation"]["generation_max_new_tokens"] = 4
    config["calibration"]["enabled"] = True

    calibration_batches = [{"input_ids": torch.tensor([[1]]), "labels": torch.tensor([[1]])}]

    def build_calibration_batches(*_args, **_kwargs):
        return calibration_batches

    def load_or_build_preallocation(
        _config,
        _model,
        _tokenizer,
        module_names,
        _module_dims,
        batches,
        _target_budget,
        _project_root,
    ):
        assert batches is calibration_batches
        assert batches
        return {name: 1 for name in module_names}, {"atom_mode": "module_proxy", "module_logs": []}

    monkeypatch.setattr(trainer_module, "_build_calibration_batches", build_calibration_batches)
    monkeypatch.setattr(trainer_module, "load_or_build_preallocation", load_or_build_preallocation)

    train(config)

    assert calibration_batches == []
