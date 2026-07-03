import json
from pathlib import Path

import torch

from dico_rank.config import load_yaml
from dico_rank.data import SFTCollator
import dico_rank.trainer as trainer_module
from dico_rank.trainer import train, _build_calibration_batches


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


def test_mid_eval_loss_logs_to_train_log_only(tmp_path: Path):
    config = load_yaml(ROOT / "configs" / "debug" / "tiny_lora.yaml")
    config["experiment_name"] = "tiny_lora_mid_eval"
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    config["training"]["max_steps"] = 2
    config["training"]["logging_steps"] = 1
    config["evaluation"]["compute_accuracy"] = False
    config["evaluation"]["mid_eval_loss_only"] = {
        "enabled": True,
        "every_n_steps": 1,
        "max_batches": 1,
    }

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_lora_mid_eval"
    train_rows = _jsonl(output_dir / "train_log.jsonl")
    mid_rows = [row for row in train_rows if row["event"] == "mid_eval_loss"]
    eval_rows = _jsonl(output_dir / "eval_log.jsonl")
    rank_history_header = (output_dir / "rank_history.csv").read_text(encoding="utf-8").splitlines()[0]

    assert [row["step"] for row in mid_rows] == [1]
    assert "eval_loss" in mid_rows[0]
    assert all(row["event"] == "final_eval" for row in eval_rows)
    assert "latest_mid_eval_loss" in rank_history_header


def test_masked_lora_state_contains_only_lora_and_rank_mask(tmp_path: Path):
    config = load_yaml(ROOT / "configs" / "debug" / "tiny_lora.yaml")
    config["experiment_name"] = "tiny_lora_state_keys"
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    config["training"]["max_steps"] = 1
    config["evaluation"]["compute_accuracy"] = False
    config["evaluation"]["mid_eval_loss_only"] = {"enabled": False}

    train(config)

    state = torch.load(
        tmp_path / "outputs" / "tiny_lora_state_keys" / "masked_lora_state.pt",
        map_location="cpu",
    )
    assert state
    assert all(
        "lora_A" in key or "lora_B" in key or "rank_mask" in key
        for key in state
    )
    assert not any("optimizer" in key or "base_layer.weight" in key for key in state)


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


def test_dico_pre_keeps_preallocation_without_generic_repair(monkeypatch, tmp_path: Path):
    config = load_yaml(ROOT / "configs" / "debug" / "tiny_lora.yaml")
    config["experiment_name"] = "tiny_dico_pre_no_repair"
    config["method"] = "dico_pre"
    config["project"]["output_dir"] = str(tmp_path / "outputs")
    config["training"]["max_steps"] = 1
    config["evaluation"]["accuracy_max_samples"] = 1
    config["evaluation"]["generation_max_new_tokens"] = 4
    config["calibration"]["enabled"] = True
    config["preallocation"]["eta"] = 0.0

    calibration_batches = [{"input_ids": torch.tensor([[1]]), "labels": torch.tensor([[1]])}]
    preallocation_by_module = {}

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
        preallocation_by_module.update({name: 1 for name in module_names})
        return dict(preallocation_by_module), {
            "atom_mode": "module_proxy",
            "module_logs": [{"module_name": name, "final_rank": 1} for name in module_names],
        }

    def fail_repair(*_args, **_kwargs):
        raise AssertionError("dico_pre must not call generic BudgetManager.repair")

    monkeypatch.setattr(trainer_module, "_build_calibration_batches", build_calibration_batches)
    monkeypatch.setattr(trainer_module, "load_or_build_preallocation", load_or_build_preallocation)
    monkeypatch.setattr(trainer_module.BudgetManager, "repair", fail_repair)

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_dico_pre_no_repair"
    initial_payload = json.loads((output_dir / "rank_allocation_initial.json").read_text(encoding="utf-8"))
    final_payload = json.loads((output_dir / "rank_allocation_final.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert initial_payload["rank_allocation"] == preallocation_by_module
    assert final_payload == preallocation_by_module
    assert initial_payload["generic_repair_applied"] is False
    assert metrics["generic_repair_applied"] is False


def _calibration_ids(records: list[dict], calibration_cfg: dict) -> list[int]:
    collator = SFTCollator(pad_token_id=0)
    batches = _build_calibration_batches(records, collator, torch.device("cpu"), calibration_cfg)
    return [int(batch["input_ids"][0, 0].item()) for batch in batches]


def test_calibration_sampling_keeps_prefix_by_default():
    records = [
        {"input_ids": [idx], "attention_mask": [1], "labels": [idx]}
        for idx in range(10)
    ]

    selected = _calibration_ids(records, {"num_samples": 4, "batch_size": 1, "seed": 1})

    assert selected == [0, 1, 2, 3]


def test_calibration_sampling_can_shuffle_with_seed():
    records = [
        {"input_ids": [idx], "attention_mask": [1], "labels": [idx]}
        for idx in range(10)
    ]

    first = _calibration_ids(records, {"num_samples": 4, "batch_size": 1, "seed": 7, "shuffle": True})
    second = _calibration_ids(records, {"num_samples": 4, "batch_size": 1, "seed": 7, "shuffle": True})
    different = _calibration_ids(records, {"num_samples": 4, "batch_size": 1, "seed": 8, "shuffle": True})

    assert first == second
    assert first != [0, 1, 2, 3]
    assert first != different
