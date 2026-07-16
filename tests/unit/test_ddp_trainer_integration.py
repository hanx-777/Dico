"""Integration tests that exercise the REAL dico.trainer.train() under a faked
multi-process Accelerator, unlike test_ddp_compat.py which only mirrors the
sharding/warmup arithmetic in isolation and cannot catch regressions in trainer.py
itself. These tests simulate num_processes/local_rank without requiring the
`accelerate` package or real multi-GPU hardware, and cover:

1. Calibration/rank-allocation must be invariant to num_processes (regression
   test for the bug where the SVD/calibration pool was built from the
   already-DDP-sharded train_records instead of the full dataset).
2. Only the main process (is_main) writes any output artifacts.
3. per_gpu_batch_size auto-scales to preserve effective batch size when unset
   under multi-process DDP.
"""
from __future__ import annotations

import json
from pathlib import Path

import dico.trainer as trainer_module
from dico.trainer import train


class _FakeAccelerator:
    """Minimal stand-in for accelerate.Accelerator, just enough for train()."""

    def __init__(self, num_processes: int, local_process_index: int):
        self.num_processes = num_processes
        self.local_process_index = local_process_index
        self.is_main_process = local_process_index == 0
        self.device = "cpu"

    def wait_for_everyone(self):
        pass

    def prepare(self, *args):
        return args if len(args) > 1 else args[0]

    def backward(self, loss):
        loss.backward()

    def unwrap_model(self, model):
        return model


def _install_fake_accelerator(monkeypatch, num_processes: int, local_process_index: int) -> None:
    monkeypatch.setattr(trainer_module, "_ACCELERATE_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        trainer_module,
        "Accelerator",
        lambda *args, **kwargs: _FakeAccelerator(num_processes, local_process_index),
        raising=False,
    )
    # Single-process-in-test simulation: whatever the "main" rank computed is what
    # every rank would receive from a real broadcast, so identity is correct here
    # as long as only the main-process call path produces non-None values.
    monkeypatch.setattr(trainer_module, "broadcast_object_list", lambda objs: objs, raising=False)
    monkeypatch.setattr(
        trainer_module, "InitProcessGroupKwargs", lambda **kwargs: kwargs, raising=False
    )


def _tiny_dico_cd_da_config(tmp_path: Path, experiment_name: str, *, train_limit: int = 2) -> dict:
    return {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": experiment_name,
        "method": "dico_cd_da",
        "rank": 1,
        "project": {"output_dir": str(tmp_path / "outputs")},
        "model": {"type": "tiny", "name_or_path": "tiny", "hidden_size": 8, "vocab_size": 64, "torch_dtype": "float32"},
        "data": {
            "source": "tiny",
            "train_path": "tiny",
            "eval_path": "tiny",
            "max_length": 16,
            "train_limit": train_limit,
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
            "allocation_method": "dico_v03",
            "top_k_atoms": 4,
            "sketch_dim": 4,
            "sketch_seed": 42,
            "answer_only": False,
            "profile_norm_mode": "streaming_estimate",
            "eta": 0.5,
            "r_min_multiplier": 0.0,
            "r_max_multiplier": 4.0,
        },
        "dico": {
            "version": "cd_da",
            "init": {"mode": "direction_anchored", "zero_B": True},
        },
    }


def test_calibration_invariant_to_num_processes(tmp_path, monkeypatch):
    """Same config, same rank allocation, regardless of simulated GPU count."""
    _install_fake_accelerator(monkeypatch, num_processes=1, local_process_index=0)
    config_1gpu = _tiny_dico_cd_da_config(tmp_path, "calib_invariance_1gpu")
    train(config_1gpu)
    rank_dict_1gpu = json.loads(
        (tmp_path / "outputs" / "calib_invariance_1gpu" / "rank_dict.json").read_text()
    )

    _install_fake_accelerator(monkeypatch, num_processes=3, local_process_index=0)
    config_3gpu = _tiny_dico_cd_da_config(tmp_path, "calib_invariance_3gpu")
    train(config_3gpu)
    rank_dict_3gpu = json.loads(
        (tmp_path / "outputs" / "calib_invariance_3gpu" / "rank_dict.json").read_text()
    )

    assert rank_dict_1gpu == rank_dict_3gpu, (
        "Rank allocation changed with num_processes alone -- calibration pool is "
        "leaking DDP sharding into the SVD/rank-allocation pipeline."
    )


def test_non_main_rank_writes_no_artifacts(tmp_path, monkeypatch):
    """Only rank 0 (is_main) may write any output file."""
    # First, run as the main rank so broadcast_object_list has a real allocation to
    # hand back when the non-main rank below calls it with [None, None].
    captured: dict = {}

    def recording_broadcast(objs):
        alloc, meta = objs
        if alloc is not None:
            captured["alloc"], captured["meta"] = alloc, meta
            return objs
        return [captured["alloc"], captured["meta"]]

    monkeypatch.setattr(trainer_module, "_ACCELERATE_AVAILABLE", True, raising=False)
    monkeypatch.setattr(trainer_module, "broadcast_object_list", recording_broadcast, raising=False)
    monkeypatch.setattr(
        trainer_module, "InitProcessGroupKwargs", lambda **kwargs: kwargs, raising=False
    )

    monkeypatch.setattr(
        trainer_module,
        "Accelerator",
        lambda *args, **kwargs: _FakeAccelerator(num_processes=3, local_process_index=0),
        raising=False,
    )
    main_config = _tiny_dico_cd_da_config(tmp_path, "writes_main")
    train(main_config)
    main_output_dir = tmp_path / "outputs" / "writes_main"
    assert (main_output_dir / "rank_dict.json").exists()
    assert (main_output_dir / "metrics.json").exists()

    monkeypatch.setattr(
        trainer_module,
        "Accelerator",
        lambda *args, **kwargs: _FakeAccelerator(num_processes=3, local_process_index=1),
        raising=False,
    )
    non_main_config = _tiny_dico_cd_da_config(tmp_path, "writes_nonmain")
    result = train(non_main_config)
    assert result == {"experiment": "writes_nonmain", "rank": 1, "is_main": False}

    non_main_output_dir = tmp_path / "outputs" / "writes_nonmain"
    written_files = (
        [p.name for p in non_main_output_dir.iterdir()] if non_main_output_dir.exists() else []
    )
    assert written_files == [], f"Non-main rank wrote unexpected files: {written_files}"


def test_per_gpu_batch_size_auto_scales_when_unset(tmp_path, monkeypatch, caplog):
    """batch_size auto-scales under multi-process DDP when per_gpu_batch_size is unset,
    instead of silently multiplying the effective batch size by num_processes."""
    _install_fake_accelerator(monkeypatch, num_processes=4, local_process_index=0)
    config = _tiny_dico_cd_da_config(tmp_path, "batch_autoscale")
    config["training"]["batch_size"] = 4  # no per_gpu_batch_size set.
    import logging

    with caplog.at_level(logging.WARNING, logger=trainer_module.LOGGER.name):
        train(config)

    assert "auto-scaling per-GPU batch_size" in caplog.text, (
        "Expected an auto-scaling warning when per_gpu_batch_size is unset under multi-process DDP"
    )


def test_ddp_manifest_records_manual_shard_and_no_drop_last(tmp_path, monkeypatch):
    _install_fake_accelerator(monkeypatch, num_processes=3, local_process_index=0)
    config = _tiny_dico_cd_da_config(tmp_path, "ddp_data_loading_manifest", train_limit=4)
    config["training"]["batch_size"] = 3
    config["training"]["gradient_accumulation_steps"] = 7
    config["training"]["per_gpu_batch_size"] = 3

    train(config)

    manifest = json.loads((tmp_path / "outputs" / "ddp_data_loading_manifest" / "run_manifest.json").read_text())
    data_loading = manifest["data_loading"]

    assert data_loading["sampler_type"] == "manual_stride_shard"
    assert data_loading["distributed_sampler_used"] is False
    assert data_loading["drop_last"] is False
    assert data_loading["shard_strategy"] == "records[local_rank::world_size]"
    assert data_loading["total_train_records"] == 4
    assert data_loading["per_rank_train_records"] == 2
    assert data_loading["dataloader_length_batches"] == 1
    assert data_loading["global_batch_size"] == 63
    assert data_loading["last_accumulation_behavior"] == "full_accumulation_from_repeating_iterator"


def test_ddp_process_group_timeout_exceeds_nccl_default(tmp_path, monkeypatch):
    """Final GSM8K/HumanEval accuracy eval runs on rank-0 only and can take far longer
    than NCCL's default 10-minute collective timeout (e.g. ~49min for the full 1319-example
    GSM8K test set at ~2.2s/sample); other ranks idle at the post-training
    `wait_for_everyone()` barrier and get killed by the watchdog before rank-0 finishes.
    Regression test for that real incident: Accelerator must be constructed with a
    process-group timeout comfortably longer than NCCL's 10-minute default."""
    from datetime import timedelta

    monkeypatch.setattr(trainer_module, "_ACCELERATE_AVAILABLE", True, raising=False)
    monkeypatch.setattr(trainer_module, "broadcast_object_list", lambda objs: objs, raising=False)
    monkeypatch.setattr(
        trainer_module, "InitProcessGroupKwargs", lambda **kwargs: kwargs, raising=False
    )

    captured_kwargs: dict = {}

    def spy_accelerator(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeAccelerator(num_processes=1, local_process_index=0)

    monkeypatch.setattr(trainer_module, "Accelerator", spy_accelerator, raising=False)

    config = _tiny_dico_cd_da_config(tmp_path, "ddp_timeout_default")
    train(config)

    [pg_kwargs] = captured_kwargs["kwargs_handlers"]
    assert pg_kwargs["timeout"] > timedelta(minutes=10), (
        "Process-group timeout must exceed NCCL's 10-minute default or rank-0's "
        "long final eval will get other ranks killed by the watchdog"
    )

    captured_kwargs.clear()
    config = _tiny_dico_cd_da_config(tmp_path, "ddp_timeout_override")
    config["training"]["ddp_timeout_minutes"] = 5
    train(config)
    [pg_kwargs] = captured_kwargs["kwargs_handlers"]
    assert pg_kwargs["timeout"] == timedelta(minutes=5)


def test_device_map_ignored_under_multiprocess_ddp(tmp_path, monkeypatch, caplog):
    """model.device_map=auto does single-process model-parallel sharding across every
    visible GPU, which conflicts with DDP (each rank would try to shard the same model
    across all GPUs simultaneously). It must be stripped before load under DDP."""
    import logging

    captured_configs = []
    original_load = trainer_module.load_tokenizer_and_model

    def spy_load(config):
        captured_configs.append(config)
        return original_load(config)

    monkeypatch.setattr(trainer_module, "load_tokenizer_and_model", spy_load)
    _install_fake_accelerator(monkeypatch, num_processes=3, local_process_index=0)

    config = _tiny_dico_cd_da_config(tmp_path, "device_map_ddp")
    config["model"]["device_map"] = "auto"

    with caplog.at_level(logging.WARNING, logger=trainer_module.LOGGER.name):
        train(config)

    assert "device_map" not in captured_configs[0]["model"]
    assert "incompatible with" in caplog.text and "DDP" in caplog.text


def test_device_map_preserved_single_process(tmp_path, monkeypatch):
    """Single-process runs (no DDP) must keep model.device_map untouched -- it's a
    legitimate way to fit an oversized model across multiple GPUs on one process."""
    captured_configs = []
    original_load = trainer_module.load_tokenizer_and_model

    def spy_load(config):
        captured_configs.append(config)
        return original_load(config)

    monkeypatch.setattr(trainer_module, "load_tokenizer_and_model", spy_load)
    _install_fake_accelerator(monkeypatch, num_processes=1, local_process_index=0)

    config = _tiny_dico_cd_da_config(tmp_path, "device_map_single")
    config["model"]["device_map"] = "auto"
    train(config)

    assert captured_configs[0]["model"]["device_map"] == "auto"
