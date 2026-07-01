import json
from pathlib import Path

import torch

import dico_rank.trainer as trainer_module
from dico_rank.preallocation import DiCoPreAllocator, build_preallocation_cache_context, load_preallocation
from dico_rank.rank_budget import get_uniform_budget
from dico_rank.trainer import _preallocation_cache_is_compatible, build_preallocation_cache


def test_preallocation_returns_all_modules_and_round_trips(tmp_path: Path):
    module_scores = {"a": 10.0, "b": 1.0}
    dims = {"a": {"in_dim": 4, "out_dim": 4}, "b": {"in_dim": 4, "out_dim": 4}}
    allocator = DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config={
            "rank": 2,
            "seed": 42,
            "preallocation": {
                "fallback_atom_mode": "module_proxy",
                "r_min": 0,
                "r_max_multiplier": 2,
                "aggregation_mode": "weighted_topk",
                "atom_weight_normalization": "none",
                "use_cost_aware_allocation": True,
            },
        },
        module_names=["a", "b"],
        module_dims=dims,
        module_scores=module_scores,
    )

    budget = get_uniform_budget(2, ["a", "b"], dims)
    result = allocator.allocate(rank_budget=budget.target_budget)
    output = tmp_path / "preallocation.json"
    allocator.save(output, result)
    loaded = load_preallocation(output)

    assert set(result.rank_allocation) == {"a", "b"}
    assert loaded["rank_allocation"] == result.rank_allocation
    assert loaded["atom_mode"] == "module_proxy"
    assert loaded["aggregation_mode"] == "weighted_topk"
    assert loaded["atom_weight_normalization"] == "none"
    assert loaded["use_cost_aware_allocation"] is True
    assert loaded["module_logs"]


def test_preallocation_json_contains_atom_logs(tmp_path: Path):
    dims = {"a": {"in_dim": 4, "out_dim": 4}}
    allocator = DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config={
            "rank": 1,
            "preallocation": {
                "fallback_atom_mode": "module_proxy",
                "aggregation_mode": "weighted_topk",
                "atom_weight_normalization": "none",
            },
        },
        module_names=["a"],
        module_dims=dims,
        module_scores={"a": 1.0},
    )
    result = allocator.allocate(rank_budget=8)
    output = tmp_path / "preallocation.json"
    allocator.save(output, result)

    payload = json.loads(output.read_text())
    assert payload["atom_mode"] == "module_proxy"
    assert payload["atom_mode_limitation"]
    assert payload["atom_logs"][0]["atom_mode"] == "module_proxy"
    assert "utility" in payload["atom_logs"][0]


def test_preallocation_cache_rejected_when_module_set_differs():
    payload = {
        "rank_allocation": {"old": 2},
        "module_logs": [{"module_name": "old"}],
        "aggregation_mode": "weighted_topk",
        "atom_weight_normalization": "none",
        "use_cost_aware_allocation": True,
    }
    config = {
        "preallocation": {
            "aggregation_mode": "weighted_topk",
            "atom_weight_normalization": "none",
            "use_cost_aware_allocation": True,
        }
    }

    assert not _preallocation_cache_is_compatible(
        payload,
        config,
        module_names=["new"],
        module_dims={"new": {"in_dim": 4, "out_dim": 4}},
    )


def test_preallocation_cache_rejected_when_dims_differ():
    payload = {
        "rank_allocation": {"a": 2},
        "module_dims": {"a": {"in_dim": 4, "out_dim": 4}},
        "module_logs": [{"module_name": "a"}],
        "aggregation_mode": "weighted_topk",
        "atom_weight_normalization": "none",
        "use_cost_aware_allocation": True,
    }
    config = {
        "preallocation": {
            "aggregation_mode": "weighted_topk",
            "atom_weight_normalization": "none",
            "use_cost_aware_allocation": True,
        }
    }

    assert not _preallocation_cache_is_compatible(
        payload,
        config,
        module_names=["a"],
        module_dims={"a": {"in_dim": 8, "out_dim": 4}},
    )


def test_preallocation_cache_accepts_matching_context():
    module_names = ["a"]
    module_dims = {"a": {"in_dim": 4, "out_dim": 4}}
    config = {
        "seed": 42,
        "rank": 2,
        "model": {"name_or_path": "/models/qwen"},
        "data": {"dataset_name": "openai/gsm8k", "dataset_config": "main"},
        "calibration": {"num_samples": 128, "seed": 42},
        "lora": {"target_modules": ["q_proj"]},
        "preallocation": {
            "aggregation_mode": "weighted_topk",
            "atom_weight_normalization": "none",
            "use_cost_aware_allocation": True,
        },
    }
    payload = {
        "rank_allocation": {"a": 2},
        "module_dims": module_dims,
        "module_logs": [{"module_name": "a"}],
        "aggregation_mode": "weighted_topk",
        "atom_weight_normalization": "none",
        "use_cost_aware_allocation": True,
        "cache_context": build_preallocation_cache_context(config, module_names, module_dims),
    }

    assert _preallocation_cache_is_compatible(payload, config, module_names, module_dims)


def test_build_preallocation_cache_does_not_write_training_artifacts(tmp_path: Path):
    config = {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": "tiny_prealloc_only",
        "method": "dico_pre",
        "rank": 1,
        "project": {"output_dir": str(tmp_path / "outputs")},
        "model": {"type": "tiny", "name_or_path": "tiny", "hidden_size": 8, "vocab_size": 128},
        "data": {
            "source": "tiny",
            "train_path": "tiny",
            "eval_path": "tiny",
            "max_length": 32,
            "train_limit": 2,
            "eval_limit": 2,
        },
        "training": {"max_steps": 2, "batch_size": 1, "gradient_accumulation_steps": 1},
        "lora": {
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_rank_multiplier": 2,
        },
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
        "calibration": {
            "enabled": False,
            "num_samples": 1,
            "seed": 42,
            "save_dir": str(tmp_path / "preallocations"),
        },
        "preallocation": {
            "fallback_atom_mode": "module_proxy",
            "aggregation_mode": "weighted_topk",
            "weighted_topk_k": "auto",
            "atom_weight_normalization": "none",
            "use_cost_aware_allocation": True,
            "r_min": 0,
            "r_max_multiplier": 2,
        },
    }

    result = build_preallocation_cache(config)

    assert result["rank_allocation"]
    assert (tmp_path / "preallocations" / "dico_pre_rank1_seed42.json").exists()
    assert not (tmp_path / "outputs" / "tiny_prealloc_only" / "metrics.json").exists()
    assert not (tmp_path / "outputs" / "tiny_prealloc_only" / "masked_lora_state.pt").exists()
    assert not (tmp_path / "outputs" / "tiny_prealloc_only" / "train_log.jsonl").exists()


def test_build_preallocation_cache_releases_calibration_batches(monkeypatch, tmp_path: Path):
    config = {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": "tiny_prealloc_only",
        "method": "dico_pre",
        "rank": 1,
        "project": {"output_dir": str(tmp_path / "outputs")},
        "model": {"type": "tiny", "name_or_path": "tiny", "hidden_size": 8, "vocab_size": 128},
        "data": {
            "source": "tiny",
            "train_path": "tiny",
            "eval_path": "tiny",
            "max_length": 32,
            "train_limit": 2,
            "eval_limit": 2,
        },
        "training": {"max_steps": 2, "batch_size": 1, "gradient_accumulation_steps": 1},
        "lora": {
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_rank_multiplier": 2,
        },
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
        "calibration": {
            "enabled": True,
            "num_samples": 1,
            "seed": 42,
            "save_dir": str(tmp_path / "preallocations"),
        },
        "preallocation": {
            "fallback_atom_mode": "module_proxy",
            "aggregation_mode": "weighted_topk",
            "weighted_topk_k": "auto",
            "atom_weight_normalization": "none",
            "use_cost_aware_allocation": True,
            "r_min": 0,
            "r_max_multiplier": 2,
        },
    }
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

    result = trainer_module.build_preallocation_cache(config)

    assert result["rank_allocation"]
    assert calibration_batches == []
