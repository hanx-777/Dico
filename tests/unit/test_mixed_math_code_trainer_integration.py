import json
from pathlib import Path

import yaml

from dico.trainer import train


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _base_tiny_config(tmp_path: Path, experiment_name: str, data_cfg: dict) -> dict:
    return {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": experiment_name,
        "method": "dico_cd_da",
        "rank": 1,
        "project": {"output_dir": str(tmp_path / "outputs")},
        "model": {"type": "tiny", "name_or_path": "tiny", "hidden_size": 8, "vocab_size": 64, "torch_dtype": "float32"},
        "data": data_cfg,
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
            "allocation_method": "covra_v05",
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


def test_train_sources_real_group_labels_flow_through_to_resolved_config(tmp_path: Path):
    # Phase K end-to-end: a config with real (math/code) data.train_sources must get
    # config["data"]["group_labels"] populated by train() before the allocator runs --
    # observable afterwards in resolved_config.yaml, since it's the same config object.
    math_path = tmp_path / "data" / "math" / "train.jsonl"
    code_path = tmp_path / "data" / "code" / "train.jsonl"
    _write_jsonl(math_path, [{"question": f"What is {i}+{i}?", "answer": f"{2*i}"} for i in range(8)])
    _write_jsonl(code_path, [{"question": f"Write function {i}", "answer": f"def f{i}(): pass"} for i in range(8)])

    data_cfg = {
        "train_sources": [
            {"path": "data/math/train.jsonl", "group": "math"},
            {"path": "data/code/train.jsonl", "group": "code"},
        ],
        "eval_path": "data/math/train.jsonl",
        "max_length": 16,
        "train_limit": 16,
        "eval_limit": 2,
    }
    config = _base_tiny_config(tmp_path, "mixed_math_code_tiny", data_cfg)
    config["calibration"]["group_sampling"] = "balanced"
    config["dico"]["split"] = {"mode": "group"}

    train(config)

    output_dir = tmp_path / "outputs" / "mixed_math_code_tiny"
    resolved = yaml.safe_load((output_dir / "resolved_config.yaml").read_text())

    group_labels = resolved["data"]["group_labels"]
    assert set(group_labels) == {"math", "code"}
    assert group_labels.count("math") == group_labels.count("code")


def test_single_source_tiny_config_never_sets_group_labels(tmp_path: Path):
    # Regression guard: an ordinary single-source config (every calibration sample
    # defaults to group="math") must NOT spuriously populate data.group_labels --
    # that would make the allocator treat a single degenerate group as "configured"
    # instead of falling through to real pseudo-group construction.
    data_cfg = {
        "source": "tiny",
        "train_path": "tiny",
        "eval_path": "tiny",
        "max_length": 16,
        "train_limit": 2,
        "eval_limit": 2,
    }
    config = _base_tiny_config(tmp_path, "single_source_tiny", data_cfg)

    train(config)

    output_dir = tmp_path / "outputs" / "single_source_tiny"
    resolved = yaml.safe_load((output_dir / "resolved_config.yaml").read_text())

    assert "group_labels" not in resolved.get("data", {})
