import json
from copy import deepcopy
from pathlib import Path

from dico_rank.trainer import build_preallocation_cache


def _tiny_svd_config(tmp_path: Path) -> dict:
    return {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": "tiny_svd_prealloc",
        "method": "dico_pre",
        "rank": 1,
        "project": {"output_dir": str(tmp_path / "outputs")},
        "model": {"type": "tiny", "name_or_path": "tiny", "hidden_size": 8, "vocab_size": 128, "torch_dtype": "float32"},
        "data": {
            "source": "tiny",
            "train_path": "tiny",
            "eval_path": "tiny",
            "max_length": 32,
            "train_limit": 2,
            "eval_limit": 2,
        },
        "training": {"max_steps": 1, "batch_size": 1, "gradient_accumulation_steps": 1},
        "lora": {
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_rank_multiplier": 2,
        },
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
        "calibration": {
            "enabled": True,
            "num_samples": 2,
            "batch_size": 1,
            "seed": 42,
            "save_dir": str(tmp_path / "preallocations"),
        },
        "preallocation": {
            "atom_mode": "svd",
            "fallback_atom_mode": "module_proxy",
            "allocation_method": "directional_budgeted",
            "aggregation_mode": "weighted_log",
            "top_k_atoms": 2,
            "sketch_dim": 4,
            "sketch_oversample": 1,
            "compute_device": "auto",
            "module_chunk_size": 32,
            "progress_logging_steps": 1,
            "sketch_seed": 42,
            "sketch_dtype": "float32",
            "answer_only": True,
            "profile_norm_mode": "exact_small",
            "beta": 1.0,
            "gamma": 1.0,
            "delta": 1.0,
            "epsilon_cov": 0.05,
            "use_soft_tail": True,
            "eta": 0.0,
            "allow_rank_beyond_selected_evidence": False,
            "evidence_selection": {
                "max_selected_atoms": "auto",
                "sparse_stop_by_coverage": False,
                "coverage_stop_threshold": 0.05,
            },
            "r_min": 0,
            "r_max_multiplier": 2,
        },
    }


def test_tiny_model_svd_atom_mode_builds_true_atom_logs(tmp_path: Path, caplog):
    config = _tiny_svd_config(tmp_path)

    caplog.set_level("INFO")
    result = build_preallocation_cache(config)

    assert result["rank_allocation"]
    prealloc_path = tmp_path / "preallocations" / "dico_pre_rank1_seed42.json"
    payload = json.loads(prealloc_path.read_text())
    assert payload["atom_mode"] == "svd"
    assert payload["aggregation_mode"] == "weighted_log"
    assert payload["profile_norm_mode"] == "exact_small"
    assert payload["atom_logs"]
    assert payload["atom_logs"][0]["atom_mode"] == "svd"
    assert payload["atom_logs"][0]["singular_value"] is not None
    assert "profile" not in payload["atom_logs"][0]
    profile_path = Path(payload["atom_logs"][0]["profile_path"])
    assert profile_path.exists()
    assert payload["module_chunk_size"] == 32
    assert payload["num_module_chunks"] == 1
    assert "compute_device" in payload
    assert "preallocation_start" in caplog.text
    assert "svd_preallocation_progress pass=sketch_pass" in caplog.text
    assert "svd_preallocation_progress pass=basis_pass" in caplog.text
    assert "svd_preallocation_progress pass=profile_pass" in caplog.text
    for row in payload["module_logs"]:
        assert row["final_rank"] <= row["selected_atom_count"]


def test_tiny_svd_atom_mode_supports_module_chunking(tmp_path: Path):
    config = _tiny_svd_config(tmp_path)
    chunked = deepcopy(config)
    chunked["preallocation"]["module_chunk_size"] = 1
    chunked["calibration"]["save_dir"] = str(tmp_path / "chunked_preallocations")

    full_result = build_preallocation_cache(config)
    chunked_result = build_preallocation_cache(chunked)

    full_payload = json.loads((tmp_path / "preallocations" / "dico_pre_rank1_seed42.json").read_text())
    chunked_payload = json.loads((tmp_path / "chunked_preallocations" / "dico_pre_rank1_seed42.json").read_text())
    assert full_result["rank_allocation"]
    assert chunked_result["rank_allocation"]
    assert len(full_payload["atom_logs"]) == len(chunked_payload["atom_logs"])
    assert [row["module_name"] for row in full_payload["atom_logs"]] == [
        row["module_name"] for row in chunked_payload["atom_logs"]
    ]
    assert chunked_payload["module_chunk_size"] == 1
    assert chunked_payload["num_module_chunks"] == len(chunked_result["module_names"])
    assert "compute_device" in chunked_payload
    assert Path(chunked_payload["atom_logs"][0]["profile_path"]).exists()
