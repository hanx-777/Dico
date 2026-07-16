import hashlib
import json
from pathlib import Path

import torch

import dico.trainer as trainer_module
from dico.lora_checkpoint import load_lora_state
from dico.lora_static import inject_static_lora
from dico.model_loader import collect_module_dims, find_target_linear_modules, load_tokenizer_and_model
from dico.preallocation import load_direction_bank
from dico.trainer import train


def _tiny_dico_cd_da_config(tmp_path: Path) -> dict:
    return {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": "tiny_dico_cd_da_da_init",
        "method": "dico_cd_da",
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
            "max_steps": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 5e-5,
            "warmup_ratio": 0.03,
            "lr_decay_ratio": 0.1,
            "weight_decay": 5e-4,
            "max_grad_norm": 1.0,
        },
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
            "num_samples": 2,
            "batch_size": 1,
            "seed": 42,
            "save_dir": str(tmp_path / "preallocations"),
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


def test_direction_anchored_init_wires_through_trainer(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_dico_cd_da_da_init"
    rank_dict = json.loads((output_dir / "rank_dict.json").read_text())
    assert rank_dict
    init_summary = json.loads((output_dir / "init_summary.json").read_text())

    assert init_summary["mode"] == "direction_anchored"
    assert init_summary["delta_w_zero"] is True
    assert init_summary["delta_w_zero_by_module"]
    assert all(init_summary["delta_w_zero_by_module"].values())

    direction_bank_path = init_summary["direction_bank_path"]
    assert direction_bank_path
    bank = load_direction_bank(direction_bank_path)
    assert bank
    for module_name, entries in bank.items():
        for entry in entries:
            assert isinstance(entry["v"], torch.Tensor)

    # v0.5 4.6.7节 + Appendix B: diagnostics and previously-missing artifacts
    # must be written alongside the existing rank_dict/init_summary outputs.
    diagnostics = json.loads((output_dir / "diagnostics.json").read_text())
    expected_keys = {
        "rank_gini",
        "param_share_gini",
        "cap_hit_ratio",
        "zero_rank_ratio",
        "type_budget_share",
        "top10_module_share",
        "mean_abs_adjacent_rank_diff",
        "anchored_rank_ratio",
        "balanced_fill_ratio",
        "budget_realized_ratio",
    }
    assert expected_keys <= set(diagnostics.keys())

    assert (output_dir / "physical_utility.json").exists()
    assert (output_dir / "normalization_stats.json").exists()
    assert (output_dir / "quota_stats.json").exists()
    assert (output_dir / "resolved_config.yaml").exists()
    seeds = json.loads((output_dir / "seeds.json").read_text())
    assert seeds["seed"] == 42


def test_final_covra_direction_anchored_init_wires_through_trainer(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_covra_full_da_init"
    config["preallocation"].update(
        {
            "allocation_method": "covra_full",
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
        }
    )

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_covra_full_da_init"
    rank_dict = json.loads((output_dir / "rank_dict.json").read_text())
    assert rank_dict
    init_summary = json.loads((output_dir / "init_summary.json").read_text())

    assert init_summary["mode"] == "direction_anchored"
    assert init_summary["delta_w_zero"] is True
    assert init_summary["direction_bank_path"]

    bank = load_direction_bank(init_summary["direction_bank_path"])
    allocated_modules = {module_name for module_name, rank in rank_dict.items() if int(rank) > 0}
    assert set(bank) <= allocated_modules
    for module_name, entries in bank.items():
        assert len(entries) <= int(rank_dict[module_name])
        assert all(entry["source"] == "certified" for entry in entries)
        assert all(isinstance(entry["v"], torch.Tensor) for entry in entries)

    preallocation_path = Path(init_summary["direction_bank_path"]).with_name("dico_v03_rank1_seed42.json")
    preallocation = json.loads(preallocation_path.read_text())
    assert preallocation["allocation_method"] == "covra_full"
    assert preallocation["utility_builder"] == "conditional"
    assert preallocation["direction_bank_path"] == init_summary["direction_bank_path"]


def test_reference_covra_writes_taxonomy_coverage_and_procurement_artifacts(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_covra_reference"
    config["preallocation"].update(
        {
            "allocation_method": "covra_v05",
            "beta": 1.0,
            "allocation_device": "cpu",
        }
    )
    config["dico"].update(
        {
            "taxonomy": {"enabled": True, "alpha": 0.05, "permutation_count": 8},
            "pseudo_group": {"enabled": False},
            "split": {"enabled": True, "mode": "sign", "physical_merge": True},
            "coverage": {
                "eps": 1e-6,
                "relative_stop_delta": 0.0,
                "window_h": 2,
                "kappa_calibration": {"enabled": False},
            },
            "procurement": {
                "mode": "density_greedy",
                "beta": 0.5,
                "reserve_queue": True,
                "relaxation_fallback": True,
            },
        }
    )

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_covra_reference"
    for filename in (
        "taxonomy_stats.json",
        "coverage_trace.json",
        "procurement_trace.json",
        "physical_utility.json",
        "normalization_stats.json",
        "quota_stats.json",
        "kappa_calibration.json",
        "rank_dict.json",
        "rank_allocation_initial.json",
        "rank_allocation_final.json",
        "init_summary.json",
        "config_resolved.yaml",
        "resolved_config.yaml",
    ):
        assert (output_dir / filename).exists(), filename

    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["preallocation"]["allocation_method"] == "covra_v05"
    assert manifest["budget"]["generic_repair_applied"] is False
    assert {"taxonomy_stats", "coverage_trace", "procurement_trace"} <= set(manifest["method_artifacts"])

    init_summary = json.loads((output_dir / "init_summary.json").read_text())
    assert init_summary["delta_w_zero"] is True
    preallocation_path = Path(init_summary["direction_bank_path"]).with_name("dico_v03_rank1_seed42.json")
    preallocation = json.loads(preallocation_path.read_text())
    assert preallocation["procurement_beta"] == 0.5


def test_train_writes_run_manifest_and_markdown_summary(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_manifest"
    config["preallocation"].update(
        {
            "allocation_method": "covra_full",
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
            "solver": "dp",
        }
    )

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_manifest"
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    summary = (output_dir / "run_summary.md").read_text()

    assert manifest["experiment_name"] == "tiny_manifest"
    assert manifest["method"] == "dico_cd_da"
    assert manifest["model"]["name_or_path"] == "tiny"
    assert manifest["model"]["model_revision"] == "UNRESOLVED"
    assert manifest["model"]["tokenizer_revision"] == "UNRESOLVED"
    assert manifest["model"]["model_revision_status"] == "UNRESOLVED"
    assert manifest["model"]["tokenizer_revision_status"] == "UNRESOLVED"
    resolved_config_path = output_dir / "config_resolved.yaml"
    resolved_config_hash = hashlib.sha256(resolved_config_path.read_bytes()).hexdigest()
    assert manifest["config"]["resolved_config_path"] == str(resolved_config_path)
    assert manifest["config"]["resolved_config_sha256"] == resolved_config_hash
    assert manifest["source_control"]["repo_root"].endswith("dico_rank_experiments")
    assert "git_commit" in manifest["source_control"]
    assert "git_branch" in manifest["source_control"]
    assert isinstance(manifest["source_control"]["git_dirty"], bool)
    assert isinstance(manifest["source_control"]["uncommitted_change_count"], int)
    assert "git_status_porcelain_sha256" in manifest["source_control"]
    assert manifest["command"]["argv"] == manifest["command_line"]
    assert manifest["command"]["cwd"]
    assert manifest["command"]["python_executable"]
    assert manifest["seeds"]["base_seed"] == 42
    assert manifest["seeds"]["model_and_lora_init_seed"] == 42
    assert manifest["seeds"]["training_rng_seed"] == 42
    assert manifest["seeds"]["calibration_seed"] == 42
    assert manifest["seeds"]["preallocation_sketch_seed"] == 42
    assert manifest["world_size"] == 1
    assert manifest["global_batch_size"] == 1
    assert manifest["optimizer_steps"] == 1
    assert manifest["warmup_steps"] >= 0
    assert manifest["scheduler"]["name"] == "cosine_with_warmup_and_floor"
    assert manifest["scheduler"]["warmup_steps"] == manifest["warmup_steps"]
    assert manifest["scheduler"]["warmup_ratio"] == 0.03
    assert manifest["scheduler"]["lr_decay_ratio"] == 0.1
    assert manifest["scheduler"]["num_training_steps"] == 1
    assert manifest["scheduler"]["optimizer_steps_source"] == "training_loop_global_step"
    assert isinstance(manifest["cuda_device_names"], list)
    assert "cuda_peak_memory_allocated_bytes" in manifest["hardware"]
    assert manifest["dependency_versions"]["torch"] == manifest["torch_version"]
    assert "python" in manifest["dependency_versions"]
    assert "transformers" in manifest["dependency_versions"]
    assert "accelerate" in manifest["dependency_versions"]
    assert "numpy" in manifest["dependency_versions"]
    assert manifest["precision"]["model_torch_dtype"] == "float32"
    assert manifest["training"]["gradient_checkpointing"] is False
    assert manifest["optimizer"]["name"] == "AdamW"
    assert manifest["training"]["max_grad_norm"] == 1.0
    assert manifest["optimizer"]["betas"] == [0.9, 0.999]
    assert manifest["optimizer"]["eps"] == 1e-8
    assert manifest["optimizer"]["param_groups"]
    assert manifest["optimizer"]["param_groups"][0]["lr"] >= 0.0
    assert manifest["optimizer"]["param_groups"][0]["initial_lr"] == 5e-5
    assert manifest["optimizer"]["param_groups"][0]["weight_decay"] == 5e-4
    assert manifest["optimizer"]["param_groups"][0]["num_params"] > 0
    opt_state = manifest["optimizer_state_estimate"]
    assert opt_state["optimizer"] == "AdamW"
    assert opt_state["trainable_params"] == manifest["parameter_counts"]["requires_grad"]
    assert opt_state["state_tensors_per_param"] == 2
    assert opt_state["state_dtype"] == "float32"
    assert opt_state["state_bytes_per_param"] == 8
    assert opt_state["estimated_state_bytes"] == opt_state["trainable_params"] * 8
    assert opt_state["includes_master_params"] is False
    assert manifest["lora"]["dropout"] == 0.0
    assert manifest["lora"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert manifest["data"]["train_count"] == 2
    assert manifest["data"]["eval_count"] == 2
    assert manifest["data"]["train_hash"]
    assert manifest["data"]["eval_hash"]
    assert manifest["data_loading"]["sampler_type"] == "single_process_full_dataset"
    assert "cosine_with_warmup_and_floor" in summary
    assert "optimizer_state_estimate" in summary
    assert manifest["data_loading"]["distributed_sampler_used"] is False
    assert manifest["data_loading"]["drop_last"] is False
    assert manifest["data_loading"]["last_accumulation_behavior"] == "full_accumulation_from_repeating_iterator"
    assert manifest["data_loading"]["per_rank_train_records"] == 2
    assert manifest["data_loading"]["dataloader_length_batches"] == 2
    assert manifest["calibration"]["num_selected_samples"] == 2
    assert len(manifest["calibration"]["sample_ids"]) == 2
    assert len(manifest["calibration"]["sample_hashes"]) == 2
    assert manifest["timing"]["calibration_sec"] >= 0.0
    assert manifest["timing"]["allocation_sec"] >= 0.0
    assert manifest["timing"]["initialization_sec"] >= 0.0
    assert manifest["timing"]["training_sec"] > 0.0
    assert manifest["timing"]["train_tokens"] > 0
    assert manifest["timing"]["train_tokens_per_sec"] > 0.0
    assert manifest["budget"]["target_budget"] == manifest["metrics"]["target_budget"]
    assert manifest["parameter_counts"]["requires_grad"] > 0
    module_budget = manifest["module_budget"]
    assert set(module_budget["modules"]) == set(manifest["target_modules_resolved"])
    first_module = next(iter(module_budget["modules"].values()))
    assert {"in_dim", "out_dim", "rank_cost", "initial_rank", "final_rank", "initial_params", "final_params"} <= set(first_module)
    assert all(row["rank_cost"] == row["in_dim"] + row["out_dim"] for row in module_budget["modules"].values())
    assert module_budget["total_initial_params"] == sum(row["initial_params"] for row in module_budget["modules"].values())
    assert module_budget["total_final_params"] == sum(row["final_params"] for row in module_budget["modules"].values())
    assert module_budget["total_final_params"] == manifest["parameter_counts"]["active_final"]
    expected_method_artifacts = {
        "budget": "budget.json",
        "rank_dict": "rank_dict.json",
        "rank_allocation_initial": "rank_allocation_initial.json",
        "rank_allocation_final": "rank_allocation_final.json",
        "init_summary": "init_summary.json",
        "diagnostics": "diagnostics.json",
        "physical_utility": "physical_utility.json",
        "normalization_stats": "normalization_stats.json",
    }
    assert expected_method_artifacts.keys() <= manifest["method_artifacts"].keys()
    assert "taxonomy_stats" not in manifest["method_artifacts"]
    assert "coverage_trace" not in manifest["method_artifacts"]
    assert "procurement_trace" not in manifest["method_artifacts"]
    assert not (output_dir / "taxonomy_stats.json").exists()
    assert not (output_dir / "coverage_trace.json").exists()
    assert not (output_dir / "procurement_trace.json").exists()
    for artifact_name, filename in expected_method_artifacts.items():
        artifact_path = output_dir / filename
        artifact = manifest["method_artifacts"][artifact_name]
        assert artifact["path"] == str(artifact_path)
        assert artifact["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert artifact["size_bytes"] == artifact_path.stat().st_size
        assert artifact["format"] == "json"
    expected_run_artifacts = {
        "train_log": ("train_log.jsonl", "jsonl"),
        "eval_log": ("eval_log.jsonl", "jsonl"),
        "metrics": ("metrics.json", "json"),
        "evaluation_protocol": ("evaluation_protocol.json", "json"),
        "run_summary": ("run_summary.md", "markdown"),
    }
    assert expected_run_artifacts.keys() <= manifest["run_artifacts"].keys()
    for artifact_name, (filename, expected_format) in expected_run_artifacts.items():
        artifact_path = output_dir / filename
        artifact = manifest["run_artifacts"][artifact_name]
        assert artifact["path"] == str(artifact_path)
        assert artifact["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert artifact["size_bytes"] == artifact_path.stat().st_size
        assert artifact["format"] == expected_format
    assert manifest["run_artifacts"]["train_log"]["num_rows"] >= 1
    assert manifest["run_artifacts"]["eval_log"]["num_rows"] >= 1
    checkpoint_path = output_dir / "masked_lora_state.pt"
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    adapter_checkpoint = manifest["checkpoint_artifacts"]["adapter_checkpoint"]
    assert adapter_checkpoint["path"] == str(checkpoint_path)
    assert adapter_checkpoint["sha256"] == checkpoint_hash
    assert adapter_checkpoint["size_bytes"] == checkpoint_path.stat().st_size
    assert adapter_checkpoint["format"] == "torch_state_dict"
    assert adapter_checkpoint["contains_base_model_weights"] is False
    assert adapter_checkpoint["checkpoint_selection_rule"] == "final_checkpoint_only"
    eval_predictions_path = output_dir / "eval_predictions.jsonl"
    eval_predictions_hash = hashlib.sha256(eval_predictions_path.read_bytes()).hexdigest()
    prediction_rows = [
        json.loads(line)
        for line in eval_predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gsm8k_artifact = manifest["evaluation_artifacts"]["gsm8k_predictions"]
    assert gsm8k_artifact["path"] == str(eval_predictions_path)
    assert gsm8k_artifact["sha256"] == eval_predictions_hash
    assert gsm8k_artifact["num_rows"] == 2
    assert {
        "question",
        "gold_answer",
        "raw_prediction",
        "prediction",
        "pred_final",
        "gold_final",
        "metric",
        "score",
    } <= set(gsm8k_artifact["required_fields"])
    assert {"raw_prediction", "prediction", "pred_final", "gold_final", "metric", "score"} <= set(prediction_rows[0])
    assert "# Run Summary: tiny_manifest" in summary
    assert "global_batch_size" in summary
    assert "resolved_config_sha256" in summary
    assert "git_commit" in summary
    assert "command_cwd" in summary
    assert "dependency_versions" in summary
    assert "base_seed" in summary
    assert "preallocation_sketch_seed" in summary
    assert "requires_grad" in summary
    assert "active_final" in summary
    assert "module_budget" in summary
    assert "method_artifacts" in summary
    assert "run_artifacts" in summary
    assert "adapter_checkpoint" in summary
    assert "training_sec" in summary
    assert "gsm8k_predictions" in summary


def test_train_applies_configured_gradient_clipping_before_optimizer_step(tmp_path: Path, monkeypatch):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_grad_clip"
    config["training"]["max_grad_norm"] = 1.0
    config["preallocation"].update(
        {
            "allocation_method": "covra_full",
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
            "solver": "dp",
        }
    )
    clip_calls: list[dict[str, object]] = []
    real_clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def record_clip_grad_norm_(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        clip_calls.append({"num_params": len(params), "max_norm": float(max_norm)})
        return real_clip_grad_norm(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(trainer_module.torch.nn.utils, "clip_grad_norm_", record_clip_grad_norm_)

    train(config)

    assert clip_calls
    assert clip_calls[0]["num_params"] > 0
    assert clip_calls[0]["max_norm"] == 1.0

    output_dir = tmp_path / "outputs" / "tiny_grad_clip"
    log_lines = (output_dir / "train_log.jsonl").read_text().splitlines()
    first_step = json.loads(next(line for line in log_lines if '"event": "train_step"' in line))
    assert first_step["max_grad_norm"] == 1.0
    assert first_step["grad_norm_before_clip"] >= 0.0


def test_trainer_checkpoint_can_be_restored_into_fresh_tiny_lora_model(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_checkpoint_restore"
    config["preallocation"].update(
        {
            "allocation_method": "covra_full",
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
            "solver": "dp",
        }
    )

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_checkpoint_restore"
    rank_dict = json.loads((output_dir / "rank_dict.json").read_text())
    _tokenizer, fresh_model = load_tokenizer_and_model(config)
    target_modules = find_target_linear_modules(fresh_model, config["lora"]["target_modules"])
    assert collect_module_dims(target_modules)
    wrapped = inject_static_lora(
        fresh_model,
        rank_dict,
        alpha=float(config["lora"]["alpha"]),
        dropout=float(config["lora"]["dropout"]),
        scaling=config["lora"].get("scaling", "alpha_over_sqrt_r"),
        lora_dtype=torch.float32,
    )
    assert wrapped

    report = load_lora_state(output_dir / "masked_lora_state.pt", fresh_model)

    assert report["loaded_keys"]
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["shape_mismatches"] == []


def test_trainer_keeps_adapter_checkpoint_fp32_when_base_dtype_is_bfloat16(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_adapter_fp32"
    config["model"]["torch_dtype"] = "bfloat16"
    config["preallocation"].update(
        {
            "allocation_method": "covra_full",
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
            "solver": "dp",
        }
    )

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_adapter_fp32"
    state = torch.load(output_dir / "masked_lora_state.pt", map_location="cpu")
    lora_dtypes = {value.dtype for key, value in state.items() if "lora_" in key}
    manifest = json.loads((output_dir / "run_manifest.json").read_text())

    assert lora_dtypes == {torch.float32}
    assert manifest["precision"]["model_torch_dtype"] == "bfloat16"
    assert manifest["precision"]["adapter_dtype"] == "float32"


def test_uniform_rank_covra_init_ablation_uses_uniform_rank_with_covra_direction_bank(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_uniform_rank_covra_init"
    config["preallocation"].update(
        {
            "allocation_method": "covra_full",
            "rank_override": "uniform_ref",
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
            "solver": "dp",
        }
    )

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_uniform_rank_covra_init"
    rank_dict = json.loads((output_dir / "rank_dict.json").read_text())
    assert set(rank_dict.values()) == {1}
    init_summary = json.loads((output_dir / "init_summary.json").read_text())
    assert init_summary["direction_bank_path"]
    preallocation_path = Path(init_summary["direction_bank_path"]).with_name("dico_v03_rank1_seed42.json")
    preallocation = json.loads(preallocation_path.read_text())
    assert preallocation["rank_override"] == "uniform_ref"
    assert preallocation["allocation_before_rank_override"]


def test_covra_rank_random_init_ablation_keeps_covra_rank_but_skips_direction_bank_init(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_covra_rank_random_init"
    config["preallocation"].update(
        {
            "allocation_method": "covra_full",
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
            "solver": "dp",
        }
    )
    config["dico"]["init"] = {"mode": "kaiming_zero_B"}

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_covra_rank_random_init"
    rank_dict = json.loads((output_dir / "rank_dict.json").read_text())
    assert rank_dict
    init_summary = json.loads((output_dir / "init_summary.json").read_text())
    assert init_summary["mode"] == "kaiming_zero_B"
    assert "direction_bank_path" not in init_summary
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert manifest["preallocation"]["allocation_method"] == "covra_full"


def test_train_writes_evaluation_protocol_manifest_with_mtbench_local_config(tmp_path: Path):
    config = _tiny_dico_cd_da_config(tmp_path)
    config["experiment_name"] = "tiny_eval_protocol"
    config["preallocation"].update(
        {
            "allocation_method": "covra_full",
            "rho": 1.0,
            "sign_split": True,
            "type_scaling": False,
            "log_compression": False,
            "solver": "dp",
        }
    )
    config.setdefault("evaluation", {}).update(
        {
            "compute_accuracy": False,
            "generation_max_new_tokens": 32,
            "mtbench_local": {
                "enabled": False,
                "judge_model": "meta-llama/Llama-3.1-70B-Instruct",
                "judge_prompt_version": "fastchat-v0.2.36",
                "conversation_template": "llama-3",
                "temperature": 0.0,
                "seed": 0,
                "swap_positions": True,
                "max_retries": 2,
            },
        }
    )

    train(config)

    output_dir = tmp_path / "outputs" / "tiny_eval_protocol"
    protocol = json.loads((output_dir / "evaluation_protocol.json").read_text())

    assert protocol["gsm8k"]["decoding"] == "greedy"
    assert protocol["gsm8k"]["do_sample"] is False
    assert protocol["gsm8k"]["max_new_tokens"] == 32
    assert protocol["humaneval"]["pass_at_k_estimator"] == "official_unbiased"
    assert protocol["checkpoint_selection"]["rule"] == "final_checkpoint_only"
    assert protocol["checkpoint_selection"]["uses_test_metric_for_selection"] is False
    assert protocol["checkpoint_selection"]["max_evaluations_per_checkpoint"] == 1
    assert protocol["mtbench_local"]["status"] == "CONFIGURED_NOT_EXECUTED"
    assert protocol["mtbench_local"]["judge_model"] == "meta-llama/Llama-3.1-70B-Instruct"
    assert protocol["mtbench_local"]["swap_positions"] is True
