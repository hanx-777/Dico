import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection_hash(sample_hashes: list[str]) -> str:
    digest = hashlib.sha256()
    for value in sample_hashes:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_valid_manifest(run_dir: Path) -> Path:
    config_path = run_dir / "config_resolved.yaml"
    train_log = run_dir / "train_log.jsonl"
    eval_log = run_dir / "eval_log.jsonl"
    metrics = run_dir / "metrics.json"
    protocol = run_dir / "evaluation_protocol.json"
    summary = run_dir / "run_summary.md"
    checkpoint = run_dir / "masked_lora_state.pt"
    predictions = run_dir / "eval_predictions.jsonl"

    _write_text(config_path, "method: dico_cd_da\n")
    _write_text(train_log, '{"event":"train_step"}\n')
    _write_text(eval_log, '{"event":"final_eval"}\n')
    _write_text(metrics, '{"final_metric": 0.5}\n')
    _write_text(protocol, '{"checkpoint_selection":{"rule":"final_checkpoint_only"}}\n')
    _write_text(summary, "# Run Summary\n")
    checkpoint.write_bytes(b"adapter-state")
    _write_text(
        predictions,
        json.dumps(
            {
                "question": "q",
                "gold_answer": "1",
                "raw_prediction": "1",
                "prediction": "1",
                "pred_final": "1",
                "gold_final": "1",
                "metric": "exact_match",
                "score": 1,
            }
        )
        + "\n",
    )

    manifest = {
        "experiment_name": "covra_seed42",
        "method": "dico_cd_da",
        "rank": 8,
        "seed": 42,
        "model": {
            "name_or_path": "meta-llama/Llama-3.1-8B-Base",
            "model_revision": "UNRESOLVED",
            "tokenizer_revision": "UNRESOLVED",
            "model_revision_status": "UNRESOLVED",
            "tokenizer_revision_status": "UNRESOLVED",
        },
        "world_size": 1,
        "global_batch_size": 64,
        "optimizer_steps": 1563,
        "warmup_steps": 47,
        "source_control": {"git_commit": "abc123", "git_dirty": True},
        "command": {
            "argv": ["scripts/run_experiment.py", "--config", "configs/dico/dico_cd_da_r8.yaml"],
            "cwd": str(ROOT),
            "python_executable": "python",
        },
        "dependency_versions": {
            "python": "3.13.0",
            "torch": "2.8.0",
            "transformers": "4.55.0",
            "accelerate": "1.10.0",
        },
        "python": "3.13.0",
        "cuda_available": False,
        "cuda_version": None,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "config": {
            "resolved_config_path": str(config_path),
            "resolved_config_sha256": _sha(config_path),
        },
        "seeds": {"base_seed": 42, "calibration_seed": 42, "preallocation_sketch_seed": 42},
        "data": {
            "dataset_name": "gsm8k",
            "train_path": "data/gsm8k/main/train.jsonl",
            "eval_path": "data/gsm8k/main/test.jsonl",
            "train_count": 8,
            "eval_count": 1,
            "train_hash": "trainhash",
            "eval_hash": "evalhash",
        },
        "data_loading": {
            "sampler_type": "single_process_full_dataset",
            "distributed_sampler_used": False,
            "drop_last": False,
            "last_accumulation_behavior": "full_accumulation_from_repeating_iterator",
            "dataloader_length_batches": 2,
            "optimizer_steps_source": "training_loop_global_step",
        },
        "calibration": {
            "num_selected_samples": 2,
            "sample_ids": ["sample-0", "sample-1"],
            "sample_hashes": ["hash0", "hash1"],
            "sample_indices": [0, 1],
            "selection_hash": _selection_hash(["hash0", "hash1"]),
        },
        "training": {
            "batch_size_per_process": 4,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": False,
            "learning_rate": 5e-5,
            "weight_decay": 5e-4,
            "max_grad_norm": 1.0,
            "lr_decay_ratio": 0.1,
        },
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "param_groups": [
                {"index": 0, "lr": 5e-5, "initial_lr": 5e-5, "weight_decay": 5e-4, "num_tensors": 2, "num_params": 6815744}
            ],
        },
        "scheduler": {
            "name": "cosine_with_warmup_and_floor",
            "warmup_steps": 47,
            "warmup_ratio": 0.03,
            "lr_decay_ratio": 0.1,
            "num_training_steps": 1563,
            "optimizer_steps_source": "training_loop_global_step",
        },
        "precision": {"model_torch_dtype": "bfloat16", "adapter_dtype": "float32"},
        "lora": {"dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "budget": {
            "target_budget": 6815744,
            "actual_budget": 6815744,
            "budget_error": 0,
            "budget_error_ratio": 0.0,
        },
        "parameter_metrics": {
            "requires_grad_params": 6815744,
            "peak_active_params": 6815744,
            "final_active_params": 6815744,
            "budget_target": 6815744,
            "budget_actual": 6815744,
            "budget_error": 0,
        },
        "parameter_counts": {
            "requires_grad": 6815744,
            "active_final": 6815744,
            "active_peak": 6815744,
        },
        "module_budget": {
            "total_final_params": 6815744,
            "modules": {
                "layers.0.q_proj": {
                    "in_dim": 4096,
                    "out_dim": 4096,
                    "rank_cost": 8192,
                    "initial_rank": 8,
                    "final_rank": 8,
                    "initial_params": 65536,
                    "final_params": 65536,
                }
            },
        },
        "optimizer_state_estimate": {
            "optimizer": "AdamW",
            "trainable_params": 6815744,
            "state_tensors_per_param": 2,
            "state_dtype": "float32",
            "state_bytes_per_param": 8,
            "estimated_state_bytes": 54525952,
            "includes_master_params": False,
        },
        "timing": {
            "calibration_sec": 1.0,
            "allocation_sec": 0.5,
            "initialization_sec": 0.2,
            "training_sec": 10.0,
            "train_tokens": 1024,
            "train_tokens_per_sec": 102.4,
        },
        "hardware": {"cuda_peak_memory_allocated_bytes": 123},
        "run_artifacts": {
            "train_log": {"path": str(train_log), "sha256": _sha(train_log), "size_bytes": train_log.stat().st_size, "format": "jsonl", "num_rows": 1},
            "eval_log": {"path": str(eval_log), "sha256": _sha(eval_log), "size_bytes": eval_log.stat().st_size, "format": "jsonl", "num_rows": 1},
            "metrics": {"path": str(metrics), "sha256": _sha(metrics), "size_bytes": metrics.stat().st_size, "format": "json"},
            "evaluation_protocol": {"path": str(protocol), "sha256": _sha(protocol), "size_bytes": protocol.stat().st_size, "format": "json"},
            "run_summary": {"path": str(summary), "sha256": _sha(summary), "size_bytes": summary.stat().st_size, "format": "markdown"},
        },
        "checkpoint_artifacts": {
            "adapter_checkpoint": {
                "path": str(checkpoint),
                "sha256": _sha(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "format": "torch_state_dict",
                "contains_base_model_weights": False,
                "checkpoint_selection_rule": "final_checkpoint_only",
            }
        },
        "evaluation_artifacts": {
            "gsm8k_predictions": {
                "path": str(predictions),
                "sha256": _sha(predictions),
                "num_rows": 1,
                "format": "jsonl",
                "required_fields": [
                    "question",
                    "gold_answer",
                    "raw_prediction",
                    "prediction",
                    "pred_final",
                    "gold_final",
                    "metric",
                    "score",
                ],
            }
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    _write_text(manifest_path, json.dumps(manifest))
    return manifest_path


def test_validate_run_manifest_accepts_complete_manifest(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    json_path = tmp_path / "validation.json"
    md_path = tmp_path / "validation.md"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "PASS"
    assert payload["summary"]["failed_checks"] == 0
    assert payload["summary"]["validated_manifests"] == 1
    assert "# Run Manifest Validation" in md_path.read_text()


def test_validate_run_manifest_accepts_cpu_smoke_batch_when_scope_is_explicit(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_scope"] = "cpu_tiny_smoke"
    manifest["global_batch_size"] = 1
    manifest["training"]["batch_size_per_process"] = 1
    manifest["training"]["gradient_accumulation_steps"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python", "scripts/validate_run_manifest.py", "--manifest", str(manifest_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_validate_run_manifest_accepts_adalora_e_vector_as_method_physical_overhead(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    manifest["method"] = "adalora"
    manifest["budget"]["adalora_final_active_params"] = 6816768
    manifest["budget"]["adalora_peak_active_params"] = 10225152
    manifest["parameter_counts"]["active_final"] = 6816768
    manifest["parameter_counts"]["active_peak"] = 10225152
    manifest["parameter_counts"]["requires_grad"] = 10225152
    manifest["parameter_metrics"]["final_active_params"] = 6816768
    manifest["parameter_metrics"]["peak_active_params"] = 10225152
    manifest["parameter_metrics"]["requires_grad_params"] = 10225152
    manifest["optimizer_state_estimate"]["trainable_params"] = 10225152
    manifest["optimizer_state_estimate"]["estimated_state_bytes"] = 10225152 * 8
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = subprocess.run(
        ["python", "scripts/validate_run_manifest.py", "--manifest", str(manifest_path)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_validate_run_manifest_fails_when_parameter_metric_contract_is_incomplete(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    del manifest["parameter_metrics"]["peak_active_params"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    json_path = tmp_path / "bad_parameter_metrics_validation.json"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    assert any(
        check["id"] == "parameter_metrics"
        for manifest_result in payload["manifests"]
        for check in manifest_result["checks"]
        if check["status"] == "FAIL"
    )
    assert "peak_active_params" in result.stderr


def test_validate_run_manifest_fails_on_artifact_sha_mismatch(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    manifest["run_artifacts"]["train_log"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    json_path = tmp_path / "bad_validation.json"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    assert any(
        check["id"] == "artifact_hashes"
        for manifest_result in payload["manifests"]
        for check in manifest_result["checks"]
        if check["status"] == "FAIL"
    )
    assert "sha256 mismatch" in result.stderr


def test_validate_run_manifest_fails_on_invalid_revision_status(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["model_revision_status"] = "MAYBE"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    json_path = tmp_path / "bad_revision_validation.json"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    assert any(
        check["id"] == "model_revision_status"
        for manifest_result in payload["manifests"]
        for check in manifest_result["checks"]
        if check["status"] == "FAIL"
    )
    assert "model_revision_status" in result.stderr


def test_validate_run_manifest_fails_when_locked_revision_value_is_unresolved(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["model_revision_status"] = "LOCKED"
    manifest["model"]["model_revision"] = "UNRESOLVED"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    json_path = tmp_path / "bad_locked_revision_validation.json"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    assert any(
        check["id"] == "model_revision_status"
        for manifest_result in payload["manifests"]
        for check in manifest_result["checks"]
        if check["status"] == "FAIL"
    )
    assert "LOCKED" in result.stderr


def test_validate_run_manifest_fails_when_unresolved_status_has_locked_value(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    manifest["model"]["tokenizer_revision_status"] = "UNRESOLVED"
    manifest["model"]["tokenizer_revision"] = "abc123"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    json_path = tmp_path / "bad_unresolved_revision_validation.json"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    assert any(
        check["id"] == "model_revision_status"
        for manifest_result in payload["manifests"]
        for check in manifest_result["checks"]
        if check["status"] == "FAIL"
    )
    assert "UNRESOLVED" in result.stderr


def test_validate_run_manifest_fails_when_audit_trail_fields_are_missing(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    del manifest["data"]["train_hash"]
    del manifest["calibration"]["selection_hash"]
    del manifest["command"]["argv"]
    del manifest["scheduler"]["optimizer_steps_source"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    json_path = tmp_path / "bad_audit_trail_validation.json"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    failing = [
        check
        for manifest_result in payload["manifests"]
        for check in manifest_result["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "required_fields" for check in failing)
    assert "data.train_hash" in result.stderr
    assert "calibration.selection_hash" in result.stderr
    assert "command.argv" in result.stderr
    assert "scheduler.optimizer_steps_source" in result.stderr


def test_validate_run_manifest_fails_when_runtime_environment_fields_are_missing(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    del manifest["dependency_versions"]["python"]
    del manifest["cuda_version"]
    del manifest["cuda_device_names"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    json_path = tmp_path / "bad_runtime_environment_validation.json"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    failing = [
        check
        for manifest_result in payload["manifests"]
        for check in manifest_result["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "required_fields" for check in failing)
    assert "dependency_versions.python" in result.stderr
    assert "cuda_version" in result.stderr
    assert "cuda_device_names" in result.stderr


def test_validate_run_manifest_fails_on_calibration_selection_hash_mismatch(tmp_path):
    manifest_path = _write_valid_manifest(tmp_path / "run")
    manifest = json.loads(manifest_path.read_text())
    manifest["calibration"]["selection_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    json_path = tmp_path / "bad_calibration_hash_validation.json"

    result = subprocess.run(
        [
            "python",
            "scripts/validate_run_manifest.py",
            "--manifest",
            str(manifest_path),
            "--json-output",
            str(json_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    assert any(
        check["id"] == "calibration_selection"
        for manifest_result in payload["manifests"]
        for check in manifest_result["checks"]
        if check["status"] == "FAIL"
    )
    assert "selection_hash" in result.stderr
