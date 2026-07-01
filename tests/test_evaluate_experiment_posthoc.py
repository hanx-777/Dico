import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from dico_rank.lora_masked import inject_masked_lora
from dico_rank.model_loader import find_target_linear_modules, load_tokenizer_and_model


ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_experiment_posthoc_updates_metrics_without_training(tmp_path: Path):
    experiment_dir = tmp_path / "outputs" / "tiny_lora"
    experiment_dir.mkdir(parents=True)
    config = {
        "_project_root": str(tmp_path),
        "seed": 42,
        "experiment_name": "tiny_lora",
        "method": "lora",
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
        "training": {"max_steps": 99, "batch_size": 1, "gradient_accumulation_steps": 1},
        "lora": {
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "max_rank_multiplier": 2,
        },
        "budget": {"mode": "equal_trainable_params", "warning_threshold": 0.01},
        "evaluation": {
            "metric": "gsm8k_accuracy",
            "protocol": "internal_zero_shot",
            "prompt_style": "sft_cot_hash",
            "answer_extraction": "strict_then_flexible",
            "compute_accuracy": True,
            "accuracy_max_samples": 2,
            "generation_max_new_tokens": 4,
            "stop_sequences": ["\nQuestion:", "<|im_end|>"],
            "max_batches": 1,
        },
    }
    (experiment_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    _tokenizer, model = load_tokenizer_and_model(config)
    module_names = [name for name, _module in find_target_linear_modules(model, config["lora"]["target_modules"])]
    allocation = {name: 1 for name in module_names}
    (experiment_dir / "rank_allocation_final.json").write_text(json.dumps(allocation), encoding="utf-8")
    inject_masked_lora(model, allocation, max_rank=2, alpha=16, dropout=0.0)
    lora_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name or "rank_mask" in name
    }
    torch.save(lora_state, experiment_dir / "masked_lora_state.pt")
    (experiment_dir / "eval_log.jsonl").write_text("", encoding="utf-8")
    (experiment_dir / "metrics.json").write_text(
        json.dumps({"method": "lora", "rank": 1, "target_budget": 123, "actual_budget": 123}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_experiment.py"), "--experiment_dir", str(experiment_dir)],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["method"] == "lora"
    assert metrics["rank"] == 1
    assert metrics["target_budget"] == 123
    assert metrics["evaluation_protocol"] == "internal_zero_shot"
    assert metrics["final_eval_accuracy"] is not None
    assert metrics["eval_total"] == 2.0
    assert (experiment_dir / "eval_predictions.jsonl").exists()
    eval_rows = [
        json.loads(line)
        for line in (experiment_dir / "eval_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert eval_rows[-1]["posthoc"] is True
    assert eval_rows[-1]["event"] == "posthoc_eval"
    assert "timestamp" in eval_rows[-1]
