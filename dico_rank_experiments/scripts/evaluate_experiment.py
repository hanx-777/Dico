#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico_rank.config import apply_overrides
from dico_rank.data import SFTCollator, limit_records, load_raw_datasets, tokenize_records
from dico_rank.evaluator import evaluate_gsm8k_accuracy, evaluate_loss
from dico_rank.logging_utils import log_eval
from dico_rank.lora_masked import inject_masked_lora
from dico_rank.model_loader import (
    load_tokenizer_and_model,
    model_device,
    model_input_device,
    select_torch_dtype,
)
from dico_rank.path_utils import resolve_project_path
from dico_rank.utils import setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing DiCo experiment output directory.")
    parser.add_argument("--experiment_dir", required=True, help="Path such as outputs/lora_r4")
    parser.add_argument("--override", action="append", default=[], help="Optional config override key=value")
    return parser.parse_args()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    setup_logging()
    experiment_dir = resolve_project_path(ROOT, args.experiment_dir)
    config_path = experiment_dir / "config_resolved.yaml"
    allocation_path = experiment_dir / "rank_allocation_final.json"
    state_path = experiment_dir / "masked_lora_state.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    if not allocation_path.exists():
        raise FileNotFoundError(f"Missing {allocation_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing {state_path}")

    config = apply_overrides(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}, args.override)
    tokenizer, model = load_tokenizer_and_model(config)
    placement_device = torch.device("cpu") if config.get("model", {}).get("type") == "tiny" else model_device(model)
    if str(placement_device) == "cpu" or not config.get("model", {}).get("device_map"):
        model.to(placement_device)
    input_device = model_input_device(model)

    final_allocation = {name: int(rank) for name, rank in _read_json(allocation_path).items()}
    max_rank = int(int(config["rank"]) * config.get("lora", {}).get("max_rank_multiplier", 2))
    inject_masked_lora(
        model,
        final_allocation,
        max_rank=max_rank,
        alpha=float(config.get("lora", {}).get("alpha", 16)),
        dropout=float(config.get("lora", {}).get("dropout", 0.0)),
        lora_dtype=select_torch_dtype(config.get("model", {}).get("torch_dtype", "bfloat16")),
    )
    state = torch.load(state_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected_lora = [name for name in unexpected if "lora_" in name or "rank_mask" in name]
    missing_lora = [name for name in missing if "lora_" in name or "rank_mask" in name]
    if unexpected_lora or missing_lora:
        raise RuntimeError(f"LoRA state mismatch; missing={missing_lora}, unexpected={unexpected_lora}")

    _train_raw, eval_raw = load_raw_datasets(config)
    data_cfg = config.get("data", {})
    eval_raw = limit_records(eval_raw, data_cfg.get("eval_limit"))
    eval_records = tokenize_records(eval_raw, tokenizer, int(data_cfg.get("max_length", 512)))
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or getattr(tokenizer, "eos_token_id", 0) or 0
    collator = SFTCollator(pad_token_id)
    evaluation_cfg = config.get("evaluation", {})
    batch_size = int(config.get("training", {}).get("batch_size", 1))

    loss_metrics = evaluate_loss(
        model,
        eval_records,
        collator,
        batch_size=batch_size,
        device=input_device,
        max_batches=int(evaluation_cfg.get("max_batches", 4)),
    )
    accuracy_samples = evaluation_cfg.get("accuracy_max_samples", data_cfg.get("eval_limit"))
    accuracy_metrics = evaluate_gsm8k_accuracy(
        model,
        tokenizer,
        eval_records,
        device=input_device,
        max_samples=accuracy_samples,
        max_new_tokens=int(evaluation_cfg.get("generation_max_new_tokens", 256)),
        stop_sequences=evaluation_cfg.get("stop_sequences"),
        extraction_mode=str(evaluation_cfg.get("answer_extraction", "strict_then_flexible")),
        prediction_path=experiment_dir / "eval_predictions.jsonl",
    )
    final_eval = {**loss_metrics, **accuracy_metrics}
    log_eval(experiment_dir / "eval_log.jsonl", {"posthoc": True, **final_eval})

    metrics_path = experiment_dir / "metrics.json"
    metrics = _read_json(metrics_path) if metrics_path.exists() else {}
    final_metric_name = str(evaluation_cfg.get("metric", "gsm8k_accuracy"))
    final_metric = final_eval["eval_accuracy"] if final_metric_name in {"accuracy", "exact_match", "gsm8k_accuracy"} else final_eval["eval_loss"]
    metrics.update(
        {
            "final_eval_loss": final_eval["eval_loss"],
            "final_eval_accuracy": final_eval["eval_accuracy"],
            "final_exact_match": final_eval["eval_exact_match"],
            "eval_correct": final_eval["eval_correct"],
            "eval_total": final_eval["eval_total"],
            "eval_sample_count": final_eval["eval_sample_count"],
            "evaluation_protocol": evaluation_cfg.get("protocol", "internal_zero_shot"),
            "evaluation_prompt_style": evaluation_cfg.get("prompt_style", "sft_cot_hash"),
            "answer_extraction": evaluation_cfg.get("answer_extraction", "strict_then_flexible"),
            "final_metric_name": final_metric_name,
            "final_metric": final_metric,
        }
    )
    write_json(metrics_path, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
