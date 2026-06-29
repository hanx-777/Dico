import argparse
import json
import logging
import os
from pathlib import Path

import datasets  # Fix Windows pyarrow+CUDA crash by importing datasets BEFORE torch
import torch

from src.allocator import allocate_module_dico
from src.calibration import collect_module_dico_profiles, normalize_profiles
from src.data import load_gsm8k, subset_dataset
from src.model_utils import find_target_linear_modules, load_model, load_tokenizer
from src.train import count_trainable_parameters, inject_lora, train_sft
from src.evaluate import evaluate_gsm8k
from src.utils import ensure_dir, setup_logging, write_json

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--method", type=str, choices=["uniform", "dico"], default="dico")
    parser.add_argument("--run_train", type=str, default="false")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Isolate output directory by method
    base_output_dir = Path(cfg.get("output_dir", "outputs"))
    output_dir = ensure_dir(base_output_dir / f"gsm8k_{args.method}")
    rank_pattern_path = output_dir / "rank_pattern.json"
    
    # 1. Load Dataset
    LOGGER.info("Loading dataset...")
    dataset = load_gsm8k("openai/gsm8k", "main")
    train_data = subset_dataset(dataset["train"], cfg.get("train_limit", 7473))
    eval_data = subset_dataset(dataset["test"], cfg.get("eval_limit", 200))
    
    # 2. Load Model & Tokenizer
    LOGGER.info("Loading model and tokenizer...")
    tokenizer = load_tokenizer(cfg["model_name_or_path"])
    model, _ = load_model(
        cfg["model_name_or_path"],
        torch_dtype=cfg.get("torch_dtype", "bf16"),
        load_in_4bit=cfg.get("load_in_4bit", False),
        load_in_8bit=cfg.get("load_in_8bit", False),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True)
    )
    
    target_suffixes = [x.strip() for x in str(cfg["target_modules"]).split(",") if x.strip()]
    target_modules = find_target_linear_modules(model, target_suffixes)
    target_names = [name for name, _ in target_modules]
    
    # 3. Calibration & Allocation (if rank pattern doesn't exist)
    if not rank_pattern_path.exists():
        if args.method == "uniform":
            LOGGER.info("Generating Uniform rank pattern...")
            avg_rank = int(cfg.get("avg_rank", 4.0))
            rank_pattern = {name: avg_rank for name in target_names}
            write_json(rank_pattern_path, rank_pattern)
            LOGGER.info(f"Uniform rank pattern saved to {rank_pattern_path}")
            
        elif args.method == "dico":
            LOGGER.info("Starting DiCo Calibration...")
            profiles = collect_module_dico_profiles(
                model,
                tokenizer,
                train_data,
                target_names,
                calibration_size=cfg.get("calibration_size", 384),
                max_length=cfg.get("max_length", 512),
                device=device,
            )
            
            normalized_profiles, importance = normalize_profiles(profiles.response_norms)
            
            module_dims = {
                name: {"cost": int(profiles.costs[i].item())}
                for i, name in enumerate(target_names)
            }
            
            allocation = allocate_module_dico(
                module_names=target_names,
                module_dims=module_dims,
                normalized_profiles=normalized_profiles,
                importance=importance,
                avg_rank=cfg.get("avg_rank", 4.0),
                budget_floor_ratio=cfg.get("budget_floor_ratio", 0.95),
                budget_max_ratio=cfg.get("budget_max_ratio", 1.0),
                coverage_eps=cfg.get("coverage_eps", 1e-3),
                max_rank_per_module=cfg.get("max_rank_per_module", 8),
                rank_decay=cfg.get("rank_decay", "sqrt"),
            )
            
            rank_pattern = allocation.rank_pattern
            write_json(rank_pattern_path, rank_pattern)
            write_json(output_dir / "allocation_debug.json", allocation.debug)
            LOGGER.info(f"Generated DiCo rank pattern saved to {rank_pattern_path}")
    else:
        LOGGER.info(f"Loading existing rank pattern from {rank_pattern_path}")
        with open(rank_pattern_path, "r", encoding="utf-8") as f:
            rank_pattern = json.load(f)

    # 4. Training
    if args.run_train.lower() == "true":
        LOGGER.info(f"Injecting LoRA adapters for method: {args.method}...")
        model = inject_lora(model, rank_pattern)
        trainable_params = count_trainable_parameters(model)
        LOGGER.info(f"Trainable parameters: {trainable_params}")
        
        # Free up memory before training
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        train_sft(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            output_dir=str(output_dir),
            max_length=cfg.get("max_length", 512),
            learning_rate=cfg.get("learning_rate", 2e-4),
            num_train_epochs=cfg.get("num_train_epochs", 1.0),
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            weight_decay=cfg.get("weight_decay", 0.0),
            warmup_ratio=cfg.get("warmup_ratio", 0.03),
            device=device,
        )
        LOGGER.info(f"Training complete for method: {args.method}!")
        
        LOGGER.info(f"Starting Evaluation on {len(eval_data)} samples...")
        evaluate_gsm8k(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_data,
            output_dir=output_dir,
            max_length=cfg.get("max_length", 512),
            max_new_tokens=cfg.get("eval_max_new_tokens", 256),
            device=device,
        )
    else:
        LOGGER.info("Skipping training (run_train=false). Exiting gracefully.")


if __name__ == "__main__":
    main()
