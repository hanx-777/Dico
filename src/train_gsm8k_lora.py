import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from src.calibration import (
    collect_dico_calibration_stats,
    collect_dico_profiles,
    compute_dico_atoms,
)
from src.data_gsm8k import build_sft_example, load_gsm8k, subset_dataset
from src.dico_allocator import allocate_dico_lite
from src.diagnostics import run_diagnostics
from src.evaluate_gsm8k import evaluate_gsm8k
from src.model_utils import (
    apply_lora_adapters,
    find_target_linear_modules,
    load_model,
    load_tokenizer,
    verify_lora_ranks,
)
from src.module_coverage_allocator import allocate_module_coverage
from src.uniform_allocator import allocate_uniform
from src.utils import (
    ensure_dir,
    read_version,
    set_seed,
    setup_logging,
    tensor_to_list,
    total_parameter_count,
    trainable_parameter_count,
    write_json,
)


LOGGER = logging.getLogger(__name__)


def collate_sft_examples(examples: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in examples)
    input_ids, attention_mask, labels = [], [], []
    for ex in examples:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(ex["attention_mask"] + [0] * pad)
        labels.append(ex["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_one_step(model, examples: List[Dict[str, Any]], lr: float, device: torch.device) -> float:
    model.to(device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found")
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    pad_token_id = 0
    batch = collate_sft_examples(examples, pad_token_id=pad_token_id)
    batch = {k: v.to(device) for k, v in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**batch)
    if outputs.loss is None:
        raise RuntimeError("Model did not return loss")
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu().item())


def train_sft(
    model,
    tokenizer,
    dataset: List[Dict[str, str]],
    max_length: int,
    lr: float,
    num_train_epochs: int,
    gradient_accumulation_steps: int,
    device: torch.device,
    use_chat_template: bool,
    enable_thinking: bool,
) -> Dict[str, Any]:
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters after LoRA injection")
    optimizer = torch.optim.AdamW(params, lr=lr)
    losses = []
    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(num_train_epochs):
        for raw in tqdm(dataset, desc="train", leave=False):
            ex = build_sft_example(
                raw["question"],
                raw["answer"],
                tokenizer,
                max_length=max_length,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            batch = collate_sft_examples([ex], pad_token_id=pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / float(gradient_accumulation_steps)
            loss.backward()
            losses.append(float(outputs.loss.detach().cpu().item()))
            step += 1
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    if step % gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"train_loss": float(sum(losses) / max(1, len(losses))), "train_steps": step}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_name", default="openai/gsm8k")
    parser.add_argument("--dataset_config", default="main")
    parser.add_argument("--method", choices=["uniform", "module_coverage", "dico"], default="uniform")
    parser.add_argument("--finetune_mode", choices=["lora", "full"], default="full")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_limit", type=int, default=512)
    parser.add_argument("--eval_limit", type=int, default=200)
    parser.add_argument("--calibration_size", type=int, default=32)
    parser.add_argument("--avg_rank", type=float, default=1)
    parser.add_argument("--top_k_atoms", type=int, default=4)
    parser.add_argument("--target_modules", default="q_proj,v_proj")
    parser.add_argument("--load_in_4bit", default="false")
    parser.add_argument("--load_in_8bit", default="false")
    parser.add_argument("--torch_dtype", default="bf16")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_chat_template", default="false")
    parser.add_argument("--enable_thinking", default="false")
    parser.add_argument("--gradient_checkpointing", default="false")
    parser.add_argument("--eval_max_new_tokens", type=int, default=256)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--exact_svd", action="store_true")
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    from src.utils import str2bool

    return str2bool(value)


def main() -> None:
    setup_logging()
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(Path(args.output_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_in_4bit = _as_bool(args.load_in_4bit)
    load_in_8bit = _as_bool(args.load_in_8bit)
    if args.finetune_mode == "full" and (load_in_4bit or load_in_8bit):
        raise RuntimeError(
            "finetune_mode=full requires unquantized model loading. "
            "Set --load_in_4bit false and --load_in_8bit false."
        )

    tokenizer = load_tokenizer(args.model_name_or_path)
    model, model_config = load_model(
        args.model_name_or_path,
        torch_dtype=args.torch_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=_as_bool(args.gradient_checkpointing),
    )
    write_json(
        output_dir / "config.json",
        {
            "args": vars(args),
            "peft_version": read_version("peft"),
            "transformers_version": read_version("transformers"),
            "torch_version": torch.__version__,
            "model_config": model_config.to_dict() if hasattr(model_config, "to_dict") else {},
        },
    )

    dataset = load_gsm8k(args.dataset_name, args.dataset_config)
    train_data = subset_dataset(dataset["train"], args.train_limit)
    eval_data = subset_dataset(dataset["test"], args.eval_limit)
    target_suffixes = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    target_modules = find_target_linear_modules(model, target_suffixes)
    target_names = [name for name, _ in target_modules]
    if not target_names:
        raise RuntimeError("No target nn.Linear modules found for suffixes %s" % target_suffixes)
    target_module_dims = {
        name: {"d_in": module.in_features, "d_out": module.out_features, "cost": module.in_features + module.out_features}
        for name, module in target_modules
    }

    rank_patterns: Dict[str, Dict[str, int]] = {}
    used_budget = 0.0
    total_budget = float(args.avg_rank) * sum(float(dims["cost"]) for dims in target_module_dims.values())
    stats = atoms = profiles = None
    if args.method == "uniform":
        rank_pattern = allocate_uniform(target_names, avg_rank=int(args.avg_rank))
        used_budget = sum(
            float(rank_pattern[name]) * float(target_module_dims[name]["cost"]) for name in target_names
        )
    elif args.method == "module_coverage":
        stats = collect_dico_calibration_stats(
            model,
            tokenizer,
            train_data,
            target_names,
            args.calibration_size,
            args.max_length,
            output_dir,
            device,
            use_chat_template=_as_bool(args.use_chat_template),
            enable_thinking=_as_bool(args.enable_thinking),
        )
        denom = stats.norm_matrix.sum(dim=1).clamp_min(1e-8)
        module_profiles = stats.norm_matrix.T / denom.view(1, -1)
        module_profiles = module_profiles - module_profiles.mean(dim=-1, keepdim=True)
        module_profiles = module_profiles / torch.linalg.norm(module_profiles, dim=-1, keepdim=True).clamp_min(1e-8)
        allocation = allocate_module_coverage(target_names, stats.module_dims, module_profiles, args.avg_rank)
        rank_pattern = allocation.rank_pattern
        used_budget = allocation.used_budget
        total_budget = allocation.total_budget
        write_json(output_dir / "allocation_debug.json", tensor_to_list(allocation.__dict__))
    else:
        stats = collect_dico_calibration_stats(
            model,
            tokenizer,
            train_data,
            target_names,
            args.calibration_size,
            args.max_length,
            output_dir,
            device,
            use_chat_template=_as_bool(args.use_chat_template),
            enable_thinking=_as_bool(args.enable_thinking),
        )
        atoms = compute_dico_atoms(stats, args.top_k_atoms, output_dir, exact_svd=args.exact_svd)
        profiles = collect_dico_profiles(
            model,
            tokenizer,
            train_data,
            stats,
            atoms,
            args.max_length,
            output_dir,
            device,
            use_chat_template=_as_bool(args.use_chat_template),
            enable_thinking=_as_bool(args.enable_thinking),
        )
        allocation = allocate_dico_lite(
            target_names,
            stats.module_dims,
            profiles.normalized_profiles,
            stats.importance,
            atoms.rho,
            avg_rank=args.avg_rank,
        )
        rank_pattern = allocation.rank_pattern
        used_budget = allocation.used_budget
        total_budget = allocation.total_budget
        write_json(output_dir / "allocation_debug.json", tensor_to_list(allocation.__dict__))
        module_baseline = allocate_module_coverage(
            target_names,
            stats.module_dims,
            profiles.module_profiles,
            args.avg_rank,
        )
        run_diagnostics(
            target_names,
            profiles.module_profiles,
            profiles.normalized_profiles,
            output_dir,
            rank_patterns={
                "uniform": allocate_uniform(target_names, avg_rank=int(args.avg_rank)),
                "module_coverage": module_baseline.rank_pattern,
                "dico": rank_pattern,
            },
        )

    rank_patterns[args.method] = rank_pattern
    (output_dir / "rank_pattern.json").write_text(json.dumps(rank_pattern, indent=2), encoding="utf-8")

    if args.finetune_mode == "lora":
        model = apply_lora_adapters(model, rank_pattern, target_names, lora_alpha=16, use_peft=True)
        verify_lora_ranks(model, rank_pattern, target_names, output_dir)
    else:
        for param in model.parameters():
            param.requires_grad = True
        write_json(
            output_dir / "full_finetune_verification.json",
            {
                "finetune_mode": "full",
                "all_parameters_trainable": True,
                "trainable_params": trainable_parameter_count(model),
                "total_params": total_parameter_count(model),
                "note": "LoRA adapters are not injected in full fine-tuning mode.",
            },
        )

    before_trainable = trainable_parameter_count(model)
    train_summary = train_sft(
        model,
        tokenizer,
        train_data,
        args.max_length,
        args.learning_rate,
        args.num_train_epochs,
        args.gradient_accumulation_steps,
        device,
        _as_bool(args.use_chat_template),
        _as_bool(args.enable_thinking),
    )
    metrics = evaluate_gsm8k(
        model,
        tokenizer,
        eval_data,
        output_dir,
        args.max_length,
        args.eval_max_new_tokens,
        device,
        use_chat_template=_as_bool(args.use_chat_template),
        enable_thinking=_as_bool(args.enable_thinking),
    )
    write_json(
        output_dir / "run_summary.json",
        {
            "method": args.method,
            "finetune_mode": args.finetune_mode,
            "eval": metrics,
            "train": train_summary,
            "trainable_params": before_trainable,
            "total_params": total_parameter_count(model),
            "used_budget": used_budget,
            "total_budget": total_budget,
        },
    )


if __name__ == "__main__":
    main()
