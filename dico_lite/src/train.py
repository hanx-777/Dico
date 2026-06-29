import logging
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Trainer, TrainingArguments

LOGGER = logging.getLogger(__name__)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CustomCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attention_mask, labels = [], [], []
        
        for ex in examples:
            pad = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(ex["attention_mask"] + [0] * pad)
            labels.append(ex["labels"] + [-100] * pad)
            
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }



def inject_lora(model: nn.Module, rank_pattern: Dict[str, int]) -> nn.Module:
    """
    Injects LoRA into the model using the given rank pattern.
    Modules with rank 0 are ignored.
    """
    target_modules = []
    filtered_pattern = {}
    
    for name, rank in rank_pattern.items():
        if rank > 0:
            target_modules.append(name)
            filtered_pattern[name] = rank
            
    if not target_modules:
        raise ValueError("No modules selected for LoRA injection (all ranks are 0).")

    config = LoraConfig(
        r=8,  # Default fallback, actually overridden by rank_pattern
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        rank_pattern=filtered_pattern,
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def train_sft(
    model: nn.Module,
    tokenizer: Any,
    train_dataset: List[Dict[str, str]],
    output_dir: str,
    max_length: int,
    learning_rate: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    weight_decay: float,
    warmup_ratio: float,
    device: torch.device,
) -> Dict[str, Any]:
    
    from transformers import Trainer, TrainingArguments
    from src.data import build_sft_example

    LOGGER.info("Tokenizing training dataset...")
    tokenized_dataset = []
    for raw in train_dataset:
        ex = build_sft_example(
            raw["question"],
            raw["answer"],
            tokenizer,
            max_length=max_length,
            use_chat_template=False,
            enable_thinking=False,
        )
        tokenized_dataset.append(ex)

    pad_token_id = getattr(tokenizer, "pad_token_id", 0) or getattr(tokenizer, "eos_token_id", 0)
    if isinstance(pad_token_id, list):
        pad_token_id = pad_token_id[0]
    if not isinstance(pad_token_id, int):
        pad_token_id = 0
        
    collator = CustomCollator(pad_token_id=pad_token_id)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        bf16=(device.type == "cuda"),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=collator,
    )

    LOGGER.info("Starting Training...")
    train_result = trainer.train()
    
    trainer.save_model(output_dir)
    
    return {
        "train_loss": train_result.training_loss,
        "global_step": train_result.global_step,
    }
