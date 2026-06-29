from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch

from dico_rank.data import SFTCollator, format_prompt, normalize_number


DEFAULT_GSM8K_STOP_SEQUENCES = ["\nQuestion:", "<|im_end|>"]
GSM8K_NUMBER_PATTERN = r"(?:[-+]?\$?|\$[-+]?)?\d[\d,]*(?:\.\d+)?\.?"
STRICT_HASH_RE = re.compile(r"####\s*(" + GSM8K_NUMBER_PATTERN + r")")
FLEXIBLE_NUMBER_RE = re.compile(GSM8K_NUMBER_PATTERN)


def evaluate_loss(
    model: torch.nn.Module,
    tokenized_records: list[dict[str, Any]],
    collator: SFTCollator,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    losses = []
    with torch.no_grad():
        total = len(tokenized_records)
        step = 0
        for start in range(0, total, batch_size):
            if max_batches is not None and step >= max_batches:
                break
            batch = collator(tokenized_records[start : start + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            if outputs.loss is not None:
                losses.append(float(outputs.loss.detach().cpu().item()))
            step += 1
    model.train()
    if not losses:
        return {"eval_loss": 0.0}
    return {"eval_loss": float(sum(losses) / len(losses))}


def extract_gsm8k_final_number(text: str, mode: str = "strict_then_flexible") -> str:
    if mode not in {"strict", "flexible", "strict_then_flexible"}:
        raise ValueError(f"Unsupported GSM8K extraction mode: {mode}")
    text = str(text)
    if mode in {"strict", "strict_then_flexible"}:
        match = STRICT_HASH_RE.search(text)
        if match:
            return normalize_number(match.group(1))
        if mode == "strict":
            return ""

    matches = FLEXIBLE_NUMBER_RE.findall(text)
    return normalize_number(matches[-1]) if matches else ""


def truncate_generation(text: str, stop_sequences: list[str] | None = None) -> str:
    stop_sequences = stop_sequences if stop_sequences is not None else DEFAULT_GSM8K_STOP_SEQUENCES
    cut = len(text)
    for sequence in stop_sequences:
        if not sequence:
            continue
        idx = text.find(sequence)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut].strip()


def evaluate_gsm8k_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    max_samples: int | None = None,
    max_new_tokens: int = 128,
    stop_sequences: list[str] | None = None,
    extraction_mode: str = "strict_then_flexible",
    prediction_path: Path | str | None = None,
) -> dict[str, float]:
    """Run generation-based GSM8K exact-match evaluation.

    The score compares the final normalized number from model generation with
    the final normalized number in the reference answer.
    """

    was_training = model.training
    model.eval()
    selected = records if max_samples is None else records[: int(max_samples)]
    correct = 0
    total = 0
    rows = []
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    with torch.no_grad():
        for record in selected:
            question = str(record.get("question", ""))
            gold_answer = str(record.get("answer", ""))
            prompt = format_prompt(question)
            encoded = tokenizer(prompt, add_special_tokens=False)
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
            attention_mask = torch.tensor([encoded.get("attention_mask", [1] * input_ids.shape[1])], dtype=torch.long, device=device)
            generate_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": int(max_new_tokens),
                "do_sample": False,
            }
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id
            generated = model.generate(**generate_kwargs)
            generated_suffix = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
            raw_prediction_text = tokenizer.decode(generated_suffix, skip_special_tokens=True)
            prediction_text = truncate_generation(raw_prediction_text, stop_sequences)
            pred_answer = extract_gsm8k_final_number(prediction_text, mode=extraction_mode)
            gold_final = extract_gsm8k_final_number(gold_answer, mode="strict_then_flexible")
            is_correct = bool(pred_answer and pred_answer == gold_final)
            correct += int(is_correct)
            total += 1
            rows.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "raw_prediction": raw_prediction_text,
                    "prediction": prediction_text,
                    "pred_final": pred_answer,
                    "gold_final": gold_final,
                    "correct": is_correct,
                    "extraction_mode": extraction_mode,
                }
            )

    if prediction_path is not None:
        path = Path(prediction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    if was_training:
        model.train()
    accuracy = float(correct / total) if total else 0.0
    return {
        "eval_accuracy": accuracy,
        "eval_exact_match": accuracy,
        "eval_correct": correct,
        "eval_total": total,
        "eval_sample_count": total,
    }
