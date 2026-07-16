from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from dico.data import SFTCollator, format_prompt, normalize_number


DEFAULT_GSM8K_STOP_SEQUENCES = ["\nQuestion:", "<|im_end|>"]
GSM8K_NUMBER_PATTERN = r"(?:[-+]?\$?|\$[-+]?)?\d[\d,]*(?:\.\d+)?\.?"
STRICT_HASH_RE = re.compile(r"####\s*(" + GSM8K_NUMBER_PATTERN + r")")
FLEXIBLE_NUMBER_RE = re.compile(GSM8K_NUMBER_PATTERN)

DEFAULT_HUMANEVAL_STOP_SEQUENCES = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert "]


def evaluate_loss(
    model: torch.nn.Module,
    tokenized_records: list[dict[str, Any]],
    collator: SFTCollator,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses = []
    try:
        with torch.inference_mode():
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
                del batch, outputs
                step += 1
    finally:
        if was_training:
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
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval: int = 50,
    batch_size: int = 1,
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
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    prediction_handle = None
    prediction_tmp_path = None
    prediction_final_path = None

    if prediction_path is not None:
        path = Path(prediction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction_final_path = path
        prediction_tmp_path = path.with_name(path.name + ".tmp")

    try:
        if prediction_tmp_path is not None:
            prediction_handle = prediction_tmp_path.open("w", encoding="utf-8")

        with torch.inference_mode():
            eval_batch_size = max(1, int(batch_size))
            for start in range(0, len(selected), eval_batch_size):
                batch_records = selected[start : start + eval_batch_size]
                encoded_batch = [
                    tokenizer(format_prompt(str(record.get("question", ""))), add_special_tokens=False)
                    for record in batch_records
                ]
                max_prompt_length = max(len(encoded["input_ids"]) for encoded in encoded_batch)
                effective_pad_id = int(pad_token_id if pad_token_id is not None else (eos_token_id or 0))
                padded_ids: list[list[int]] = []
                padded_masks: list[list[int]] = []
                for encoded in encoded_batch:
                    ids = list(encoded["input_ids"])
                    pad_length = max_prompt_length - len(ids)
                    # Decoder-only batched generation must use left padding.
                    padded_ids.append([effective_pad_id] * pad_length + ids)
                    padded_masks.append([0] * pad_length + [1] * len(ids))
                input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
                attention_mask = torch.tensor(padded_masks, dtype=torch.long, device=device)
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
                for row_index, record in enumerate(batch_records):
                    question = str(record.get("question", ""))
                    gold_answer = str(record.get("answer", ""))
                    generated_suffix = generated[row_index, max_prompt_length:].detach().cpu().tolist()
                    raw_prediction_text = tokenizer.decode(generated_suffix, skip_special_tokens=True)
                    prediction_text = truncate_generation(raw_prediction_text, stop_sequences)
                    pred_answer = extract_gsm8k_final_number(prediction_text, mode=extraction_mode)
                    gold_final = extract_gsm8k_final_number(gold_answer, mode="strict_then_flexible")
                    is_correct = bool(pred_answer and pred_answer == gold_final)
                    correct += int(is_correct)
                    total += 1
                    row = {
                        "question": question,
                        "gold_answer": gold_answer,
                        "raw_prediction": raw_prediction_text,
                        "prediction": prediction_text,
                        "pred_final": pred_answer,
                        "gold_final": gold_final,
                        "correct": is_correct,
                        "metric": "exact_match",
                        "score": 1.0 if is_correct else 0.0,
                        "extraction_mode": extraction_mode,
                        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0},
                        "batch_size": eval_batch_size,
                        "padding_side": "left",
                        "stop_sequences": stop_sequences if stop_sequences is not None else DEFAULT_GSM8K_STOP_SEQUENCES,
                    }
                    if prediction_handle is not None:
                        prediction_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                del input_ids, attention_mask, generated
                if progress_callback is not None:
                    interval = max(1, int(progress_interval))
                    if total % interval == 0 or total == len(selected) or total - len(batch_records) < interval <= total:
                        progress_callback(
                            {
                                "eval_correct": correct,
                                "eval_total": total,
                                "eval_accuracy": float(correct / total) if total else 0.0,
                            }
                        )
    except BaseException:
        if prediction_handle is not None and not prediction_handle.closed:
            prediction_handle.close()
        if prediction_tmp_path is not None:
            prediction_tmp_path.unlink(missing_ok=True)
        raise
    else:
        if prediction_handle is not None:
            prediction_handle.close()
            prediction_tmp_path.replace(prediction_final_path)
    finally:
        if prediction_handle is not None and not prediction_handle.closed:
            prediction_handle.close()
        if was_training:
            model.train()
    accuracy = float(correct / total) if total else 0.0
    return {
        "eval_accuracy": accuracy,
        "eval_exact_match": accuracy,
        "eval_correct": correct,
        "eval_total": total,
        "eval_sample_count": total,
        "eval_decoding": "greedy",
        "eval_batch_size": max(1, int(batch_size)),
        "eval_padding_side": "left",
    }


def extract_humaneval_completion(text: str, stop_sequences: list[str] | None = None) -> str:
    return truncate_generation(
        text, stop_sequences if stop_sequences is not None else DEFAULT_HUMANEVAL_STOP_SEQUENCES
    )


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Official HumanEval unbiased pass@k estimator.

    This is the estimator used by OpenAI's HumanEval evaluation:
    ``1 - comb(n-c, k) / comb(n, k)`` when there are enough incorrect samples,
    otherwise the estimate is 1.0.
    """

    n = int(num_samples)
    c = int(num_correct)
    k = int(k)
    if n <= 0 or k <= 0:
        return 0.0
    if c < 0 or c > n:
        raise ValueError(f"num_correct must be in [0, num_samples], got c={c}, n={n}")
    if n - c < k:
        return 1.0
    return float(1.0 - math.comb(n - c, k) / math.comb(n, k))


def _run_humaneval_program(program: str, timeout_seconds: float) -> bool:
    """Execute a completed HumanEval program in its own subprocess.

    A subprocess (not an in-process exec) plus a hard timeout is the minimum
    guard needed against a generated completion that infinite-loops, crashes
    the interpreter, or otherwise misbehaves -- it isolates the parent
    evaluation process without requiring a full container sandbox.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(program)
        script_path = Path(handle.name)
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            timeout=float(timeout_seconds),
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return False
    finally:
        script_path.unlink(missing_ok=True)


def evaluate_humaneval_pass_at_1(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    max_samples: int | None = None,
    max_new_tokens: int = 256,
    stop_sequences: list[str] | None = None,
    timeout_seconds: float = 5.0,
    prediction_path: Path | str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_interval: int = 50,
) -> dict[str, float]:
    """Run greedy-decoding HumanEval pass@1 evaluation.

    Each record must provide "prompt" (the function signature + docstring the
    model completes), "test" (the HumanEval `check(candidate)` harness), and
    "entry_point" (the function name `check` calls). A completion counts as
    correct iff `prompt + completion + test + check(entry_point)` executes with
    exit code 0 in a fresh subprocess within timeout_seconds.
    """

    was_training = model.training
    model.eval()
    selected = records if max_samples is None else records[: int(max_samples)]
    correct = 0
    total = 0
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    prediction_handle = None
    prediction_tmp_path = None
    prediction_final_path = None

    if prediction_path is not None:
        path = Path(prediction_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction_final_path = path
        prediction_tmp_path = path.with_name(path.name + ".tmp")

    try:
        if prediction_tmp_path is not None:
            prediction_handle = prediction_tmp_path.open("w", encoding="utf-8")

        with torch.inference_mode():
            for record in selected:
                task_id = str(record.get("task_id", ""))
                prompt = str(record.get("prompt", ""))
                test_code = str(record.get("test", ""))
                entry_point = str(record.get("entry_point", ""))
                encoded = tokenizer(prompt, add_special_tokens=False)
                input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
                attention_mask = torch.tensor(
                    [encoded.get("attention_mask", [1] * input_ids.shape[1])], dtype=torch.long, device=device
                )
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
                raw_completion = tokenizer.decode(generated_suffix, skip_special_tokens=True)
                completion = extract_humaneval_completion(raw_completion, stop_sequences)
                program = f"{prompt}{completion}\n\n{test_code}\n\ncheck({entry_point})\n"
                is_correct = _run_humaneval_program(program, timeout_seconds)
                correct += int(is_correct)
                total += 1
                row = {
                    "task_id": task_id,
                    "raw_completion": raw_completion,
                    "completion": completion,
                    "correct": is_correct,
                    "metric": "task_success",
                    "score": 1.0 if is_correct else 0.0,
                    "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0},
                    "pass_at_k_estimator": "official_unbiased",
                }
                if prediction_handle is not None:
                    prediction_handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                del input_ids, attention_mask, generated
                if progress_callback is not None:
                    interval = max(1, int(progress_interval))
                    if total % interval == 0 or total == len(selected):
                        progress_callback(
                            {
                                "eval_correct": correct,
                                "eval_total": total,
                                "eval_accuracy": float(correct / total) if total else 0.0,
                            }
                        )
    except BaseException:
        if prediction_handle is not None and not prediction_handle.closed:
            prediction_handle.close()
        if prediction_tmp_path is not None:
            prediction_tmp_path.unlink(missing_ok=True)
        raise
    else:
        if prediction_handle is not None:
            prediction_handle.close()
            prediction_tmp_path.replace(prediction_final_path)
    finally:
        if prediction_handle is not None and not prediction_handle.closed:
            prediction_handle.close()
        if was_training:
            model.train()
    pass_at_1 = estimate_pass_at_k(total, correct, 1)
    return {
        "eval_accuracy": pass_at_1,
        "eval_pass_at_1": pass_at_1,
        "eval_pass_at_1_estimator": "official_unbiased",
        "eval_num_samples_per_task": 1,
        "eval_correct": correct,
        "eval_total": total,
        "eval_sample_count": total,
    }
