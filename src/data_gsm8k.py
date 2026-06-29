import re
from typing import Any, Dict, List, Optional

import torch


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_number(text: str) -> str:
    value = str(text).strip()
    value = value.replace(",", "").replace("$", "").replace(" ", "")
    while value.endswith(".") and value.count(".") <= 1:
        value = value[:-1]
    return value


def _numbers(text: str) -> List[str]:
    return [normalize_number(match.group(0)) for match in NUMBER_RE.finditer(text)]


def extract_final_answer(text: str) -> str:
    if "####" not in text:
        raise ValueError("GSM8K gold answer must contain ####")
    tail = text.split("####", 1)[1]
    numbers = _numbers(tail)
    if not numbers:
        raise ValueError("No numeric final answer found after ####")
    return numbers[-1]


def extract_prediction_answer(text: str) -> str:
    if "####" in text:
        tail = text.split("####", 1)[1]
        numbers = _numbers(tail)
        if numbers:
            return numbers[-1]
    numbers = _numbers(text)
    return numbers[-1] if numbers else ""


def format_gsm8k_prompt(
    question: str,
    use_chat_template: bool = False,
    tokenizer: Optional[Any] = None,
    enable_thinking: bool = False,
) -> str:
    plain = (
        "Question:\n"
        f"{question}\n\n"
        "Please solve the problem step by step and put the final answer after ####.\n\n"
        "Answer:\n"
    )
    if not use_chat_template or tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return plain
    messages = [{"role": "user", "content": plain}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_sft_example(
    question: str,
    answer: str,
    tokenizer: Any,
    max_length: int,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    prompt = format_gsm8k_prompt(
        question,
        use_chat_template=use_chat_template,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
    )
    answer_text = answer.strip()
    if getattr(tokenizer, "eos_token", None):
        answer_text = answer_text + " " + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = ([-100] * prompt_len + input_ids[prompt_len:])[: len(input_ids)]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_ids_tensor": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask_tensor": torch.tensor(attention_mask, dtype=torch.long),
        "labels_tensor": torch.tensor(labels, dtype=torch.long),
        "prompt_text": prompt,
        "answer_text": answer_text,
    }


def load_gsm8k(
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    split: Optional[str] = None,
):
    from datasets import load_dataset

    if split is None:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name, dataset_config, split=split)


def subset_dataset(dataset: Any, limit: Optional[int]) -> List[Dict[str, str]]:
    if limit is None:
        limit = len(dataset)
    return [dataset[i] for i in range(min(limit, len(dataset)))]
