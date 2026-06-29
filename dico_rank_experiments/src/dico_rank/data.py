from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import torch


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


class TinyTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._token_to_id = {self.pad_token: 0, self.eos_token: 1}
        self._id_to_token = {0: self.pad_token, 1: self.eos_token}

    def _tokenize(self, text: str) -> list[str]:
        return text.replace("\n", " \n ").split()

    def _id_for_token(self, token: str) -> int:
        if token not in self._token_to_id:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        return self._token_to_id[token]

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ids = [self._id_for_token(token) for token in self._tokenize(text)]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for idx in ids:
            idx = int(idx)
            if skip_special_tokens and idx in {self.pad_token_id, self.eos_token_id}:
                continue
            tokens.append(self._id_to_token.get(idx, f"<unk{idx}>"))
        return " ".join(tokens).replace(" \n ", "\n")


def normalize_number(text: str) -> str:
    return str(text).strip().replace(",", "").replace("$", "").replace(" ", "").rstrip(".")


def extract_final_answer(text: str) -> str:
    tail = text.split("####", 1)[1] if "####" in text else text
    numbers = [normalize_number(match.group(0)) for match in NUMBER_RE.finditer(tail)]
    return numbers[-1] if numbers else ""


def format_prompt(question: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Please solve the problem step by step and put the final answer after ####.\n\n"
        "Answer:\n"
    )


def build_sft_example(
    question: str,
    answer: str,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    prompt = format_prompt(question)
    answer_text = answer.strip()
    if getattr(tokenizer, "eos_token", None):
        answer_text += " " + tokenizer.eos_token
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
        "question": question,
        "answer": answer,
    }


def tiny_dataset() -> list[dict[str, str]]:
    return [
        {"question": "Tom has 1 apple and gets 1 more. How many?", "answer": "Tom has 2.\n#### 2"},
        {"question": "Mia has 3 pens and loses 1. How many?", "answer": "Mia has 2.\n#### 2"},
        {"question": "A box has 2 red balls and 2 blue balls. How many balls?", "answer": "There are 4.\n#### 4"},
        {"question": "Kai reads 5 pages and then 1 page. How many pages?", "answer": "Kai reads 6.\n#### 6"},
    ]


def _read_records(path: Path) -> list[dict[str, str]]:
    if path.is_dir():
        candidates = [
            path / "data.jsonl",
            path / "data.json",
            path / "train.jsonl",
            path / "train.json",
            path / "train-00000-of-00001.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                return _read_records(candidate)
        raise ValueError(f"No supported dataset file found in directory {path}")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".parquet":
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError("datasets is required to read local parquet data files") from exc
        dataset = load_dataset("parquet", data_files=str(path), split="train")
        return [dict(row) for row in dataset]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    raise ValueError(f"Unsupported JSON dataset shape in {path}")


def _resolve_data_path(config: dict[str, Any], path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    if value.is_absolute():
        return value
    project_root = Path(config.get("_project_root", Path.cwd()))
    return project_root / value


def load_raw_datasets(config: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data_cfg = config.get("data", {})
    if data_cfg.get("source") == "tiny" or data_cfg.get("train_path") == "tiny":
        data = tiny_dataset()
        return data, data
    train_path = _resolve_data_path(config, data_cfg.get("train_path"))
    eval_path = _resolve_data_path(config, data_cfg.get("eval_path"))
    if train_path and train_path.exists():
        train = _read_records(train_path)
        eval_data = _read_records(eval_path) if eval_path and eval_path.exists() else train[:]
        return train, eval_data
    dataset_name = data_cfg.get("dataset_name", "openai/gsm8k")
    dataset_config = data_cfg.get("dataset_config", "main")
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "datasets is required for non-tiny data. Install requirements or set data.source=tiny."
        ) from exc
    import os
    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/root/hf_cache/datasets")
    dataset = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    return list(dataset["train"]), list(dataset["test"])


def limit_records(records: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    return records if limit is None else records[: int(limit)]


def tokenize_records(records: list[dict[str, str]], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    return [build_sft_example(row["question"], row["answer"], tokenizer, max_length) for row in records]


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
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


def batch_iter(records: list[dict[str, Any]], batch_size: int, collator: SFTCollator):
    idx = 0
    while True:
        batch = []
        for _ in range(batch_size):
            batch.append(records[idx % len(records)])
            idx += 1
        yield collator(batch)
