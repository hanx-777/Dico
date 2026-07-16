from __future__ import annotations

import json
import hashlib
import os
import random
import re
import time
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


def format_prompt(question: str, group: str = "math") -> str:
    if group == "code":
        # The math template's "put the final answer after ####" instruction is
        # meaningless (actively misleading) for code-completion targets -- code
        # answers aren't a single extractable number -- so code-group records get a
        # plain instruction/response template instead.
        return f"Question:\n{question}\n\nAnswer:\n"
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
    group: str = "math",
) -> dict[str, Any]:
    prompt = format_prompt(question, group=group)
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
        "group": group,
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
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
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


def stable_record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_hash(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(stable_record_hash(record).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resolve_data_path(config: dict[str, Any], path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    if value.is_absolute():
        return value
    project_root = Path(config.get("_project_root", Path.cwd()))
    return project_root / value


def load_multi_source_train_records(config: dict[str, Any]) -> list[dict[str, str]]:
    """§6.5 mixed math+code: load and concatenate `data.train_sources`, tagging each
    record with its source's `group` (consumed downstream by trainer.py to build real
    per-sample calibration group labels -- see Phase K of the CovRA v0.6.2 alignment).

    Each source is `{path, group, limit}`; `limit` (optional) takes that source's
    first N records, matching this repo's existing no-shuffle convention for
    `data.train_limit` (reproducible, order-stable across runs). Concatenation order
    follows `train_sources`' own list order.
    """
    data_cfg = config.get("data", {})
    sources = data_cfg.get("train_sources") or []
    records: list[dict[str, str]] = []
    for source in sources:
        source_path = _resolve_data_path(config, source.get("path"))
        if source_path is None or not source_path.exists():
            raise FileNotFoundError(
                f"data.train_sources entry {source!r} points to a path that does not exist: "
                f"{source_path}. For the CodeFeedback source, run: python scripts/download_codefeedback.py"
            )
        source_records = _read_records(source_path)
        limit = source.get("limit")
        if limit is not None:
            source_records = source_records[: int(limit)]
        group = str(source.get("group", "unlabeled"))
        for record in source_records:
            record = dict(record)
            record["_group"] = group
            records.append(record)
    return records


def load_raw_datasets(config: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data_cfg = config.get("data", {})
    if data_cfg.get("source") == "tiny" or data_cfg.get("train_path") == "tiny":
        data = tiny_dataset()
        return data, data
    if data_cfg.get("train_sources"):
        train = load_multi_source_train_records(config)
        eval_path = _resolve_data_path(config, data_cfg.get("eval_path"))
        eval_data = _read_records(eval_path) if eval_path and eval_path.exists() else train[:]
        return train, eval_data
    train_path = _resolve_data_path(config, data_cfg.get("train_path"))
    eval_path = _resolve_data_path(config, data_cfg.get("eval_path"))
    if train_path and train_path.exists():
        train = _read_records(train_path)
        eval_data = _read_records(eval_path) if eval_path and eval_path.exists() else train[:]
        return train, eval_data

    # Local file not found — give a clear error rather than silently downloading.
    _configured = data_cfg.get("train_path", "<not set>")
    _resolved = str(train_path) if train_path else "<could not resolve path>"
    raise FileNotFoundError(
        f"\nTraining data not found.\n"
        f"  Configured path : {_configured}\n"
        f"  Resolved path   : {_resolved}\n"
        f"\n"
        f"To download MetaMathQA-100K (the default training set), run:\n"
        f"  python scripts/download_data.py\n"
        f"\n"
        f"If your server cannot reach HuggingFace directly, use a mirror:\n"
        f"  python scripts/download_data.py --hf-endpoint https://hf-mirror.com\n"
        f"\n"
        f"If you want to use a different local file, set data.train_path in your config\n"
        f"to the relative or absolute path of a JSONL file with 'question'/'answer' fields.\n"
    )


def load_humaneval_records(config: dict[str, Any]) -> list[dict[str, str]]:
    """Load local HumanEval problems for the §6.5 mixed math+code evaluation.

    Follows the same local-jsonl convention as data/gsm8k/main/*.jsonl: one
    problem per line with fields task_id, prompt, test, entry_point. Raises
    rather than silently skipping when data.eval_datasets names "humaneval"
    but no local file is configured/present, since a missing eval set should
    never be reported as a passing (or absent) score.
    """
    data_cfg = config.get("data", {})
    path = _resolve_data_path(config, data_cfg.get("humaneval_path", "data/humaneval/test.jsonl"))
    if path is None or not path.exists():
        raise FileNotFoundError(
            "data.eval_datasets includes 'humaneval' but no local HumanEval file was found at "
            f"{path}. Provide a jsonl file with one HumanEval problem per line (fields: task_id, "
            "prompt, test, entry_point) -- matching the data/gsm8k/main/*.jsonl convention -- and "
            "set data.humaneval_path if using a non-default location."
        )
    return _read_records(path)


def limit_records(records: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    return records if limit is None else records[: int(limit)]


def order_records(
    records: list[dict[str, str]],
    *,
    shuffle: bool,
    dataset_seed: int,
) -> list[dict[str, str]]:
    """Return the public, method-independent training order.

    The fixed 100K subset is selected before this function is called.  Shuffling
    therefore changes only exposure order, never subset membership.
    """
    ordered = list(records)
    if shuffle:
        random.Random(int(dataset_seed)).shuffle(ordered)
    return ordered


def tokenize_records(records: list[dict[str, str]], tokenizer: Any, max_length: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(records):
        group = row.get("_group", "math")
        row_hash = stable_record_hash(dict(row))
        example = build_sft_example(row["question"], row["answer"], tokenizer, max_length, group=group)
        example["sample_index"] = index
        example["sample_id"] = f"{group}:{index}:{row_hash[:12]}"
        example["sample_hash"] = row_hash
        examples.append(example)
    return examples


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    stable_kwargs = {
        str(key): value
        for key, value in dict(init_kwargs or {}).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "init_kwargs": stable_kwargs,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tokenize_records_cached(
    records: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
    cache_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Tokenize with a deterministic, atomic on-disk cache.

    The cache key includes the ordered data hash, tokenizer fingerprint, prompt
    template version, and max length.  A cache hit therefore cannot change the
    public sample order or tokenization semantics.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    order_hash = dataset_hash(records)
    tokenizer_hash = _tokenizer_fingerprint(tokenizer)
    key_payload = {
        "sample_order_hash": order_hash,
        "tokenizer_fingerprint": tokenizer_hash,
        "prompt_template_version": "sft_cot_hash_v1",
        "max_length": int(max_length),
    }
    cache_key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_path = cache_root / f"{cache_key}.pt"
    metadata = {
        **key_payload,
        "cache_key": cache_key,
        "cache_path": str(cache_path),
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return list(payload["records"]), {**metadata, "cache_hit": True}

    lock_path = cache_root / f".{cache_key}.lock"
    owns_lock = False
    try:
        while not cache_path.exists():
            try:
                descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(descriptor)
                owns_lock = True
                break
            except FileExistsError:
                try:
                    if time.time() - lock_path.stat().st_mtime > 3600:
                        lock_path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.2)
        if not owns_lock:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            return list(payload["records"]), {**metadata, "cache_hit": True, "waited_for_cache": True}

        tokenized = tokenize_records(records, tokenizer, int(max_length))
        temporary = cache_root / f".{cache_key}.{os.getpid()}.tmp"
        torch.save({"records": tokenized, "metadata": metadata}, temporary)
        os.replace(temporary, cache_path)
        return tokenized, {**metadata, "cache_hit": False, "waited_for_cache": False}
    finally:
        if owns_lock:
            lock_path.unlink(missing_ok=True)
        (cache_root / f".{cache_key}.{os.getpid()}.tmp").unlink(missing_ok=True)


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
