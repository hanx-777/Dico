import json
from pathlib import Path

from dico.data import (
    build_sft_example,
    format_prompt,
    load_multi_source_train_records,
    load_raw_datasets,
    tokenize_records,
)


class _TinyTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_format_prompt_uses_math_template_by_default():
    prompt = format_prompt("What is 2+2?")
    assert "put the final answer after ####" in prompt


def test_format_prompt_code_group_skips_math_specific_instruction():
    prompt = format_prompt("Write a function that reverses a string.", group="code")
    assert "####" not in prompt
    assert "Write a function that reverses a string." in prompt


def test_build_sft_example_records_its_group():
    example = build_sft_example("q", "a", _TinyTokenizer(), max_length=32, group="code")
    assert example["group"] == "code"

    default_example = build_sft_example("q", "a", _TinyTokenizer(), max_length=32)
    assert default_example["group"] == "math"


def test_load_multi_source_train_records_tags_and_concatenates_in_order(tmp_path):
    math_path = tmp_path / "data" / "metamathqa" / "train.jsonl"
    code_path = tmp_path / "data" / "codefeedback" / "train.jsonl"
    _write_jsonl(math_path, [{"question": f"m{i}", "answer": f"ma{i}"} for i in range(5)])
    _write_jsonl(code_path, [{"question": f"c{i}", "answer": f"ca{i}"} for i in range(5)])

    config = {
        "_project_root": str(tmp_path),
        "data": {
            "train_sources": [
                {"path": "data/metamathqa/train.jsonl", "group": "math", "limit": 3},
                {"path": "data/codefeedback/train.jsonl", "group": "code", "limit": 2},
            ]
        },
    }

    records = load_multi_source_train_records(config)

    assert [r["_group"] for r in records] == ["math", "math", "math", "code", "code"]
    assert [r["question"] for r in records] == ["m0", "m1", "m2", "c0", "c1"]


def test_load_multi_source_train_records_raises_clear_error_for_missing_source(tmp_path):
    config = {
        "_project_root": str(tmp_path),
        "data": {"train_sources": [{"path": "data/codefeedback/train.jsonl", "group": "code"}]},
    }

    try:
        load_multi_source_train_records(config)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "download_codefeedback" in str(exc)


def test_load_raw_datasets_dispatches_to_train_sources_when_present(tmp_path):
    math_path = tmp_path / "data" / "metamathqa" / "train.jsonl"
    code_path = tmp_path / "data" / "codefeedback" / "train.jsonl"
    _write_jsonl(math_path, [{"question": "m0", "answer": "ma0"}])
    _write_jsonl(code_path, [{"question": "c0", "answer": "ca0"}])
    eval_path = tmp_path / "data" / "gsm8k" / "test.jsonl"
    _write_jsonl(eval_path, [{"question": "eq0", "answer": "ea0"}])

    config = {
        "_project_root": str(tmp_path),
        "data": {
            "train_sources": [
                {"path": "data/metamathqa/train.jsonl", "group": "math"},
                {"path": "data/codefeedback/train.jsonl", "group": "code"},
            ],
            "eval_path": "data/gsm8k/test.jsonl",
        },
    }

    train, eval_data = load_raw_datasets(config)

    assert [r["_group"] for r in train] == ["math", "code"]
    assert eval_data == [{"question": "eq0", "answer": "ea0"}]


def test_load_raw_datasets_single_source_config_is_unaffected(tmp_path):
    # Regression guard: an ordinary (non-train_sources) config must behave exactly as
    # before -- no "_group"/"group" leakage into records that never opted into it.
    train_path = tmp_path / "train.jsonl"
    _write_jsonl(train_path, [{"question": "q0", "answer": "a0"}])

    config = {"_project_root": str(tmp_path), "data": {"train_path": "train.jsonl"}}

    train, _eval_data = load_raw_datasets(config)

    assert "_group" not in train[0]


def test_tokenize_records_defaults_group_to_math_when_untagged():
    records = [{"question": "q", "answer": "a"}]  # no "_group" key at all

    tokenized = tokenize_records(records, _TinyTokenizer(), max_length=32)

    assert tokenized[0]["group"] == "math"


def test_tokenize_records_threads_group_tag_through():
    records = [{"question": "q", "answer": "a", "_group": "code"}]

    tokenized = tokenize_records(records, _TinyTokenizer(), max_length=32)

    assert tokenized[0]["group"] == "code"
