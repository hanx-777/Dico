import json
import time
from pathlib import Path

import torch

from dico.data import TinyTokenizer
from dico.evaluator import _run_humaneval_program, estimate_pass_at_k, evaluate_humaneval_pass_at_1


class FixedAnswerModel(torch.nn.Module):
    def __init__(self, answer_texts: list[str], tokenizer: TinyTokenizer):
        super().__init__()
        self.answer_ids = [
            tokenizer(answer_text, add_special_tokens=False)["input_ids"] for answer_text in answer_texts
        ]
        self.calls = 0

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128, do_sample=False, **kwargs):
        del attention_mask, max_new_tokens, do_sample, kwargs
        idx = min(self.calls, len(self.answer_ids) - 1)
        self.calls += 1
        suffix = torch.tensor([self.answer_ids[idx]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def test_run_humaneval_program_passes_on_correct_solution():
    program = (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n\n"
        "check(add)\n"
    )

    assert _run_humaneval_program(program, timeout_seconds=5.0) is True


def test_run_humaneval_program_fails_on_incorrect_solution():
    program = (
        "def add(a, b):\n"
        "    return a - b\n\n"
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n\n"
        "check(add)\n"
    )

    assert _run_humaneval_program(program, timeout_seconds=5.0) is False


def test_run_humaneval_program_times_out_on_infinite_loop():
    program = "while True:\n    pass\n"

    started = time.monotonic()
    result = _run_humaneval_program(program, timeout_seconds=0.5)
    elapsed = time.monotonic() - started

    assert result is False
    assert elapsed < 3.0  # generous upper bound so a slow CI runner doesn't flake


def test_estimate_pass_at_k_matches_official_unbiased_formula():
    assert estimate_pass_at_k(num_samples=1, num_correct=1, k=1) == 1.0
    assert estimate_pass_at_k(num_samples=1, num_correct=0, k=1) == 0.0
    assert estimate_pass_at_k(num_samples=5, num_correct=2, k=1) == 0.4
    assert abs(estimate_pass_at_k(num_samples=5, num_correct=2, k=2) - 0.7) < 1e-12


def test_evaluate_humaneval_pass_at_1_scores_correct_and_incorrect_completions(tmp_path: Path):
    tokenizer = TinyTokenizer()
    # Both prompts are single-line (colon-body-on-same-line) so the tokenizer's
    # whitespace-collapsing round trip can't mangle indentation.
    model = FixedAnswerModel(["b", "999"], tokenizer)
    records = [
        {
            "task_id": "add",
            "prompt": "def add(a, b): return a + ",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
            "entry_point": "add",
        },
        {
            "task_id": "sub",
            "prompt": "def sub(a, b): return a - ",
            "test": "def check(candidate):\n    assert candidate(5, 3) == 2\n",
            "entry_point": "sub",
        },
    ]

    metrics = evaluate_humaneval_pass_at_1(
        model,
        tokenizer,
        records,
        device=torch.device("cpu"),
        max_new_tokens=8,
        prediction_path=tmp_path / "humaneval_predictions.jsonl",
    )

    assert metrics["eval_correct"] == 1
    assert metrics["eval_total"] == 2
    assert metrics["eval_pass_at_1"] == 0.5
    assert metrics["eval_pass_at_1_estimator"] == "official_unbiased"
    assert metrics["eval_num_samples_per_task"] == 1
    assert metrics["eval_accuracy"] == 0.5
    rows = [json.loads(line) for line in (tmp_path / "humaneval_predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["task_id"] == "add"
    assert rows[0]["correct"] is True
    assert rows[0]["metric"] == "task_success"
    assert rows[0]["score"] == 1.0
    assert rows[0]["decoding"] == {"do_sample": False, "temperature": 0.0, "top_p": 1.0}
    assert rows[0]["pass_at_k_estimator"] == "official_unbiased"
    assert rows[1]["task_id"] == "sub"
    assert rows[1]["correct"] is False
    assert rows[1]["score"] == 0.0
