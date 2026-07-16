import json
from pathlib import Path

import torch

from dico.data import SFTCollator
from dico.data import TinyTokenizer
from dico.evaluator import evaluate_gsm8k_accuracy, evaluate_loss


class TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, labels=None):
        del input_ids, attention_mask, labels
        return type("Output", (), {"loss": self.param * 0.0 + 2.0})()


class FixedAnswerModel(torch.nn.Module):
    def __init__(self, answer_texts: list[str], tokenizer: TinyTokenizer):
        super().__init__()
        self.answer_ids = [
            tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            for answer_text in answer_texts
        ]
        self.calls = 0

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128, do_sample=False, **kwargs):
        assert torch.is_inference_mode_enabled()
        del attention_mask, max_new_tokens, do_sample, kwargs
        idx = min(self.calls, len(self.answer_ids) - 1)
        self.calls += 1
        suffix = torch.tensor([self.answer_ids[idx]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


def test_gsm8k_accuracy_extracts_final_number_and_writes_predictions(tmp_path: Path):
    tokenizer = TinyTokenizer()
    model = FixedAnswerModel(
        [
            "reasoning #### 2 then extra 999<|im_end|>",
            "no numeric answer",
            "ignored #### 4",
        ],
        tokenizer,
    )
    records = [
        {"question": "one plus one?", "answer": "The answer is 2.\n#### 2"},
        {"question": "two plus two?", "answer": "The answer is 4.\n#### 4"},
        {"question": "three plus one?", "answer": "The answer is 4.\n#### 4"},
    ]

    metrics = evaluate_gsm8k_accuracy(
        model,
        tokenizer,
        records,
        device=torch.device("cpu"),
        max_samples=2,
        max_new_tokens=64,
        stop_sequences=["<|im_end|>"],
        extraction_mode="strict_then_flexible",
        prediction_path=tmp_path / "predictions.jsonl",
    )

    assert metrics["eval_accuracy"] == 0.5
    assert metrics["eval_exact_match"] == 0.5
    assert metrics["eval_correct"] == 1.0
    assert metrics["eval_total"] == 2.0
    assert metrics["eval_sample_count"] == 2.0
    rows = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["raw_prediction"].endswith("<|im_end|>")
    assert rows[0]["prediction"] == "reasoning #### 2 then extra 999"
    assert rows[0]["pred_final"] == "2"
    assert rows[0]["gold_final"] == "2"
    assert rows[0]["correct"] is True
    assert rows[0]["metric"] == "exact_match"
    assert rows[0]["score"] == 1.0
    assert rows[0]["extraction_mode"] == "strict_then_flexible"
    assert rows[0]["decoding"] == {"do_sample": False, "temperature": 0.0, "top_p": 1.0}
    assert rows[0]["stop_sequences"] == ["<|im_end|>"]
    assert rows[1]["pred_final"] == ""
    assert rows[1]["correct"] is False
    assert rows[1]["score"] == 0.0


def test_evaluate_loss_restores_original_training_state():
    records = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [1, 2]}]
    collator = SFTCollator(pad_token_id=0)

    train_model = TinyLossModel()
    train_model.train()
    train_metrics = evaluate_loss(
        train_model,
        records,
        collator,
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert train_metrics["eval_loss"] == 2.0
    assert train_model.training is True

    eval_model = TinyLossModel()
    eval_model.eval()
    eval_metrics = evaluate_loss(
        eval_model,
        records,
        collator,
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert eval_metrics["eval_loss"] == 2.0
    assert eval_model.training is False
