from __future__ import annotations

import torch

from dico.data import TinyTokenizer
from dico.evaluator import evaluate_gsm8k_accuracy


class BatchAwareModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.answer = tokenizer("#### 2", add_special_tokens=False)["input_ids"]
        self.batch_sizes = []

    def generate(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        self.batch_sizes.append(input_ids.shape[0])
        suffix = torch.tensor(self.answer, dtype=input_ids.dtype, device=input_ids.device)
        suffix = suffix.unsqueeze(0).expand(input_ids.shape[0], -1)
        return torch.cat([input_ids, suffix], dim=1)


def test_gsm8k_batch_four_matches_batch_one():
    records = [{"question": f"q{i}", "answer": "#### 2"} for i in range(5)]
    tokenizer1 = TinyTokenizer()
    tokenizer1.padding_side = "right"
    model1 = BatchAwareModel(tokenizer1)
    result1 = evaluate_gsm8k_accuracy(model1, tokenizer1, records, torch.device("cpu"), batch_size=1)

    tokenizer4 = TinyTokenizer()
    tokenizer4.padding_side = "right"
    model4 = BatchAwareModel(tokenizer4)
    result4 = evaluate_gsm8k_accuracy(model4, tokenizer4, records, torch.device("cpu"), batch_size=4)

    assert result1["eval_accuracy"] == result4["eval_accuracy"] == 1.0
    assert model4.batch_sizes == [4, 1]
    assert tokenizer4.padding_side == "right"
