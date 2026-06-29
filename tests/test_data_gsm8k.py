import pytest

from tests.tiny_model import TinyTokenizer

from src.data_gsm8k import (
    build_sft_example,
    extract_final_answer,
    extract_prediction_answer,
    format_gsm8k_prompt,
    normalize_number,
)
from src.calibration import build_shifted_answer_mask


def test_gold_answer_requires_hash_delimiter():
    assert extract_final_answer("work\n#### $1,234.00") == "1234.00"
    with pytest.raises(ValueError, match="####"):
        extract_final_answer("work only final is 42")


def test_prediction_prefers_hash_answer_then_falls_back_to_last_number():
    assert extract_prediction_answer("noise 99 #### 12 apples") == "12"
    assert extract_prediction_answer("reason 3 then answer $4.50.") == "4.50"


def test_build_sft_example_masks_prompt_and_shifted_answer_positions():
    tokenizer = TinyTokenizer()
    question = "A has 1 and gets 2. How many?"
    answer = "A has 3.\n#### 3"
    example = build_sft_example(question, answer, tokenizer, max_length=128)
    prompt = format_gsm8k_prompt(question)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    assert all(label == -100 for label in example["labels"][:prompt_len])
    assert any(label != -100 for label in example["labels"][prompt_len:])
    assert normalize_number("$1,200.") == "1200"

    mask = build_shifted_answer_mask(
        labels=example["labels_tensor"].unsqueeze(0),
        attention_mask=example["attention_mask_tensor"].unsqueeze(0),
    )
    assert mask.shape == example["labels_tensor"].unsqueeze(0).shape
    assert int(mask.sum().item()) == sum(label != -100 for label in example["labels"][1:])
    assert not bool(mask[0, -1].item())
