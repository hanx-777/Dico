from dico_rank.evaluator import extract_gsm8k_final_number, truncate_generation


def test_extract_gsm8k_final_number_handles_strict_hash_answers():
    cases = {
        "#### 1,234": "1234",
        "#### $56": "56",
        "#### -7": "-7",
        "#### 12.5": "12.5",
        "#### 56.": "56",
    }

    for text, expected in cases.items():
        assert extract_gsm8k_final_number(text) == expected


def test_extract_gsm8k_final_number_falls_back_to_last_number():
    assert extract_gsm8k_final_number("reasoning 1 2 final 4") == "4"


def test_extract_gsm8k_final_number_prefers_strict_answer_over_later_numbers():
    assert extract_gsm8k_final_number("work... #### 2 then random 999") == "2"


def test_extract_gsm8k_final_number_returns_empty_for_no_number():
    assert extract_gsm8k_final_number("no number here") == ""


def test_truncate_generation_uses_earliest_stop_sequence():
    text = "answer #### 2<|im_end|>\nQuestion: next 999"

    assert truncate_generation(text, ["\nQuestion:", "<|im_end|>"]) == "answer #### 2"
