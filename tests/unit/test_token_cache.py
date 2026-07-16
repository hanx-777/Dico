from __future__ import annotations

from dico.data import TinyTokenizer, tokenize_records_cached


def test_token_cache_is_deterministic_and_reports_hit(tmp_path):
    records = [
        {"question": "1+1?", "answer": "#### 2"},
        {"question": "2+2?", "answer": "#### 4"},
    ]
    tokenizer = TinyTokenizer()

    first, first_meta = tokenize_records_cached(records, tokenizer, 32, tmp_path)
    second, second_meta = tokenize_records_cached(records, tokenizer, 32, tmp_path)

    assert first == second
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True
    assert first_meta["cache_key"] == second_meta["cache_key"]
    assert second_meta["sample_order_hash"] == first_meta["sample_order_hash"]
