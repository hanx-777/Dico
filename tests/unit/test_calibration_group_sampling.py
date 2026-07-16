import torch

import dico.trainer as trainer_module
from dico.data import SFTCollator


def _tokenized_record(question: str, group: str) -> dict:
    return {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "labels": [1, 2, 3],
        "question": question,
        "answer": "a",
        "group": group,
    }


def test_build_calibration_batches_returns_selected_in_same_order_as_batches():
    records = [_tokenized_record(f"q{i}", "math") for i in range(6)]
    collator = SFTCollator(pad_token_id=0)

    batches, selected = trainer_module._build_calibration_batches(
        records, collator, torch.device("cpu"), {"num_samples": 4, "batch_size": 2},
    )

    assert len(selected) == 4
    assert selected == records[:4]  # no shuffle by default -> prefix, matches batches' content
    assert sum(b["input_ids"].shape[0] for b in batches) == len(selected)


def test_stratified_group_sample_draws_evenly_across_groups():
    records = [_tokenized_record(f"m{i}", "math") for i in range(20)] + [
        _tokenized_record(f"c{i}", "code") for i in range(5)
    ]

    selected = trainer_module._stratified_group_sample(records, calibration_limit=10, seed=42)

    groups = [r["group"] for r in selected]
    assert groups.count("math") == 5
    assert groups.count("code") == 5


def test_stratified_group_sample_caps_at_available_pool_size_per_group():
    # "code" only has 3 records available, fewer than the even 5/5 split would want.
    records = [_tokenized_record(f"m{i}", "math") for i in range(20)] + [
        _tokenized_record(f"c{i}", "code") for i in range(3)
    ]

    selected = trainer_module._stratified_group_sample(records, calibration_limit=10, seed=42)

    groups = [r["group"] for r in selected]
    assert groups.count("code") == 3  # capped, not padded with duplicates
    assert groups.count("math") == 5


def test_build_calibration_batches_uses_balanced_group_sampling_when_configured():
    records = [_tokenized_record(f"m{i}", "math") for i in range(20)] + [
        _tokenized_record(f"c{i}", "code") for i in range(20)
    ]
    collator = SFTCollator(pad_token_id=0)

    _batches, selected = trainer_module._build_calibration_batches(
        records, collator, torch.device("cpu"),
        {"num_samples": 8, "batch_size": 2, "group_sampling": "balanced", "seed": 1},
    )

    groups = [r["group"] for r in selected]
    assert groups.count("math") == 4
    assert groups.count("code") == 4


def test_stratified_group_sample_is_deterministic_for_a_fixed_seed():
    records = [_tokenized_record(f"m{i}", "math") for i in range(20)] + [
        _tokenized_record(f"c{i}", "code") for i in range(20)
    ]

    selected_a = trainer_module._stratified_group_sample(records, calibration_limit=8, seed=7)
    selected_b = trainer_module._stratified_group_sample(records, calibration_limit=8, seed=7)

    assert [r["question"] for r in selected_a] == [r["question"] for r in selected_b]
