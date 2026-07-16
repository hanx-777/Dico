"""Unit tests for the custom cosine-with-warmup-and-floor LR scheduler.

GoRA's protocol decays cosine LR down to 10% of peak, not to 0 (the default
behavior of transformers.get_cosine_schedule_with_warmup, which this repo no
longer depends on for scheduling).
"""
from __future__ import annotations

import pytest
import torch

from dico.trainer import build_cosine_schedule_with_warmup_and_floor


def _run_schedule(
    num_warmup_steps,
    num_training_steps,
    lr_decay_ratio,
    peak_lr=1.0,
    decay_over_post_warmup_steps=False,
):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr)
    scheduler = build_cosine_schedule_with_warmup_and_floor(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_decay_ratio=lr_decay_ratio,
        decay_over_post_warmup_steps=decay_over_post_warmup_steps,
    )
    lrs = []
    for _ in range(num_training_steps + 1):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return lrs


def test_warmup_ramps_linearly_to_peak():
    lrs = _run_schedule(num_warmup_steps=10, num_training_steps=100, lr_decay_ratio=0.1)
    assert lrs[0] == 0.0
    assert lrs[5] == pytest.approx(0.5)
    assert lrs[10] == pytest.approx(1.0)


def test_decays_toward_floor_with_locked_gora_denominator():
    lrs = _run_schedule(num_warmup_steps=10, num_training_steps=100, lr_decay_ratio=0.1)
    expected = 0.1 + 0.9 * 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * 0.9))
    assert lrs[-1] == pytest.approx(expected)
    # never dips below the floor after warmup completes
    assert min(lrs[10:]) >= 0.1 - 1e-9


def test_zero_decay_ratio_matches_legacy_full_decay_to_zero():
    lrs = _run_schedule(num_warmup_steps=10, num_training_steps=100, lr_decay_ratio=0.0)
    assert lrs[-1] == pytest.approx(0.5 * (1.0 + __import__("math").cos(__import__("math").pi * 0.9)))


def test_reference_covra_decay_reaches_configured_floor_at_final_step():
    lrs = _run_schedule(
        num_warmup_steps=10,
        num_training_steps=100,
        lr_decay_ratio=0.1,
        decay_over_post_warmup_steps=True,
    )

    assert lrs[-1] == pytest.approx(0.1)
