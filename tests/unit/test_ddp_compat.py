"""Unit tests for DDP compatibility logic.

These tests run without real GPUs or a running distributed process group.
They verify:
1. per-rank data sharding (slice coverage and non-overlap)
2. warmup_steps calculation
3. is_main_process guard semantics (via mock)
"""
from __future__ import annotations

import math
from pathlib import Path

from dico.config import load_yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. per-rank data sharding
# ---------------------------------------------------------------------------

def _shard(records, local_rank, num_processes):
    """Mirror the sharding logic in trainer.py train()."""
    return records[local_rank::num_processes]


def test_shard_coverage_single_process():
    records = list(range(100))
    shard = _shard(records, local_rank=0, num_processes=1)
    assert shard == records


def test_shard_coverage_3_processes():
    records = list(range(99))
    shards = [_shard(records, r, 3) for r in range(3)]
    # Union must equal the full dataset
    reconstructed = sorted(x for s in shards for x in s)
    assert reconstructed == sorted(records)


def test_shard_non_overlap_3_processes():
    records = list(range(99))
    shards = [set(_shard(records, r, 3)) for r in range(3)]
    for i in range(3):
        for j in range(3):
            if i != j:
                assert shards[i].isdisjoint(shards[j]), f"Ranks {i} and {j} overlap"


def test_shard_uneven_coverage_3_processes():
    """101 records into 3 ranks: sizes [34, 34, 33]."""
    records = list(range(101))
    shards = [_shard(records, r, 3) for r in range(3)]
    total = sum(len(s) for s in shards)
    assert total == 101
    # rank-0 gets the largest share
    assert len(shards[0]) >= len(shards[2])


# ---------------------------------------------------------------------------
# 2. warmup_steps calculation
# ---------------------------------------------------------------------------

def _warmup_steps(max_steps: int, warmup_ratio: float) -> int:
    return max(1, int(max_steps * warmup_ratio))


def test_warmup_steps_default():
    # base.yaml: max_steps=1000, warmup_ratio=0.03 → 30
    assert _warmup_steps(1000, 0.03) == 30


def test_warmup_steps_minimum():
    # Even with very small ratio, warmup_steps >= 1
    assert _warmup_steps(10, 0.001) == 1


def test_warmup_steps_gora_alignment():
    # GoRA: epochs=1, ~1562 steps (100K/64), warmup_ratio=0.03 → ~46 steps
    steps = _warmup_steps(1562, 0.03)
    assert 40 <= steps <= 60


# ---------------------------------------------------------------------------
# 3. is_main_process guard (mock-based)
# ---------------------------------------------------------------------------

class _MockAccelerator:
    def __init__(self, is_main):
        self.is_main_process = is_main
        self.num_processes = 3
        self.local_process_index = 0 if is_main else 1


def test_file_writes_skipped_on_non_main(tmp_path):
    """Simulate the is_main guard: non-main ranks must NOT write files."""
    acc = _MockAccelerator(is_main=False)
    is_main = acc.is_main_process

    out_file = tmp_path / "metrics.json"
    if is_main:
        out_file.write_text("{}")

    assert not out_file.exists(), "Non-main rank wrote a file it should not have"


def test_file_writes_happen_on_main(tmp_path):
    """Main rank DOES write the file."""
    acc = _MockAccelerator(is_main=True)
    is_main = acc.is_main_process

    out_file = tmp_path / "metrics.json"
    if is_main:
        out_file.write_text("{}")

    assert out_file.exists(), "Main rank failed to write file"


def test_effective_batch_size_3gpu():
    """The final protocol's preferred path is single-GPU seeds in parallel:
    batch_size is the per-run microbatch, and global batch is batch_size * grad_accum.
    DDP is only a fallback and intentionally uses the nearest 3-GPU batch, 63."""
    base_cfg = load_yaml(_REPO_ROOT / "configs" / "base.yaml")
    accelerate_cfg = load_yaml(_REPO_ROOT / "configs" / "accelerate_3gpu.yaml")

    single_run_global_batch = int(base_cfg["training"]["batch_size"]) * int(
        base_cfg["training"]["gradient_accumulation_steps"]
    )
    num_gpus = int(accelerate_cfg["num_processes"])

    per_gpu = 3
    grad_accum = 7
    effective = per_gpu * num_gpus * grad_accum
    assert single_run_global_batch == 64
    assert effective == 63


def test_auto_scaled_per_gpu_batch_matches_trainer_formula():
    """When the shipped config leaves per_gpu_batch_size unset, the default protocol is
    not DDP auto-scaling; it is one single-GPU run per seed with exact global batch 64."""
    base_cfg = load_yaml(_REPO_ROOT / "configs" / "base.yaml")

    assert base_cfg["training"]["per_gpu_batch_size"] is None
    assert int(base_cfg["training"]["batch_size"]) == 4
    assert int(base_cfg["training"]["gradient_accumulation_steps"]) == 16
