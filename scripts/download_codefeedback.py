#!/usr/bin/env python3
"""Download and prepare the CodeFeedback dataset for dico_rank_experiments' §6.5
mixed math+code experiment (configs/dico/mixed_math_code_r8.yaml).

Currently supported datasets:
  - CodeFeedback-Filtered-Instruction (m-a-p/CodeFeedback-Filtered-Instruction, HuggingFace)

Usage
-----
# Download and subsample to 50K rows (default target: data/codefeedback/train.jsonl)
python scripts/download_codefeedback.py

# Custom sample count / seed
python scripts/download_codefeedback.py --limit 50000 --seed 42

# Custom data root
python scripts/download_codefeedback.py --data-dir /path/to/data

# Use a HuggingFace mirror (e.g., on servers that cannot reach hf.co directly)
python scripts/download_codefeedback.py --hf-endpoint https://hf-mirror.com

# Only check whether data exists, do not download
python scripts/download_codefeedback.py --check-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Dataset spec
# ---------------------------------------------------------------------------

CODEFEEDBACK_HF_REPO = "m-a-p/CodeFeedback-Filtered-Instruction"
CODEFEEDBACK_SPLIT = "train"
CODEFEEDBACK_RELPATH = Path("data/codefeedback/train.jsonl")
DEFAULT_LIMIT = 50000
DEFAULT_SEED = 42


def _field_map(row: dict) -> dict:
    """Map CodeFeedback-Filtered-Instruction HuggingFace fields (query/answer) to the
    project's question/answer SFT convention -- already the same names as GoRA's
    MetaMathQA convention, just query->question."""
    return {
        "question": row.get("query", ""),
        "answer": row.get("answer", ""),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_exists(target: Path) -> bool:
    if target.exists():
        try:
            lines = [l for l in target.read_text(encoding="utf-8").splitlines() if l.strip()]
            count = len(lines)
            print(f"OK: {target} already exists ({count:,} records). Skipping download.")
            return True
        except Exception as exc:
            print(f"WARNING: {target} exists but could not be parsed ({exc}). Will re-download.")
            return False
    return False


def _download_codefeedback(target: Path, limit: int, seed: int, hf_endpoint: str | None, hf_cache: str | None) -> None:
    """Download CodeFeedback from HuggingFace, subsample to `limit` rows with a fixed
    seed (for reproducibility across runs), and write JSONL to target."""
    print(f"Downloading {CODEFEEDBACK_HF_REPO} ({CODEFEEDBACK_SPLIT} split) ...")

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"  Using HF endpoint: {hf_endpoint}")

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: The 'datasets' package is required to download CodeFeedback.\n"
            "       Install it with:  pip install datasets\n"
            "       Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    kwargs: dict = {"split": CODEFEEDBACK_SPLIT}
    if hf_cache:
        kwargs["cache_dir"] = hf_cache

    try:
        ds = load_dataset(CODEFEEDBACK_HF_REPO, **kwargs)
    except Exception as exc:
        print(
            f"ERROR: Failed to download {CODEFEEDBACK_HF_REPO}: {exc}\n"
            "\n"
            "Possible remedies:\n"
            "  1. Check your network / proxy settings.\n"
            "  2. Use a mirror:  --hf-endpoint https://hf-mirror.com\n"
            "  3. Download manually and place at:\n"
            f"       {target}\n"
            "     with one JSON object per line containing 'question' and 'answer' fields.",
            file=sys.stderr,
        )
        sys.exit(1)

    total = len(ds)
    if int(limit) < total:
        rng = random.Random(int(seed))
        indices = sorted(rng.sample(range(total), int(limit)))
        ds = ds.select(indices)
        print(f"  Subsampled {int(limit):,} of {total:,} rows (seed={seed}).")
    else:
        print(f"  Using all {total:,} rows (limit={limit:,} >= dataset size).")

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Writing {len(ds):,} records to {target} ...")
    with target.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(json.dumps(_field_map(row), ensure_ascii=False) + "\n")

    written = sum(1 for l in target.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"OK: Wrote {written:,} records to {target}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare the CodeFeedback dataset for dico_rank_experiments."
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT),
        help="Project root for resolving relative data paths (default: repo root).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Number of rows to subsample (default: {DEFAULT_LIMIT}, matching the "
             "'CodeFeedback-50K' naming used in configs/dico/mixed_math_code_r8.yaml).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Subsample seed (default: 42).")
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help=(
            "Override HuggingFace endpoint URL, e.g. https://hf-mirror.com "
            "(can also be set via HF_ENDPOINT env var)."
        ),
    )
    parser.add_argument(
        "--hf-cache",
        default=os.environ.get("HF_DATASETS_CACHE"),
        help="HuggingFace datasets cache directory (can also be set via HF_DATASETS_CACHE env var).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check whether the local data files exist; do not download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    target = data_root / CODEFEEDBACK_RELPATH

    if _check_exists(target):
        return

    if args.check_only:
        print(
            f"ERROR: {target} does not exist.\n"
            "       Run without --check-only to download it.",
            file=sys.stderr,
        )
        sys.exit(1)

    _download_codefeedback(target, args.limit, args.seed, args.hf_endpoint, args.hf_cache)


if __name__ == "__main__":
    main()
