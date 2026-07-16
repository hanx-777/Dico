#!/usr/bin/env python3
"""Download and prepare datasets for dico_rank_experiments.

Currently supported datasets:
  - MetaMathQA-100K  (meta-math/MetaMathQA, HuggingFace)

Usage
-----
# Download MetaMathQA-100K (default target: data/metamathqa/train.jsonl)
python scripts/download_data.py

# Custom data root
python scripts/download_data.py --data-dir /path/to/data

# Use a HuggingFace mirror (e.g., on servers that cannot reach hf.co directly)
python scripts/download_data.py --hf-endpoint https://hf-mirror.com

# Only check whether data exists, do not download
python scripts/download_data.py --check-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Dataset spec
# ---------------------------------------------------------------------------

METAMATHQA_HF_REPO = "meta-math/MetaMathQA"
METAMATHQA_SPLIT = "train"
METAMATHQA_RELPATH = Path("data/metamathqa/train.jsonl")


def _field_map(row: dict) -> dict:
    """Map MetaMathQA HuggingFace fields to the project's question/answer convention."""
    return {
        "question": row.get("query", row.get("question", "")),
        "answer": row.get("response", row.get("answer", "")),
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


def _download_metamathqa(target: Path, hf_endpoint: str | None, hf_cache: str | None) -> None:
    """Download MetaMathQA from HuggingFace and write JSONL to target."""
    print(f"Downloading {METAMATHQA_HF_REPO} ({METAMATHQA_SPLIT} split) ...")

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"  Using HF endpoint: {hf_endpoint}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        print(
            "ERROR: The 'datasets' package is required to download MetaMathQA.\n"
            "       Install it with:  pip install datasets\n"
            "       Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    kwargs: dict = {"split": METAMATHQA_SPLIT}
    if hf_cache:
        kwargs["cache_dir"] = hf_cache

    try:
        ds = load_dataset(METAMATHQA_HF_REPO, **kwargs)
    except Exception as exc:
        print(
            f"ERROR: Failed to download {METAMATHQA_HF_REPO}: {exc}\n"
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

    target.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Writing {len(ds):,} records to {target} ...")
    with target.open("w", encoding="utf-8") as fh:
        for row in ds:
            fh.write(json.dumps(_field_map(row), ensure_ascii=False) + "\n")

    # Verify write
    written = sum(1 for l in target.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"OK: Wrote {written:,} records to {target}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for dico_rank_experiments."
    )
    parser.add_argument(
        "--data-dir",
        default=str(ROOT),
        help="Project root for resolving relative data paths (default: repo root).",
    )
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
    target = data_root / METAMATHQA_RELPATH

    if _check_exists(target):
        return

    if args.check_only:
        print(
            f"ERROR: {target} does not exist.\n"
            "       Run without --check-only to download it.",
            file=sys.stderr,
        )
        sys.exit(1)

    _download_metamathqa(target, args.hf_endpoint, args.hf_cache)


if __name__ == "__main__":
    main()
