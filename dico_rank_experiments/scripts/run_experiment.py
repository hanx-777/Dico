#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico_rank.config import apply_overrides, load_yaml
from dico_rank.path_utils import resolve_project_path
from dico_rank.trainer import train
from dico_rank.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DiCo rank experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_yaml(resolve_project_path(ROOT, args.config))
    config = apply_overrides(config, args.override)
    metrics = train(config)
    print(metrics)


if __name__ == "__main__":
    main()
