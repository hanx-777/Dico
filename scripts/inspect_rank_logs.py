#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico_rank.path_utils import resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_history", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = resolve_project_path(ROOT, args.rank_history)
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    latest = max(int(row["step"]) for row in rows) if rows else 0
    for row in rows:
        if int(row["step"]) == latest:
            print(f"{row['module_name']}\t{row['active_rank']}\tparams={row['total_active_params']}")


if __name__ == "__main__":
    main()
