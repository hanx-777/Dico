from __future__ import annotations

import csv
from pathlib import Path


def read_rank_history(path: Path | str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def latest_rank_by_module(path: Path | str) -> dict[str, int]:
    rows = read_rank_history(path)
    latest_step = max(int(row["step"]) for row in rows) if rows else 0
    return {
        row["module_name"]: int(row["active_rank"])
        for row in rows
        if int(row["step"]) == latest_step
    }
