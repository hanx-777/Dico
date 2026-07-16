from __future__ import annotations

from typing import Mapping


def summarize_rank_migration(before: Mapping[str, int], after: Mapping[str, int]) -> list[dict[str, int | str]]:
    rows = []
    for name in sorted(set(before) | set(after)):
        old = int(before.get(name, 0))
        new = int(after.get(name, 0))
        rows.append({"module_name": name, "before": old, "after": new, "delta": new - old})
    return rows
