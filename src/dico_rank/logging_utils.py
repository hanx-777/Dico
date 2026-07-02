from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from dico_rank.dynamic_allocation import rank_distance
from dico_rank.rank_budget import BudgetManager
from dico_rank.utils import append_jsonl, ensure_dir


RANK_HISTORY_FIELDS = [
    "step",
    "module_name",
    "active_rank",
    "max_rank",
    "module_score",
    "total_active_rank",
    "total_active_params",
    "target_budget",
    "budget_error_ratio",
    "rank_distance_from_initial",
    "rank_distance_from_preallocation",
    "latest_mid_eval_loss",
]


def init_rank_history(path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=RANK_HISTORY_FIELDS).writeheader()


def append_rank_history(
    path: Path | str,
    step: int,
    allocation: Mapping[str, int],
    max_rank: int,
    module_scores: Mapping[str, float],
    budget_manager: BudgetManager,
    target_budget: int,
    initial_allocation: Mapping[str, int],
    preallocation: Mapping[str, int] | None = None,
    latest_mid_eval_loss: float | None = None,
) -> None:
    budget = budget_manager.describe(allocation, target_budget)
    distance_initial = rank_distance(allocation, initial_allocation)
    distance_pre = rank_distance(allocation, preallocation) if preallocation is not None else None
    with Path(path).open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANK_HISTORY_FIELDS)
        for name, rank in allocation.items():
            writer.writerow(
                {
                    "step": int(step),
                    "module_name": name,
                    "active_rank": int(rank),
                    "max_rank": int(max_rank),
                    "module_score": module_scores.get(name),
                    "total_active_rank": budget["total_active_rank"],
                    "total_active_params": budget["actual_budget_paramcount"],
                    "target_budget": budget["target_budget_paramcount"],
                    "budget_error_ratio": budget["budget_error_ratio"],
                    "rank_distance_from_initial": distance_initial,
                    "rank_distance_from_preallocation": distance_pre,
                    "latest_mid_eval_loss": latest_mid_eval_loss,
                }
            )


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def with_log_defaults(payload: Mapping[str, Any], default_event: str) -> dict[str, Any]:
    row = dict(payload)
    row.setdefault("timestamp", current_timestamp())
    row.setdefault("event", default_event)
    return row


def log_train(path: Path | str, payload: Mapping[str, Any]) -> None:
    append_jsonl(path, with_log_defaults(payload, "train"))


def log_eval(path: Path | str, payload: Mapping[str, Any]) -> None:
    append_jsonl(path, with_log_defaults(payload, "eval"))
