#!/usr/bin/env python
"""Aggregate metrics.json across seeds of the same experiment into mean/std.

Groups experiment output directories by their name with a trailing "_seedN"
suffix stripped (e.g. lora_r8_protocol_aligned_seed42/43/44 all belong to the
group "lora_r8_protocol_aligned"), then reports mean/std of the requested
metrics.json fields across the seeds found in each group.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FIELDS = [
    "final_metric",
    "final_eval_accuracy",
    "final_eval_loss",
    "best_metric",
    "actual_budget",
    "budget_ratio",
]

_SEED_SUFFIX = re.compile(r"_seed\d+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/covra_main_3seed")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help=f"metrics.json fields to aggregate (default: {DEFAULT_FIELDS})",
    )
    parser.add_argument("--json", default=None, help="Optional path to also write the summary as JSON")
    return parser.parse_args()


def group_name(experiment_name: str) -> str:
    return _SEED_SUFFIX.sub("", experiment_name)


def load_all_metrics(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for metrics_path in sorted(output_dir.glob("*/metrics.json")):
        try:
            records.append(json.loads(metrics_path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[summarize_seeds] skipping unreadable {metrics_path}: {exc}", file=sys.stderr)
    return records


def summarize(records: list[dict[str, Any]], fields: list[str]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        experiment = record.get("experiment", "")
        groups.setdefault(group_name(experiment), []).append(record)

    summary: dict[str, dict[str, Any]] = {}
    for name, group_records in groups.items():
        group_records.sort(key=lambda r: r.get("seed", 0))
        field_stats: dict[str, Any] = {}
        for field in fields:
            values = [r[field] for r in group_records if isinstance(r.get(field), (int, float))]
            if not values:
                field_stats[field] = {"mean": None, "std": None, "n": 0}
                continue
            field_stats[field] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        summary[name] = {
            "method": group_records[0].get("method"),
            "seeds": [r.get("seed") for r in group_records],
            "fields": field_stats,
        }
    return summary


def print_summary(summary: dict[str, dict[str, Any]], fields: list[str]) -> None:
    for name, group in sorted(summary.items()):
        print(f"\n{name} (method={group['method']}, seeds={group['seeds']})")
        for field in fields:
            stats = group["fields"][field]
            if stats["n"] == 0:
                print(f"  {field}: no data")
                continue
            print(f"  {field}: mean={stats['mean']:.6g} std={stats['std']:.6g} (n={stats['n']})")


def main() -> None:
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not output_dir.exists():
        raise SystemExit(f"Output dir does not exist: {output_dir}")

    records = load_all_metrics(output_dir)
    if not records:
        raise SystemExit(f"No metrics.json files found under {output_dir}/*/metrics.json")

    summary = summarize(records, args.fields)
    print_summary(summary, args.fields)

    if args.json:
        json_path = Path(args.json)
        json_path.write_text(json.dumps(summary, indent=2))
        print(f"\n[summarize_seeds] wrote {json_path}")


if __name__ == "__main__":
    main()
