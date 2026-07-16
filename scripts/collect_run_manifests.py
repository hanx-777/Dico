#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_SEED_SUFFIX = re.compile(r"_seed\d+$")

DEFAULT_FIELDS = [
    "final_metric",
    "final_eval_accuracy",
    "budget_error",
    "budget_error_ratio",
    "actual_budget",
    "requires_grad",
    "active_final",
    "active_peak",
    "global_batch_size",
    "optimizer_steps",
    "run_elapsed_sec",
]


def group_name(experiment_name: str) -> str:
    return _SEED_SUFFIX.sub("", str(experiment_name))


def _get_nested(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


FIELD_PATHS = {
    "final_metric": "metrics.final_metric",
    "final_eval_accuracy": "metrics.final_eval_accuracy",
    "budget_error": "budget.budget_error",
    "budget_error_ratio": "budget.budget_error_ratio",
    "actual_budget": "budget.actual_budget",
    "requires_grad": "parameter_counts.requires_grad",
    "active_final": "parameter_counts.active_final",
    "active_peak": "parameter_counts.active_peak",
    "requires_grad_params": "parameter_metrics.requires_grad_params",
    "peak_active_params": "parameter_metrics.peak_active_params",
    "final_active_params": "parameter_metrics.final_active_params",
    "budget_target": "parameter_metrics.budget_target",
    "budget_actual": "parameter_metrics.budget_actual",
    "global_batch_size": "global_batch_size",
    "optimizer_steps": "optimizer_steps",
    "run_elapsed_sec": "timing.run_elapsed_sec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate run_manifest.json files by seed group.")
    parser.add_argument("--output-dir", default="outputs/e01_llama3_r8_main")
    parser.add_argument("--json-output", default="reports/run_manifest_summary.json")
    parser.add_argument("--markdown-output", default="reports/run_manifest_summary.md")
    parser.add_argument("--fields", nargs="+", default=DEFAULT_FIELDS)
    return parser.parse_args()


def load_manifests(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(output_dir.glob("*/run_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[collect_run_manifests] skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        payload["_manifest_path"] = str(path)
        records.append(payload)
    return records


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def summarize(records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        name = group_name(str(record.get("experiment_name", "")))
        groups.setdefault(name, []).append(record)

    result: dict[str, Any] = {}
    for name, group_records in sorted(groups.items()):
        group_records.sort(key=lambda row: int(row.get("seed", 0)))
        field_stats = {}
        for field in fields:
            path = FIELD_PATHS.get(field, field)
            values = [
                float(value)
                for record in group_records
                for value in [_get_nested(record, path)]
                if isinstance(value, (int, float))
            ]
            field_stats[field] = _stats(values)
        result[name] = {
            "method": group_records[0].get("method"),
            "rank": group_records[0].get("rank"),
            "seeds": [int(row.get("seed")) for row in group_records if row.get("seed") is not None],
            "n": len(group_records),
            "manifest_paths": [row.get("_manifest_path") for row in group_records],
            "fields": field_stats,
        }
    return result


def _fmt_mean_std(stats: dict[str, Any]) -> str:
    if stats.get("n", 0) == 0 or stats.get("mean") is None:
        return "-"
    return f"{stats['mean']:.6g}±{stats['std']:.6g}"


def _empty_stats() -> dict[str, Any]:
    return {"mean": None, "std": None, "min": None, "max": None, "n": 0}


def _field(fields: dict[str, dict[str, Any]], *names: str) -> dict[str, Any]:
    for name in names:
        stats = fields.get(name)
        if stats is not None:
            return stats
    return _empty_stats()


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        (
            "| group | method | seeds | final_metric mean±std | budget_error mean | "
            "requires_grad mean | active_final mean | active_peak mean | global_batch |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, group in sorted(summary.items()):
        fields = group["fields"]
        final_metric_stats = _field(fields, "final_metric")
        budget_error_stats = _field(fields, "budget_error")
        requires_grad_stats = _field(fields, "requires_grad", "requires_grad_params")
        active_final_stats = _field(fields, "active_final", "final_active_params")
        active_peak_stats = _field(fields, "active_peak", "peak_active_params")
        global_batch_stats = _field(fields, "global_batch_size")
        budget_error = budget_error_stats["mean"] if budget_error_stats["n"] else None
        requires_grad = requires_grad_stats["mean"] if requires_grad_stats["n"] else None
        active_final = active_final_stats["mean"] if active_final_stats["n"] else None
        active_peak = active_peak_stats["mean"] if active_peak_stats["n"] else None
        global_batch = global_batch_stats["mean"] if global_batch_stats["n"] else None
        lines.append(
            f"| {name} | {group.get('method')} | {','.join(str(seed) for seed in group['seeds'])} | "
            f"{_fmt_mean_std(final_metric_stats)} | "
            f"{budget_error if budget_error is not None else '-'} | "
            f"{requires_grad if requires_grad is not None else '-'} | "
            f"{active_final if active_final is not None else '-'} | "
            f"{active_peak if active_peak is not None else '-'} | "
            f"{global_batch if global_batch is not None else '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not output_dir.exists():
        raise SystemExit(f"Output dir does not exist: {output_dir}")
    records = load_manifests(output_dir)
    if not records:
        raise SystemExit(f"No run_manifest.json files found under {output_dir}/*/run_manifest.json")
    groups = summarize(records, list(args.fields))
    payload = {"output_dir": str(output_dir), "run_count": len(records), "groups": groups}
    json_path = Path(args.json_output)
    md_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(groups), encoding="utf-8")
    print(f"[collect_run_manifests] wrote {json_path}")
    print(f"[collect_run_manifests] wrote {md_path}")


if __name__ == "__main__":
    main()
