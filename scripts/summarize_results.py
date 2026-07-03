#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico_rank.path_utils import resolve_project_path


METHOD_LABELS = {
    "lora": "LoRA",
    "dico_pre": "DiCo-eta",
    "dico_dynamic": "DiCo-Dynamic",
    "dico_predynamic": "DiCo-D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


def load_metrics(output_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(output_dir.glob("*/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        run_name = path.parent.name
        experiment = str(row.get("experiment") or run_name)
        seed = row.get("seed")
        experiment_base = experiment
        if "__seed" in run_name:
            experiment_base, seed_text = run_name.rsplit("__seed", 1)
            experiment = experiment_base
            if seed is None:
                try:
                    seed = int(seed_text)
                except ValueError:
                    seed = seed_text
        row["run_name"] = run_name
        row["experiment"] = experiment
        row["experiment_base"] = experiment_base
        row["seed"] = seed
        row["target_paramcount"] = row.get("target_budget_paramcount", row.get("target_budget"))
        row["actual_paramcount"] = row.get(
            "actual_budget_paramcount",
            row.get("actual_budget", row.get("total_params")),
        )
        row["ratio_paramcount"] = row.get("budget_ratio_paramcount", row.get("budget_ratio"))
        row["rank_beyond_evidence_ratio"] = (
            (row.get("evidence_relaxation") or {}).get("rank_beyond_evidence_ratio")
            if isinstance(row.get("evidence_relaxation"), dict)
            else row.get("rank_beyond_evidence_ratio")
        )
        rows.append(row)
    return rows


def _is_multiseed(rows: list[dict]) -> bool:
    return any(row.get("seed") is not None or "__seed" in str(row.get("run_name", "")) for row in rows)


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("experiment_base") or row.get("experiment")), []).append(row)

    aggregate = []
    metrics = [
        ("final_eval_accuracy", "accuracy"),
        ("final_eval_loss", "loss"),
        ("actual_paramcount", "actual_paramcount"),
        ("ratio_paramcount", "budget_ratio_paramcount"),
        ("rank_beyond_evidence_ratio", "rank_beyond_evidence_ratio"),
    ]
    for experiment in sorted(grouped):
        group = grouped[experiment]
        first = group[0]
        seeds = sorted(str(row.get("seed")) for row in group if row.get("seed") is not None)
        out: dict[str, Any] = {
            "experiment": experiment,
            "method": first.get("method"),
            "rank": first.get("rank"),
            "n": len(group),
            "seeds": " ".join(seeds),
        }
        for source_key, label in metrics:
            values = [_to_float(row.get(source_key)) for row in group]
            clean = [value for value in values if value is not None]
            out[f"{label}_mean"] = _mean(clean)
            out[f"{label}_std"] = _std(clean)
        aggregate.append(out)
    return aggregate


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_csv(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "experiment",
        "experiment_base",
        "seed",
        "method",
        "rank",
        "evaluation_protocol",
        "evaluation_prompt_style",
        "eval_sample_count",
        "final_eval_loss",
        "best_eval_loss",
        "final_eval_accuracy",
        "final_exact_match",
        "eval_correct",
        "eval_total",
        "final_metric_name",
        "final_metric",
        "best_metric",
        "target_paramcount",
        "actual_paramcount",
        "ratio_paramcount",
        "total_params",
        "total_active_rank",
        "rank_beyond_evidence_ratio",
        "budget_error",
        "budget_error_ratio",
    ]
    _write_csv(output_dir / "summary_per_run.csv", rows, fields)
    if _is_multiseed(rows):
        aggregate_fields = [
            "experiment",
            "method",
            "rank",
            "n",
            "seeds",
            "accuracy_mean",
            "accuracy_std",
            "loss_mean",
            "loss_std",
            "actual_paramcount_mean",
            "actual_paramcount_std",
            "budget_ratio_paramcount_mean",
            "budget_ratio_paramcount_std",
            "rank_beyond_evidence_ratio_mean",
            "rank_beyond_evidence_ratio_std",
        ]
        _write_csv(output_dir / "summary.csv", _aggregate_rows(rows), aggregate_fields)
    else:
        _write_csv(output_dir / "summary.csv", rows, fields)


def _delta_loss(method_loss, lora_loss):
    if method_loss is None or lora_loss in {None, 0}:
        return ""
    return f"{(float(lora_loss) - float(method_loss)) / float(lora_loss) * 100:.2f}%"


def write_md(output_dir: Path, rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row.get("method"), int(row.get("rank", 0))), []).append(row)

    def agg(key, field):
        vals = [_to_float(r.get(field)) for r in groups.get(key, [])]
        return _mean([v for v in vals if v is not None])

    def fmt(val):
        return f"{val:.4f}" if isinstance(val, float) else str(val) if val is not None else ""

    lines = [
        "| Method | r=4 Internal GSM8K Acc | r=8 Internal GSM8K Acc | r=4 Loss | r=8 Loss | r=4 Params | r=8 Params | Delta Loss r=4 | Delta Loss r=8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lora4 = agg(("lora", 4), "final_eval_loss")
    lora8 = agg(("lora", 8), "final_eval_loss")
    for method in ["lora", "dico_pre", "dico_dynamic", "dico_predynamic"]:
        lines.append(
            "| {label} | {r4_acc} | {r8_acc} | {r4_loss} | {r8_loss} | {r4_params} | {r8_params} | {d4} | {d8} |".format(
                label=METHOD_LABELS[method],
                r4_acc=fmt(agg((method, 4), "final_eval_accuracy")),
                r8_acc=fmt(agg((method, 8), "final_eval_accuracy")),
                r4_loss=fmt(agg((method, 4), "final_eval_loss")),
                r8_loss=fmt(agg((method, 8), "final_eval_loss")),
                r4_params=fmt(agg((method, 4), "actual_paramcount")),
                r8_params=fmt(agg((method, 8), "actual_paramcount")),
                d4=_delta_loss(agg((method, 4), "final_eval_loss"), lora4),
                d8=_delta_loss(agg((method, 8), "final_eval_loss"), lora8),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(ROOT, args.output_dir)
    rows = load_metrics(output_dir)
    write_csv(output_dir, rows)
    write_md(output_dir, rows)


if __name__ == "__main__":
    main()
