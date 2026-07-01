#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico_rank.path_utils import resolve_project_path


METHOD_LABELS = {
    "lora": "LoRA",
    "dico_pre": "DiCo-Pre",
    "dico_dynamic": "DiCo-Dynamic",
    "dico_predynamic": "DiCo-PreDynamic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


def load_metrics(output_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(output_dir.glob("*/metrics.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_csv(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
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
        "total_params",
        "total_active_rank",
        "budget_error",
        "budget_error_ratio",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _delta_loss(method_loss, lora_loss):
    if method_loss is None or lora_loss in {None, 0}:
        return ""
    return f"{(float(lora_loss) - float(method_loss)) / float(lora_loss) * 100:.2f}%"


def write_md(output_dir: Path, rows: list[dict]) -> None:
    by_key = {(row.get("method"), int(row.get("rank", 0))): row for row in rows}
    lines = [
        "| Method | r=4 Internal GSM8K Acc | r=8 Internal GSM8K Acc | r=4 Loss | r=8 Loss | r=4 Params | r=8 Params | Delta Loss r=4 | Delta Loss r=8 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lora4 = by_key.get(("lora", 4), {}).get("final_eval_loss")
    lora8 = by_key.get(("lora", 8), {}).get("final_eval_loss")
    for method in ["lora", "dico_pre", "dico_dynamic", "dico_predynamic"]:
        r4 = by_key.get((method, 4), {})
        r8 = by_key.get((method, 8), {})
        lines.append(
            "| {label} | {r4_acc} | {r8_acc} | {r4_loss} | {r8_loss} | {r4_params} | {r8_params} | {d4} | {d8} |".format(
                label=METHOD_LABELS[method],
                r4_acc=r4.get("final_eval_accuracy", ""),
                r8_acc=r8.get("final_eval_accuracy", ""),
                r4_loss=r4.get("final_eval_loss", ""),
                r8_loss=r8.get("final_eval_loss", ""),
                r4_params=r4.get("total_params", r4.get("actual_budget", "")),
                r8_params=r8.get("total_params", r8.get("actual_budget", "")),
                d4=_delta_loss(r4.get("final_eval_loss"), lora4),
                d8=_delta_loss(r8.get("final_eval_loss"), lora8),
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
