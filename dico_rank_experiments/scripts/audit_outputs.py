#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico_rank.path_utils import resolve_project_path


EXPECTED_EXPERIMENTS = {
    "lora_r4": {"method": "lora", "rank": 4},
    "lora_r8": {"method": "lora", "rank": 8},
    "dico_pre_r4": {"method": "dico_pre", "rank": 4},
    "dico_pre_r8": {"method": "dico_pre", "rank": 8},
    "dico_dynamic_r4": {"method": "dico_dynamic", "rank": 4, "move_ratio": 0.20},
    "dico_dynamic_r8": {"method": "dico_dynamic", "rank": 8, "move_ratio": 0.20},
    "dico_predynamic_r4": {"method": "dico_predynamic", "rank": 4, "move_ratio": 0.10},
    "dico_predynamic_r8": {"method": "dico_predynamic", "rank": 8, "move_ratio": 0.10},
}

COMMON_FILES = [
    "config_resolved.yaml",
    "metrics.json",
    "budget.json",
    "rank_allocation_initial.json",
    "rank_allocation_final.json",
    "rank_history.csv",
    "train_log.jsonl",
    "eval_log.jsonl",
]

PREALLOC_METHODS = {"dico_pre", "dico_predynamic"}
DYNAMIC_METHODS = {"dico_dynamic", "dico_predynamic"}
BUDGET_WARNING_THRESHOLD = 0.01


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _append_missing_file_criticals(exp_dir: Path, experiment: str, critical: list[str]) -> None:
    for filename in COMMON_FILES:
        if not (exp_dir / filename).exists():
            critical.append(f"{experiment}: missing required file {filename}")


def _audit_config(
    experiment: str,
    expected: dict[str, Any],
    config: dict[str, Any],
    critical: list[str],
) -> None:
    method = config.get("method")
    rank = config.get("rank")
    if method != expected["method"]:
        critical.append(f"{experiment}: config method={method!r}, expected {expected['method']!r}")
    if int(rank) != int(expected["rank"]):
        critical.append(f"{experiment}: config rank={rank!r}, expected {expected['rank']!r}")

    dynamic_cfg = config.get("dynamic", {})
    if expected["method"] in DYNAMIC_METHODS:
        if not dynamic_cfg.get("enabled", False):
            critical.append(f"{experiment}: dynamic.enabled must be true")
        expected_move = float(expected["move_ratio"])
        actual_move = _as_float(dynamic_cfg.get("move_ratio"))
        if abs(actual_move - expected_move) > 1e-9:
            critical.append(f"{experiment}: move_ratio={actual_move}, expected {expected_move}")
    else:
        if dynamic_cfg.get("enabled", False):
            critical.append(f"{experiment}: dynamic.enabled should be false")

    if expected["method"] in PREALLOC_METHODS:
        pre_cfg = config.get("preallocation", {})
        if pre_cfg.get("aggregation_mode") != "weighted_topk":
            critical.append(f"{experiment}: preallocation.aggregation_mode must be weighted_topk")
        if pre_cfg.get("atom_weight_normalization") != "none":
            critical.append(f"{experiment}: preallocation.atom_weight_normalization must be none")
        if pre_cfg.get("use_cost_aware_allocation") is not True:
            critical.append(f"{experiment}: preallocation.use_cost_aware_allocation must be true")


def _audit_budget(
    experiment: str,
    budget: dict[str, Any],
    critical: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    ratio = _as_float(budget.get("budget_error_ratio"))
    over_budget = bool(budget.get("over_budget", False))
    if over_budget:
        critical.append(f"{experiment}: actual_budget exceeds target_budget")
    if ratio > BUDGET_WARNING_THRESHOLD:
        warnings.append(f"{experiment}: budget_error_ratio={ratio:.6f} exceeds {BUDGET_WARNING_THRESHOLD}")
    if budget.get("warning"):
        warnings.append(f"{experiment}: budget warning: {budget['warning']}")
    return {
        "target_budget": budget.get("target_budget"),
        "actual_budget": budget.get("actual_budget"),
        "budget_error_ratio": ratio,
        "over_budget": over_budget,
        "budget_warning": budget.get("warning"),
    }


def _audit_evaluation(
    experiment: str,
    exp_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    evaluation_cfg = config.get("evaluation", {})
    if not evaluation_cfg.get("compute_accuracy", True):
        return {"evaluation_enabled": False}

    protocol = metrics.get("evaluation_protocol")
    prompt_style = metrics.get("evaluation_prompt_style")
    sample_count = _as_int(metrics.get("eval_sample_count", metrics.get("eval_total")))
    eval_total = _as_int(metrics.get("eval_total"))
    eval_limit = config.get("data", {}).get("eval_limit")
    expected_samples = _as_int(eval_limit) if eval_limit is not None else eval_total
    scope = f"{expected_samples}-sample subset" if eval_limit is not None else "full eval set"

    if protocol is None:
        warnings.append(f"{experiment}: metrics.json missing evaluation_protocol")
    elif protocol != "internal_zero_shot":
        warnings.append(f"{experiment}: evaluation_protocol={protocol!r}, expected 'internal_zero_shot'")
    if metrics.get("final_eval_accuracy") is None:
        warnings.append(f"{experiment}: metrics.json missing final_eval_accuracy")
    if metrics.get("eval_correct") is None:
        warnings.append(f"{experiment}: metrics.json missing eval_correct")
    if metrics.get("eval_total") is None:
        warnings.append(f"{experiment}: metrics.json missing eval_total")
    if eval_total and expected_samples and eval_total != expected_samples:
        warnings.append(f"{experiment}: eval_total={eval_total}, expected {expected_samples} from config")
    if sample_count and eval_total and sample_count != eval_total:
        warnings.append(f"{experiment}: eval_sample_count={sample_count}, eval_total={eval_total}")

    prediction_path = exp_dir / "eval_predictions.jsonl"
    prediction_rows = _read_jsonl(prediction_path)
    if not prediction_path.exists():
        warnings.append(f"{experiment}: evaluation.compute_accuracy=true but eval_predictions.jsonl is missing")
    elif eval_total and len(prediction_rows) != eval_total:
        warnings.append(f"{experiment}: eval_predictions.jsonl has {len(prediction_rows)} rows but eval_total={eval_total}")

    return {
        "evaluation_enabled": True,
        "evaluation_protocol": protocol,
        "evaluation_prompt_style": prompt_style,
        "eval_sample_count": sample_count,
        "eval_total": eval_total,
        "eval_scope": scope,
        "prediction_rows": len(prediction_rows),
    }


def _rank_map(payload: Any) -> dict[str, int]:
    if isinstance(payload, dict) and "rank_allocation" in payload:
        payload = payload["rank_allocation"]
    if not isinstance(payload, dict):
        return {}
    return {str(name): int(rank) for name, rank in payload.items()}


def _is_uniform(allocation: dict[str, int]) -> bool:
    return len(set(allocation.values())) <= 1 if allocation else False


def _audit_preallocation_metadata(
    experiment: str,
    output_dir: Path,
    initial_payload: dict[str, Any],
    metrics: dict[str, Any],
    critical: list[str],
) -> dict[str, Any]:
    metadata = metrics.get("preallocation") or {}
    aggregation_mode = initial_payload.get("aggregation_mode") or metadata.get("aggregation_mode")
    atom_norm = initial_payload.get("atom_weight_normalization") or metadata.get("atom_weight_normalization")
    cost_aware = initial_payload.get("use_cost_aware_allocation")
    if cost_aware is None:
        cost_aware = metadata.get("use_cost_aware_allocation")
    atom_mode = initial_payload.get("atom_mode") or metadata.get("atom_mode")
    limitation = initial_payload.get("atom_mode_limitation") or metadata.get("atom_mode_limitation")
    preallocation_path = metadata.get("preallocation_path")

    if aggregation_mode != "weighted_topk":
        critical.append(f"{experiment}: missing/invalid aggregation_mode={aggregation_mode!r}")
    if atom_norm != "none":
        critical.append(f"{experiment}: missing/invalid atom_weight_normalization={atom_norm!r}")
    if cost_aware is not True:
        critical.append(f"{experiment}: use_cost_aware_allocation must be true")
    if not initial_payload.get("module_logs") and not metadata.get("module_logs"):
        critical.append(f"{experiment}: missing module_logs for preallocation audit")
    if atom_mode == "module_proxy" and not limitation:
        critical.append(f"{experiment}: module_proxy atom_mode requires atom_mode_limitation")
    if not metadata:
        critical.append(f"{experiment}: metrics.json missing nested preallocation metadata")
    if preallocation_path:
        path = Path(preallocation_path)
        if not path.is_absolute():
            path = output_dir.parent / path
        if not path.exists():
            critical.append(f"{experiment}: preallocation_path does not exist: {preallocation_path}")
    return {
        "aggregation_mode": aggregation_mode,
        "atom_weight_normalization": atom_norm,
        "use_cost_aware_allocation": cost_aware,
        "atom_mode": atom_mode,
        "atom_mode_limitation": limitation,
        "preallocation_source": metadata.get("preallocation_source"),
        "preallocation_path": preallocation_path,
    }


def _audit_dynamic(
    experiment: str,
    exp_dir: Path,
    method: str,
    config: dict[str, Any],
    rank_history: list[dict[str, str]],
    critical: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    dynamic_path = exp_dir / "dynamic_adjustments.jsonl"
    adjustments = _read_jsonl(dynamic_path)
    if method not in DYNAMIC_METHODS:
        return {"dynamic_adjustment_count": len(adjustments), "dynamic_steps": [row.get("step") for row in adjustments]}

    if not dynamic_path.exists():
        critical.append(f"{experiment}: missing dynamic_adjustments.jsonl")
        return {"dynamic_adjustment_count": 0, "dynamic_steps": []}
    if not adjustments:
        critical.append(f"{experiment}: dynamic_adjustments.jsonl is empty")

    max_steps = _as_int(config.get("training", {}).get("max_steps"))
    ratios = config.get("dynamic", {}).get("update_ratios", [0.2, 0.4, 0.6])
    expected_steps = {max(1, math.ceil(float(ratio) * max_steps)) for ratio in ratios} if max_steps else set()
    actual_steps = {_as_int(row.get("step")) for row in adjustments}
    unexpected = sorted(step for step in actual_steps if expected_steps and step not in expected_steps)
    if unexpected:
        critical.append(f"{experiment}: dynamic adjustments at unexpected steps {unexpected}; expected {sorted(expected_steps)}")

    if method == "dico_predynamic":
        if any(row.get("rank_distance_from_preallocation") is None for row in adjustments):
            critical.append(f"{experiment}: missing rank_distance_from_preallocation in dynamic adjustments")
        history_has_distance = any(
            row.get("rank_distance_from_preallocation") not in {None, ""}
            for row in rank_history
        )
        if not history_has_distance:
            critical.append(f"{experiment}: rank_history.csv missing rank_distance_from_preallocation values")
    elif method == "dico_dynamic":
        if any(row.get("rank_distance_from_initial") is None for row in adjustments):
            warnings.append(f"{experiment}: dynamic adjustments missing rank_distance_from_initial")

    return {
        "dynamic_adjustment_count": len(adjustments),
        "dynamic_steps": sorted(actual_steps),
        "expected_dynamic_steps": sorted(expected_steps),
    }


def _audit_summary_files(output_dir: Path, warnings: list[str]) -> dict[str, bool]:
    summary = {}
    for filename in ["summary.csv", "summary.md"]:
        exists = (output_dir / filename).exists()
        summary[filename] = exists
        if not exists:
            warnings.append(f"missing {filename}; run scripts/summarize_results.py after experiments")
    return summary


def audit_outputs(output_dir: Path | str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    critical: list[str] = []
    warnings: list[str] = []
    experiments: dict[str, Any] = {}

    for experiment, expected in EXPECTED_EXPERIMENTS.items():
        exp_dir = output_dir / experiment
        if not exp_dir.exists():
            critical.append(f"{experiment}: Missing experiment directory {exp_dir}")
            experiments[experiment] = {"status": "missing"}
            continue

        _append_missing_file_criticals(exp_dir, experiment, critical)
        exp_report: dict[str, Any] = {"status": "present"}

        try:
            config = _read_yaml(exp_dir / "config_resolved.yaml")
            _audit_config(experiment, expected, config, critical)
            exp_report["method"] = config.get("method")
            exp_report["rank"] = config.get("rank")
        except Exception as exc:
            critical.append(f"{experiment}: could not read config_resolved.yaml: {exc}")
            config = {}

        try:
            metrics = _read_json(exp_dir / "metrics.json")
            exp_report["metrics_method"] = metrics.get("method")
            exp_report["metrics_rank"] = metrics.get("rank")
            exp_report["final_eval_accuracy"] = metrics.get("final_eval_accuracy")
        except Exception as exc:
            critical.append(f"{experiment}: could not read metrics.json: {exc}")
            metrics = {}

        exp_report.update(_audit_evaluation(experiment, exp_dir, config, metrics, warnings))

        try:
            budget = _read_json(exp_dir / "budget.json")
            exp_report.update(_audit_budget(experiment, budget, critical, warnings))
        except Exception as exc:
            critical.append(f"{experiment}: could not read budget.json: {exc}")

        try:
            initial_payload = _read_json(exp_dir / "rank_allocation_initial.json")
            initial_allocation = _rank_map(initial_payload)
            exp_report["initial_total_rank"] = sum(initial_allocation.values())
            if expected["method"] == "dico_dynamic" and not _is_uniform(initial_allocation):
                critical.append(f"{experiment}: DiCo-Dynamic initial allocation must be uniform")
            if expected["method"] in PREALLOC_METHODS:
                exp_report["preallocation"] = _audit_preallocation_metadata(
                    experiment,
                    output_dir,
                    initial_payload,
                    metrics,
                    critical,
                )
        except Exception as exc:
            critical.append(f"{experiment}: could not read rank_allocation_initial.json: {exc}")
            initial_allocation = {}

        try:
            final_payload = _read_json(exp_dir / "rank_allocation_final.json")
            final_allocation = _rank_map(final_payload)
            exp_report["final_total_rank"] = sum(final_allocation.values())
        except Exception as exc:
            critical.append(f"{experiment}: could not read rank_allocation_final.json: {exc}")

        rank_history = _read_csv(exp_dir / "rank_history.csv")
        if not rank_history:
            critical.append(f"{experiment}: rank_history.csv is empty or unreadable")

        exp_report.update(
            _audit_dynamic(
                experiment,
                exp_dir,
                expected["method"],
                config,
                rank_history,
                critical,
                warnings,
            )
        )
        experiments[experiment] = exp_report

    summary_files = _audit_summary_files(output_dir, warnings)
    status = "fail" if critical else ("warning" if warnings else "pass")
    return {
        "status": status,
        "critical": critical,
        "warnings": warnings,
        "experiments": experiments,
        "summary_files": summary_files,
    }


def write_report(output_dir: Path | str, report: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# DiCo Rank Experiments Audit Report",
        "",
        f"Status: **{report['status'].upper()}**",
        "",
        "## Critical",
    ]
    if report["critical"]:
        lines.extend([f"- {item}" for item in report["critical"]])
    else:
        lines.append("- None")
    lines.extend(["", "## Warnings"])
    if report["warnings"]:
        lines.extend([f"- {item}" for item in report["warnings"]])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Experiments",
            "",
            "| Experiment | Method | Rank | Eval Protocol | Eval Scope | Accuracy | Budget Error Ratio | Dynamic Steps |",
            "| --- | --- | ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for name, row in report["experiments"].items():
        lines.append(
            "| {name} | {method} | {rank} | {protocol} | {scope} | {accuracy} | {ratio} | {steps} |".format(
                name=name,
                method=row.get("method", row.get("status", "")),
                rank=row.get("rank", ""),
                protocol=row.get("evaluation_protocol", ""),
                scope=row.get("eval_scope", ""),
                accuracy=row.get("final_eval_accuracy", ""),
                ratio=row.get("budget_error_ratio", ""),
                steps=row.get("dynamic_steps", ""),
            )
        )
    (output_dir / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DiCo rank experiment outputs.")
    parser.add_argument("--output_dir", default="outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = resolve_project_path(ROOT, args.output_dir)
    report = audit_outputs(output_dir)
    write_report(output_dir, report)
    print(f"audit status: {report['status']}")
    print(f"wrote {output_dir / 'audit_report.md'}")
    print(f"wrote {output_dir / 'audit_report.json'}")
    return 1 if report["critical"] else 0


if __name__ == "__main__":
    sys.exit(main())
