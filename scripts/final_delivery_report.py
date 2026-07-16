#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


REPORT_PATHS = {
    "readme": "README.md",
    "directory_structure": "reports/directory_structure.md",
    "changed_files": "reports/changed_files.md",
    "static_acceptance": "reports/static_acceptance.md",
    "baseline_status": "reports/baseline_status.md",
    "e00_readiness": "reports/e00_readiness.md",
    "experiment_matrix": "reports/experiment_matrix.md",
    "method_audit": "reports/audit/method_implementation_audit.md",
    "protocol_audit": "reports/audit/experiment_protocol_audit.md",
    "status_matrix": "reports/audit/status_matrix.md",
}


def _read_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required report is missing: {relative_path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_report(script: str) -> None:
    """Regenerate volatile readiness artifacts before aggregating them."""
    result = subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to refresh {script} before final delivery aggregation:\n{result.stderr or result.stdout}"
        )


def _git_modified_files() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return [f"[git status unavailable] {result.stderr.strip()}"]
    files: list[str] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        files.append(line)
    return files


def _requirements_by_status(report: dict[str, Any], status: str) -> list[str]:
    return [
        str(row.get("id"))
        for row in report.get("requirements", [])
        if str(row.get("status")) == status
    ]


def _static_acceptance_skips(report: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "id": str(row.get("id")),
            "message": str(row.get("message")),
        }
        for row in report.get("checks", [])
        if str(row.get("status")) == "SKIP"
    ]


def _protocol_preflight_skips(report: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for config in report.get("configs", []):
        config_path = str(config.get("path"))
        for row in config.get("checks", []):
            if str(row.get("status")) != "SKIP":
                continue
            check_id = str(row.get("id"))
            entry = merged.setdefault(
                check_id,
                {
                    "id": check_id,
                    "message": str(row.get("message")),
                    "configs": [],
                },
            )
            entry["configs"].append(config_path)
    return [merged[key] for key in sorted(merged)]


def build_payload(test_result: str | None = None) -> dict[str, Any]:
    # E00 readiness depends on the current code/config tree and must never be
    # inferred from a checked-in JSON produced before the latest edits.
    _refresh_report("scripts/e00_readiness.py")
    baseline = _read_json("reports/baseline_status.json")
    static_acceptance = _read_json("reports/static_acceptance.json")
    e00 = _read_json("reports/e00_readiness.json")
    matrix = _read_json("reports/experiment_matrix.json")
    protocol_preflight = _read_json("reports/protocol_preflight.json")
    method_audit = _read_json("reports/audit/method_implementation_audit.json")
    protocol_audit = _read_json("reports/audit/experiment_protocol_audit.json")
    status_matrix = _read_json("reports/audit/status_matrix.json")

    external_blockers = [
        str(row.get("display_name") or row.get("method"))
        for row in baseline.get("baselines", [])
        if str(row.get("status")) == "BLOCKED_BY_UNRESOLVED_PROTOCOL"
    ]
    gpu_not_executed = _requirements_by_status(status_matrix, "NOT_EXECUTED")
    not_implemented = [
        str(row.get("method"))
        for row in baseline.get("baselines", [])
        if str(row.get("status")) == "NOT_IMPLEMENTED"
    ]
    experiment_commands = {
        str(row.get("id")): list(row.get("commands", []))
        for row in matrix.get("experiments", [])
    }
    experiment_blockers = {
        str(row.get("id")): list(row.get("blocked_items", []))
        for row in matrix.get("experiments", [])
        if row.get("blocked_items")
    }
    status = (
        "COMPLETE"
        if not external_blockers and not gpu_not_executed and not not_implemented
        else "NOT_COMPLETE_GPU_AND_EXTERNAL_PROTOCOL_PENDING"
    )
    return {
        "summary": {
            "status": status,
            "static_acceptance_status": static_acceptance.get("summary", {}).get("status"),
            "static_acceptance_skipped_checks": static_acceptance.get("summary", {}).get("skipped_checks"),
            "e00_readiness_status": e00.get("summary", {}).get("status"),
            "baseline_status_counts": _count_statuses(baseline.get("baselines", [])),
            "method_audit_status_counts": method_audit.get("requirements_by_status", {}),
            "protocol_audit_status_counts": protocol_audit.get("requirements_by_status", {}),
            "status_matrix_counts": status_matrix.get("requirements_by_status", {}),
        },
        "test_evidence": {
            "latest": test_result or "not supplied; run pytest -q and regenerate with --test-result",
            "note": "This report records supplied test evidence; it does not run tests or GPU jobs itself.",
        },
        "gpu_not_executed": gpu_not_executed,
        "static_acceptance_skipped_checks": _static_acceptance_skips(static_acceptance),
        "protocol_preflight_skipped_checks": _protocol_preflight_skips(protocol_preflight),
        "external_protocol_blockers": external_blockers,
        "not_implemented_baselines": not_implemented,
        "experiment_commands": experiment_commands,
        "experiment_blockers": experiment_blockers,
        "delivery_artifacts": REPORT_PATHS,
        "modified_files": _git_modified_files(),
        "unexecuted_gpu_tests": [
            "E00 single-GPU LoRA/CovRA/AdaLoRA/GoRA-public/GoRA-BM pilot on A800",
            "E01-E10 formal/recommended GPU training matrix",
            "DDP fallback runtime validation after single-GPU OOM",
            "MTBench-local scoring with target local 70B judge",
        ],
        "unresolved_external_protocols": external_blockers,
    }


def _count_statuses(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Final Delivery Report",
        "",
        f"- status: `{payload['summary']['status']}`",
        f"- static_acceptance_status: `{payload['summary']['static_acceptance_status']}`",
        f"- e00_readiness_status: `{payload['summary']['e00_readiness_status']}`",
        f"- latest_test_evidence: `{payload['test_evidence']['latest']}`",
        "",
        "## Remaining blockers",
        "",
        f"- GPU not executed: `{', '.join(payload['gpu_not_executed']) or '-'}`",
        f"- External protocol blockers: `{', '.join(payload['external_protocol_blockers']) or '-'}`",
        f"- Not implemented baselines: `{', '.join(payload['not_implemented_baselines']) or '-'}`",
        "",
        "## Static acceptance skipped checks",
        "",
    ]
    skipped_static = payload.get("static_acceptance_skipped_checks", [])
    if skipped_static:
        for row in skipped_static:
            lines.append(f"- `{row['id']}`: {row['message']}")
    else:
        lines.append("- `-`")
    lines.extend(
        [
            "",
            "## Protocol preflight skipped checks",
            "",
        ]
    )
    skipped_protocol = payload.get("protocol_preflight_skipped_checks", [])
    if skipped_protocol:
        for row in skipped_protocol:
            config_count = len(row.get("configs", []))
            lines.append(f"- `{row['id']}`: {row['message']} ({config_count} config(s))")
    else:
        lines.append("- `-`")
    lines.extend(
        [
            "",
            "## Unexecuted GPU tests",
            "",
        ]
    )
    unexecuted_gpu_tests = payload.get("unexecuted_gpu_tests", [])
    if unexecuted_gpu_tests:
        lines.extend(f"- {item}" for item in unexecuted_gpu_tests)
    else:
        lines.append("- `-`")
    lines.extend(
        [
            "",
        ]
    )
    lines.extend(
        [
        "## Delivery artifacts",
        "",
        "| item | path |",
        "|---|---|",
        ]
    )
    for key, path in payload["delivery_artifacts"].items():
        lines.append(f"| {key} | {path} |")
    lines.extend(
        [
            "",
            "## Experiment commands",
            "",
        ]
    )
    for exp_id, commands in payload["experiment_commands"].items():
        lines.append(f"### {exp_id}")
        lines.append("")
        if commands:
            for command in commands:
                lines.append(f"- `{command}`")
        else:
            lines.append("- `[blocked/no command]`")
        blockers = payload["experiment_blockers"].get(exp_id, [])
        if blockers:
            lines.append(f"- blockers: `{', '.join(blockers)}`")
        lines.append("")
    lines.extend(
        [
            "## Modified files",
            "",
        ]
    )
    if payload["modified_files"]:
        lines.extend(f"- `{item}`" for item in payload["modified_files"])
    else:
        lines.append("- `git status --short` clean")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a final handoff/delivery report from existing audit artifacts.")
    parser.add_argument("--json-output", default="reports/final_delivery.json")
    parser.add_argument("--markdown-output", default="reports/final_delivery.md")
    parser.add_argument("--test-result", default=None, help="Latest verified test command/result string to record.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(test_result=args.test_result)
    json_path = Path(args.json_output)
    md_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[final_delivery_report] wrote {json_path}")
    print(f"[final_delivery_report] wrote {md_path}")


if __name__ == "__main__":
    main()
