#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["added", "modified", "deleted", "renamed", "untracked", "other"]


def _run_git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _category_from_status(status: str) -> str:
    if status == "??":
        return "untracked"
    if "R" in status:
        return "renamed"
    if "D" in status:
        return "deleted"
    if "A" in status:
        return "added"
    if "M" in status:
        return "modified"
    return "other"


def _parse_status_line(line: str) -> dict[str, str]:
    raw_status = line[:2]
    path_text = line[3:].strip()
    category = _category_from_status(raw_status)
    old_path = ""
    path = path_text
    if " -> " in path_text:
        old_path, path = path_text.split(" -> ", 1)
    return {
        "category": category,
        "raw_status": raw_status,
        "path": path,
        "old_path": old_path,
        "display": path_text,
    }


def build_payload() -> dict[str, Any]:
    files = [_parse_status_line(line) for line in _run_git_status()]
    by_category = {category: 0 for category in CATEGORIES}
    for row in files:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1

    return {
        "summary": {
            "repo_root": str(ROOT),
            "total_changed_files": len(files),
        },
        "by_category": by_category,
        "files": files,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Changed Files Report",
        "",
        f"- repo_root: `{payload['summary']['repo_root']}`",
        f"- total_changed_files: `{payload['summary']['total_changed_files']}`",
        "",
        "## Counts",
        "",
        "| category | count |",
        "|---|---:|",
    ]
    for category in CATEGORIES:
        lines.append(f"| {category} | {payload['by_category'].get(category, 0)} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "| category | status | path | old_path |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["files"]:
        old_path = row["old_path"] or "-"
        lines.append(f"| {row['category']} | `{row['raw_status']}` | `{row['path']}` | `{old_path}` |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a structured report of changed files in the current worktree.")
    parser.add_argument("--json-output", default="reports/changed_files.json")
    parser.add_argument("--markdown-output", default="reports/changed_files.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload()
    json_path = Path(args.json_output)
    md_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"[changed_files_report] wrote {json_path}")
    print(f"[changed_files_report] wrote {md_path}")


if __name__ == "__main__":
    main()
