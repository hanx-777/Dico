#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

INCLUDED_ROOTS = [
    "configs",
    "scripts",
    "src/dico",
    "tests",
    "reports",
]

EXCLUDED_ROOTS = [
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
    "__pycache__",
    "outputs",
]

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".DS_Store",
    "__pycache__",
}

EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDED_PARTS:
        return True
    return path.suffix in EXCLUDED_SUFFIXES


def _entry_for(path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    return {
        "path": relative,
        "type": "directory" if path.is_dir() else "file",
    }


def build_payload() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []

    for root_name in INCLUDED_ROOTS:
        root_path = ROOT / root_name
        if not root_path.exists():
            continue
        roots.append(_entry_for(root_path))
        for path in sorted(root_path.rglob("*")):
            if _is_excluded(path):
                continue
            entries.append(_entry_for(path))

    return {
        "summary": {
            "root": str(ROOT),
            "included_roots": INCLUDED_ROOTS,
            "excluded_roots": EXCLUDED_ROOTS,
            "entry_count": len(entries),
        },
        "roots": roots,
        "entries": entries,
        "excluded_roots": EXCLUDED_ROOTS,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Final Directory Structure",
        "",
        f"- root: `{payload['summary']['root']}`",
        f"- entry_count: `{payload['summary']['entry_count']}`",
        f"- excluded_roots: `{', '.join(payload['excluded_roots'])}`",
        "",
        "## Included roots",
        "",
    ]
    for root in payload["roots"]:
        lines.append(f"- `{root['path']}`")
    lines.extend(["", "## Tree", ""])
    for entry in payload["entries"]:
        prefix = "📁" if entry["type"] == "directory" else "📄"
        lines.append(f"- {prefix} `{entry['path']}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the final deliverable directory structure report.")
    parser.add_argument("--json-output", default="reports/directory_structure.json")
    parser.add_argument("--markdown-output", default="reports/directory_structure.md")
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
    print(f"[directory_structure] wrote {json_path}")
    print(f"[directory_structure] wrote {md_path}")


if __name__ == "__main__":
    main()
