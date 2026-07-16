#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.baselines import BASELINE_STATUS_VALUES, baseline_status_matrix, render_baseline_status_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write baseline implementation/protocol status reports.")
    parser.add_argument("--json-output", default="reports/baseline_status.json")
    parser.add_argument("--markdown-output", default="reports/baseline_status.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_output)
    md_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status_values": sorted(BASELINE_STATUS_VALUES),
        "baselines": baseline_status_matrix(),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_baseline_status_markdown(), encoding="utf-8")
    print(f"[baseline_status] wrote {json_path}")
    print(f"[baseline_status] wrote {md_path}")


if __name__ == "__main__":
    main()
