#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


QUERY_FIELDS = (
    "index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
)
CSV_FIELDS = (
    "timestamp",
    "gpu_index",
    "gpu_uuid",
    "utilization_gpu_percent",
    "memory_used_mib",
    "memory_total_mib",
    "power_draw_w",
    "temperature_c",
)


def _number(value: str) -> float | None:
    normalized = value.strip()
    if normalized in {"", "N/A", "[N/A]"}:
        return None
    return float(normalized)


def parse_sample(line: str) -> dict[str, Any]:
    values = [item.strip() for item in next(csv.reader([line]))]
    if len(values) != 7:
        raise ValueError(f"Unexpected nvidia-smi sample: {line!r}")
    return {
        "gpu_index": int(values[0]),
        "gpu_uuid": values[1],
        "utilization_gpu_percent": _number(values[2]),
        "memory_used_mib": _number(values[3]),
        "memory_total_mib": _number(values[4]),
        "power_draw_w": _number(values[5]),
        "temperature_c": _number(values[6]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one-second nvidia-smi samples for a CovRA run.")
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()
        while not stopped:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--id",
                    str(args.gpu_id),
                    f"--query-gpu={QUERY_FIELDS}",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                row = parse_sample(result.stdout.splitlines()[0])
                writer.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), **row})
                handle.flush()
            time.sleep(max(0.1, float(args.interval)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
