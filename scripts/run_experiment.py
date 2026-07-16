#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.config import apply_overrides, load_yaml, validate_known_config_fields
from dico.path_utils import resolve_project_path
from dico.trainer import train
from dico.utils import setup_logging

try:
    from accelerate import notebook_launcher
    _NOTEBOOK_LAUNCHER_AVAILABLE = True
except ImportError:
    notebook_launcher = None
    _NOTEBOOK_LAUNCHER_AVAILABLE = False


def _default_num_processes() -> int:
    """Avoid recursively spawning DDP workers inside an external launcher rank."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or "LOCAL_RANK" in os.environ:
        return 1
    return int(os.environ.get("NUM_GPUS", "1"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one DiCo rank experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and print a summary without loading a model or training.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=_default_num_processes(),
        help=(
            "If >1, spawn this many DDP processes internally via accelerate.notebook_launcher "
            "instead of requiring an external `accelerate launch` wrapper. Lets this single "
            "script do multi-GPU DDP training on its own (default: $NUM_GPUS or 1)."
        ),
    )
    parser.add_argument(
        "--main-process-port",
        type=int,
        default=int(os.environ.get("MASTER_PORT", "29500")),
        help="Port for inter-process communication when --num-processes > 1 (default: $MASTER_PORT or 29500).",
    )
    return parser.parse_args()


def _train_entrypoint(config: dict) -> None:
    """Module-level (picklable) target for accelerate.notebook_launcher's spawned processes."""
    metrics = train(config)
    print(metrics)


def print_config_summary(config: dict) -> None:
    print(
        "[run] "
        f"experiment_name={config.get('experiment_name')} "
        f"method={config.get('method')} "
        f"rank={config.get('rank')} "
        f"seed={config.get('seed')} "
        f"output_dir={config.get('project', {}).get('output_dir')} "
        f"lora_injection={config.get('lora', {}).get('injection')} "
        f"lora_scaling={config.get('lora', {}).get('scaling')} "
        f"dico_version={config.get('dico', {}).get('version')} "
        f"config_path={config.get('_config_path')}"
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_yaml(resolve_project_path(ROOT, args.config))
    config = apply_overrides(config, args.override)
    validate_known_config_fields(config)
    print_config_summary(config)
    if args.dry_run or os.environ.get("DRY_RUN") == "1":
        resolved = {
            "experiment_name": config.get("experiment_name"),
            "method": config.get("method"),
            "rank": config.get("rank"),
            "seed": config.get("seed"),
            "output_dir": config.get("project", {}).get("output_dir"),
            "config_path": config.get("_config_path"),
            "overrides": args.override,
        }
        print("[dry-run] " + json.dumps(resolved, sort_keys=True))
        return

    if args.num_processes > 1:
        # No external `accelerate launch` wrapper available: spawn the DDP processes
        # ourselves. Must happen before any CUDA initialization in this (parent) process.
        if not _NOTEBOOK_LAUNCHER_AVAILABLE:
            raise SystemExit(
                "--num-processes > 1 requires the `accelerate` package to spawn DDP processes "
                "via accelerate.notebook_launcher (pip install accelerate)."
            )
        print(f"[run] launching {args.num_processes} DDP processes via accelerate.notebook_launcher")
        notebook_launcher(
            _train_entrypoint,
            args=(config,),
            num_processes=args.num_processes,
            use_port=str(args.main_process_port),
        )
        return

    metrics = train(config)
    print(metrics)


if __name__ == "__main__":
    main()
