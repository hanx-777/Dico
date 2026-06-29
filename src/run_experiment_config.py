import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


SKIP_CLI_KEYS = {"summary_fields"}


def load_experiment_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "common" not in config or "experiments" not in config:
        raise ValueError("Config must contain top-level 'common' and 'experiments' keys")
    return config


def build_run_args(config: Dict[str, Any], experiment: str) -> List[Dict[str, Any]]:
    experiments = config.get("experiments", {})
    if experiment not in experiments:
        raise ValueError("Unknown experiment '%s'. Available: %s" % (experiment, sorted(experiments)))
    section = experiments[experiment]
    common = dict(config.get("common", {}))
    defaults = dict(section.get("defaults", {}))
    runs = section.get("runs", [])
    if not runs:
        raise ValueError("Experiment '%s' has no runs" % experiment)

    merged_runs = []
    for run in runs:
        merged = {}
        merged.update(common)
        merged.update(defaults)
        merged.update(run)
        _validate_run_args(merged)
        merged_runs.append(merged)
    return merged_runs


def _validate_run_args(args: Dict[str, Any]) -> None:
    required = ["model_name_or_path", "method", "output_dir"]
    missing = [key for key in required if not args.get(key)]
    if missing:
        raise ValueError("Run is missing required keys: %s" % missing)
    if args["model_name_or_path"] == "/path/to/Qwen3-8B":
        raise ValueError("Please edit configs/mvp_gsm8k.json and set a real model_name_or_path")
    if args.get("finetune_mode", "full") == "full" and (
        _as_bool(args.get("load_in_4bit", False)) or _as_bool(args.get("load_in_8bit", False))
    ):
        raise ValueError("finetune_mode=full requires load_in_4bit=false and load_in_8bit=false")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def to_train_cli_args(run_args: Dict[str, Any]) -> List[str]:
    cli = [sys.executable, "-m", "src.train_gsm8k_lora"]
    for key, value in run_args.items():
        if key in SKIP_CLI_KEYS or value is None:
            continue
        cli.extend(["--" + key, _format_cli_value(value)])
    return cli


def run_experiment(config_path: Path, experiment: str, dry_run: bool = False) -> List[Dict[str, Any]]:
    config = load_experiment_config(config_path)
    runs = build_run_args(config, experiment)
    root = Path(__file__).resolve().parents[1]
    for run in runs:
        command = to_train_cli_args(run)
        print("+ " + " ".join(command), flush=True)
        if not dry_run:
            subprocess.run(command, cwd=root, check=True)
    if not dry_run:
        print_summary(runs)
    return runs


def print_summary(runs: List[Dict[str, Any]]) -> None:
    print("method\texact_match\ttrainable_params\tused_budget\toutput_dir")
    for run in runs:
        output_dir = Path(run["output_dir"])
        eval_path = output_dir / "eval_results.json"
        summary_path = output_dir / "run_summary.json"
        eval_data = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        print(
            "%s\t%s\t%s\t%s\t%s"
            % (
                run["method"],
                eval_data.get("exact_match"),
                summary.get("trainable_params"),
                summary.get("used_budget"),
                output_dir,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DiCo-LoRA MVP experiments from one JSON config.")
    parser.add_argument("--config", default="configs/mvp_gsm8k.json")
    parser.add_argument("--experiment", choices=["smoke", "mvp"], default="smoke")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(Path(args.config), args.experiment, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
