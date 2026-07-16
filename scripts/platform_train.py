#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from shutil import which
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


DEFAULT_MODEL_PATH = Path("/ai/lxw/lxw/Meta-Llama-3.1-8B")
DEFAULT_SEEDS = "42,43,44"
# We have 3xA800 and 3 seeds per config: each seed gets its own GPU and runs a normal
# self-contained single-GPU train() (training + its own final eval) concurrently with its
# siblings, instead of DDP-parallelizing one seed's job across all 3 GPUs. This keeps all 3
# GPUs busy during both training and eval (the final GSM8K/HumanEval accuracy eval only ever
# runs on one process) and avoids NCCL/accelerate entirely for the main batch. True DDP for a
# single experiment remains available via `scripts/run_ddp.sh` if a future model doesn't fit
# on one GPU.
DEFAULT_NUM_GPUS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 16  # effective batch = batch_size x grad_accum = 64, matches GoRA exactly
CONFIGS = [
    "configs/dico/lora_r8.yaml",
    "configs/dico/adalora_r8.yaml",
    "configs/dico/gora_public_r8.yaml",
    "configs/dico/dico_cd_da_r8.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Platform launcher for the three main DiCo experiments.")
    parser.add_argument("--python-bin", default=os.environ.get("DICO_PYTHON_BIN"))
    parser.add_argument("--conda-bin", default=os.environ.get("DICO_CONDA_BIN", "/ai/lxw/lxw/miniconda3/bin/conda"))
    parser.add_argument("--conda-env", default=os.environ.get("DICO_CONDA_ENV", "dico-rank"))
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    parser.add_argument("--output-dir", default=os.environ.get("DICO_OUTPUT_DIR", "outputs/covra_main_3seed"))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", str(DEFAULT_NUM_GPUS))),
        help=(
            "Number of GPUs to fan seeds out across: each config's seeds run this many at a "
            "time as independent single-GPU jobs (one seed per GPU, no DDP), then the next "
            f"chunk of seeds starts (default {DEFAULT_NUM_GPUS}). Set to 1 for fully "
            "sequential single-GPU runs."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("DICO_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))),
        help=f"training.batch_size for every (single-GPU) job (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=int(os.environ.get("DICO_GRAD_ACCUM", str(DEFAULT_GRAD_ACCUM))),
        help=f"training.gradient_accumulation_steps for every job (default {DEFAULT_GRAD_ACCUM}).",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=int(os.environ.get("DICO_CALIBRATION_BATCH_SIZE", "4")),
    )
    parser.add_argument(
        "--child-num-processes",
        type=int,
        default=1,
        help="Process count inside each independent seed job (formal profiles require 1).",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help=(
            "Config path to run. Can be passed multiple times. "
            "Defaults to the aligned LoRA/AdaLoRA/GoRA-public/CovRA r8 configs when omitted."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument(
        "--seeds",
        default=os.environ.get("DICO_SEEDS", DEFAULT_SEEDS),
        help=f"Comma-separated seeds to run for each main config (default: {DEFAULT_SEEDS}).",
    )
    args = parser.parse_args()

    if not args.cuda_visible_devices:
        args.cuda_visible_devices = ",".join(str(i) for i in range(args.num_gpus))

    return args


def parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer seed")
    return seeds


def chunked(seq: list[int], size: int) -> list[list[int]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def experiment_name_for_config(config_path: str) -> str:
    # Imported lazily: this launcher script picks/activates the conda env that has the
    # project's dependencies (e.g. PyYAML) for the *child* run_experiment.py process, so
    # it must itself stay importable even when run under a bare interpreter that doesn't
    # have those dependencies -- the experiment_name lookup here is a nice-to-have, not
    # essential (falls back to the config file's stem).
    try:
        from dico.config import load_yaml

        config = load_yaml(ROOT / config_path)
        value = config.get("experiment_name")
        if value:
            return str(value)
    except Exception:
        pass
    return Path(config_path).stem


def job_experiment_name(config_path: str, seed: int) -> str:
    return f"{experiment_name_for_config(config_path)}_seed{int(seed)}"


def validate_config_paths(config_paths: list[str]) -> None:
    try:
        from dico.config import load_yaml, validate_known_config_fields
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "platform_train.py cannot validate experiment configs because a required "
            f"package is unavailable in the parent interpreter: {exc}. Run with a "
            "project environment or pass --python-bin for child execution after fixing "
            "the parent environment."
        ) from exc
    failures: list[str] = []
    for config_path in config_paths:
        try:
            config = load_yaml(ROOT / config_path)
            validate_known_config_fields(config)
        except Exception as exc:
            failures.append(f"{config_path}: {exc}")
    if failures:
        raise SystemExit("Config validation failed before launching jobs:\n" + "\n".join(failures))


def seed_overrides(config_path: str, seed: int, output_dir: str) -> list[str]:
    experiment_name = job_experiment_name(config_path, seed)
    overrides = [
        f"experiment_name={experiment_name}",
        f"seed={int(seed)}",
        f"preallocation.sketch_seed={int(seed)}",
        f"calibration.save_dir={output_dir}/preallocations/{experiment_name}",
    ]
    from dico.config import load_yaml

    config = load_yaml(ROOT / config_path)
    if (
        str(config.get("method")) in {"dico_cd", "dico_cd_da"}
        and str(config.get("preallocation", {}).get("allocation_method")) == "covra_v05"
    ):
        overrides.append(f"calibration.seed={int(seed)}")
    return overrides


def candidate_python_bins(args: argparse.Namespace) -> list[Path]:
    candidates: list[Path] = []
    if args.python_bin:
        candidates.append(Path(args.python_bin))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "bin" / "python")
    for base in (
        Path("/ai/lxw/lxw/miniconda3"),
        Path("/root/miniconda3"),
        Path("/opt/conda"),
        Path("/usr/local/miniconda3"),
        Path("/ai/lxw/lxw/anaconda3"),
        Path("/root/anaconda3"),
    ):
        candidates.append(base / "envs" / str(args.conda_env) / "bin" / "python")
    conda_exe = which("conda")
    if conda_exe:
        try:
            result = subprocess.run(
                [conda_exe, "info", "--base"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                candidates.append(Path(result.stdout.strip()) / "envs" / str(args.conda_env) / "bin" / "python")
        except OSError:
            pass
    candidates.append(Path(sys.executable))
    return candidates


def resolve_python_bin(args: argparse.Namespace) -> Path:
    if args.python_bin:
        python_bin = Path(args.python_bin)
        if not python_bin.exists():
            raise SystemExit(f"Python interpreter does not exist: {python_bin}")
        return python_bin
    for candidate in candidate_python_bins(args):
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {path}" for path in candidate_python_bins(args))
    raise SystemExit(f"Could not find a Python interpreter for env {args.conda_env!r}. Searched:\n{searched}")


def _base_experiment_args(config_path: str, args: argparse.Namespace) -> list[str]:
    return [
        "scripts/run_experiment.py",
        "--config",
        config_path,
        "--num-processes",
        str(args.child_num_processes),
        "--override",
        f"model.name_or_path={Path(args.model_path)}",
        "--override",
        f"project.output_dir={args.output_dir}",
        "--override",
        f"training.batch_size={args.batch_size}",
        "--override",
        f"training.gradient_accumulation_steps={args.grad_accum}",
        "--override",
        f"calibration.batch_size={args.calibration_batch_size}",
    ]


def build_experiment_args(config_path: str, args: argparse.Namespace, seed: int) -> list[str]:
    command = _base_experiment_args(config_path, args)
    for override in seed_overrides(config_path, seed, str(args.output_dir)):
        command.extend(["--override", override])
    return command


def build_command(config_path: str, args: argparse.Namespace, python_bin: Path, seed: int) -> list[str]:
    return [str(python_bin), *build_experiment_args(config_path, args, seed)]


def build_conda_shell_command(config_path: str, args: argparse.Namespace, seed: int) -> str:
    full_argv = ["python", *build_experiment_args(config_path, args, seed)]
    run_command = shlex.join(full_argv)
    conda_bin = shlex.quote(str(args.conda_bin))
    conda_sh = shlex.quote(str(Path(args.conda_bin).resolve().parents[1] / "etc" / "profile.d" / "conda.sh"))
    return "\n".join(
        [
            "set -e",
            f'eval "$({conda_bin} shell.bash hook)" || source {conda_sh}',
            f"conda activate {shlex.quote(str(args.conda_env))}",
            run_command,
        ]
    )


def validate_model_path(model_path: Path, skip_check: bool) -> None:
    if skip_check:
        return
    if not model_path.exists():
        raise SystemExit(f"Model path does not exist: {model_path}")
    if not (model_path / "config.json").exists():
        raise SystemExit(f"Model path is missing config.json: {model_path}")


def build_job_chunks(args: argparse.Namespace, python_bin: Path | None) -> list[list[dict]]:
    """One chunk per (config, seed-group): each chunk's jobs run concurrently, one per GPU."""
    seeds = parse_seeds(args.seeds)
    gpu_pool = [item.strip() for item in str(args.cuda_visible_devices).split(",") if item.strip()]
    if len(gpu_pool) < args.num_gpus:
        raise SystemExit(
            f"--cuda-visible-devices has fewer entries ({len(gpu_pool)}) than --num-gpus "
            f"({args.num_gpus}): {gpu_pool}"
        )
    gpu_pool = gpu_pool[:args.num_gpus]

    chunks: list[list[dict]] = []
    config_paths = list(args.configs or CONFIGS)
    for config_path in config_paths:
        for seed_group in chunked(seeds, args.num_gpus):
            jobs = []
            for slot, seed in enumerate(seed_group):
                command = (
                    build_command(config_path, args, python_bin, seed)
                    if python_bin is not None
                    else build_conda_shell_command(config_path, args, seed)
                )
                jobs.append({
                    "config_path": config_path,
                    "seed": seed,
                    "gpu_id": gpu_pool[slot],
                    "command": command,
                    "name": job_experiment_name(config_path, seed),
                })
            chunks.append(jobs)
    return chunks


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    validate_model_path(model_path, args.skip_model_check or args.dry_run)
    python_bin = resolve_python_bin(args) if args.python_bin else None
    validate_config_paths(list(args.configs or CONFIGS))

    job_chunks = build_job_chunks(args, python_bin)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")

    if args.dry_run:
        print(f"[platform] cwd={ROOT}")
        if python_bin is not None:
            print(f"[platform] python={python_bin}")
        else:
            print(f"[platform] conda_bin={args.conda_bin}")
            print(f"[platform] conda_env={args.conda_env}")
        print(
            f"[platform] num_gpus={args.num_gpus} (parallel single-GPU workers) "
            f"batch_size={args.batch_size} grad_accum={args.grad_accum} "
            f"effective_batch={args.batch_size * args.grad_accum}"
        )
        for chunk in job_chunks:
            config_path = chunk[0]["config_path"]
            seeds_in_chunk = [job["seed"] for job in chunk]
            print(f"[platform] config={config_path} seeds={seeds_in_chunk}")
            for job in chunk:
                rendered = shlex.join(job["command"]) if isinstance(job["command"], list) else job["command"]
                print(f"[platform] gpu={job['gpu_id']} " + rendered)
        return

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    monitor_enabled = os.environ.get("COVRA_GPU_MONITOR", "1") != "0" and which("nvidia-smi") is not None

    for chunk in job_chunks:
        config_path = chunk[0]["config_path"]
        print(
            f"[platform] launching {len(chunk)} seed(s) in parallel for {config_path}: "
            f"{[(job['seed'], job['gpu_id']) for job in chunk]}",
            flush=True,
        )
        procs = []
        for job in chunk:
            job_env = env.copy()
            job_env["CUDA_VISIBLE_DEVICES"] = job["gpu_id"]
            job_env["NUM_GPUS"] = str(args.child_num_processes)
            log_path = logs_dir / f"{job['name']}.log"
            log_fh = open(log_path, "w")
            if isinstance(job["command"], list):
                popen_args = job["command"]
            else:
                popen_args = ["bash", "-lc", job["command"]]
            proc = subprocess.Popen(popen_args, cwd=ROOT, env=job_env, stdout=log_fh, stderr=subprocess.STDOUT)
            monitor_proc = None
            if monitor_enabled:
                monitor_path = Path(args.output_dir) / job["name"] / "gpu_monitor.csv"
                if not monitor_path.is_absolute():
                    monitor_path = ROOT / monitor_path
                monitor_proc = subprocess.Popen(
                    [
                        sys.executable,
                        "scripts/gpu_monitor.py",
                        "--gpu-id",
                        str(job["gpu_id"]),
                        "--output",
                        str(monitor_path),
                        "--interval",
                        "1.0",
                    ],
                    cwd=ROOT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            print(
                f"[platform]   pid={proc.pid} gpu={job['gpu_id']} seed={job['seed']} "
                f"log={log_path.relative_to(ROOT)}",
                flush=True,
            )
            procs.append((job, proc, log_fh, log_path, monitor_proc))

        failures = []
        for job, proc, log_fh, log_path, monitor_proc in procs:
            ret = proc.wait()
            if monitor_proc is not None:
                monitor_proc.terminate()
                try:
                    monitor_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    monitor_proc.kill()
                    monitor_proc.wait()
            log_fh.close()
            status = "OK" if ret == 0 else f"FAILED (exit {ret})"
            print(
                f"[platform]   seed={job['seed']} gpu={job['gpu_id']} {status} "
                f"log={log_path.relative_to(ROOT)}",
                flush=True,
            )
            if ret != 0:
                failures.append((job, ret, log_path))

        if failures:
            lines = "\n".join(
                f"  seed={job['seed']} gpu={job['gpu_id']} exit={ret} log={log_path}"
                for job, ret, log_path in failures
            )
            raise SystemExit(
                f"[platform] {len(failures)}/{len(chunk)} job(s) failed for {config_path}:\n{lines}"
            )


if __name__ == "__main__":
    main()
