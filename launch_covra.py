#!/usr/bin/env python3
"""CovRA 单文件无参数启动器。

适用于“只能运行一个 Python 文件，且启动命令不能附带任何参数”的算力平台。
平台最终只需要执行：

    python launch_covra.py

本文件只负责平台启动：检查路径、创建目录、设置环境变量、调用现有
`scripts/platform_train.py`，并把输出同时写到控制台和日志。它不会修改 CovRA
方法逻辑，也不会在这里改变实验超参数。
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# =============================================================================
# 手动配置区 / MANUAL CONFIGURATION - 需要在 A800 服务器上修改
# =============================================================================

# 1) Conda 环境。这里要指向已经安装 torch/transformers/accelerate/datasets/vLLM
#    等依赖的 conda 环境里的 Python。
CONDA_ENV_NAME = os.environ.get("COVRA_CONDA_ENV_NAME", "dico-rank")
_SERVER_CONDA_PYTHON = Path(f"/ai/lxw/lxw/miniconda3/envs/{CONDA_ENV_NAME}/bin/python")
CONDA_ENV_PYTHON = Path(
    os.environ.get(
        "COVRA_CONDA_ENV_PYTHON",
        str(_SERVER_CONDA_PYTHON if _SERVER_CONDA_PYTHON.exists() else Path(sys.executable)),
    )
)

# 2) 项目根目录。该目录必须包含 `scripts/platform_train.py`、
#    `scripts/run_experiment.py`、`configs/` 和 `src/`。
_SERVER_PROJECT_ROOT = Path("/ai/lxw/lxw/dico_rank_experiments")
PROJECT_ROOT = Path(
    os.environ.get(
        "COVRA_PROJECT_ROOT",
        str(_SERVER_PROJECT_ROOT if _SERVER_PROJECT_ROOT.exists() else Path(__file__).resolve().parent),
    )
)

# 3) 模型路径。请填写服务器上已下载好的 Llama-3.1-8B base 本地目录。
MODEL_PATH = Path(
    os.environ.get(
        "COVRA_MODEL_PATH",
        "/ai/lxw/lxw/Meta-Llama-3.1-8B",
    )
)

# 4) 启动前要检查的数据路径。这些检查不会重写 config，只是避免数据缺失时浪费队列时间。
DATA_ROOT = Path(os.environ.get("COVRA_DATA_ROOT", str(PROJECT_ROOT / "data")))
DATA_PATHS_TO_CHECK = (
    DATA_ROOT / "metamathqa" / "train.jsonl",
    DATA_ROOT / "gsm8k" / "main" / "test.jsonl",
)

# 5) 启动 profile。平台命令无需参数；如需 E02，可在平台环境变量中设置
#    COVRA_PROFILE=e02_strict_budget。旧 gora_bw 不再进入任何正式 profile。
PROFILE = os.environ.get("COVRA_PROFILE", "e01_aligned")
PROFILES = {
    "e01_aligned": (
        Path("configs/dico/lora_r8.yaml"),
        Path("configs/dico/adalora_r8.yaml"),
        Path("configs/dico/gora_public_r8.yaml"),
        Path("configs/dico/dico_cd_da_r8.yaml"),
    ),
    "e02_strict_budget": (
        Path("configs/dico/lora_r8.yaml"),
        Path("configs/dico/gora_bm_r8.yaml"),
        Path("configs/dico/dico_cd_da_r8.yaml"),
    ),
}
if PROFILE not in PROFILES:
    raise SystemExit(f"未知 COVRA_PROFILE={PROFILE!r}；可选值：{sorted(PROFILES)}")
CONFIG_FILES = PROFILES[PROFILE]

# 6) seed 与 GPU。默认把三张 A800 当作三个独立单卡 run 并行使用，
#    通过 batch_size × grad_accum 保持 global batch 64。
SEEDS = (42, 43, 44)
GPU_IDS = ("0", "1", "2")

# 7) 输出和日志目录。目录会自动创建。
DEFAULT_OUTPUT_NAME = (
    "e01_llama3_r8_aligned_sdpa_v4"
    if PROFILE == "e01_aligned"
    else "e02_llama3_r8_strict_budget_sdpa_v4"
)
OUTPUT_DIR = Path(os.environ.get("COVRA_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / DEFAULT_OUTPUT_NAME)))
LOG_DIR = Path(os.environ.get("COVRA_LOG_DIR", str(PROJECT_ROOT / "logs")))
LOG_FILE = LOG_DIR / "launch_covra.log"

# 8) 协议 batch 设置。这里镜像正式 config / platform 协议。
#    不要在这里根据结果调参；如果协议本身变化，应先改 config 与实验计划。
BATCH_SIZE = int(os.environ.get("COVRA_BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("COVRA_GRAD_ACCUM", "16"))
CALIBRATION_BATCH_SIZE = int(os.environ.get("COVRA_CALIBRATION_BATCH_SIZE", "4"))

# 9) 可选安全/调试开关。正式训练保持 DRY_RUN=False。
DRY_RUN = os.environ.get("COVRA_DRY_RUN", "0") == "1"
SKIP_MODEL_CHECK = os.environ.get("COVRA_SKIP_MODEL_CHECK", "0") == "1"
SKIP_DATA_CHECK = os.environ.get("COVRA_SKIP_DATA_CHECK", "0") == "1"


# =============================================================================
# 启动器实现区：下面只负责平台启动，不包含方法逻辑
# =============================================================================


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _die(message: str, code: int = 2) -> None:
    print(f"[launch_covra] ERROR: {message}", file=sys.stderr)
    raise SystemExit(code)


def _check_file(path: Path, label: str) -> None:
    if not path.is_file():
        _die(f"{label} does not exist or is not a file: {path}")


def _check_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        _die(f"{label} does not exist or is not a directory: {path}")


def check_required_paths() -> None:
    """正式提交长训练前，检查固定启动路径是否存在。"""

    _check_dir(PROJECT_ROOT, "PROJECT_ROOT")
    _check_file(CONDA_ENV_PYTHON, "CONDA_ENV_PYTHON")
    _check_file(PROJECT_ROOT / "scripts" / "platform_train.py", "platform launcher")
    _check_file(PROJECT_ROOT / "scripts" / "run_experiment.py", "training entrypoint")

    if not (SKIP_MODEL_CHECK or DRY_RUN):
        _check_dir(MODEL_PATH, "MODEL_PATH")

    if not (SKIP_DATA_CHECK or DRY_RUN):
        for data_path in DATA_PATHS_TO_CHECK:
            _check_file(data_path, "dataset path")

    for config_path in CONFIG_FILES:
        _check_file(PROJECT_ROOT / config_path, "config")

    if not GPU_IDS:
        _die("GPU_IDS is empty; set at least one GPU id")
    if not SEEDS:
        _die("SEEDS is empty; set at least one seed")


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def build_command() -> list[str]:
    """构造正式项目训练命令。

    这里调用 `platform_train.py`，不重新实现调度逻辑。这样方法逻辑、config
    解析、seed override 和 run manifest 都仍走项目原有正式路径。
    """

    command = [
        str(CONDA_ENV_PYTHON),
        "scripts/platform_train.py",
        "--python-bin",
        str(CONDA_ENV_PYTHON),
        "--model-path",
        str(MODEL_PATH),
        "--output-dir",
        str(OUTPUT_DIR),
        "--cuda-visible-devices",
        ",".join(str(gpu) for gpu in GPU_IDS),
        "--num-gpus",
        str(len(GPU_IDS)),
        "--child-num-processes",
        "1",
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--batch-size",
        str(BATCH_SIZE),
        "--grad-accum",
        str(GRAD_ACCUM),
        "--calibration-batch-size",
        str(CALIBRATION_BATCH_SIZE),
    ]
    for config_file in CONFIG_FILES:
        command.extend(["--config", str(config_file)])
    if DRY_RUN:
        command.append("--dry-run")
    if SKIP_MODEL_CHECK:
        command.append("--skip-model-check")
    return command


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env["COVRA_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["COVRA_MODEL_PATH"] = str(MODEL_PATH)
    env["COVRA_DATA_ROOT"] = str(DATA_ROOT)
    env["COVRA_OUTPUT_DIR"] = str(OUTPUT_DIR)
    env["COVRA_CONDA_ENV_NAME"] = str(CONDA_ENV_NAME)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in GPU_IDS)
    # The parent fans out three independent jobs.  Each child is deliberately a
    # one-process run even when the platform exports NUM_GPUS=3 globally.
    env["NUM_GPUS"] = "1"
    return env


def _write_header(log_fh, command: list[str]) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    header_lines = [
        f"[launch_covra] started_at={timestamp}",
        f"[launch_covra] project_root={PROJECT_ROOT}",
        f"[launch_covra] conda_env_python={CONDA_ENV_PYTHON}",
        f"[launch_covra] model_path={MODEL_PATH}",
        f"[launch_covra] data_root={DATA_ROOT}",
        f"[launch_covra] output_dir={OUTPUT_DIR}",
        f"[launch_covra] profile={PROFILE}",
        f"[launch_covra] log_file={LOG_FILE}",
        f"[launch_covra] configs={[str(path) for path in CONFIG_FILES]}",
        f"[launch_covra] seeds={list(SEEDS)} gpu_ids={list(GPU_IDS)}",
        f"[launch_covra] batch_size={BATCH_SIZE} grad_accum={GRAD_ACCUM} effective_batch={BATCH_SIZE * GRAD_ACCUM}",
        "[launch_covra] command=" + shlex.join(command),
        "",
    ]
    for line in header_lines:
        print(line, flush=True)
        log_fh.write(line + "\n")
    log_fh.flush()


def run_and_tee(command: list[str]) -> int:
    env = build_environment()
    with LOG_FILE.open("w", encoding="utf-8") as log_fh:
        _write_header(log_fh, command)
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_fh.write(line)
            log_fh.flush()
        return_code = process.wait()
        footer = f"\n[launch_covra] child_exit_code={return_code}\n"
        print(footer, end="", flush=True)
        log_fh.write(footer)
        return int(return_code)


def main() -> int:
    try:
        check_required_paths()
        ensure_directories()
        command = build_command()
        return run_and_tee(command)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return int(code)
    except KeyboardInterrupt:
        print("\n[launch_covra] interrupted by user", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
