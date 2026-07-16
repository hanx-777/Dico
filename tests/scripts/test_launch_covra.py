from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "launch_covra.py"


def _load_launch_covra():
    spec = importlib.util.spec_from_file_location("launch_covra", LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_covra_is_zero_argument_platform_wrapper_with_manual_top_config():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "MANUAL CONFIGURATION" in text
    assert "CONDA_ENV_PYTHON" in text
    assert "PROJECT_ROOT" in text
    assert "MODEL_PATH" in text
    assert "DATA_PATHS_TO_CHECK" in text
    assert "CONFIG_FILES" in text
    assert "SEEDS" in text
    assert "GPU_IDS" in text
    assert "OUTPUT_DIR" in text
    assert "argparse" not in text
    assert "subprocess.Popen" in text
    assert "sys.exit(main())" in text


def test_launch_covra_builds_platform_train_command_without_changing_method_hyperparams(tmp_path, monkeypatch):
    module = _load_launch_covra()
    project_root = tmp_path / "project"
    config_path = project_root / "configs" / "dico" / "dico_cd_da_r8.yaml"
    platform_train = project_root / "scripts" / "platform_train.py"
    python_bin = tmp_path / "env" / "bin" / "python"
    model_path = tmp_path / "models" / "llama"
    data_path = tmp_path / "data" / "metamathqa" / "train.jsonl"
    for path in (config_path.parent, platform_train.parent, python_bin.parent, model_path, data_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    config_path.write_text("method: dico_cd_da\n", encoding="utf-8")
    platform_train.write_text("# launcher target\n", encoding="utf-8")
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    data_path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(module, "CONDA_ENV_PYTHON", python_bin)
    monkeypatch.setattr(module, "MODEL_PATH", model_path)
    monkeypatch.setattr(module, "DATA_PATHS_TO_CHECK", (data_path,))
    monkeypatch.setattr(module, "CONFIG_FILES", (Path("configs/dico/dico_cd_da_r8.yaml"),))
    monkeypatch.setattr(module, "SEEDS", (42,))
    monkeypatch.setattr(module, "GPU_IDS", ("0",))
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(module, "BATCH_SIZE", 4)
    monkeypatch.setattr(module, "GRAD_ACCUM", 16)

    command = module.build_command()

    assert command[:4] == [
        str(python_bin),
        "scripts/platform_train.py",
        "--python-bin",
        str(python_bin),
    ]
    assert "--config" in command
    assert "configs/dico/dico_cd_da_r8.yaml" in command
    assert command[command.index("--model-path") + 1] == str(model_path)
    assert command[command.index("--output-dir") + 1] == str(tmp_path / "outputs")
    assert command[command.index("--cuda-visible-devices") + 1] == "0"
    assert command[command.index("--seeds") + 1] == "42"
    assert command[command.index("--batch-size") + 1] == "4"
    assert command[command.index("--grad-accum") + 1] == "16"
    assert "--override" not in command


def test_launch_covra_forces_each_seed_child_to_one_process(monkeypatch):
    module = _load_launch_covra()
    command = module.build_command()
    environment = module.build_environment()

    assert command[command.index("--child-num-processes") + 1] == "1"
    assert environment["NUM_GPUS"] == "1"


def test_launch_covra_uses_fresh_sdpa_output_version():
    module = _load_launch_covra()
    assert module.DEFAULT_OUTPUT_NAME == "e01_llama3_r8_aligned_sdpa_v4"


def test_launch_covra_streams_output_to_console_and_log_and_returns_child_code(tmp_path, monkeypatch, capsys):
    module = _load_launch_covra()
    project_root = tmp_path / "project"
    config_path = project_root / "configs" / "dico" / "dico_cd_da_r8.yaml"
    platform_train = project_root / "scripts" / "platform_train.py"
    run_experiment = project_root / "scripts" / "run_experiment.py"
    python_bin = tmp_path / "env" / "bin" / "python"
    model_path = tmp_path / "models" / "llama"
    data_path = tmp_path / "data" / "metamathqa" / "train.jsonl"
    for path in (config_path.parent, platform_train.parent, python_bin.parent, model_path, data_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    config_path.write_text("method: dico_cd_da\n", encoding="utf-8")
    platform_train.write_text("# launcher target\n", encoding="utf-8")
    run_experiment.write_text("# training target\n", encoding="utf-8")
    data_path.write_text("{}\n", encoding="utf-8")
    python_bin.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "print('child stdout marker')",
                "print('child stderr marker', file=sys.stderr)",
                "sys.exit(7)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    python_bin.chmod(0o755)

    monkeypatch.setattr(module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(module, "CONDA_ENV_PYTHON", python_bin)
    monkeypatch.setattr(module, "MODEL_PATH", model_path)
    monkeypatch.setattr(module, "DATA_PATHS_TO_CHECK", (data_path,))
    monkeypatch.setattr(module, "CONFIG_FILES", (Path("configs/dico/dico_cd_da_r8.yaml"),))
    monkeypatch.setattr(module, "SEEDS", (42,))
    monkeypatch.setattr(module, "GPU_IDS", ("0",))
    monkeypatch.setattr(module, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(module, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(module, "LOG_FILE", tmp_path / "logs" / "launch.log")

    code = module.main()

    captured = capsys.readouterr()
    assert code == 7
    assert "child stdout marker" in captured.out
    assert "child stderr marker" in captured.out
    log_text = (tmp_path / "logs" / "launch.log").read_text(encoding="utf-8")
    assert "child stdout marker" in log_text
    assert "child stderr marker" in log_text


def test_launch_covra_fails_before_subprocess_when_required_path_is_missing(tmp_path, monkeypatch):
    module = _load_launch_covra()
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path / "missing_project")
    monkeypatch.setattr(module, "CONDA_ENV_PYTHON", tmp_path / "missing_python")
    monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "missing_model")
    monkeypatch.setattr(module, "DATA_PATHS_TO_CHECK", ())

    try:
        module.check_required_paths()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("check_required_paths should abort before launching subprocess")


def test_python_can_parse_launch_covra():
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(LAUNCHER)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
