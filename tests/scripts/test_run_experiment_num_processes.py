"""Tests for scripts/run_experiment.py's --num-processes flag, which lets this single
entry-point script spawn DDP processes itself via accelerate.notebook_launcher --
for environments that can only invoke this one file (no external `accelerate launch`
wrapper, no shell scripts)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_experiment as run_experiment_module  # noqa: E402


def _fake_config(**overrides):
    config = {
        "experiment_name": "fake_experiment",
        "method": "lora",
        "rank": 8,
        "seed": 42,
        "project": {"output_dir": "outputs/fake"},
        "lora": {"injection": "static", "scaling": "alpha_over_sqrt_r"},
        "dico": {"version": "cd_da"},
        "_config_path": "configs/fake.yaml",
    }
    config.update(overrides)
    return config


def test_num_processes_defaults_to_env_var(monkeypatch):
    monkeypatch.setenv("NUM_GPUS", "3")
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", "configs/fake.yaml", "--dry-run"])
    args = run_experiment_module.parse_args()
    assert args.num_processes == 3


def test_num_processes_defaults_to_one_without_env_var(monkeypatch):
    monkeypatch.delenv("NUM_GPUS", raising=False)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", "configs/fake.yaml", "--dry-run"])
    args = run_experiment_module.parse_args()
    assert args.num_processes == 1


def test_external_ddp_rank_never_spawns_nested_processes(monkeypatch):
    monkeypatch.setenv("NUM_GPUS", "3")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", "configs/fake.yaml", "--dry-run"])
    args = run_experiment_module.parse_args()
    assert args.num_processes == 1


def test_dry_run_short_circuits_even_with_num_processes(monkeypatch, capsys):
    monkeypatch.setattr(run_experiment_module, "load_yaml", lambda path: _fake_config())
    calls = []
    monkeypatch.setattr(
        run_experiment_module,
        "notebook_launcher",
        lambda *a, **k: calls.append((a, k)),
        raising=False,
    )
    monkeypatch.setattr(run_experiment_module, "_NOTEBOOK_LAUNCHER_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        sys, "argv", ["run_experiment.py", "--config", "configs/fake.yaml", "--dry-run", "--num-processes", "3"]
    )
    run_experiment_module.main()
    assert calls == [], "notebook_launcher must not be invoked on a --dry-run"
    assert "[dry-run]" in capsys.readouterr().out


def test_num_processes_greater_than_one_calls_notebook_launcher(monkeypatch):
    fake_config = _fake_config()
    monkeypatch.setattr(run_experiment_module, "load_yaml", lambda path: fake_config)
    calls = []
    monkeypatch.setattr(
        run_experiment_module,
        "notebook_launcher",
        lambda fn, args, num_processes, use_port: calls.append(
            {"fn": fn, "args": args, "num_processes": num_processes, "use_port": use_port}
        ),
        raising=False,
    )
    monkeypatch.setattr(run_experiment_module, "_NOTEBOOK_LAUNCHER_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiment.py",
            "--config",
            "configs/fake.yaml",
            "--num-processes",
            "3",
            "--main-process-port",
            "12345",
        ],
    )

    run_experiment_module.main()

    assert len(calls) == 1
    call = calls[0]
    assert call["fn"] is run_experiment_module._train_entrypoint
    assert call["args"] == (fake_config,)
    assert call["num_processes"] == 3
    assert call["use_port"] == "12345"


def test_num_processes_without_accelerate_raises_clear_error(monkeypatch):
    monkeypatch.setattr(run_experiment_module, "load_yaml", lambda path: _fake_config())
    monkeypatch.setattr(run_experiment_module, "_NOTEBOOK_LAUNCHER_AVAILABLE", False, raising=False)
    monkeypatch.setattr(
        sys, "argv", ["run_experiment.py", "--config", "configs/fake.yaml", "--num-processes", "2"]
    )

    try:
        run_experiment_module.main()
        assert False, "expected SystemExit"
    except SystemExit as exc:
        assert "accelerate" in str(exc)
