from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


def test_platform_train_dry_run_prints_three_seed_main_experiments(tmp_path):
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    python_bin = tmp_path / "env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            "scripts/platform_train.py",
            "--dry-run",
            "--python-bin",
            str(python_bin),
            "--model-path",
            str(model_dir),
            "--num-gpus",
            "1",
            "--batch-size",
            "4",
            "--grad-accum",
            "16",
            "--calibration-batch-size",
            "4",
            "--cuda-visible-devices",
            "0",
            "--seeds",
            "42,43,44",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(python_bin) in result.stdout
    assert "configs/dico/lora_r8.yaml" in result.stdout
    assert "configs/dico/adalora_r8.yaml" in result.stdout
    assert "configs/dico/gora_public_r8.yaml" in result.stdout
    assert "configs/dico/dico_cd_da_r8.yaml" in result.stdout
    command_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("[platform] ") and "scripts/run_experiment.py" in line
    ]
    assert len(command_lines) == 12
    assert f"model.name_or_path={model_dir}" in result.stdout
    assert "training.batch_size=4" in result.stdout
    assert "training.gradient_accumulation_steps=16" in result.stdout
    assert "calibration.batch_size=4" in result.stdout
    for seed in (42, 43, 44):
        assert f"seed={seed}" in result.stdout
        assert f"preallocation.sketch_seed={seed}" in result.stdout
    assert result.stdout.count("calibration.seed=42") == 1
    assert result.stdout.count("calibration.seed=43") == 1
    assert result.stdout.count("calibration.seed=44") == 1
    assert "--num-processes 1" in result.stdout
    for experiment_name in (
        "lora_r8_protocol_aligned_seed42",
        "adalora_r8_protocol_aligned_seed42",
        "gora_public_r8_aligned_sdpa_v4_seed43",
        "dico_cd_da_r8_protocol_aligned_seed44",
    ):
        assert f"experiment_name={experiment_name}" in result.stdout
        assert f"calibration.save_dir=outputs/covra_main_3seed/preallocations/{experiment_name}" in result.stdout
    # Note: device_map is no longer hardcoded to {"":0} in platform_train.py;
    # it now falls back to the value set in base.yaml (device_map: auto).
    assert 'model.device_map={"":0}' not in result.stdout


def test_experiment_name_for_config_survives_missing_project_dependencies(monkeypatch):
    """platform_train.py's job is to pick/activate the conda env that has the project's
    deps (PyYAML, dico, ...) for the *child* run_experiment.py process, so it must stay
    importable and functional even when run under a bare interpreter that lacks them --
    regression test for a real failure: `ModuleNotFoundError: No module named 'yaml'`
    when platform_train.py itself was launched with such an interpreter."""
    import platform_train

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml" or name.startswith("dico"):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod_name in list(sys.modules):
        if mod_name == "yaml" or mod_name.startswith("dico"):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    result = platform_train.experiment_name_for_config("configs/dico/lora_r8.yaml")
    assert result == "lora_r8"


def test_seed_overrides_sync_calibration_seed_only_for_reference_covra():
    import platform_train

    reference = platform_train.seed_overrides("configs/dico/dico_cd_da_r8.yaml", 43, "outputs/test")
    experimental = platform_train.seed_overrides(
        "configs/dico/dico_cd_da_r8_covra_full_experimental.yaml", 43, "outputs/test"
    )
    lora = platform_train.seed_overrides("configs/dico/lora_r8.yaml", 43, "outputs/test")

    assert "calibration.seed=43" in reference
    assert "calibration.seed=43" not in experimental
    assert "calibration.seed=43" not in lora
    for overrides in (reference, experimental, lora):
        assert "seed=43" in overrides
        assert "preallocation.sketch_seed=43" in overrides


def test_platform_train_dry_run_uses_conda_shell_when_python_bin_is_not_set(tmp_path):
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            "scripts/platform_train.py",
            "--dry-run",
            "--skip-model-check",
            "--model-path",
            str(model_dir),
            "--num-gpus",
            "1",
            "--conda-bin",
            "/ai/lxw/lxw/miniconda3/bin/conda",
            "--conda-env",
            "dico-rank",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'eval "$(/ai/lxw/lxw/miniconda3/bin/conda shell.bash hook)"' in result.stdout
    assert "conda activate dico-rank" in result.stdout
    assert "python scripts/run_experiment.py" in result.stdout


def test_platform_train_dry_run_defaults_to_3way_parallel_seeds(tmp_path):
    """The platform has 3xA800 and 3 seeds per config; by default platform_train.py should
    run each config's 3 seeds concurrently as independent single-GPU jobs (one seed per GPU),
    not wrap them in accelerate/DDP -- see the pivot documented in
    dico_platform_train_multigpu_default_2026_07.md."""
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    python_bin = tmp_path / "env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            "scripts/platform_train.py",
            "--dry-run",
            "--python-bin",
            str(python_bin),
            "--model-path",
            str(model_dir),
            "--seeds",
            "42,43,44",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "num_gpus=3 (parallel single-GPU workers)" in result.stdout
    assert "effective_batch=64" in result.stdout
    assert "training.batch_size=4" in result.stdout
    assert "training.gradient_accumulation_steps=16" in result.stdout
    assert "accelerate" not in result.stdout
    assert "training.per_gpu_batch_size" not in result.stdout

    for config_path in (
        "configs/dico/lora_r8.yaml",
        "configs/dico/adalora_r8.yaml",
        "configs/dico/gora_public_r8.yaml",
        "configs/dico/dico_cd_da_r8.yaml",
    ):
        assert f"config={config_path} seeds=[42, 43, 44]" in result.stdout

    # Each config's 3 seeds must be spread across 3 distinct GPUs, seed order -> gpu order.
    gpu_lines = [line for line in result.stdout.splitlines() if line.startswith("[platform] gpu=")]
    assert len(gpu_lines) == 12
    for config_start in range(0, 12, 3):
        block = gpu_lines[config_start:config_start + 3]
        assert [line.split()[1] for line in block] == ["gpu=0", "gpu=1", "gpu=2"]
        for line, seed in zip(block, (42, 43, 44)):
            assert f"seed={seed}" in line

    command_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("[platform] ") and "scripts/run_experiment.py" in line
    ]
    assert len(command_lines) == 12


def test_platform_train_dry_run_num_gpus_1_is_fully_sequential(tmp_path):
    """--num-gpus 1 chunks seeds into groups of 1, i.e. the old fully-sequential behavior."""
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    python_bin = tmp_path / "env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            "scripts/platform_train.py",
            "--dry-run",
            "--python-bin",
            str(python_bin),
            "--model-path",
            str(model_dir),
            "--num-gpus",
            "1",
            "--seeds",
            "42,43,44",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    gpu_lines = [line for line in result.stdout.splitlines() if line.startswith("[platform] gpu=")]
    assert len(gpu_lines) == 12
    assert all(line.split()[1] == "gpu=0" for line in gpu_lines)


def test_platform_train_accepts_custom_config_list_for_experiment_matrix(tmp_path):
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    python_bin = tmp_path / "env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    result = subprocess.run(
        [
            "python",
            "scripts/platform_train.py",
            "--dry-run",
            "--python-bin",
            str(python_bin),
            "--model-path",
            str(model_dir),
            "--num-gpus",
            "1",
            "--seeds",
            "42",
            "--config",
            "configs/ablations/covra_independent.yaml",
            "--config",
            "configs/ablations/covra_module_scalar.yaml",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "config=configs/ablations/covra_independent.yaml seeds=[42]" in result.stdout
    assert "config=configs/ablations/covra_module_scalar.yaml seeds=[42]" in result.stdout
    assert "configs/dico/lora_r8.yaml" not in result.stdout


def test_platform_train_dry_run_rejects_unknown_config_fields(tmp_path):
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    python_bin = tmp_path / "env" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    bad_config = tmp_path / "bad_config.yaml"
    bad_config.write_text(
        "\n".join(
            [
                f"inherits: {ROOT / 'configs' / 'dico' / 'lora_r8.yaml'}",
                "experiment_name: bad_unknown_field",
                "training:",
                "  typo_batch_size: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "python",
            "scripts/platform_train.py",
            "--dry-run",
            "--python-bin",
            str(python_bin),
            "--model-path",
            str(model_dir),
            "--num-gpus",
            "1",
            "--seeds",
            "42",
            "--config",
            str(bad_config),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    assert "training.typo_batch_size" in result.stderr
    assert "bad_config.yaml" in result.stderr


def _fake_seed_marker_script(tmp_path: Path, fail_seed: int | None) -> Path:
    """A fast fake `--python-bin` that stands in for `python scripts/run_experiment.py`:
    reads its own argv for `seed=NN`, writes a start/end timestamp + CUDA_VISIBLE_DEVICES
    marker file, sleeps briefly (to make concurrency observable), then exits 0 (or 1 for
    fail_seed)."""
    script = tmp_path / "fake_python.sh"
    markers_dir = tmp_path / "markers"
    markers_dir.mkdir(exist_ok=True)
    script.write_text(
        "\n".join([
            "#!/usr/bin/env bash",
            "set -e",
            # Leading "(" on the case pattern is required for this to parse under bash 3.2
            # (macOS default) when nested inside a $(...) command substitution.
            'seed=$(for a in "$@"; do case "$a" in (seed=*) echo "${a#seed=}";; esac; done | head -1)',
            f'marker="{markers_dir}/seed_${{seed}}.marker"',
            'start=$(python3 -c "import time; print(time.time())")',
            "sleep 0.3",
            'end=$(python3 -c "import time; print(time.time())")',
            'echo "start=$start end=$end cuda=$CUDA_VISIBLE_DEVICES" > "$marker"',
            f'if [ "$seed" = "{fail_seed}" ]; then exit 1; fi',
            "exit 0",
        ]),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _parse_marker(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    fields = dict(item.split("=", 1) for item in text.split())
    return {"start": float(fields["start"]), "end": float(fields["end"]), "cuda": fields["cuda"]}


def test_platform_train_runs_seeds_concurrently_on_distinct_gpus(tmp_path):
    """Real (non-dry-run) execution: the 3 seeds of one config must actually overlap in
    wall-clock time (proving concurrent subprocess.Popen, not a sequential loop) and each
    must see a distinct CUDA_VISIBLE_DEVICES."""
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    fake_python = _fake_seed_marker_script(tmp_path, fail_seed=None)

    result = subprocess.run(
        [
            "python",
            "scripts/platform_train.py",
            "--python-bin",
            str(fake_python),
            "--model-path",
            str(model_dir),
            "--skip-model-check",
            "--seeds",
            "42,43,44",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    markers = {
        seed: _parse_marker(tmp_path / "markers" / f"seed_{seed}.marker")
        for seed in (42, 43, 44)
    }
    assert {m["cuda"] for m in markers.values()} == {"0", "1", "2"}

    # Concurrency check: every job's window must overlap the others' windows (all 3 windows
    # share a common point in time), which cannot happen if they ran one at a time.
    latest_start = max(m["start"] for m in markers.values())
    earliest_end = min(m["end"] for m in markers.values())
    assert latest_start < earliest_end, (
        f"Seeds did not run concurrently: {markers}"
    )

    for seed in (42, 43, 44):
        for experiment_name in (
            "lora_r8_protocol_aligned",
            "adalora_r8_protocol_aligned",
                "gora_public_r8_aligned_sdpa_v4",
            "dico_cd_da_r8_protocol_aligned",
        ):
            log_path = ROOT / "logs" / f"{experiment_name}_seed{seed}.log"
            assert log_path.exists(), f"missing log file {log_path}"
            log_path.unlink()


def test_platform_train_failed_seed_does_not_kill_siblings_but_aborts_next_config(tmp_path):
    """One seed failing must not kill the other concurrently-running seeds in the same
    chunk, but must stop platform_train.py before it launches the next config."""
    model_dir = tmp_path / "Meta-Llama-3.1-8B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    fake_python = _fake_seed_marker_script(tmp_path, fail_seed=43)

    result = subprocess.run(
        [
            "python",
            "scripts/platform_train.py",
            "--python-bin",
            str(fake_python),
            "--model-path",
            str(model_dir),
            "--skip-model-check",
            "--seeds",
            "42,43,44",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "seed=43" in combined and "exit=1" in combined

    # Siblings 42/44 in the same (first) chunk must still have completed.
    markers_dir = tmp_path / "markers"
    assert (markers_dir / "seed_42.marker").exists()
    assert (markers_dir / "seed_44.marker").exists()

    # The second config (adalora_r8) and later configs must never have launched.
    assert not (ROOT / "logs" / "adalora_r8_protocol_aligned_seed42.log").exists()
    assert not (ROOT / "logs" / "gora_public_r8_aligned_sdpa_v4_seed42.log").exists()

    for name in ("lora_r8_protocol_aligned_seed42.log", "lora_r8_protocol_aligned_seed43.log", "lora_r8_protocol_aligned_seed44.log"):
        path = ROOT / "logs" / name
        if path.exists():
            path.unlink()
