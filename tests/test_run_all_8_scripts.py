import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(args, tmp_path, extra_env=None):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "HF_HOME": str(tmp_path / "hf_cache"),
        }
    )
    env.pop("HF_ENDPOINT", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def test_shell_scripts_pass_bash_syntax_check():
    scripts = [
        "scripts/run_all_8.sh",
        "scripts/run_all_8_experiments.sh",
        "scripts/run_all_8_nohup.sh",
        "scripts/env_hf_mirror.sh",
        "scripts/lib/hf_env.sh",
        "scripts/lib/runtime.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", script], cwd=ROOT, check=True)


def test_nohup_dry_run_uses_default_hf_mirror(tmp_path):
    result = run_script(
        ["scripts/run_all_8.sh", "--nohup", "--override", "model.name_or_path=/m"],
        tmp_path,
    )

    assert "nohup_mode=1" in result.stdout
    assert "HF_ENDPOINT=https://hf-mirror.com" in result.stdout
    assert "model.name_or_path=/m" in result.stdout


def test_nohup_dry_run_without_train_args_has_no_empty_argument(tmp_path):
    result = run_script(["scripts/run_all_8.sh", "--nohup"], tmp_path)

    assert "nohup_mode=1" in result.stdout
    assert result.stdout.rstrip().endswith("train_args:")


def test_output_dir_flag_overrides_project_output_override(tmp_path):
    result = run_script(
        [
            "scripts/run_all_8.sh",
            "--output_dir",
            "outputs_new",
            "--override",
            "project.output_dir=outputs_old",
            "--override",
            "training.max_steps=2",
        ],
        tmp_path,
    )

    assert "output_dir=outputs_new" in result.stdout
    assert "project.output_dir=outputs_new" in result.stdout
    assert "calibration.save_dir=outputs_new/preallocations" in result.stdout
    assert "outputs_old" not in result.stdout


def test_output_dir_flag_without_train_args_is_safe_under_nounset(tmp_path):
    result = run_script(
        ["scripts/run_all_8.sh", "--output_dir", "outputs_new"],
        tmp_path,
    )

    assert "output_dir=outputs_new" in result.stdout
    assert "project.output_dir=outputs_new" in result.stdout
    assert "calibration.save_dir=outputs_new/preallocations" in result.stdout


def test_output_dir_flag_preserves_explicit_calibration_save_dir(tmp_path):
    result = run_script(
        [
            "scripts/run_all_8.sh",
            "--output_dir",
            "outputs_new",
            "--override",
            "calibration.save_dir=/tmp/prealloc",
            "--override",
            "training.max_steps=2",
        ],
        tmp_path,
    )

    assert "project.output_dir=outputs_new" in result.stdout
    assert "calibration.save_dir=/tmp/prealloc" in result.stdout
    assert "calibration.save_dir=outputs_new/preallocations" not in result.stdout


def test_no_hf_mirror_does_not_set_endpoint(tmp_path):
    result = run_script(
        ["scripts/run_all_8.sh", "--no_hf_mirror", "--override", "training.max_steps=2"],
        tmp_path,
    )

    assert "HF_ENDPOINT=\n" in result.stdout
