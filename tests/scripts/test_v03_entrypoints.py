from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["DRY_RUN"] = "1"
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_run_experiment_supports_dry_run_without_training():
    result = _run(
        [
            "python",
            "scripts/run_experiment.py",
            "--config",
            "configs/dico/dico_cd_da_r8.yaml",
            "--dry-run",
            "--override",
            "training.max_steps=1",
        ]
    )

    assert result.returncode == 0, result.stderr
    assert "[dry-run]" in result.stdout
    assert "experiment_name=dico_cd_da_r8_protocol_aligned" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_v03_shell_wrappers_honor_dry_run_env():
    wrappers = [
        "scripts/run_lora_r8.sh",
        "scripts/run_gora_bw.sh",
        "scripts/run_dico_cd.sh",
        "scripts/run_dico_cd_da.sh",
        "scripts/run_mixed_math_code.sh",
        "scripts/run_ablation_covra_independent.sh",
        "scripts/run_ablation_init.sh",
        "scripts/run_ablation_no_sign_split.sh",
        "scripts/run_ablation_random_init.sh",
    ]

    for wrapper in wrappers:
        result = _run(["bash", wrapper, "--override", "training.max_steps=1"])
        assert result.returncode == 0, f"{wrapper}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert "[dry-run]" in result.stdout, wrapper


def test_legacy_ablation_aliases_are_explicitly_labelled():
    aliases = {
        "scripts/run_ablation_taxonomy.sh": "legacy alias",
        "scripts/run_ablation_coverage.sh": "legacy alias",
    }
    for wrapper, label in aliases.items():
        text = (ROOT / wrapper).read_text(encoding="utf-8")
        assert label in text
