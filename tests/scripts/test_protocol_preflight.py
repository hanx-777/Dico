import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_protocol_preflight_writes_machine_and_markdown_reports(tmp_path):
    json_path = tmp_path / "protocol_preflight.json"
    md_path = tmp_path / "protocol_preflight.md"

    result = subprocess.run(
        [
            "python",
            "scripts/protocol_preflight.py",
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "PASS"
    assert payload["summary"]["failed_checks"] == 0
    assert payload["summary"]["checked_configs"] >= 1
    check_ids = {
        check["id"]
        for config in payload["configs"]
        for check in config["checks"]
    }
    assert {
        "strict_schema",
        "single_gpu_global_batch_64",
        "covra_candidate_budget",
        "adapter_fp32_base_bf16",
        "budget_policy",
        "dropout_zero",
        "gradient_checkpointing_off",
        "legacy_fields_isolated",
        "unresolved_fields_marked",
        "covra_module_scalar_template",
    } <= check_ids

    markdown = md_path.read_text()
    assert "# Protocol Preflight" in markdown
    assert "| config | status | failed_checks |" in markdown
    assert "configs/dico/dico_cd_da_r8.yaml" in markdown


def test_protocol_preflight_fails_fast_on_candidate_budget_violation(tmp_path):
    bad_config = tmp_path / "bad_covra.yaml"
    bad_config.write_text(
        "\n".join(
            [
                f"inherits: {ROOT / 'configs' / 'dico' / 'dico_cd_da_r8.yaml'}",
                "experiment_name: bad_covra_k7",
                "preallocation:",
                "  top_k_atoms: 7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "bad_preflight.json"
    md_path = tmp_path / "bad_preflight.md"
    result = subprocess.run(
        [
            "python",
            "scripts/protocol_preflight.py",
            "--config",
            str(bad_config),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    failed = [
        check
        for config in payload["configs"]
        for check in config["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "covra_candidate_budget" for check in failed)
    assert "top_k_atoms" in result.stderr


def test_protocol_preflight_fails_on_covra_calibration_sample_mismatch(tmp_path):
    bad_config = tmp_path / "bad_covra_calibration.yaml"
    bad_config.write_text(
        "\n".join(
            [
                f"inherits: {ROOT / 'configs' / 'dico' / 'dico_cd_da_r8.yaml'}",
                "experiment_name: bad_covra_calibration",
                "calibration:",
                "  num_samples: 128",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "bad_calibration_preflight.json"
    md_path = tmp_path / "bad_calibration_preflight.md"
    result = subprocess.run(
        [
            "python",
            "scripts/protocol_preflight.py",
            "--config",
            str(bad_config),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    assert payload["summary"]["status"] == "FAIL"
    failed = [
        check
        for config in payload["configs"]
        for check in config["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "covra_calibration_samples" for check in failed)
    assert "num_samples" in result.stderr


def test_protocol_preflight_fails_on_evaluation_protocol_mismatch(tmp_path):
    bad_config = tmp_path / "bad_eval_protocol.yaml"
    bad_config.write_text(
        "\n".join(
            [
                f"inherits: {ROOT / 'configs' / 'dico' / 'dico_cd_da_r8.yaml'}",
                "experiment_name: bad_eval_protocol",
                "evaluation:",
                "  humaneval_num_samples_per_task: 5",
                "  mtbench_local:",
                "    judge_prompt_version: ''",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "bad_eval_preflight.json"
    md_path = tmp_path / "bad_eval_preflight.md"
    result = subprocess.run(
        [
            "python",
            "scripts/protocol_preflight.py",
            "--config",
            str(bad_config),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    failed = [
        check
        for config in payload["configs"]
        for check in config["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "evaluation_protocol" for check in failed)
    assert "humaneval_num_samples_per_task" in result.stderr
    assert "judge_prompt_version" in result.stderr


def test_protocol_preflight_fails_when_unlocked_model_revision_is_not_marked(tmp_path):
    bad_config = tmp_path / "bad_unmarked_revision.yaml"
    bad_config.write_text(
        "\n".join(
            [
                f"inherits: {ROOT / 'configs' / 'dico' / 'dico_cd_da_r8.yaml'}",
                "experiment_name: bad_unmarked_revision",
                "protocol:",
                "  unresolved_fields:",
                "    - field: preallocation.rho",
                "      status: provisional",
                "      reason: retained test row",
                "    - field: preallocation.response_agg_groups",
                "      status: provisional",
                "      reason: retained test row",
                "    - field: preallocation.r_min_multiplier",
                "      status: provisional",
                "      reason: retained test row",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "bad_revision_preflight.json"
    md_path = tmp_path / "bad_revision_preflight.md"
    result = subprocess.run(
        [
            "python",
            "scripts/protocol_preflight.py",
            "--config",
            str(bad_config),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    failed = [
        check
        for config in payload["configs"]
        for check in config["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "unresolved_fields_marked" for check in failed)
    assert "model.revision" in result.stderr
    assert "model.tokenizer_revision" in result.stderr


def test_protocol_preflight_fails_when_max_steps_do_not_match_train_volume(tmp_path):
    bad_config = tmp_path / "bad_steps.yaml"
    bad_config.write_text(
        "\n".join(
            [
                f"inherits: {ROOT / 'configs' / 'dico' / 'lora_r8.yaml'}",
                "experiment_name: bad_steps",
                "data:",
                "  train_limit: 64000",
                "training:",
                "  max_steps: 1563",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "bad_steps_preflight.json"
    md_path = tmp_path / "bad_steps_preflight.md"
    result = subprocess.run(
        [
            "python",
            "scripts/protocol_preflight.py",
            "--config",
            str(bad_config),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    failed = [
        check
        for config in payload["configs"]
        for check in config["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "max_steps_match_train_volume" for check in failed)
    assert "expected max_steps=1000" in result.stderr
    assert "train_samples=64000" in result.stderr


def test_protocol_preflight_fails_on_budget_policy_mismatch(tmp_path):
    bad_config = tmp_path / "bad_budget_policy.yaml"
    bad_config.write_text(
        "\n".join(
            [
                f"inherits: {ROOT / 'configs' / 'dico' / 'dico_cd_da_r8.yaml'}",
                "experiment_name: bad_budget_policy",
                "budget:",
                "  mode: nominal_rank",
                "  warning_threshold: 0.05",
                "  enforce_target_ratio: 0.99",
                "  enforce_min_ratio: 0.90",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "bad_budget_preflight.json"
    md_path = tmp_path / "bad_budget_preflight.md"
    result = subprocess.run(
        [
            "python",
            "scripts/protocol_preflight.py",
            "--config",
            str(bad_config),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    failed = [
        check
        for config in payload["configs"]
        for check in config["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "budget_policy" for check in failed)
    assert "equal_trainable_params" in result.stderr
    assert "enforce_target_ratio" in result.stderr


def test_protocol_preflight_fails_on_covra_module_scalar_template_mismatch(tmp_path):
    bad_config = tmp_path / "bad_covra_m_template.yaml"
    bad_config.write_text(
        "\n".join(
            [
                f"inherits: {ROOT / 'configs' / 'ablations' / 'covra_module_scalar.yaml'}",
                "experiment_name: bad_covra_m_template",
                "preallocation:",
                "  module_scalar_template_normalization: not_sum_to_module_energy",
                "  module_scalar_template:",
                "    - 1.0",
                "    - 2.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_path = tmp_path / "bad_covra_m_template_preflight.json"
    md_path = tmp_path / "bad_covra_m_template_preflight.md"
    result = subprocess.run(
        [
            "python",
            "scripts/protocol_preflight.py",
            "--config",
            str(bad_config),
            "--json-output",
            str(json_path),
            "--markdown-output",
            str(md_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(json_path.read_text())
    failed = [
        check
        for config in payload["configs"]
        for check in config["checks"]
        if check["status"] == "FAIL"
    ]
    assert any(check["id"] == "covra_module_scalar_template" for check in failed)
    assert "module_scalar_template" in result.stderr
    assert "sum_to_module_energy" in result.stderr
