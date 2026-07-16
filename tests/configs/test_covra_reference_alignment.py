from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dico.config import load_yaml, validate_known_config_fields


ROOT = Path(__file__).resolve().parents[2]
BASELINE_RESOLVED_SHA256 = {
    "lora_r8.yaml": "19b7f28bff4756aaf707470e60f81490af895024eddf042d80e67a4f81845b9c",
    "adalora_r8.yaml": "d3f1341efb98afd8f931f959d22c7336dda003b9f5f3da996be7df989fd2f7f6",
    "gora_public_r8.yaml": "74e12608178fec4d0aa6aef9f654ffa8aa9940a3ca65026f7063d4c9fee6cd1d",
    "gora_bm_r8.yaml": "4bbb15da727018ac0367b9dee88518fcf3a39e3d93168bb1031c0398cd258642",
}


def _load(relative_path: str) -> dict:
    config = load_yaml(ROOT / relative_path)
    validate_known_config_fields(config)
    return config


def test_formal_covra_matches_reference_resolved_protocol() -> None:
    config = _load("configs/dico/dico_cd_da_r8.yaml")

    assert config["method"] == "dico_cd_da"
    assert config["preallocation"]["allocation_method"] == "covra_v05"
    assert config["data"]["max_length"] == 512
    assert config["data"]["shuffle"] is False
    assert config["training"]["max_steps"] == 1562
    assert config["training"]["batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["training"]["learning_rate"] == 5e-5
    assert config["training"]["weight_decay"] == 5e-4
    assert config["training"]["max_grad_norm"] is None
    assert config["training"]["gradient_checkpointing"] is True
    assert config["training"]["auto_warmup_steps"] == 0
    assert config["lora"]["dropout"] == 0.05
    assert config["lora"]["adapter_dtype"] == "bfloat16"
    assert config["calibration"]["num_samples"] == 256
    assert config["calibration"]["batch_size"] == 4
    assert config["preallocation"]["top_k_atoms"] == 8
    assert config["preallocation"]["sketch_dim"] == 16
    assert config["preallocation"]["eta"] == 0.98
    assert config["preallocation"]["r_min_multiplier"] == 0.25
    assert config["preallocation"]["r_max_multiplier"] == 4
    assert config["preallocation"]["beta"] == 1.0
    assert config["preallocation"]["compute_device"] == "auto"
    assert config["preallocation"]["allocation_device"] == "cpu"
    assert config["dico"]["procurement"]["beta"] == 0.5
    assert config["evaluation"]["batch_size"] == 1
    assert config["evaluation"]["compute_loss"] is True
    assert config["model"]["attn_implementation"] is None


def test_covra_reference_base_is_isolated_from_shared_dico_base() -> None:
    reference = ROOT / "configs" / "dico" / "covra_reference_base.yaml"
    raw_text = reference.read_text(encoding="utf-8")

    assert "inherits:" not in raw_text
    assert _load("configs/dico/dico_cd_da_r8.yaml")["_config_path"] == str(
        (ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml").resolve()
    )


def test_formal_covra_variants_keep_their_distinct_initialization_and_data_semantics() -> None:
    direction_anchored = _load("configs/dico/dico_cd_da_r8.yaml")
    zero_b = _load("configs/dico/dico_cd_r8.yaml")
    mixed = _load("configs/dico/mixed_math_code_r8.yaml")

    assert direction_anchored["dico"]["init"]["mode"] == "direction_anchored"
    assert direction_anchored["dico"]["init"]["zero_B"] is True
    assert zero_b["dico"]["init"]["mode"] == "kaiming_zero_B"
    assert mixed["preallocation"]["allocation_method"] == "covra_v05"
    assert mixed["calibration"]["num_samples"] == 256
    assert mixed["dico"]["split"]["mode"] == "group"


def test_final_covra_ablations_and_r32_pilot_inherit_experimental_protocol() -> None:
    experimental = _load("configs/dico/dico_cd_da_r8_covra_full_experimental.yaml")
    assert experimental["preallocation"]["allocation_method"] == "covra_full"
    assert experimental["preallocation"]["solver"] == "dp"
    assert experimental["data"]["max_length"] == 1024

    paths = [
        "configs/ablations/covra_independent.yaml",
        "configs/ablations/covra_module_scalar.yaml",
        "configs/ablations/covra_rank_random_init.yaml",
        "configs/ablations/global_only.yaml",
        "configs/ablations/grouped_only.yaml",
        "configs/ablations/no_log_compression.yaml",
        "configs/ablations/no_sign_split.yaml",
        "configs/ablations/no_type_scaling.yaml",
        "configs/ablations/proportional_rounding.yaml",
        "configs/ablations/random_init.yaml",
        "configs/ablations/uniform_rank_covra_init.yaml",
        "configs/dico/dico_cd_da_r32_pilot.yaml",
    ]
    for path in paths:
        config = _load(path)
        assert config["data"]["max_length"] == 1024, path
        assert config["preallocation"]["allocation_method"] in {
            "covra_full",
            "covra_independent",
            "covra_module_scalar",
        }, path


def test_protected_baseline_resolved_configs_are_byte_for_byte_stable_canonically() -> None:
    for filename, expected_sha256 in BASELINE_RESOLVED_SHA256.items():
        config = _load(f"configs/dico/{filename}")
        canonical = {
            key: value
            for key, value in config.items()
            if key not in {"_config_path", "_project_root"}
        }
        payload = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        assert hashlib.sha256(payload).hexdigest() == expected_sha256, filename
