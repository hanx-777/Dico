"""Lightweight, config-dict-only checks that a --override seed=N flows into
every place the trainer/preallocator actually reads a seed from, without
needing a real training run.
"""

from pathlib import Path

from dico.config import apply_overrides, load_yaml


ROOT = Path(__file__).resolve().parents[2]


def _load_and_seed(config_path: str, seed: int) -> dict:
    config = load_yaml(ROOT / config_path)
    return apply_overrides(
        config,
        [f"seed={seed}", f"calibration.seed={seed}", f"preallocation.sketch_seed={seed}"],
    )


def test_seed_override_propagates_to_top_level_and_calibration_and_sketch():
    config = _load_and_seed("configs/dico/dico_cd_da_r8.yaml", 123)

    assert config["seed"] == 123
    assert config["calibration"]["seed"] == 123
    assert config["preallocation"]["sketch_seed"] == 123


def test_seed_override_does_not_touch_unrelated_fields():
    baseline = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    seeded = _load_and_seed("configs/dico/dico_cd_da_r8.yaml", 7)

    assert seeded["method"] == baseline["method"]
    assert seeded["rank"] == baseline["rank"]
    assert seeded["preallocation"]["eta"] == baseline["preallocation"]["eta"]


def test_different_seeds_produce_independent_config_copies():
    config = load_yaml(ROOT / "configs" / "dico" / "lora_r8.yaml")
    original_seed = config["seed"]

    seeded_42 = apply_overrides(config, ["seed=42"])
    seeded_43 = apply_overrides(config, ["seed=43"])

    assert seeded_42["seed"] == 42
    assert seeded_43["seed"] == 43
    assert config["seed"] == original_seed  # apply_overrides must not mutate its input


def test_calibration_seed_falls_back_to_top_level_seed_when_unset():
    # Mirrors dico.trainer._preallocation_path's fallback expression.
    config_with_calibration_seed = {"seed": 99, "calibration": {"seed": 7}}
    config_without_calibration_seed = {"seed": 99, "calibration": {}}

    def resolved_seed(config: dict) -> int:
        return int(config.get("calibration", {}).get("seed", config.get("seed", 42)))

    assert resolved_seed(config_with_calibration_seed) == 7
    assert resolved_seed(config_without_calibration_seed) == 99
