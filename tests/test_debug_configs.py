from pathlib import Path

from dico_rank.config import load_yaml


def test_debug_configs_are_tiny_and_offline():
    root = Path(__file__).resolve().parents[1]
    for name in ["tiny_lora.yaml", "tiny_dico_dynamic.yaml"]:
        cfg = load_yaml(root / "configs" / "debug" / name)
        assert cfg["model"]["type"] == "tiny"
        assert cfg["data"]["source"] == "tiny"
        assert cfg["training"]["max_steps"] <= 3


def test_all_eight_experiment_configs_exist():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "lora_r4.yaml",
        "lora_r8.yaml",
        "dico_pre_r4.yaml",
        "dico_pre_r8.yaml",
        "dico_dynamic_r4.yaml",
        "dico_dynamic_r8.yaml",
        "dico_predynamic_r4.yaml",
        "dico_predynamic_r8.yaml",
    }
    actual = {path.name for path in (root / "configs" / "experiments").glob("*.yaml")}
    assert expected <= actual


def test_lora_eta98_and_ablation_configs_load():
    root = Path(__file__).resolve().parents[1]
    for name in ["lora_r4_eta98.yaml", "lora_r8_eta98.yaml"]:
        cfg = load_yaml(root / "configs" / "experiments" / name)
        assert cfg["method"] == "lora"
        assert cfg["budget"]["enforce_target_ratio"] == 0.98

    expected_ablations = {
        "dico_pre_r8_no_relaxation.yaml",
        "dico_pre_r8_eta100.yaml",
        "dico_pre_r8_answer_full.yaml",
        "dico_pre_r8_random.yaml",
        "dico_predynamic_r8_move20.yaml",
    }
    ablation_dir = root / "configs" / "experiments" / "ablations"
    found_ablations = {path.name for path in ablation_dir.glob("*.yaml") if not path.name.startswith("._")}
    assert expected_ablations == found_ablations
    for name in expected_ablations:
        cfg = load_yaml(ablation_dir / name)
        assert cfg["rank"] == 8
        assert cfg["method"] in {"dico_pre", "dico_predynamic"}


def test_base_config_disables_training_time_eval_by_default():
    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(root / "configs" / "base.yaml")

    assert cfg["training"]["eval_steps"] == 0
