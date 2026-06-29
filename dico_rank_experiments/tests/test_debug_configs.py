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
