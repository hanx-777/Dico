from pathlib import Path

from dico_rank.config import load_yaml


ALLOCATOR_3X3_COMBINATIONS = {
    ("marginal_curve", "budget_guardrails"),
    ("marginal_curve", "layer_diffusion"),
    ("marginal_curve", "concentration_penalty"),
    ("prototype_bundle", "budget_guardrails"),
    ("prototype_bundle", "layer_diffusion"),
    ("prototype_bundle", "concentration_penalty"),
    ("soft_slot", "budget_guardrails"),
    ("soft_slot", "layer_diffusion"),
    ("soft_slot", "concentration_penalty"),
}


def test_debug_configs_are_tiny_and_offline():
    root = Path(__file__).resolve().parents[1]
    for name in ["tiny_lora.yaml", "tiny_dico_pre.yaml", "tiny_dico_dynamic.yaml"]:
        cfg = load_yaml(root / "configs" / "debug" / name)
        assert cfg["model"]["type"] == "tiny"
        assert cfg["data"]["source"] == "tiny"
        assert cfg["training"]["max_steps"] <= 3


def test_main_experiment_configs_include_static_and_optional_dynamic_variants():
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


def test_pre_allocator_3x3_configs_cover_mainstream_r8_matrix():
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs" / "experiments" / "allocator_3x3"
    paths = sorted(config_dir.glob("*.yaml"))

    assert len(paths) == 9

    seen = set()
    for path in paths:
        cfg = load_yaml(path)
        allocator = cfg["preallocation"]["rank_allocator"]
        combo = (allocator["atom_to_rank"], allocator["smoothing"])
        seen.add(combo)

        assert cfg["method"] == "dico_pre"
        assert cfg["rank"] == 8
        assert cfg["rank_strategy"]["init"] == "dico_pre"
        assert cfg["lora"]["alpha"] == 16
        assert cfg["lora"]["dropout"] == 0.05
        assert cfg["preallocation"]["eta"] == 0.98
        assert cfg["preallocation"]["allocation_method"] == "directional_budgeted"
        assert allocator["utility"] == {
            "align_gamma": 1.0,
            "use_log1p": True,
            "type_normalization": "median",
        }
        assert allocator["marginal_curve"]["decay"] == "sqrt"
        assert allocator["prototype_bundle"] == {
            "similarity_threshold": 0.8,
            "residual_weight": 0.25,
        }
        assert allocator["soft_slot"] == {
            "temperature": 1.0,
            "slot_decay": 0.15,
        }
        assert allocator["budget_guardrails"]["max_rank_per_module"] is None
        assert allocator["budget_guardrails"]["layer_cap_multiplier"] == 1.8
        assert allocator["budget_guardrails"]["type_cap_multiplier"] == 2.0
        assert allocator["layer_diffusion"]["kernel"] == [0.25, 0.50, 0.25]
        assert allocator["concentration_penalty"]["lambda"] == 0.02

    assert seen == ALLOCATOR_3X3_COMBINATIONS


def test_base_config_disables_training_time_eval_by_default():
    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(root / "configs" / "base.yaml")

    assert cfg["training"]["eval_steps"] == 0
    assert cfg["dynamic"]["enabled"] is False
