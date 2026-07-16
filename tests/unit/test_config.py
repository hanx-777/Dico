from pathlib import Path

import pytest

from dico.config import apply_overrides, deep_merge, load_yaml, save_yaml, validate_known_config_fields


# --- deep_merge --------------------------------------------------------------


def test_deep_merge_overrides_scalar_leaves():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"a": 10, "b": {"c": 20}}

    result = deep_merge(base, override)

    assert result == {"a": 10, "b": {"c": 20, "d": 3}}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    override = {"a": {"b": 2}}

    result = deep_merge(base, override)

    assert base == {"a": {"b": 1}}
    assert override == {"a": {"b": 2}}
    assert result == {"a": {"b": 2}}


def test_deep_merge_replaces_lists_instead_of_concatenating():
    base = {"kernel": [0.25, 0.50, 0.25]}
    override = {"kernel": [1.0]}

    result = deep_merge(base, override)

    assert result["kernel"] == [1.0]


def test_deep_merge_replaces_dict_with_scalar_when_override_is_not_a_dict():
    base = {"a": {"nested": 1}}
    override = {"a": "scalar"}

    result = deep_merge(base, override)

    assert result == {"a": "scalar"}


# --- load_yaml / inherits chain ----------------------------------------------


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_yaml_single_file_has_no_inherits_key_leftover(tmp_path):
    configs_dir = tmp_path / "configs"
    _write_yaml(configs_dir / "base.yaml", "a: 1\nb: 2\n")

    result = load_yaml(configs_dir / "base.yaml")

    assert result["a"] == 1
    assert result["b"] == 2
    assert "inherits" not in result


def test_load_yaml_resolves_two_hop_inherits_chain(tmp_path):
    configs_dir = tmp_path / "configs"
    _write_yaml(configs_dir / "base.yaml", "a: 1\nnested:\n  x: 1\n  y: 1\n")
    _write_yaml(
        configs_dir / "methods" / "child.yaml",
        "inherits: ../base.yaml\nnested:\n  x: 2\n",
    )

    result = load_yaml(configs_dir / "methods" / "child.yaml")

    assert result["a"] == 1
    assert result["nested"] == {"x": 2, "y": 1}


def test_load_yaml_resolves_three_hop_inherits_chain(tmp_path):
    configs_dir = tmp_path / "configs"
    _write_yaml(configs_dir / "base.yaml", "a: 1\nb: 1\nc: 1\n")
    _write_yaml(configs_dir / "debug" / "tiny.yaml", "inherits: ../base.yaml\nb: 2\n")
    _write_yaml(configs_dir / "debug" / "tiny_child.yaml", "inherits: tiny.yaml\nc: 3\n")

    result = load_yaml(configs_dir / "debug" / "tiny_child.yaml")

    assert result["a"] == 1
    assert result["b"] == 2
    assert result["c"] == 3


def test_load_yaml_injects_config_path_and_project_root(tmp_path):
    configs_dir = tmp_path / "configs"
    _write_yaml(configs_dir / "methods" / "child.yaml", "a: 1\n")

    result = load_yaml(configs_dir / "methods" / "child.yaml")

    assert result["_config_path"] == str((configs_dir / "methods" / "child.yaml").resolve())
    assert result["_project_root"] == str(tmp_path.resolve())


def test_load_yaml_empty_file_returns_only_injected_keys(tmp_path):
    configs_dir = tmp_path / "configs"
    _write_yaml(configs_dir / "empty.yaml", "")

    result = load_yaml(configs_dir / "empty.yaml")

    assert set(result) == {"_config_path", "_project_root"}


# --- apply_overrides ----------------------------------------------------------


def test_apply_overrides_sets_dotted_path_leaf():
    config = {"a": {"b": 1}}

    result = apply_overrides(config, ["a.b=2"])

    assert result["a"]["b"] == 2
    assert config["a"]["b"] == 1  # original untouched


def test_apply_overrides_creates_missing_intermediate_dicts():
    config = {}

    result = apply_overrides(config, ["a.b.c=1"])

    assert result == {"a": {"b": {"c": 1}}}


def test_apply_overrides_parses_yaml_scalar_types():
    config = {}

    result = apply_overrides(config, ["seed=42", "flag=true", "ratio=0.98", "name=lora"])

    assert result["seed"] == 42
    assert result["flag"] is True
    assert result["ratio"] == 0.98
    assert result["name"] == "lora"


def test_apply_overrides_applies_multiple_overrides_in_order_last_wins():
    config = {}

    result = apply_overrides(config, ["a=1", "a=2"])

    assert result["a"] == 2


def test_apply_overrides_raises_without_equals_sign():
    with pytest.raises(ValueError):
        apply_overrides({}, ["not_an_override"])


def test_apply_overrides_raises_when_path_crosses_non_dict_value():
    config = {"a": 1}

    with pytest.raises(ValueError):
        apply_overrides(config, ["a.b=2"])


# --- save_yaml -----------------------------------------------------------------


def test_save_yaml_round_trips_through_load_yaml(tmp_path):
    config = {"method": "dico_cd_da", "rank": 8, "nested": {"x": 1}}
    out_path = tmp_path / "resolved" / "config_resolved.yaml"

    save_yaml(out_path, config)
    reloaded = load_yaml(out_path)

    assert reloaded["method"] == "dico_cd_da"
    assert reloaded["rank"] == 8
    assert reloaded["nested"] == {"x": 1}


# --- schema validation --------------------------------------------------------


def test_validate_known_config_fields_rejects_unknown_nested_key():
    config = {
        "method": "lora",
        "rank": 8,
        "training": {"batch_size": 4, "typo_batch_size": 4},
    }

    with pytest.raises(ValueError, match="training.typo_batch_size"):
        validate_known_config_fields(config)


def test_validate_known_config_fields_accepts_train_source_entries():
    config = {
        "method": "lora",
        "rank": 8,
        "data": {
            "train_sources": [
                {"path": "data/metamathqa/train.jsonl", "group": "math", "limit": 100000},
                {"path": "data/codefeedback/train.jsonl", "group": "code", "limit": 100000},
            ]
        },
    }

    validate_known_config_fields(config)
