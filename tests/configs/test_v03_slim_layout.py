from pathlib import Path
from typing import Any

from dico.config import load_yaml, validate_known_config_fields


ROOT = Path(__file__).resolve().parents[2]


def test_legacy_experiment_families_are_removed():
    removed_paths = [
        ROOT / "configs" / "methods",
        ROOT / "configs" / "debug",
        ROOT / "configs" / "ablations" / "single_factor",
        ROOT / "configs" / "ablations" / "allocator_grid",
        ROOT / "configs" / "extensions" / "allocator",
        ROOT / "scripts" / "run",
        ROOT / "src" / "dico" / "dynamic_allocation.py",
        ROOT / "src" / "dico" / "rank_allocator.py",
        ROOT / "scripts" / "run_dico_v027.sh",
        ROOT / "configs" / "dico" / "dico_v027_r8.yaml",
        ROOT / "configs" / "dico" / "gora_original.yaml",
    ]

    assert [path for path in removed_paths if path.exists()] == []


def test_v03_configs_only_use_protocol_aligned_methods():
    allowed_methods = {"lora", "rs_lora", "adalora", "gora_bw", "gora_public", "gora_bm", "dico_cd", "dico_cd_da"}
    config_paths = sorted((ROOT / "configs" / "dico").glob("*.yaml"))
    config_paths += sorted((ROOT / "configs" / "ablations").glob("*.yaml"))
    assert config_paths

    for path in config_paths:
        cfg = load_yaml(path)
        method = cfg.get("method")
        if method is not None:
            assert method in allowed_methods, path
        assert cfg.get("method") not in {"dico_pre", "dico_dynamic", "dico_predynamic", "dico_v027"}


def test_covra_main_config_uses_reference_protocol_defaults():
    cfg = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    r_max = int(cfg["rank"] * float(cfg["preallocation"]["r_max_multiplier"]))

    assert cfg["calibration"]["num_samples"] == 256
    assert cfg["lora"]["dropout"] == 0.05
    assert cfg["lora"]["adapter_dtype"] == "bfloat16"
    assert cfg["training"]["gradient_checkpointing"] is True
    assert cfg["preallocation"]["allocation_method"] == "covra_v05"
    assert cfg["preallocation"]["top_k_atoms"] == 8
    assert cfg["preallocation"]["top_k_atoms"] < r_max
    assert cfg["preallocation"]["sketch_dim"] == 16
    assert cfg["preallocation"]["beta"] == 1.0
    assert cfg["dico"]["taxonomy"]["enabled"] is True
    assert cfg["dico"]["procurement"]["beta"] == 0.5
    assert "legacy_covra_v05" not in cfg["dico"]
    assert cfg["evaluation"]["mtbench_local"]["judge_prompt_version"] == "fastchat-v0.2.36"
    assert cfg["evaluation"]["mtbench_local"]["judge_model"]
    assert cfg["evaluation"]["mtbench_local"]["swap_positions"] is True

    unresolved = {row["field"]: row for row in cfg["protocol"]["unresolved_fields"]}
    for field in {"model.revision", "model.tokenizer_revision"}:
        assert field in unresolved
        assert unresolved[field]["status"] == "unresolved"
        assert unresolved[field]["reason"]


def test_all_final_covra_configs_have_enough_candidates_for_rank_cap():
    config_paths = sorted((ROOT / "configs" / "dico").glob("*.yaml"))
    config_paths += sorted((ROOT / "configs" / "ablations").glob("*.yaml"))

    for path in config_paths:
        cfg = load_yaml(path)
        method = cfg.get("method")
        allocation_method = cfg.get("preallocation", {}).get("allocation_method")
        if method in {"dico_cd", "dico_cd_da"} and allocation_method in {
            "covra_full",
            "covra_independent",
            "covra_module_scalar",
        }:
            r_max = int(cfg["rank"] * float(cfg["preallocation"]["r_max_multiplier"]))
            assert int(cfg["preallocation"]["top_k_atoms"]) >= r_max, path


def test_r32_pilot_config_exists_and_uses_k128_rank_cap():
    path = ROOT / "configs" / "dico" / "dico_cd_da_r32_pilot.yaml"
    cfg = load_yaml(path)
    r_max = int(cfg["rank"] * float(cfg["preallocation"]["r_max_multiplier"]))

    assert cfg["rank"] == 32
    assert r_max == 128
    assert cfg["preallocation"]["top_k_atoms"] >= 128
    assert cfg["preallocation"]["top_k_atoms"] >= r_max
    assert cfg["preallocation"]["sketch_dim"] >= cfg["preallocation"]["top_k_atoms"]
    assert cfg["training"]["max_steps"] == 1
    assert cfg["evaluation"]["compute_accuracy"] is False


def test_covra_module_scalar_config_declares_fixed_rank_template():
    cfg = load_yaml(ROOT / "configs" / "ablations" / "covra_module_scalar.yaml")
    preallocation = cfg["preallocation"]
    r_max = int(cfg["rank"] * float(preallocation["r_max_multiplier"]))

    assert preallocation["allocation_method"] == "covra_module_scalar"
    assert preallocation["module_scalar_template_formula"] == "w_j = 1 / j for j=1..r_max"
    assert preallocation["module_scalar_template_normalization"] == "sum_to_module_energy"
    assert preallocation["module_scalar_template"] == [1.0 / float(rank) for rank in range(1, r_max + 1)]
    assert all(
        preallocation["module_scalar_template"][idx] >= preallocation["module_scalar_template"][idx + 1]
        for idx in range(len(preallocation["module_scalar_template"]) - 1)
    )


def test_formal_configs_do_not_contain_unknown_schema_fields():
    config_paths = sorted((ROOT / "configs" / "dico").glob("*.yaml"))
    config_paths += sorted((ROOT / "configs" / "ablations").glob("*.yaml"))

    for path in config_paths:
        validate_known_config_fields(load_yaml(path))


def test_adalora_config_uses_svd_triplet_and_records_peak_final_ranks():
    cfg = load_yaml(ROOT / "configs" / "dico" / "adalora_r8.yaml")

    assert cfg["method"] == "adalora"
    assert cfg["rank"] == 8
    assert cfg["adalora"]["init_rank"] == 12
    assert cfg["adalora"]["target_rank"] == 8
    assert cfg["adalora"]["tinit"] == 150
    assert cfg["adalora"]["tfinal"] == 900
    assert cfg["lora"]["injection"] == "adalora"
    assert cfg["adalora"]["beta1"] == cfg["adalora"]["beta2"] == 0.85
    assert cfg["training"]["learning_rate"] == 5e-4
    assert cfg["lora"]["max_rank_multiplier"] == 1.5
    assert cfg["calibration"]["enabled"] is False


def _without_experiment_identity(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_experiment_identity(item)
            for key, item in value.items()
            if key not in {"experiment_name", "_config_path"}
        }
    if isinstance(value, list):
        return [_without_experiment_identity(item) for item in value]
    return value


def _flatten_config_leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"experiment_name", "_config_path", "ablation"}:
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten_config_leaves(item, child_prefix))
        return flattened
    return {prefix: value}


def test_required_ablation_configs_have_unique_non_identity_definitions():
    required = [
        "covra_independent",
        "covra_module_scalar",
        "global_only",
        "grouped_only",
        "no_sign_split",
        "no_type_scaling",
        "no_log_compression",
        "proportional_rounding",
        "random_init",
        "uniform_rank_covra_init",
        "covra_rank_random_init",
    ]
    normalized: dict[str, Any] = {}
    for name in required:
        path = ROOT / "configs" / "ablations" / f"{name}.yaml"
        assert path.exists(), path
        normalized[name] = _without_experiment_identity(load_yaml(path))
        ablation = normalized[name].get("ablation", {})
        assert ablation.get("id") == name
        assert ablation.get("expected_difference"), path

    duplicates: list[tuple[str, str]] = []
    for left_index, left in enumerate(required):
        for right in required[left_index + 1 :]:
            if normalized[left] == normalized[right]:
                duplicates.append((left, right))
    assert duplicates == []


def test_ablation_metadata_declares_every_non_identity_config_difference():
    required = [
        "covra_independent",
        "covra_module_scalar",
        "global_only",
        "grouped_only",
        "no_sign_split",
        "no_type_scaling",
        "no_log_compression",
        "proportional_rounding",
        "random_init",
        "uniform_rank_covra_init",
        "covra_rank_random_init",
    ]
    for name in required:
        path = ROOT / "configs" / "ablations" / f"{name}.yaml"
        cfg = load_yaml(path)
        ablation = cfg["ablation"]
        reference = load_yaml(ROOT / ablation["reference_config"])
        cfg_leaves = _flatten_config_leaves(cfg)
        ref_leaves = _flatten_config_leaves(reference)
        diffs = {
            key
            for key in set(cfg_leaves) | set(ref_leaves)
            if cfg_leaves.get(key) != ref_leaves.get(key)
        }
        declared = ablation.get("controlled_difference_fields") or [ablation["single_factor"]]
        declared_fields = {str(field) for field in declared}
        undeclared = {
            field
            for field in diffs
            if field not in declared_fields and field.split(".")[-1] not in declared_fields
        }
        assert undeclared == set(), f"{path}: undeclared config differences {sorted(undeclared)}"
