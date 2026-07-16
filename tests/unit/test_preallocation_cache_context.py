from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from dico.config import load_yaml
from dico.preallocation import build_preallocation_cache_context
from dico.trainer import _preallocation_metadata_from_payload


ROOT = Path(__file__).resolve().parents[2]
MODULE_NAMES = ["model.layers.0.self_attn.q_proj"]
MODULE_DIMS = {MODULE_NAMES[0]: {"in_dim": 8, "out_dim": 8}}


def _context(config: dict) -> dict:
    return build_preallocation_cache_context(config, MODULE_NAMES, MODULE_DIMS)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("taxonomy", "alpha", 0.01),
        ("pseudo_group", "max_k", 8),
        ("split", "mode", "group"),
        ("coverage", "window_h", 0),
        ("coverage", "relative_stop_delta", 0.0),
        ("procurement", "beta", 1.0),
        ("init", "mode", "kaiming_zero_B"),
    ],
)
def test_covra_method_controls_are_part_of_cache_context(section: str, key: str, value: object) -> None:
    config = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    changed = deepcopy(config)
    changed["dico"][section][key] = value

    assert _context(changed) != _context(config)


def test_preallocation_beta_is_part_of_cache_context() -> None:
    config = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    changed = deepcopy(config)
    changed["preallocation"]["beta"] = 0.25

    assert _context(changed) != _context(config)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("preallocation", "lambda_cov", 0.5),
        ("preallocation", "response_agg_groups", 8),
        ("preallocation", "sketch_oversample", 32),
        ("preallocation", "allocation_device", "auto"),
        ("calibration", "group_sampling", "balanced"),
        ("data", "max_length", 256),
        ("data", "shuffle", True),
        ("data", "dataset_seed", 43),
    ],
)
def test_profile_and_allocation_inputs_are_part_of_cache_context(
    section: str,
    key: str,
    value: object,
) -> None:
    config = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    changed = deepcopy(config)
    changed[section][key] = value

    assert _context(changed) != _context(config)


def test_top_level_seed_and_train_sources_are_part_of_cache_context() -> None:
    config = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    seeded = deepcopy(config)
    seeded["seed"] = 43
    mixed = deepcopy(config)
    mixed["data"]["train_sources"] = [{"path": "data/other.jsonl", "group": "other", "limit": 10}]

    assert _context(seeded) != _context(config)
    assert _context(mixed) != _context(config)


def test_reference_covra_artifact_metadata_survives_preallocation_payload_extraction(tmp_path) -> None:
    config = load_yaml(ROOT / "configs" / "dico" / "dico_cd_da_r8.yaml")
    payload = {
        "rank_allocation": {MODULE_NAMES[0]: 2},
        "module_logs": [{"module_name": MODULE_NAMES[0], "final_rank": 2}],
        "atom_mode": "svd",
        "allocation_method": "covra_v05",
        "taxonomy_stats": {"consensus": 1},
        "coverage_trace": [{"step": 1}],
        "procurement_trace": [{"source": "certified"}],
        "kappa_calibration": {"q_proj": {"fallback_h0": False}},
        "physical_utility": {"direction-0": 1.0},
        "normalization_stats": {"q_proj": {"median": 1.0}},
        "module_quota": {MODULE_NAMES[0]: 2.0},
        "procurement_beta": 0.5,
        "r_min": 2,
        "balanced_fill_ratio": 0.0,
        "budget_gap_ratio": 0.02,
    }

    metadata = _preallocation_metadata_from_payload(
        payload,
        config,
        tmp_path / "preallocation.json",
        source="computed",
    )

    for key in (
        "taxonomy_stats",
        "coverage_trace",
        "procurement_trace",
        "kappa_calibration",
        "physical_utility",
        "normalization_stats",
        "module_quota",
        "procurement_beta",
        "r_min",
        "balanced_fill_ratio",
        "budget_gap_ratio",
    ):
        assert metadata[key] == payload[key]
