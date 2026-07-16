"""Unit tests for the trainer's output/log/preallocation-cache directory
naming logic, without running a real training loop.
"""

from pathlib import Path

import dico.trainer as trainer_module
from dico.path_utils import resolve_project_path


def test_resolve_path_joins_relative_paths_under_project_root():
    project_root = Path("/project")

    assert trainer_module._resolve_path(project_root, "outputs") == project_root / "outputs"
    assert trainer_module._resolve_path(project_root, "outputs/dico_v03") == project_root / "outputs/dico_v03"


def test_resolve_path_preserves_absolute_paths():
    project_root = Path("/project")
    absolute = Path("/tmp/external/outputs")

    assert trainer_module._resolve_path(project_root, absolute) == absolute


def test_preallocation_path_uses_calibration_save_dir_and_seed():
    project_root = Path("/project")
    config = {
        "calibration": {"save_dir": "outputs/preallocations", "seed": 7},
        "seed": 42,
    }

    path = trainer_module._preallocation_path(config, project_root, rank=8)

    assert path == project_root / "outputs/preallocations/dico_v03_rank8_seed7.json"


def test_preallocation_path_falls_back_to_default_save_dir_and_top_level_seed():
    project_root = Path("/project")
    config = {}

    path = trainer_module._preallocation_path(config, project_root, rank=4)

    assert path == project_root / "outputs/preallocations/dico_v03_rank4_seed42.json"


def test_preallocation_path_respects_per_experiment_save_dir_naming():
    project_root = Path("/project")
    config = {
        "calibration": {"save_dir": "outputs/dico_v03/preallocations/dico_cd_da_r8/seed43", "seed": 43},
    }

    path = trainer_module._preallocation_path(config, project_root, rank=8)

    assert path == project_root / "outputs/dico_v03/preallocations/dico_cd_da_r8/seed43/dico_v03_rank8_seed43.json"


def test_resolve_project_path_matches_trainer_output_dir_convention(tmp_path):
    experiment_dir = resolve_project_path(tmp_path, "outputs/dico_v03/dico_cd_da_r8_protocol_aligned")

    assert experiment_dir == tmp_path / "outputs" / "dico_v03" / "dico_cd_da_r8_protocol_aligned"
