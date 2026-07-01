from pathlib import Path

from dico_rank.path_utils import resolve_project_path


def test_resolve_project_path_uses_project_root_for_relative_paths():
    root = Path(__file__).resolve().parents[1]

    assert resolve_project_path(root, "configs/experiments/lora_r4.yaml") == root / "configs/experiments/lora_r4.yaml"
    assert resolve_project_path(root, "outputs") == root / "outputs"
    assert resolve_project_path(root, "outputs/lora_r4") == root / "outputs/lora_r4"
    assert (
        resolve_project_path(root, "outputs/dico_dynamic_r4/rank_history.csv")
        == root / "outputs/dico_dynamic_r4/rank_history.csv"
    )


def test_resolve_project_path_preserves_absolute_paths(tmp_path):
    absolute = tmp_path / "external" / "outputs"

    assert resolve_project_path(Path("/project"), absolute) == absolute
