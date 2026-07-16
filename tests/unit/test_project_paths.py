from pathlib import Path

from dico.path_utils import extract_layer_index, resolve_project_path


def test_resolve_project_path_uses_project_root_for_relative_paths():
    root = Path(__file__).resolve().parents[2]

    assert resolve_project_path(root, "configs/dico/lora_r8.yaml") == root / "configs/dico/lora_r8.yaml"
    assert resolve_project_path(root, "outputs") == root / "outputs"
    assert resolve_project_path(root, "outputs/dico_v03") == root / "outputs/dico_v03"
    assert (
        resolve_project_path(root, "outputs/dico_v03/dico_cd_da_r8_protocol_aligned/rank_history.csv")
        == root / "outputs/dico_v03/dico_cd_da_r8_protocol_aligned/rank_history.csv"
    )


def test_resolve_project_path_preserves_absolute_paths(tmp_path):
    absolute = tmp_path / "external" / "outputs"

    assert resolve_project_path(Path("/project"), absolute) == absolute


def test_extract_layer_index_matches_hf_and_bare_layer_names():
    assert extract_layer_index("layers.0.q_proj") == 0
    assert extract_layer_index("model.layers.12.self_attn.q_proj") == 12
    assert extract_layer_index("layer.3.mlp") == 3


def test_extract_layer_index_returns_none_for_unresolvable_names():
    assert extract_layer_index("m") is None
    assert extract_layer_index("cheap") is None
    assert extract_layer_index("q_proj") is None
