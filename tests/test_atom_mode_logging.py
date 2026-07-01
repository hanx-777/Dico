import json
from pathlib import Path

from dico_rank.preallocation import DiCoPreAllocator


def test_module_proxy_fallback_is_not_logged_as_svd(tmp_path: Path):
    allocator = DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config={"rank": 1, "preallocation": {"atom_mode": "svd", "fallback_atom_mode": "module_proxy"}},
        module_names=["m"],
        module_dims={"m": {"in_dim": 2, "out_dim": 2}},
        module_scores={"m": 1.0},
    )

    result = allocator.allocate(rank_budget=4)
    path = tmp_path / "preallocation.json"
    allocator.save(path, result)

    payload = json.loads(path.read_text())
    assert payload["atom_mode"] == "module_proxy"
    assert "does not distinguish true SVD" in payload["atom_mode_limitation"]
    assert payload["atom_logs"][0]["atom_mode"] == "module_proxy"
    assert "does not distinguish true SVD" in payload["atom_logs"][0]["atom_mode_limitation"]
