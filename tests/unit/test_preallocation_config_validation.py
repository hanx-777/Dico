import pytest

from dico.preallocation import DiCoPreAllocator


def _config(
    top_k_atoms: int,
    r_max_multiplier: float = 4.0,
    allocation_method: str = "covra_full",
):
    return {
        "rank": 1,
        "preallocation": {
            "atom_mode": "svd",
            "allocation_method": allocation_method,
            "top_k_atoms": top_k_atoms,
            "r_max_multiplier": r_max_multiplier,
        },
    }


def test_covra_preallocator_rejects_fewer_candidates_than_r_max():
    with pytest.raises(ValueError, match="top_k_atoms.*r_max"):
        DiCoPreAllocator(
            model=None,
            tokenizer=None,
            config=_config(top_k_atoms=3),
            module_names=["m"],
            module_dims={"m": {"in_dim": 2, "out_dim": 2}},
        )


def test_covra_preallocator_accepts_top_k_equal_to_r_max():
    allocator = DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config=_config(top_k_atoms=4),
        module_names=["m"],
        module_dims={"m": {"in_dim": 2, "out_dim": 2}},
    )

    assert allocator._r_max() == 4


def test_covra_v05_accepts_reference_top_k_below_r_max():
    allocator = DiCoPreAllocator(
        model=None,
        tokenizer=None,
        config=_config(top_k_atoms=8, r_max_multiplier=4.0, allocation_method="covra_v05")
        | {"rank": 8},
        module_names=["m"],
        module_dims={"m": {"in_dim": 2, "out_dim": 2}},
    )

    assert allocator._r_max() == 32
