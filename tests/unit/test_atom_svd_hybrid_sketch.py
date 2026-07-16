import logging

import torch

import dico.atom_svd as atom_svd_module
from dico.atom_svd import extract_svd_atom_records
from dico.model_loader import TinyDecoderOnlyLM, collect_module_dims, find_target_linear_modules


def _make_batch(vocab_size: int, seq_len: int, batch_size: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _tiny_setup():
    torch.manual_seed(0)
    model = TinyDecoderOnlyLM(vocab_size=32, hidden_size=8)
    model.eval()
    target_modules = find_target_linear_modules(model, ["q_proj"])
    module_names = [name for name, _ in target_modules]
    module_dims = collect_module_dims(target_modules)
    return model, module_names, module_dims


def _base_pre_cfg(module_names, **overrides):
    cfg = {
        "top_k_atoms": 2,
        "sketch_dim": 3,
        "sketch_seed": 42,
        "answer_only": False,
        "profile_norm_mode": "streaming_estimate",
        "module_chunk_size": len(module_names),
        "progress_logging_steps": 1,
    }
    cfg.update(overrides)
    return cfg


def test_extract_svd_atom_records_uses_configured_group_labels_for_hybrid_sketch(tmp_path):
    model, module_names, module_dims = _tiny_setup()
    batches = [
        _make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=1),
        _make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=2),
    ]
    # 4 total samples across the two batches -> 2 real group labels, one per sample.
    group_labels = ["math", "code", "math", "code"]

    _atoms, diagnostics = extract_svd_atom_records(
        model,
        module_names,
        module_dims,
        batches,
        _base_pre_cfg(module_names),
        rank=1,
        profile_path=tmp_path / "profile.pt",
        group_labels=group_labels,
    )

    assert diagnostics["sketch_group_source"] == "configured"
    assert diagnostics["response_agg_group_count"] == 2


def test_extract_svd_atom_records_falls_back_to_random_blocks_on_length_mismatch(tmp_path, caplog):
    model, module_names, module_dims = _tiny_setup()
    batches = [_make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=1)]  # 2 samples total

    with caplog.at_level(logging.WARNING, logger=atom_svd_module.LOGGER.name):
        _atoms, diagnostics = extract_svd_atom_records(
            model,
            module_names,
            module_dims,
            batches,
            _base_pre_cfg(module_names, response_agg_groups=3),
            rank=1,
            profile_path=tmp_path / "profile.pt",
            group_labels=["only_one_label"],  # length mismatch: 1 label, 2 samples
        )

    assert diagnostics["sketch_group_source"] == "random_block"
    # response_agg_groups=3 is requested, but only 2 calibration samples exist -- at
    # most 2 distinct blocks can actually appear (each sample gets exactly one block).
    assert diagnostics["response_agg_group_count"] == 2
    assert "response_agg_group_label_length_mismatch" in caplog.text


def test_lambda_cov_is_configurable_and_recorded_on_atoms(tmp_path):
    model, module_names, module_dims = _tiny_setup()
    batches = [_make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=1)]

    atoms, diagnostics = extract_svd_atom_records(
        model,
        module_names,
        module_dims,
        batches,
        _base_pre_cfg(module_names, lambda_cov=0.5),
        rank=1,
        profile_path=tmp_path / "profile.pt",
    )

    assert diagnostics["hybrid_lambda"] == 0.5
    assert atoms, "expected at least one direction atom"
    for atom in atoms:
        assert atom.lambda_cov == 0.5


def test_sketch_block_mode_global_and_grouped_only_are_recorded(tmp_path):
    model, module_names, module_dims = _tiny_setup()
    batches = [_make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=1)]

    _global_atoms, global_diag = extract_svd_atom_records(
        model,
        module_names,
        module_dims,
        batches,
        _base_pre_cfg(module_names, sketch_block_mode="global_only"),
        rank=1,
        profile_path=tmp_path / "global.pt",
    )
    _grouped_atoms, grouped_diag = extract_svd_atom_records(
        model,
        module_names,
        module_dims,
        batches,
        _base_pre_cfg(module_names, sketch_block_mode="grouped_only"),
        rank=1,
        profile_path=tmp_path / "grouped.pt",
    )

    assert global_diag["sketch_block_mode"] == "global_only"
    assert grouped_diag["sketch_block_mode"] == "grouped_only"


def test_hybrid_sketch_is_deterministic_across_repeated_calls(tmp_path):
    model, module_names, module_dims = _tiny_setup()
    batches = [
        _make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=1),
        _make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=2),
    ]
    pre_cfg = _base_pre_cfg(module_names, response_agg_groups=4)

    atoms_a, _ = extract_svd_atom_records(
        model, module_names, module_dims, batches, pre_cfg, rank=1, profile_path=tmp_path / "a.pt",
    )
    atoms_b, _ = extract_svd_atom_records(
        model, module_names, module_dims, batches, pre_cfg, rank=1, profile_path=tmp_path / "b.pt",
    )

    assert len(atoms_a) == len(atoms_b)
    for a, b in zip(atoms_a, atoms_b):
        assert torch.allclose(a.u, b.u, atol=0.0, rtol=0.0)
        assert torch.allclose(a.v_tilde, b.v_tilde, atol=0.0, rtol=0.0)
        assert torch.allclose(a.v, b.v, atol=0.0, rtol=0.0)
        assert torch.allclose(a.profile, b.profile, atol=0.0, rtol=0.0)


def test_hybrid_concat_recovers_canceling_opposite_sign_group_directions():
    """3.2.1节's motivating example: two groups with opposing-sign responses along the
    same direction cancel in the plain pooled aggregate, but the hybrid grouped
    construction [lambda*Y_agg | Y^(1) | Y^(2)] still recovers the direction because the
    group blocks contribute it with full (squared, not cancelled) energy. This exercises
    the exact formula extract_svd_atom_records uses (torch.cat + SVD), not a paraphrase.
    """
    out_dim, s = 6, 4
    direction = torch.zeros(out_dim, s)
    direction[0, 0] = 1.0  # a single strong rank-one direction

    y_group_1 = direction.clone()
    y_group_2 = -direction.clone()
    y_agg = y_group_1 + y_group_2  # cancels to exactly zero

    lambda_cov = 1.0

    # Old (pre-hybrid) behavior: SVD of the pooled aggregate alone sees nothing.
    pooled_singular = torch.linalg.svdvals(y_agg)
    assert float(pooled_singular.max().item()) < 1e-8, "aggregate should cancel to (near) zero"

    # New hybrid behavior: concatenating the group blocks recovers the direction.
    y_hat = torch.cat([lambda_cov * y_agg, y_group_1, y_group_2], dim=1)
    hybrid_singular = torch.linalg.svdvals(y_hat)
    assert float(hybrid_singular.max().item()) > 0.5, (
        "hybrid grouped sketch must recover a direction the pooled aggregate cancelled"
    )
