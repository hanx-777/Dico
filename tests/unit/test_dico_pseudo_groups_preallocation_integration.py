import torch

from dico.model_loader import TinyDecoderOnlyLM, collect_module_dims, find_target_linear_modules
from dico.preallocation import DiCoPreAllocator


def _fixed_batch(vocab_size: int, seq_len: int, batch_size: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _tiny_allocator_config(tmp_path, data_overrides: dict | None = None) -> dict:
    config = {
        "_project_root": str(tmp_path),
        "seed": 42,
        "rank": 1,
        "method": "dico_cd_da",
        "data": dict(data_overrides or {}),
        "calibration": {"save_dir": str(tmp_path / "preallocations"), "seed": 42},
        "preallocation": {
            "atom_mode": "svd",
            "top_k_atoms": 4,
            "sketch_dim": 4,
            "sketch_seed": 42,
            "answer_only": False,
            "profile_norm_mode": "streaming_estimate",
            "eta": 0.5,
            "r_min_multiplier": 0.0,
            "r_max_multiplier": 4.0,
        },
        "dico": {"version": "cd_da", "init": {"mode": "direction_anchored", "zero_B": True}},
    }
    return config


def _build_allocator(tmp_path, data_overrides=None):
    torch.manual_seed(0)
    model = TinyDecoderOnlyLM(vocab_size=64, hidden_size=8)
    model.eval()
    target_modules = find_target_linear_modules(model, ["q_proj"])
    module_names = [name for name, _ in target_modules]
    module_dims = collect_module_dims(target_modules)

    batch_a = _fixed_batch(vocab_size=64, seq_len=5, batch_size=4, seed=11)
    batch_b = _fixed_batch(vocab_size=64, seq_len=5, batch_size=4, seed=999)
    batches = [batch_a] * 4 + [batch_b] * 4

    config = _tiny_allocator_config(tmp_path, data_overrides)
    allocator = DiCoPreAllocator(
        model=model,
        tokenizer=None,
        config=config,
        module_names=module_names,
        module_dims=module_dims,
    )
    allocator.collect_calibration_statistics(batches)
    return allocator


def test_pseudo_groups_used_when_no_group_labels_configured(tmp_path):
    allocator = _build_allocator(tmp_path, data_overrides=None)

    result = allocator.allocate(rank_budget=8)

    taxonomy_stats = result.diagnostics["taxonomy_stats"]
    assert taxonomy_stats["group_source"] == "pseudo"
    assert taxonomy_stats["num_groups"] >= 2

    # Phase F wiring: kappa-null calibration diagnostics get computed per module type
    # and threaded all the way through the real allocator pipeline.
    kappa_calibration = result.diagnostics["kappa_calibration"]
    assert kappa_calibration, "expected at least one module type's kappa calibration result"
    for row in kappa_calibration.values():
        assert "fallback_h0" in row
        assert "ks_pvalue" in row


def test_pseudo_groups_disabled_falls_back_to_single_group(tmp_path):
    allocator = _build_allocator(tmp_path, data_overrides=None)
    allocator.config.setdefault("dico", {})["pseudo_group"] = {"enabled": False}

    result = allocator.allocate(rank_budget=8)

    taxonomy_stats = result.diagnostics["taxonomy_stats"]
    assert taxonomy_stats["group_source"] == "single"
    assert taxonomy_stats["num_groups"] == 1


def test_real_group_labels_are_used_when_length_matches_samples(tmp_path):
    # _build_allocator feeds 4 repeats of batch_a (batch_size=4) then 4 repeats of
    # batch_b (batch_size=4), i.e. 32 total calibration samples split 16/16.
    allocator = _build_allocator(
        tmp_path,
        data_overrides={"group_labels": ["a"] * 16 + ["b"] * 16},
    )

    result = allocator.allocate(rank_budget=8)

    taxonomy_stats = result.diagnostics["taxonomy_stats"]
    assert taxonomy_stats["group_source"] == "configured"
    assert taxonomy_stats["num_groups"] == 2
