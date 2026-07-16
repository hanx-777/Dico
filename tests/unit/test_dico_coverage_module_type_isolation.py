import torch

from dico.model_loader import TinyDecoderOnlyLM, collect_module_dims, find_target_linear_modules
from dico.preallocation import DiCoPreAllocator


def _make_batch(vocab_size: int, seq_len: int, batch_size: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _repeated_calibration_batches(vocab_size: int, seq_len: int, batch_size: int, repeats: int) -> list:
    # Repeating one fixed batch drives the signed-profile alignment statistic to 1.0,
    # which reliably clears the permutation-test threshold and yields "consensus"
    # atoms instead of "noise" -- needed so coverage certification has candidates
    # to isolate by module type.
    fixed_batch = _make_batch(vocab_size, seq_len, batch_size, seed=7)
    return [fixed_batch for _ in range(repeats)]


def _tiny_allocator_config(tmp_path) -> dict:
    return {
        "_project_root": str(tmp_path),
        "seed": 42,
        "rank": 1,
        "method": "dico_cd_da",
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


def test_coverage_certification_is_isolated_per_module_type(tmp_path):
    torch.manual_seed(0)
    model = TinyDecoderOnlyLM(vocab_size=64, hidden_size=8)
    model.eval()

    target_modules = find_target_linear_modules(model, ["q_proj", "up_proj"])
    module_names = [name for name, _ in target_modules]
    module_dims = collect_module_dims(target_modules)
    assert len(module_names) >= 2, "expected at least two distinct module types to test isolation"

    batches = _repeated_calibration_batches(vocab_size=64, seq_len=5, batch_size=4, repeats=6)

    config = _tiny_allocator_config(tmp_path)
    allocator = DiCoPreAllocator(
        model=model,
        tokenizer=None,
        config=config,
        module_names=module_names,
        module_dims=module_dims,
    )
    allocator.collect_calibration_statistics(batches)
    result = allocator.allocate(rank_budget=8)

    coverage_trace = result.diagnostics["coverage_trace"]
    assert coverage_trace, "expected at least one coverage certification step"

    module_types_seen = {row["module_type"] for row in coverage_trace}
    assert module_types_seen == {"q_proj", "up_proj"}

    # Each module type's certification steps must only ever select candidates
    # belonging to that same type: isolation means no cross-type interleaving.
    for row in coverage_trace:
        assert row["module"].split(".")[-1] == row["module_type"]
