import torch

from dico.atom_svd import extract_svd_atom_records
from dico.model_loader import TinyDecoderOnlyLM, collect_module_dims, find_target_linear_modules
from dico.sketch import make_random_projection


def _make_batch(vocab_size: int, seq_len: int, batch_size: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def test_profile_pass_uses_sketch_domain_with_per_sample_token_average(tmp_path):
    torch.manual_seed(0)
    model = TinyDecoderOnlyLM(vocab_size=32, hidden_size=8)
    model.eval()

    target_modules = find_target_linear_modules(model, ["q_proj"])
    module_names = [name for name, _ in target_modules]
    module_dims = collect_module_dims(target_modules)
    module = dict(model.named_modules())[module_names[0]]

    batches = [
        _make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=1),
        _make_batch(vocab_size=32, seq_len=5, batch_size=2, seed=2),
    ]

    sketch_dim = 3  # < hidden_size(8), so the sketch domain differs from the full domain
    sketch_seed = 42
    pre_cfg = {
        "top_k_atoms": 2,
        "sketch_dim": sketch_dim,
        "sketch_seed": sketch_seed,
        "answer_only": False,
        "profile_norm_mode": "streaming_estimate",
        "module_chunk_size": len(module_names),
        "progress_logging_steps": 1,
    }

    atoms, _diagnostics = extract_svd_atom_records(
        model,
        module_names,
        module_dims,
        batches,
        pre_cfg,
        rank=1,
        profile_path=tmp_path / "profile.pt",
    )

    assert atoms, "expected at least one direction atom"

    for atom in atoms:
        assert atom.v_tilde is not None
        assert atom.v is not None
        assert atom.profile is not None

        in_dim = int(module_dims[atom.module_name]["in_dim"])
        offset = module_names.index(atom.module_name)
        s = min(sketch_dim, in_dim)
        omega = make_random_projection(in_dim, s, sketch_seed + offset, dtype=torch.float32)

        expected = []
        activations: list[torch.Tensor] = []
        grad_outputs: list[torch.Tensor] = []

        def hook(_module, inputs, output):
            output.retain_grad()
            activations.append(inputs[0].detach())
            grad_outputs.append(output)

        handle = module.register_forward_hook(hook)
        try:
            for batch in batches:
                activations.clear()
                grad_outputs.clear()
                model.zero_grad(set_to_none=True)
                out = model(**batch)
                out.loss.backward()
                activation = activations[0]
                grad = grad_outputs[0].grad
                for sample_idx in range(activation.shape[0]):
                    a_tokens = activation[sample_idx]
                    g_tokens = grad[sample_idx]
                    sketch_tokens = a_tokens @ omega
                    projection = torch.sum((g_tokens @ atom.u) * (sketch_tokens @ atom.v_tilde))
                    # 4.3节: π_{m,k}^{(i)} = (1/T_i) Σ_t (...) -- averaged over this sample's tokens.
                    expected.append(float(projection.item()) / a_tokens.shape[0])
        finally:
            handle.remove()
            model.zero_grad(set_to_none=True)

        expected_tensor = torch.tensor(expected, dtype=torch.float32)
        assert torch.allclose(atom.profile.float(), expected_tensor, atol=1e-4, rtol=1e-3), (
            "signed profile must equal the per-sample-averaged sketch-domain projection "
            "using atom.v_tilde, not a full-dimensional approximation"
        )
