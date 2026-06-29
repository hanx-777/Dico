import json
from pathlib import Path

import torch

from tests.tiny_model import TinyDecoderOnlyLM, TinyTokenizer

from src.calibration import (
    collect_dico_calibration_stats,
    compute_dico_atoms,
    collect_dico_profiles,
)
from src.data_gsm8k import build_sft_example
from src.dico_allocator import allocate_dico_lite
from src.evaluate_gsm8k import evaluate_gsm8k
from src.model_utils import (
    apply_lora_adapters,
    find_target_linear_modules,
    verify_lora_ranks,
)
from src.train_gsm8k_lora import train_one_step


def test_tiny_e2e_pipeline_runs_without_qwen_or_downloads(tmp_path: Path):
    torch.manual_seed(0)
    tokenizer = TinyTokenizer()
    raw_examples = [
        {"question": "Tom has 1 apple and gets 1 more. How many?", "answer": "Tom has 2.\n#### 2"},
        {"question": "Mia has 3 pens and loses 1. How many?", "answer": "Mia has 2.\n#### 2"},
    ]
    for item in raw_examples:
        build_sft_example(item["question"], item["answer"], tokenizer, max_length=64)

    model = TinyDecoderOnlyLM(vocab_size=max(128, tokenizer.vocab_size + 8), hidden_size=16)
    target_modules = find_target_linear_modules(model, ["q_proj", "v_proj"])
    target_names = [name for name, _module in target_modules]
    assert target_names == ["layers.0.q_proj", "layers.0.v_proj"]

    stats = collect_dico_calibration_stats(
        model=model,
        tokenizer=tokenizer,
        dataset=raw_examples,
        target_module_names=target_names,
        calibration_size=2,
        max_length=64,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert (tmp_path / "calibration_pass1.pt").exists()
    assert (tmp_path / "mask_debug" / "sample_000.json").exists()
    mask_debug = json.loads((tmp_path / "mask_debug" / "sample_000.json").read_text())
    assert mask_debug["shifted_answer_token_count"] > 0
    assert "Question:" not in mask_debug["decoded_loss_bearing_tokens"]

    atoms = compute_dico_atoms(stats, top_k_atoms=2, output_dir=tmp_path)
    profiles = collect_dico_profiles(
        model=model,
        tokenizer=tokenizer,
        dataset=raw_examples,
        stats=stats,
        atoms=atoms,
        max_length=64,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )
    allocation = allocate_dico_lite(
        module_names=stats.module_names,
        module_dims=stats.module_dims,
        normalized_profiles=profiles.normalized_profiles,
        importance=stats.importance,
        rho=atoms.rho,
        avg_rank=1,
    )
    rank_pattern = allocation.rank_pattern
    assert any(rank > 0 for rank in rank_pattern.values())

    lora_model = apply_lora_adapters(
        model,
        rank_pattern=rank_pattern,
        target_module_names=stats.module_names,
        lora_alpha=4,
        use_peft=False,
    )
    verification = verify_lora_ranks(
        lora_model,
        requested_rank_pattern=rank_pattern,
        all_target_module_names=stats.module_names,
        output_dir=tmp_path,
    )
    assert (tmp_path / "lora_rank_verification.json").exists()
    for name, requested in rank_pattern.items():
        assert verification[name]["requested_rank"] == requested
        assert verification[name]["actual_rank"] == requested

    examples = [build_sft_example(x["question"], x["answer"], tokenizer, max_length=64) for x in raw_examples]
    loss = train_one_step(lora_model, examples, lr=1e-3, device=torch.device("cpu"))
    assert torch.isfinite(torch.tensor(loss))

    metrics = evaluate_gsm8k(
        model=lora_model,
        tokenizer=tokenizer,
        dataset=raw_examples,
        output_dir=tmp_path,
        max_length=64,
        max_new_tokens=4,
        device=torch.device("cpu"),
    )
    assert metrics["num_total"] == 2
    assert (tmp_path / "eval_results.json").exists()
    assert (tmp_path / "eval_predictions.jsonl").exists()
