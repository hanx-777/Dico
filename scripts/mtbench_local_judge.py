#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dico.mtbench_local import evaluate_mtbench_local, load_jsonl, write_mtbench_protocol


def _build_hf_judge(
    *,
    judge_model: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
) -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "MTBench-local executed mode requires torch and transformers. "
            "Use --dry-run to validate protocol only, or install the GPU runtime dependencies."
        ) from exc

    torch.manual_seed(int(seed))
    tokenizer = AutoTokenizer.from_pretrained(judge_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    def judge(prompt: str) -> str:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generate_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": float(temperature) > 0.0,
            "temperature": float(temperature) if float(temperature) > 0.0 else None,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}
        with torch.inference_mode():
            generated = model.generate(**generate_kwargs)
        suffix = generated[0, encoded["input_ids"].shape[1] :].detach().cpu().tolist()
        return str(tokenizer.decode(suffix, skip_special_tokens=True))

    return judge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or dry-run MTBench-local single-answer judging with a locked local judge protocol."
    )
    parser.add_argument("--questions-jsonl", required=True, help="FastChat-style MTBench question JSONL.")
    parser.add_argument("--answers-jsonl", required=True, help="FastChat-style model answer JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for protocol, raw judgments, and metrics.")
    parser.add_argument("--judge-model", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--judge-prompt-version", default="fastchat-v0.2.36")
    parser.add_argument("--conversation-template", default="llama-3")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--swap-positions", action="store_true", help="Record pairwise-position-swap intent; single-answer scoring does not use it.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write protocol only; do not load or call the judge model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_jsonl(args.questions_jsonl)
    answers = load_jsonl(args.answers_jsonl)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        write_mtbench_protocol(
            output_dir=output,
            status="DRY_RUN_CONFIGURED",
            judge_model=str(args.judge_model),
            judge_prompt_version=str(args.judge_prompt_version),
            conversation_template=str(args.conversation_template),
            temperature=float(args.temperature),
            seed=int(args.seed),
            max_retries=int(args.max_retries),
            num_questions=len(questions),
            num_answers=len(answers),
            swap_positions=bool(args.swap_positions),
        )
        print("[mtbench_local_judge] DRY_RUN_CONFIGURED; no judge model was loaded and no score was produced")
        return

    judge = _build_hf_judge(
        judge_model=str(args.judge_model),
        temperature=float(args.temperature),
        seed=int(args.seed),
        max_new_tokens=int(args.max_new_tokens),
    )
    payload = evaluate_mtbench_local(
        questions=questions,
        answers=answers,
        judge=judge,
        output_dir=output,
        judge_model=str(args.judge_model),
        judge_prompt_version=str(args.judge_prompt_version),
        conversation_template=str(args.conversation_template),
        temperature=float(args.temperature),
        seed=int(args.seed),
        max_retries=int(args.max_retries),
        swap_positions=bool(args.swap_positions),
    )
    print(
        "[mtbench_local_judge] EXECUTED "
        f"score={payload['metrics']['mtbench_local_score']:.4f} "
        f"scored={payload['metrics']['num_scored']}/{payload['metrics']['num_questions']}"
    )


if __name__ == "__main__":
    main()
