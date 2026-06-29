import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from src.data import extract_final_answer, extract_prediction_answer, format_gsm8k_prompt
from src.utils import ensure_dir

LOGGER = logging.getLogger(__name__)

def evaluate_gsm8k(
    model,
    tokenizer,
    dataset: List[Dict[str, str]],
    output_dir: Path,
    max_length: int,
    max_new_tokens: int,
    device: torch.device,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    output_dir = ensure_dir(Path(output_dir))
    model.to(device)
    model.eval()
    predictions = []
    correct = 0
    with torch.no_grad():
        for item in tqdm(dataset, desc="eval", leave=False):
            prompt = format_gsm8k_prompt(
                item["question"],
                use_chat_template=use_chat_template,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
            )
            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids, device=device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
            pred_text = tokenizer.decode(continuation, skip_special_tokens=True)
            try:
                gold = extract_final_answer(item["answer"])
            except ValueError:
                gold = ""
            pred = extract_prediction_answer(pred_text)
            is_correct = bool(gold) and gold == pred
            correct += int(is_correct)
            predictions.append(
                {
                    "question": item["question"],
                    "gold": item["answer"],
                    "prediction": pred_text,
                    "extracted_gold": gold,
                    "extracted_pred": pred,
                    "correct": is_correct,
                }
            )
    total = len(dataset)
    metrics = {
        "exact_match": float(correct / total) if total else 0.0,
        "num_correct": correct,
        "num_total": total,
    }
    
    (output_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "eval_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    LOGGER.info(f"Evaluation Complete. Accuracy: {metrics['exact_match']*100:.2f}% ({correct}/{total})")
    return metrics
