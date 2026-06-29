import csv
import json
from pathlib import Path

import importlib.util


def load_summary_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "summarize_results.py"
    spec = importlib.util.spec_from_file_location("summarize_results", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_includes_internal_gsm8k_accuracy_metadata(tmp_path: Path):
    summary = load_summary_module()
    exp_dir = tmp_path / "lora_r4"
    exp_dir.mkdir()
    (exp_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment": "lora_r4",
                "method": "lora",
                "rank": 4,
                "evaluation_protocol": "internal_zero_shot",
                "evaluation_prompt_style": "sft_cot_hash",
                "eval_sample_count": 200,
                "final_eval_accuracy": 0.25,
                "final_exact_match": 0.25,
                "eval_correct": 50,
                "eval_total": 200,
                "final_eval_loss": 1.5,
                "best_eval_loss": 1.4,
            }
        ),
        encoding="utf-8",
    )

    rows = summary.load_metrics(tmp_path)
    summary.write_csv(tmp_path, rows)
    summary.write_md(tmp_path, rows)

    with (tmp_path / "summary.csv").open("r", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["evaluation_protocol"] == "internal_zero_shot"
    assert csv_rows[0]["evaluation_prompt_style"] == "sft_cot_hash"
    assert csv_rows[0]["eval_sample_count"] == "200"
    assert csv_rows[0]["final_eval_accuracy"] == "0.25"
    assert "Internal GSM8K Acc" in (tmp_path / "summary.md").read_text(encoding="utf-8")
