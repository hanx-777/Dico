"""Tests for scripts/summarize_seeds.py's grouping and mean/std aggregation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import summarize_seeds as summarize_seeds_module  # noqa: E402


def _write_metrics(tmp_path: Path, experiment: str, method: str, seed: int, **fields) -> None:
    exp_dir = tmp_path / experiment
    exp_dir.mkdir()
    payload = {"experiment": experiment, "method": method, "seed": seed, **fields}
    (exp_dir / "metrics.json").write_text(json.dumps(payload))


def test_group_name_strips_seed_suffix():
    assert summarize_seeds_module.group_name("lora_r8_protocol_aligned_seed42") == "lora_r8_protocol_aligned"
    assert summarize_seeds_module.group_name("no_seed_suffix") == "no_seed_suffix"


def test_summarize_computes_mean_and_std_per_group(tmp_path):
    for seed, acc in zip([42, 43, 44], [0.30, 0.32, 0.34]):
        _write_metrics(
            tmp_path,
            f"lora_r8_protocol_aligned_seed{seed}",
            "lora",
            seed,
            final_metric=acc,
            final_eval_accuracy=acc,
        )
    for seed, acc in zip([42, 43, 44], [0.40, 0.40, 0.40]):
        _write_metrics(
            tmp_path,
            f"dico_cd_da_r8_protocol_aligned_seed{seed}",
            "dico_cd_da",
            seed,
            final_metric=acc,
            final_eval_accuracy=acc,
        )

    records = summarize_seeds_module.load_all_metrics(tmp_path)
    summary = summarize_seeds_module.summarize(records, ["final_metric", "final_eval_accuracy"])

    assert set(summary.keys()) == {"lora_r8_protocol_aligned", "dico_cd_da_r8_protocol_aligned"}

    lora_group = summary["lora_r8_protocol_aligned"]
    assert lora_group["method"] == "lora"
    assert lora_group["seeds"] == [42, 43, 44]
    assert lora_group["fields"]["final_metric"]["n"] == 3
    assert abs(lora_group["fields"]["final_metric"]["mean"] - 0.32) < 1e-9
    assert lora_group["fields"]["final_metric"]["std"] > 0.0

    dico_group = summary["dico_cd_da_r8_protocol_aligned"]
    assert dico_group["fields"]["final_metric"]["mean"] == 0.40
    assert dico_group["fields"]["final_metric"]["std"] == 0.0


def test_summarize_handles_missing_field_gracefully(tmp_path):
    _write_metrics(tmp_path, "lora_r8_protocol_aligned_seed42", "lora", 42, final_metric=0.3)
    records = summarize_seeds_module.load_all_metrics(tmp_path)
    summary = summarize_seeds_module.summarize(records, ["final_metric", "missing_field"])
    assert summary["lora_r8_protocol_aligned"]["fields"]["missing_field"]["n"] == 0
    assert summary["lora_r8_protocol_aligned"]["fields"]["missing_field"]["mean"] is None
