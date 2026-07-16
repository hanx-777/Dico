import json
from pathlib import Path

import pytest

from dico.config import load_yaml
from dico.data import load_humaneval_records, load_raw_datasets


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_base_config_uses_project_local_gsm8k_paths():
    root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(root / "configs" / "base.yaml")

    assert cfg["data"]["train_path"] == "data/metamathqa/train.jsonl"
    assert cfg["data"]["eval_path"] == "data/gsm8k/main/test.jsonl"
    assert cfg["data"]["eval_limit"] is None
    assert cfg["evaluation"]["accuracy_max_samples"] is None
    assert cfg["model"]["torch_dtype"] == "bfloat16"
    assert cfg["model"]["load_in_8bit"] is False
    assert cfg["model"]["load_in_4bit"] is False
    assert cfg["training"]["batch_size"] == 4
    assert cfg["training"]["gradient_accumulation_steps"] == 16
    assert cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation_steps"] == 64
    assert cfg["calibration"]["batch_size"] == 4
    assert cfg["calibration"]["num_samples"] == 1024
    assert cfg["calibration"]["shuffle"] is False
    assert cfg["preallocation"]["compute_device"] == "auto"
    assert cfg["preallocation"]["module_chunk_size"] == 32
    assert cfg["preallocation"]["progress_logging_steps"] == 1
    assert (root / cfg["data"]["train_path"]).exists()
    assert (root / cfg["data"]["eval_path"]).exists()


def test_local_gsm8k_jsonl_counts_and_schema():
    root = Path(__file__).resolve().parents[2]
    train_rows = _jsonl_rows(root / "data" / "gsm8k" / "main" / "train.jsonl")
    eval_rows = _jsonl_rows(root / "data" / "gsm8k" / "main" / "test.jsonl")

    assert len(train_rows) == 7473
    assert len(eval_rows) == 1319
    assert {"question", "answer"} <= set(train_rows[0])
    assert {"question", "answer"} <= set(eval_rows[0])
    assert "####" in train_rows[0]["answer"]
    assert "####" in eval_rows[0]["answer"]


def test_load_raw_datasets_resolves_local_paths_from_project_root(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(root / "configs" / "base.yaml")

    monkeypatch.chdir(tmp_path)
    train_rows, eval_rows = load_raw_datasets(cfg)

    assert len(train_rows) == 395000
    assert len(eval_rows) == 1319


def test_load_humaneval_records_raises_clear_error_when_local_file_is_missing(tmp_path):
    cfg = {"_project_root": str(tmp_path), "data": {"eval_datasets": ["gsm8k", "humaneval"]}}

    with pytest.raises(FileNotFoundError, match="humaneval"):
        load_humaneval_records(cfg)


def test_load_humaneval_records_reads_local_jsonl_when_present(tmp_path):
    humaneval_dir = tmp_path / "data" / "humaneval"
    humaneval_dir.mkdir(parents=True)
    row = {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b): return a + ",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
        "entry_point": "add",
    }
    (humaneval_dir / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    cfg = {"_project_root": str(tmp_path), "data": {"eval_datasets": ["gsm8k", "humaneval"]}}

    records = load_humaneval_records(cfg)

    assert records == [row]
