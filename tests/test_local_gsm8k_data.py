import json
from pathlib import Path

from dico_rank.config import load_yaml
from dico_rank.data import load_raw_datasets


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_base_config_uses_project_local_gsm8k_paths():
    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(root / "configs" / "base.yaml")

    assert cfg["data"]["train_path"] == "data/gsm8k/main/train.jsonl"
    assert cfg["data"]["eval_path"] == "data/gsm8k/main/test.jsonl"
    assert cfg["data"]["eval_limit"] is None
    assert cfg["evaluation"]["accuracy_max_samples"] is None
    assert cfg["model"]["torch_dtype"] == "bfloat16"
    assert cfg["model"]["load_in_8bit"] is False
    assert cfg["model"]["load_in_4bit"] is False
    assert cfg["training"]["batch_size"] == 4
    assert cfg["training"]["gradient_accumulation_steps"] == 2
    assert cfg["calibration"]["batch_size"] == 4
    assert (root / cfg["data"]["train_path"]).exists()
    assert (root / cfg["data"]["eval_path"]).exists()


def test_local_gsm8k_jsonl_counts_and_schema():
    root = Path(__file__).resolve().parents[1]
    train_rows = _jsonl_rows(root / "data" / "gsm8k" / "main" / "train.jsonl")
    eval_rows = _jsonl_rows(root / "data" / "gsm8k" / "main" / "test.jsonl")

    assert len(train_rows) == 7473
    assert len(eval_rows) == 1319
    assert {"question", "answer"} <= set(train_rows[0])
    assert {"question", "answer"} <= set(eval_rows[0])
    assert "####" in train_rows[0]["answer"]
    assert "####" in eval_rows[0]["answer"]


def test_load_raw_datasets_resolves_local_paths_from_project_root(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(root / "configs" / "base.yaml")

    monkeypatch.chdir(tmp_path)
    train_rows, eval_rows = load_raw_datasets(cfg)

    assert len(train_rows) == 7473
    assert len(eval_rows) == 1319
