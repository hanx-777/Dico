from __future__ import annotations

import json

from dico.stage_metrics import StageMetricsRecorder
from dico.trainer import _preallocation_timing_manifest


def test_stage_metrics_records_cpu_stage_without_claiming_cuda(tmp_path):
    path = tmp_path / "stage_metrics.jsonl"
    recorder = StageMetricsRecorder(path, enabled=True)
    token = recorder.begin("tokenization")
    recorder.end(token, details={"records": 2})

    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload["stage"] == "tokenization"
    assert payload["wall_sec"] >= 0
    assert payload["cuda_available"] is False
    assert payload["cuda_allocated_bytes"] is None
    assert payload["details"] == {"records": 2}


def test_gora_timing_does_not_double_count_pseudoinverse_as_rank_allocation():
    timing = _preallocation_timing_manifest(
        {
            "gradient_collection_sec": 7.0,
            "rank_allocation_sec": 2.0,
            "pseudoinverse_init_sec": 3.0,
        },
        wall_sec=12.5,
    )

    assert timing == {
        "calibration_sec": 7.0,
        "allocation_sec": 2.0,
        "initialization_sec": 3.0,
        "unattributed_sec": 0.5,
    }
