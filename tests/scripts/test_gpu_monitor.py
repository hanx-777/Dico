from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    path = ROOT / "scripts" / "gpu_monitor.py"
    spec = importlib.util.spec_from_file_location("gpu_monitor", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_monitor_parses_nvidia_smi_csv_row():
    module = _load_module()
    row = module.parse_sample("0, GPU-abc, 73, 64000, 81920, 245.5, 61")
    assert row["gpu_index"] == 0
    assert row["gpu_uuid"] == "GPU-abc"
    assert row["utilization_gpu_percent"] == 73.0
    assert row["memory_used_mib"] == 64000.0
    assert row["memory_total_mib"] == 81920.0
