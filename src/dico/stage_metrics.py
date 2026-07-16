from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class StageToken:
    stage: str
    started_at: float


class StageMetricsRecorder:
    """Append-only stage timing and CUDA-memory recorder.

    CUDA fields stay null on CPU.  This distinction prevents local smoke tests
    from being misreported as target-GPU verification.
    """

    def __init__(self, path: str | Path, *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = bool(enabled)
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def begin(self, stage: str) -> StageToken:
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except RuntimeError:
                pass
        return StageToken(str(stage), time.perf_counter())

    def end(self, token: StageToken, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
        cuda_available = bool(torch.cuda.is_available())
        payload: dict[str, Any] = {
            "stage": token.stage,
            "wall_sec": float(time.perf_counter() - token.started_at),
            "cuda_available": cuda_available,
            "cuda_device": None,
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "details": details or {},
        }
        if cuda_available:
            try:
                device = torch.cuda.current_device()
                payload.update(
                    {
                        "cuda_device": int(device),
                        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    }
                )
            except RuntimeError:
                pass
        if self.enabled:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        return payload

    def record_completed(
        self,
        stage: str,
        wall_sec: float,
        *,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a nested stage whose wall time was measured by its algorithm.

        Memory values are observations at the parent-stage boundary; details must
        identify that peak scope so they are not mistaken for independently reset
        CUDA peaks.
        """
        token = StageToken(str(stage), time.perf_counter() - max(0.0, float(wall_sec)))
        nested_details = {"memory_peak_scope": "parent_stage", **(details or {})}
        return self.end(token, details=nested_details)
