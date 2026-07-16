from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


LORA_STATE_SUFFIXES = ("lora_A", "lora_E", "lora_B", "rank_mask")


def _is_lora_state_key(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in LORA_STATE_SUFFIXES)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if _is_lora_state_key(name)
    }


def save_lora_state(path: str | Path, model: nn.Module) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = lora_state_dict(model)
    torch.save(state, path)
    return {"path": str(path), "saved_keys": sorted(state)}


def load_lora_state(path: str | Path, model: nn.Module, *, strict: bool = True) -> dict[str, Any]:
    path = Path(path)
    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(f"LoRA checkpoint must contain a state dict, got {type(loaded)!r}")
    current = model.state_dict()
    expected_keys = sorted(name for name in current if _is_lora_state_key(name))
    loaded_keys: list[str] = []
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    with torch.no_grad():
        for key in expected_keys:
            if key not in loaded:
                missing_keys.append(key)
                continue
            source = loaded[key]
            target = current[key]
            if tuple(source.shape) != tuple(target.shape):
                shape_mismatches.append(
                    {
                        "key": key,
                        "checkpoint_shape": list(source.shape),
                        "model_shape": list(target.shape),
                    }
                )
                continue
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            loaded_keys.append(key)
    for key in sorted(loaded):
        if key not in current or not _is_lora_state_key(key):
            unexpected_keys.append(key)
    report = {
        "path": str(path),
        "loaded_keys": loaded_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatches": shape_mismatches,
    }
    if strict and (missing_keys or unexpected_keys or shape_mismatches):
        raise ValueError(f"LoRA checkpoint did not match model: {report}")
    return report
