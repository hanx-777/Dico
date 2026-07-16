from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dico.utils import read_json, write_json


def save_allocation_artifact(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    serializable = {}
    tensors = {}
    for key, value in payload.items():
        if torch.is_tensor(value):
            tensors[key] = value.detach().cpu()
        else:
            serializable[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path.with_suffix(".json"), serializable)
    if tensors:
        torch.save(tensors, path.with_suffix(".pt"))


def load_allocation_artifact(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = read_json(path.with_suffix(".json"))
    tensor_path = path.with_suffix(".pt")
    if tensor_path.exists():
        payload.update(torch.load(tensor_path, map_location="cpu"))
    return payload
