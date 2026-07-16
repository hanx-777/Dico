from __future__ import annotations

import re
from pathlib import Path


def resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Resolve CLI paths relative to the project root, preserving absolute paths."""
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(project_root).resolve() / value


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers?\.(\d+)(?:\.|$)")


def extract_layer_index(module_name: str) -> int | None:
    """Extract the transformer layer index from a dotted module path.

    Matches HF-style names like "model.layers.12.self_attn.q_proj" or the
    bare "layers.0.q_proj" form used in tests. Returns None (not an error)
    for module names with no resolvable layer segment, so callers can fall
    back to layer-agnostic behavior instead of crashing.
    """
    match = _LAYER_INDEX_RE.search(module_name)
    if match is None:
        return None
    return int(match.group(1))
