from __future__ import annotations

from pathlib import Path


def resolve_project_path(project_root: str | Path, path: str | Path) -> Path:
    """Resolve CLI paths relative to the project root, preserving absolute paths."""
    value = Path(path)
    if value.is_absolute():
        return value
    return Path(project_root).resolve() / value
