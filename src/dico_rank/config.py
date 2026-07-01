from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inherits = data.pop("inherits", None)
    if inherits:
        parent = (path.parent / inherits).resolve()
        data = deep_merge(load_yaml(parent), data)
    resolved = path.resolve()
    config_root = next((parent for parent in resolved.parents if parent.name == "configs"), resolved.parent)
    data["_config_path"] = str(resolved)
    data["_project_root"] = str(config_root.parent)
    return data


def parse_override_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    result = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        path, raw_value = item.split("=", 1)
        target = result
        parts = path.split(".")
        for key in parts[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise ValueError(f"Override path crosses non-dict value: {path}")
        target[parts[-1]] = parse_override_value(raw_value)
    return result


def save_yaml(path: Path | str, config: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
