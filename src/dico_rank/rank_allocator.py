from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping


EPS = 1e-12


@dataclass(frozen=True)
class NormalizedAtom:
    module_name: str
    atom_index: int
    utility: float
    raw_utility: float
    selected: bool
    module_type: str
    layer: int
    profile: tuple[float, ...] | None = None


@dataclass
class RankEvidence:
    values: dict[str, list[float]]
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def next_value(self, module_name: str, rank_offset: int) -> float:
        values = self.values.get(module_name, [])
        if rank_offset < len(values):
            return max(0.0, float(values[rank_offset]))
        return 0.0


@dataclass(frozen=True)
class RankAllocatorResult:
    allocation: dict[str, int]
    module_logs: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    warning: str | None = None


def default_rank_allocator_config() -> dict[str, Any]:
    return {
        "atom_to_rank": "marginal_curve",
        "smoothing": "layer_diffusion",
        "utility": {
            "align_gamma": 1.0,
            "use_log1p": True,
            "type_normalization": "median",
        },
        "marginal_curve": {"decay": "sqrt", "geometric_lambda": 0.75},
        "prototype_bundle": {"similarity_threshold": 0.8, "residual_weight": 0.25},
        "soft_slot": {"temperature": 1.0, "slot_decay": 0.15},
        "cost_beta": 0.5,
        "budget_guardrails": {
            "max_rank_per_module": None,
            "layer_cap_multiplier": 1.8,
            "type_cap_multiplier": 2.0,
            "type_budget_bounds": None,
        },
        "layer_diffusion": {"kernel": [0.25, 0.50, 0.25]},
        "concentration_penalty": {"lambda": 0.02},
    }


def merge_rank_allocator_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = default_rank_allocator_config()
    for key, value in (config or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def infer_layer(module_name: str, row: Mapping[str, Any] | None = None) -> int:
    row = row or {}
    for key in ("layer", "layer_idx", "ell"):
        if row.get(key) is not None:
            return int(row[key])
    match = re.search(r"(?:^|\.)layers?\.(\d+)(?:\.|$)", module_name)
    if match:
        return int(match.group(1))
    return 0


def infer_type(module_name: str, row: Mapping[str, Any] | None = None) -> str:
    row = row or {}
    for key in ("type", "module_type", "tau"):
        if row.get(key):
            return str(row[key])
    return module_name.split(".")[-1]


def _first_present(row: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _as_profile(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        flat = []
        for item in value:
            if isinstance(item, (list, tuple)):
                flat.extend(float(x) for x in item)
            else:
                flat.append(float(item))
        return tuple(flat)
    return None


def _base_utility(row: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    utility_cfg = config.get("utility", {})
    gain = _first_present(row, ("g_sel", "selected_gain", "coverage_gain", "gain"))
    align = _first_present(row, ("align", "alignment"))
    if gain is None:
        return max(0.0, float(row.get("utility", 0.0) or 0.0))
    align_value = 1.0 if align is None else max(0.0, float(align))
    gamma = float(utility_cfg.get("align_gamma", 1.0))
    value = max(0.0, float(gain)) * (align_value**gamma)
    if bool(utility_cfg.get("use_log1p", True)):
        return math.log1p(value)
    return value


def normalize_atoms(
    atom_logs: list[Mapping[str, Any]],
    module_dims: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    use_raw_utility: bool = False,
) -> list[NormalizedAtom]:
    atoms: list[NormalizedAtom] = []
    for row in atom_logs:
        module_name = str(_first_present(row, ("module", "module_name", "module_id")) or "")
        if module_name not in module_dims:
            continue
        selected = bool(row.get("selected", False))
        if not selected:
            continue
        profile = _as_profile(_first_present(row, ("pi", "response_profile", "signed_response")))
        atoms.append(
            NormalizedAtom(
                module_name=module_name,
                atom_index=int(_first_present(row, ("atom_index", "atom_id", "k")) or 0),
                utility=max(0.0, float(row.get("utility", 0.0) or 0.0))
                if use_raw_utility
                else _base_utility(row, config),
                raw_utility=max(0.0, float(row.get("utility", 0.0) or 0.0)),
                selected=selected,
                module_type=infer_type(module_name, row),
                layer=infer_layer(module_name, row),
                profile=profile,
            )
        )

    mode = str(config.get("utility", {}).get("type_normalization", "none"))
    if use_raw_utility or mode == "none" or not atoms:
        return atoms

    by_type: dict[str, list[float]] = {}
    for atom in atoms:
        by_type.setdefault(atom.module_type, []).append(atom.utility)
    normalized: list[NormalizedAtom] = []
    shifts: dict[str, float] = {}
    scales: dict[str, float] = {}
    for module_type, values in by_type.items():
        sorted_values = sorted(values)
        median = sorted_values[len(sorted_values) // 2]
        if mode == "median":
            shifts[module_type] = 0.0
            scales[module_type] = max(median, EPS)
        elif mode == "zscore_iqr":
            q1 = sorted_values[len(sorted_values) // 4]
            q3 = sorted_values[(3 * len(sorted_values)) // 4]
            shifts[module_type] = median
            scales[module_type] = max(q3 - q1, EPS)
        else:
            shifts[module_type] = 0.0
            scales[module_type] = 1.0
    for atom in atoms:
        value = (atom.utility - shifts[atom.module_type]) / scales[atom.module_type]
        normalized.append(
            NormalizedAtom(
                module_name=atom.module_name,
                atom_index=atom.atom_index,
                utility=max(0.0, value),
                raw_utility=atom.raw_utility,
                selected=atom.selected,
                module_type=atom.module_type,
                layer=atom.layer,
                profile=atom.profile,
            )
        )
    return normalized


def _decay(index: int, config: Mapping[str, Any]) -> float:
    mode = str(config.get("marginal_curve", {}).get("decay", "sqrt"))
    j = int(index) + 1
    if mode == "none":
        return 1.0
    if mode == "geometric":
        return float(config.get("marginal_curve", {}).get("geometric_lambda", 0.75)) ** (j - 1)
    return 1.0 / math.sqrt(j)


def build_marginal_curve_evidence(
    atoms: list[NormalizedAtom],
    module_names: list[str],
    r_max: int,
    allow_relaxed_tail: bool,
    config: Mapping[str, Any],
) -> RankEvidence:
    grouped = {name: [] for name in module_names}
    for atom in atoms:
        grouped.setdefault(atom.module_name, []).append(atom.utility)
    values: dict[str, list[float]] = {}
    for name in module_names:
        utilities = sorted(grouped.get(name, []), reverse=True)
        if allow_relaxed_tail and utilities:
            avg = sum(utilities) / len(utilities)
            while len(utilities) < int(r_max):
                utilities.append(avg)
        values[name] = [utility * _decay(idx, config) for idx, utility in enumerate(utilities[: int(r_max)])]
    return RankEvidence(values=values)


def _cosine(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= EPS or right_norm <= EPS:
        return 0.0
    return dot / (left_norm * right_norm)


def build_prototype_bundle_evidence(
    atoms: list[NormalizedAtom],
    module_names: list[str],
    config: Mapping[str, Any],
) -> RankEvidence:
    threshold = float(config.get("prototype_bundle", {}).get("similarity_threshold", 0.8))
    residual_weight = float(config.get("prototype_bundle", {}).get("residual_weight", 0.25))
    values = {name: [] for name in module_names}
    metadata = {name: {"bundle_count": 0, "prototype_warnings": []} for name in module_names}
    grouped: dict[str, list[NormalizedAtom]] = {name: [] for name in module_names}
    for atom in atoms:
        grouped.setdefault(atom.module_name, []).append(atom)

    for name, module_atoms in grouped.items():
        bundles: list[dict[str, Any]] = []
        for atom in sorted(module_atoms, key=lambda item: (-item.utility, item.atom_index)):
            if atom.profile is None:
                metadata[name]["prototype_warnings"].append("missing_profile")
                bundles.append({"center": atom, "members": [atom]})
                continue
            assigned = False
            for bundle in bundles:
                center = bundle["center"]
                if center.profile is not None and _cosine(atom.profile, center.profile) >= threshold:
                    bundle["members"].append(atom)
                    assigned = True
                    break
            if not assigned:
                bundles.append({"center": atom, "members": [atom]})
        bundle_values = []
        for bundle in bundles:
            center = bundle["center"]
            members = bundle["members"]
            max_u = max(atom.utility for atom in members)
            residual = sum(atom.utility * (1.0 - _cosine(atom.profile, center.profile)) for atom in members)
            bundle_values.append(max_u + residual_weight * residual)
        values[name] = sorted(bundle_values, reverse=True)
        metadata[name]["bundle_count"] = len(bundle_values)
    return RankEvidence(values=values, metadata=metadata)


def build_soft_slot_evidence(
    atoms: list[NormalizedAtom],
    module_names: list[str],
    r_max: int,
    config: Mapping[str, Any],
) -> RankEvidence:
    temperature = max(float(config.get("soft_slot", {}).get("temperature", 1.0)), EPS)
    slot_decay = float(config.get("soft_slot", {}).get("slot_decay", 0.15))
    values = {name: [0.0 for _ in range(int(r_max))] for name in module_names}
    for atom in atoms:
        logits = [(atom.utility - slot_decay * (slot + 1)) / temperature for slot in range(int(r_max))]
        max_logit = max(logits) if logits else 0.0
        weights = [math.exp(logit - max_logit) for logit in logits]
        denom = sum(weights) or 1.0
        for slot, weight in enumerate(weights):
            values.setdefault(atom.module_name, [0.0 for _ in range(int(r_max))])
            values[atom.module_name][slot] += (weight / denom) * atom.utility
    return RankEvidence(values=values)


def apply_layer_diffusion(
    evidence: RankEvidence,
    module_names: list[str],
    module_meta: Mapping[str, dict[str, Any]],
    config: Mapping[str, Any],
) -> RankEvidence:
    kernel = list(config.get("layer_diffusion", {}).get("kernel", [0.25, 0.50, 0.25]))
    if len(kernel) != 3:
        kernel = [0.25, 0.50, 0.25]
    by_type_layer: dict[tuple[str, int], str] = {}
    for name in module_names:
        meta = module_meta[name]
        by_type_layer[(str(meta["type"]), int(meta["layer"]))] = name
    values = {name: list(evidence.values.get(name, [])) for name in module_names}
    max_len = max([len(row) for row in values.values()] + [0])
    smoothed = {name: [0.0 for _ in range(max_len)] for name in module_names}
    for name in module_names:
        meta = module_meta[name]
        module_type = str(meta["type"])
        layer = int(meta["layer"])
        neighbors = [(layer - 1, kernel[0]), (layer, kernel[1]), (layer + 1, kernel[2])]
        available = [
            (by_type_layer[(module_type, neighbor_layer)], weight)
            for neighbor_layer, weight in neighbors
            if (module_type, neighbor_layer) in by_type_layer
        ]
        mass = sum(weight for _neighbor, weight in available) or 1.0
        for slot in range(max_len):
            smoothed[name][slot] = sum(
                (weight / mass) * (values[neighbor][slot] if slot < len(values[neighbor]) else 0.0)
                for neighbor, weight in available
            )
    return RankEvidence(values=smoothed, metadata=evidence.metadata)


def hhi(allocation: Mapping[str, int], costs: Mapping[str, int]) -> float:
    total = sum(int(allocation[name]) * int(costs[name]) for name in allocation)
    if total <= 0:
        return 0.0
    return sum(((int(allocation[name]) * int(costs[name])) / total) ** 2 for name in allocation)


def gini(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(max(0.0, float(value)) for value in values)
    total = sum(sorted_values)
    if total <= 0:
        return 0.0
    weighted = sum((idx + 1) * value for idx, value in enumerate(sorted_values))
    n = len(sorted_values)
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def _layer_total_variation(allocation: Mapping[str, int], module_meta: Mapping[str, dict[str, Any]]) -> float:
    by_type: dict[str, dict[int, int]] = {}
    for name, rank in allocation.items():
        meta = module_meta[name]
        by_type.setdefault(str(meta["type"]), {})[int(meta["layer"])] = int(rank)
    total = 0.0
    for rows in by_type.values():
        layers = sorted(rows)
        for left, right in zip(layers, layers[1:]):
            total += abs(rows[right] - rows[left])
    return total


def _type_budget_share(allocation: Mapping[str, int], costs: Mapping[str, int], module_meta: Mapping[str, dict[str, Any]]) -> dict[str, float]:
    total = sum(int(allocation[name]) * int(costs[name]) for name in allocation)
    by_type: dict[str, int] = {}
    for name, rank in allocation.items():
        module_type = str(module_meta[name]["type"])
        by_type[module_type] = by_type.get(module_type, 0) + int(rank) * int(costs[name])
    if total <= 0:
        return {module_type: 0.0 for module_type in by_type}
    return {module_type: value / total for module_type, value in sorted(by_type.items())}


def _violates_guardrails(
    name: str,
    next_rank: int,
    allocation: Mapping[str, int],
    costs: Mapping[str, int],
    module_meta: Mapping[str, dict[str, Any]],
    config: Mapping[str, Any],
    target_budget: int,
) -> bool:
    guard = config.get("budget_guardrails", {})
    max_rank = guard.get("max_rank_per_module")
    if max_rank is not None and int(next_rank) > int(max_rank):
        return True
    candidate = dict(allocation)
    candidate[name] = int(next_rank)
    layer_cap = guard.get("layer_cap_multiplier")
    if layer_cap is not None:
        layer_budgets: dict[int, int] = {}
        for module_name, rank in candidate.items():
            layer = int(module_meta[module_name]["layer"])
            layer_budgets[layer] = layer_budgets.get(layer, 0) + int(rank) * int(costs[module_name])
        if layer_budgets:
            avg_layer = max(sum(layer_budgets.values()) / max(len(layer_budgets), 1), EPS)
            if layer_budgets[int(module_meta[name]["layer"])] > float(layer_cap) * avg_layer:
                return True
    type_cap = guard.get("type_cap_multiplier")
    if type_cap is not None:
        type_budgets: dict[str, int] = {}
        for module_name, rank in candidate.items():
            module_type = str(module_meta[module_name]["type"])
            type_budgets[module_type] = type_budgets.get(module_type, 0) + int(rank) * int(costs[module_name])
        if type_budgets:
            avg_type = max(sum(type_budgets.values()) / max(len(type_budgets), 1), EPS)
            if type_budgets[str(module_meta[name]["type"])] > float(type_cap) * avg_type:
                return True
    bounds = guard.get("type_budget_bounds") or {}
    if bounds:
        shares = _type_budget_share(candidate, costs, module_meta)
        for module_type, (_min_share, max_share) in bounds.items():
            if max_share is not None and shares.get(module_type, 0.0) > float(max_share):
                return True
    return False


def _allocation_diagnostics(
    allocation: Mapping[str, int],
    costs: Mapping[str, int],
    module_meta: Mapping[str, dict[str, Any]],
    target_budget: int,
    atom_to_rank: str,
    smoothing: str,
    warnings: list[str],
) -> dict[str, Any]:
    actual = sum(int(allocation[name]) * int(costs[name]) for name in allocation)
    shares = [(int(allocation[name]) * int(costs[name])) / actual for name in allocation] if actual else []
    return {
        "actual_budget": actual,
        "budget_ratio": float(actual / target_budget) if target_budget else 0.0,
        "atom_to_rank": atom_to_rank,
        "smoothing": smoothing,
        "hhi": hhi(allocation, costs),
        "gini": gini(shares),
        "layer_total_variation": _layer_total_variation(allocation, module_meta),
        "type_budget_share": _type_budget_share(allocation, costs, module_meta),
        "num_nonzero_modules": sum(1 for value in allocation.values() if int(value) > 0),
        "max_rank": max([int(value) for value in allocation.values()] + [0]),
        "warnings": list(warnings),
    }


def _generic_allocate(
    atoms: list[NormalizedAtom],
    module_names: list[str],
    costs: Mapping[str, int],
    module_meta: Mapping[str, dict[str, Any]],
    target_budget: int,
    eta: float,
    r_min: int,
    r_max: int,
    allow_rank_beyond_selected_evidence: bool,
    evidence: RankEvidence,
    config: Mapping[str, Any],
) -> RankAllocatorResult:
    atom_to_rank = str(config.get("atom_to_rank", "marginal_curve"))
    smoothing = str(config.get("smoothing", "none"))
    cost_beta = float(config.get("cost_beta", 0.5))
    allocation = {name: int(r_min) for name in module_names}
    purchased_evidence = {name: 0 for name in module_names}
    relaxed_ranks = {name: 0 for name in module_names}
    purchased_slots = {name: [] for name in module_names}
    warnings: list[str] = []

    if smoothing == "layer_diffusion":
        evidence = apply_layer_diffusion(evidence, module_names, module_meta, config)

    def total() -> int:
        return sum(allocation[name] * int(costs[name]) for name in module_names)

    while True:
        actual = total()
        best: tuple[float, int, str, str] | None = None
        best_name: str | None = None
        for name in module_names:
            if allocation[name] >= int(r_max):
                continue
            if actual + int(costs[name]) > int(target_budget):
                continue
            if smoothing == "budget_guardrails" and _violates_guardrails(
                name, allocation[name] + 1, allocation, costs, module_meta, config, target_budget
            ):
                continue
            rank_offset = max(0, allocation[name] - int(r_min))
            base_value = evidence.next_value(name, rank_offset)
            if base_value <= 0.0:
                continue
            base_score = base_value / (max(int(costs[name]), 1) ** cost_beta)
            score = base_score
            if smoothing == "concentration_penalty":
                before = hhi(allocation, costs)
                candidate = dict(allocation)
                candidate[name] += 1
                delta_hhi = hhi(candidate, costs) - before
                score = base_score - float(config.get("concentration_penalty", {}).get("lambda", 0.02)) * delta_hhi
            if score <= 0.0:
                continue
            meta = module_meta[name]
            key = (score, -int(meta["layer"]), str(meta["type"]), name)
            if best is None or key > best:
                best = key
                best_name = name
        if best_name is None:
            break
        allocation[best_name] += 1
        purchased_evidence[best_name] += 1
        purchased_slots[best_name].append(allocation[best_name])

    selected_counts = {name: 0 for name in module_names}
    selected_utilities = {name: [] for name in module_names}
    for atom in atoms:
        selected_counts[atom.module_name] = selected_counts.get(atom.module_name, 0) + 1
        selected_utilities.setdefault(atom.module_name, []).append(atom.utility)
    avg_density = {
        name: ((sum(selected_utilities.get(name, [])) / len(selected_utilities[name])) if selected_utilities.get(name) else 0.0)
        / (max(int(costs[name]), 1) ** cost_beta)
        for name in module_names
    }
    target_min = int(float(eta) * int(target_budget))
    while total() < target_min:
        actual = total()
        candidates = []
        for name in module_names:
            evidence_cap = selected_counts.get(name, 0)
            if not allow_rank_beyond_selected_evidence and allocation[name] >= max(int(r_min), evidence_cap):
                continue
            if allocation[name] >= int(r_max):
                continue
            if actual + int(costs[name]) > int(target_budget):
                continue
            if smoothing == "budget_guardrails" and _violates_guardrails(
                name, allocation[name] + 1, allocation, costs, module_meta, config, target_budget
            ):
                continue
            candidates.append(name)
        if not candidates:
            break
        best_name = max(
            candidates,
            key=lambda name: (
                avg_density[name],
                -int(module_meta[name]["layer"]),
                str(module_meta[name]["type"]),
                -int(costs[name]),
                name,
            ),
        )
        allocation[best_name] += 1
        relaxed_ranks[best_name] += 1
    if total() < target_min:
        warnings.append(f"selected evidence constraints prevented reaching eta target; actual_budget={total()} min_budget={target_min}")

    module_logs = []
    for name in module_names:
        rank = int(allocation[name])
        selected_count = int(selected_counts.get(name, 0))
        beyond = max(0, rank - selected_count)
        meta = dict(evidence.metadata.get(name, {}))
        module_logs.append(
            {
                "module_name": name,
                "module_utility": sum(selected_utilities.get(name, [])),
                "rank_cost": int(costs[name]),
                "cost_aware_score": avg_density[name],
                "continuous_rank": None,
                "r_tilde": None,
                "floor_rank": int(r_min),
                "final_rank": rank,
                "selected_atom_count": selected_count,
                "selected_evidence_count": selected_count,
                "selected_atom_utilities": selected_utilities.get(name, []),
                "purchased_evidence_rank": purchased_evidence[name],
                "evidence_relaxation_rank": relaxed_ranks[name],
                "rank_beyond_selected_evidence": beyond,
                "rank_beyond_evidence": beyond,
                "rank_beyond_evidence_ratio": float(beyond / rank) if rank else 0.0,
                "final_parameter_count": rank * int(costs[name]),
                "final_budget": rank * int(costs[name]),
                "allocation_method": "rank_allocator",
                "purchased_slots": purchased_slots[name],
                **meta,
            }
        )
    diagnostics = _allocation_diagnostics(allocation, costs, module_meta, target_budget, atom_to_rank, smoothing, warnings)
    warning = "; ".join(warnings) if warnings else None
    return RankAllocatorResult(allocation=allocation, module_logs=module_logs, diagnostics=diagnostics, warning=warning)


def _legacy_allocate(
    atoms: list[NormalizedAtom],
    module_names: list[str],
    costs: Mapping[str, int],
    module_meta: Mapping[str, dict[str, Any]],
    target_budget: int,
    eta: float,
    r_min: int,
    r_max: int,
    beta: float,
    allow_rank_beyond_selected_evidence: bool,
) -> RankAllocatorResult:
    allocation = {name: int(r_min) for name in module_names}
    purchased_evidence = {name: 0 for name in module_names}
    relaxed_ranks = {name: 0 for name in module_names}
    selected_utilities = {name: [] for name in module_names}
    selected_atoms = []
    for atom in atoms:
        selected_utilities.setdefault(atom.module_name, []).append(atom.utility)
        selected_atoms.append((atom.module_name, atom.atom_index, atom.utility))

    def total() -> int:
        return sum(allocation[name] * int(costs[name]) for name in module_names)

    for module_name, _atom_index, utility in sorted(
        selected_atoms,
        key=lambda item: (
            item[2] / max(int(costs[item[0]]), 1) ** float(beta),
            item[2],
            -int(costs[item[0]]),
            item[0],
            -item[1],
        ),
        reverse=True,
    ):
        if allocation[module_name] >= int(r_max):
            continue
        if total() + int(costs[module_name]) > int(target_budget):
            continue
        allocation[module_name] += 1
        purchased_evidence[module_name] += 1

    target_min = int(float(eta) * int(target_budget))
    avg_density = {}
    for name in module_names:
        utilities = selected_utilities.get(name, [])
        avg_utility = sum(utilities) / len(utilities) if utilities else 0.0
        avg_density[name] = avg_utility / max(int(costs[name]), 1) ** float(beta)

    while total() < target_min:
        actual = total()
        candidates = []
        for name in module_names:
            evidence_cap = len(selected_utilities.get(name, []))
            if not allow_rank_beyond_selected_evidence and allocation[name] >= max(int(r_min), evidence_cap):
                continue
            if allocation[name] >= int(r_max):
                continue
            if actual + int(costs[name]) > int(target_budget):
                continue
            candidates.append(name)
        if not candidates:
            break
        best = max(candidates, key=lambda name: (avg_density[name], -int(costs[name]), name))
        allocation[best] += 1
        relaxed_ranks[best] += 1

    warnings = []
    if total() < target_min:
        warnings.append(f"selected evidence constraints prevented reaching eta target; actual_budget={total()} min_budget={target_min}")
    module_logs = []
    for name in module_names:
        rank = int(allocation[name])
        selected_count = len(selected_utilities.get(name, []))
        beyond = max(0, rank - selected_count)
        module_logs.append(
            {
                "module_name": name,
                "module_utility": sum(selected_utilities.get(name, [])),
                "rank_cost": int(costs[name]),
                "cost_aware_score": avg_density[name],
                "continuous_rank": None,
                "r_tilde": None,
                "floor_rank": int(r_min),
                "final_rank": rank,
                "selected_atom_count": selected_count,
                "selected_evidence_count": selected_count,
                "selected_atom_utilities": selected_utilities.get(name, []),
                "purchased_evidence_rank": purchased_evidence[name],
                "evidence_relaxation_rank": relaxed_ranks[name],
                "rank_beyond_selected_evidence": beyond,
                "rank_beyond_evidence": beyond,
                "rank_beyond_evidence_ratio": float(beyond / rank) if rank else 0.0,
                "final_parameter_count": rank * int(costs[name]),
                "final_budget": rank * int(costs[name]),
                "allocation_method": "directional_budgeted",
            }
        )
    diagnostics = _allocation_diagnostics(allocation, costs, module_meta, target_budget, "legacy_atom_purchase", "none", warnings)
    return RankAllocatorResult(
        allocation=allocation,
        module_logs=module_logs,
        diagnostics=diagnostics,
        warning="; ".join(warnings) if warnings else None,
    )


def allocate_rank_pattern(
    atom_logs: list[Mapping[str, Any]],
    module_dims: Mapping[str, Mapping[str, Any]],
    costs: Mapping[str, int],
    target_budget: int,
    eta: float,
    r_min: int,
    r_max: int,
    allow_rank_beyond_selected_evidence: bool,
    config: Mapping[str, Any] | None = None,
) -> RankAllocatorResult:
    cfg = merge_rank_allocator_config(config)
    module_names = list(module_dims.keys())
    module_meta = {
        name: {"type": infer_type(name), "layer": infer_layer(name)}
        for name in module_names
    }
    atom_to_rank = str(cfg.get("atom_to_rank", "marginal_curve"))
    smoothing = str(cfg.get("smoothing", "layer_diffusion"))
    atoms = normalize_atoms(
        atom_logs,
        module_dims,
        cfg,
        use_raw_utility=atom_to_rank == "legacy_atom_purchase",
    )
    if atom_to_rank == "legacy_atom_purchase":
        return _legacy_allocate(
            atoms,
            module_names,
            costs,
            module_meta,
            target_budget,
            eta,
            r_min,
            r_max,
            float(cfg.get("cost_beta", cfg.get("beta", 1.0))),
            allow_rank_beyond_selected_evidence,
        )
    if atom_to_rank == "prototype_bundle":
        evidence = build_prototype_bundle_evidence(atoms, module_names, cfg)
    elif atom_to_rank == "soft_slot":
        evidence = build_soft_slot_evidence(atoms, module_names, r_max, cfg)
    else:
        evidence = build_marginal_curve_evidence(atoms, module_names, r_max, allow_rank_beyond_selected_evidence, cfg)
    if smoothing == "none":
        cfg = dict(cfg)
        cfg["smoothing"] = "none"
    return _generic_allocate(
        atoms,
        module_names,
        costs,
        module_meta,
        target_budget,
        eta,
        r_min,
        r_max,
        allow_rank_beyond_selected_evidence,
        evidence,
        cfg,
    )
