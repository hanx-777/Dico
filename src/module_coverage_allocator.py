from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from src.dico_allocator import _append_basis, _orthogonal_residual


@dataclass
class ModuleCoverageResult:
    rank_pattern: Dict[str, int]
    selected_modules: List[Dict[str, Any]]
    allocation_steps: List[Dict[str, Any]]
    total_budget: float
    used_budget: float


def allocate_module_coverage(
    module_names: List[str],
    module_dims: Dict[str, Dict[str, int]],
    module_profiles: torch.Tensor,
    avg_rank: float = 1,
    max_rank_per_module: int = 1,
    eps: float = 1e-8,
) -> ModuleCoverageResult:
    if module_profiles.ndim != 2:
        raise ValueError("module_profiles must have shape [M, N]")
    total_budget = float(avg_rank) * sum(float(module_dims[m]["cost"]) for m in module_names)
    used_budget = 0.0
    rank_pattern = {name: 0 for name in module_names}
    basis: List[torch.Tensor] = []
    selected: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []

    while True:
        best = None
        for module_index, module_name in enumerate(module_names):
            if rank_pattern[module_name] >= max_rank_per_module:
                continue
            cost = float(module_dims[module_name]["cost"])
            if used_budget + cost > total_budget + eps:
                continue
            profile = module_profiles[module_index].to(dtype=torch.float32)
            residual = _orthogonal_residual(profile, basis)
            coverage = float(torch.dot(residual, residual).item())
            score = coverage / cost
            candidate = {
                "module_name": module_name,
                "module_index": module_index,
                "cost": cost,
                "coverage": coverage,
                "score": score,
            }
            if best is None or score > best["score"]:
                best = candidate
        if best is None or best["score"] <= eps:
            break
        module_name = best["module_name"]
        rank_pattern[module_name] += 1
        used_budget += best["cost"]
        _append_basis(module_profiles[best["module_index"]].to(dtype=torch.float32), basis, eps)
        selected.append(best)
        steps.append(dict(best, used_budget=used_budget, total_budget=total_budget))

    return ModuleCoverageResult(rank_pattern, selected, steps, total_budget, used_budget)
