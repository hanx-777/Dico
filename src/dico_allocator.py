from dataclasses import dataclass
import math
from typing import Any, Dict, List

import torch


@dataclass
class AllocationResult:
    rank_pattern: Dict[str, int]
    selected_atoms: List[Dict[str, Any]]
    allocation_steps: List[Dict[str, Any]]
    total_budget: float
    used_budget: float
    importance: Dict[str, float]
    rho: Dict[str, torch.Tensor]
    scores: List[Dict[str, Any]]


def _orthogonal_residual(profile: torch.Tensor, basis: List[torch.Tensor]) -> torch.Tensor:
    residual = profile.clone()
    for q in basis:
        residual = residual - q * torch.dot(q, residual)
    return residual


def _append_basis(profile: torch.Tensor, basis: List[torch.Tensor], eps: float) -> None:
    residual = _orthogonal_residual(profile, basis)
    norm = torch.linalg.norm(residual)
    if float(norm) > eps:
        basis.append(residual / norm)


def allocate_dico_lite(
    module_names: List[str],
    module_dims: Dict[str, Dict[str, int]],
    normalized_profiles: torch.Tensor,
    importance: Dict[str, float],
    rho: Dict[str, torch.Tensor],
    avg_rank: float = 1,
    beta: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-8,
) -> AllocationResult:
    if normalized_profiles.ndim != 3:
        raise ValueError("normalized_profiles must have shape [M, K, N]")
    total_budget = float(avg_rank) * sum(float(module_dims[m]["cost"]) for m in module_names)
    used_budget = 0.0
    rank_pattern = {name: 0 for name in module_names}
    basis: List[torch.Tensor] = []
    selected: List[Dict[str, Any]] = []
    steps: List[Dict[str, Any]] = []
    all_scores: List[Dict[str, Any]] = []
    max_k = normalized_profiles.shape[1]

    while True:
        best = None
        round_scores: List[Dict[str, Any]] = []
        for module_index, module_name in enumerate(module_names):
            atom_index = rank_pattern[module_name]
            if atom_index >= max_k:
                continue
            cost = float(module_dims[module_name]["cost"])
            if used_budget + cost > total_budget + eps:
                continue
            profile = normalized_profiles[module_index, atom_index].to(dtype=torch.float32)
            residual = _orthogonal_residual(profile, basis)
            coverage = float(torch.dot(residual, residual).item())
            rho_value = float(rho[module_name][atom_index].item())
            importance_value = float(importance.get(module_name, 0.0))
            importance_weight = math.exp(importance_value)
            score = (
                (max(importance_weight, eps) ** beta)
                * (max(rho_value, eps) ** gamma)
                * coverage
                / cost
            )
            candidate = {
                "module_name": module_name,
                "module_index": module_index,
                "atom_index": atom_index,
                "cost": cost,
                "coverage": coverage,
                "rho": rho_value,
                "importance": importance_value,
                "importance_weight": importance_weight,
                "score": score,
            }
            round_scores.append(candidate)
            if best is None or score > best["score"]:
                best = candidate
        all_scores.extend(round_scores)
        if best is None or best["score"] <= eps:
            break

        module_name = best["module_name"]
        atom_index = best["atom_index"]
        rank_pattern[module_name] += 1
        used_budget += best["cost"]
        profile = normalized_profiles[best["module_index"], atom_index].to(dtype=torch.float32)
        _append_basis(profile, basis, eps)
        selected.append(best)
        steps.append(dict(best, used_budget=used_budget, total_budget=total_budget))

    return AllocationResult(
        rank_pattern=rank_pattern,
        selected_atoms=selected,
        allocation_steps=steps,
        total_budget=total_budget,
        used_budget=used_budget,
        importance=importance,
        rho=rho,
        scores=all_scores,
    )
