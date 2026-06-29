import math
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class AllocationResult:
    rank_pattern: Dict[str, int]
    debug: Dict[str, Any]


def _orthonormalize(profiles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    profiles: [k, N]
    returns: [k, N] orthonormal basis
    """
    U, S, Vh = torch.linalg.svd(profiles, full_matrices=False)
    tol = eps * max(profiles.shape) * S[0]
    rank = (S > tol).sum().item()
    return Vh[:rank]


def _residual_coverage(Q: torch.Tensor, v: torch.Tensor) -> float:
    """
    Q: [r, N] orthonormal basis
    v: [N] vector
    returns: ||v - Q^T Q v||^2
    """
    if Q.shape[0] == 0:
        return float(torch.sum(v ** 2).item())
    v_proj = v @ Q.T
    v_reconstructed = v_proj @ Q
    residual = v - v_reconstructed
    return float(torch.sum(residual ** 2).item())


def allocate_module_dico(
    module_names: List[str],
    module_dims: Dict[str, Dict[str, int]],
    normalized_profiles: torch.Tensor,  # [M, N]
    importance: torch.Tensor,           # [M]
    avg_rank: float,
    budget_floor_ratio: float = 0.95,
    budget_max_ratio: float = 1.0,
    coverage_eps: float = 1e-3,
    max_rank_per_module: int = 8,
    rank_decay: str = "sqrt",
) -> AllocationResult:
    
    num_modules = len(module_names)
    assert normalized_profiles.shape[0] == num_modules
    assert importance.shape[0] == num_modules
    
    costs = torch.tensor([module_dims[name]["cost"] for name in module_names], dtype=torch.float32)
    total_cost = costs.sum().item()
    budget_ref = float(avg_rank) * total_cost
    
    budget_min = budget_ref * budget_floor_ratio
    budget_max = budget_ref * budget_max_ratio
    
    rank_pattern = {name: 0 for name in module_names}
    used_budget = 0.0
    
    Q = torch.empty((0, normalized_profiles.shape[1]), dtype=torch.float32)
    step = 0
    tail_switch_step = None
    
    debug_log = []

    def current_importance(m: int) -> float:
        r = rank_pattern[module_names[m]]
        imp = float(importance[m].item())
        if rank_decay == "sqrt":
            return imp / math.sqrt(r + 1)
        elif rank_decay == "linear":
            return imp / (r + 1)
        elif rank_decay == "exp":
            return imp * (0.5 ** r)
        return imp

    while used_budget < budget_max:
        best_module = -1
        best_score = -1.0
        best_residual = 0.0
        
        mode = "coverage" if tail_switch_step is None else "tail"
        
        for m in range(num_modules):
            name = module_names[m]
            if rank_pattern[name] >= max_rank_per_module:
                continue
            
            c = float(costs[m].item())
            if used_budget + c > budget_max:
                continue
                
            imp = current_importance(m)
            
            if mode == "coverage":
                residual = _residual_coverage(Q, normalized_profiles[m])
                score = (residual * imp) / c
            else:
                residual = 0.0
                score = imp / c
                
            if score > best_score:
                best_score = score
                best_module = m
                best_residual = residual

        if best_module == -1:
            # Cannot add anything without violating constraints
            break
            
        if mode == "coverage" and best_residual < coverage_eps and used_budget < budget_min:
            tail_switch_step = step
            mode = "tail"
            # Re-evaluate in tail mode immediately
            continue
        elif mode == "coverage" and best_residual < coverage_eps and used_budget >= budget_min:
            # We hit target coverage and satisfied budget_min
            break
            
        name = module_names[best_module]
        c = float(costs[best_module].item())
        
        rank_pattern[name] += 1
        used_budget += c
        
        if mode == "coverage":
            Q_new = torch.cat([Q, normalized_profiles[best_module].unsqueeze(0)], dim=0)
            Q = _orthonormalize(Q_new)
            
        debug_log.append({
            "step": step,
            "mode": mode,
            "selected": name,
            "score": best_score,
            "residual": best_residual,
            "rank": rank_pattern[name],
            "used_budget": used_budget
        })
        step += 1

    return AllocationResult(
        rank_pattern=rank_pattern,
        debug={
            "used_budget": used_budget,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "budget_ref": budget_ref,
            "tail_switch_step": tail_switch_step,
            "total_steps": step,
            "log": debug_log,
        }
    )
