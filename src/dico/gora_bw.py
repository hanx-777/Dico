from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from dico.rank_budget import compute_total_lora_params, module_rank_cost


@dataclass(frozen=True)
class GoRABWResult:
    rank_dict: dict[str, int]
    target_budget: int
    realized_params: int
    r_min: int
    r_max: int
    advantages: dict[str, float]
    trace: list[dict[str, object]]


def compute_gora_importance(weight: torch.Tensor, grad: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(weight.detach().float() * grad.detach().float())).item())


def allocate_gora_bw(
    weights: Mapping[str, torch.Tensor],
    grads: Mapping[str, torch.Tensor],
    module_dims: Mapping[str, Mapping[str, int]],
    r_ref: int = 8,
    eta: float = 0.98,
) -> GoRABWResult:
    names = list(module_dims.keys())
    importances = {name: compute_gora_importance(weights[name], grads[name]) for name in names}
    total_importance = sum(importances.values())
    if total_importance <= 0.0:
        advantages = {name: 1.0 / max(len(names), 1) for name in names}
    else:
        advantages = {name: importances[name] / total_importance for name in names}
    r_min = int(round(float(r_ref) / 2.0))
    r_max = int(round(float(r_ref) * 4.0))
    smooth = {
        name: math.sqrt(float(module_dims[name]["out_dim"])) + math.sqrt(float(module_dims[name]["in_dim"]))
        for name in names
    }
    b_value = sum(smooth[name] * int(r_ref) for name in names)
    rank_dict = {}
    for name in names:
        continuous = b_value * advantages[name] / max(smooth[name], 1.0e-12)
        rank_dict[name] = max(r_min, min(r_max, int(round(continuous))))
    target_budget = sum(int(r_ref) * module_rank_cost(module_dims[name]) for name in names)
    lower = int(math.ceil(float(eta) * target_budget))
    trace: list[dict[str, object]] = []

    def total() -> int:
        return compute_total_lora_params(rank_dict, module_dims)

    while total() > target_budget:
        candidates = [name for name in names if rank_dict[name] > r_min]
        if not candidates:
            break
        victim = min(candidates, key=lambda name: (advantages[name], -module_rank_cost(module_dims[name]), name))
        before = total()
        rank_dict[victim] -= 1
        trace.append({"action": "decrement", "module": victim, "budget_before": before, "budget_after": total()})

    while total() < lower:
        candidates = [
            name
            for name in names
            if rank_dict[name] < r_max and total() + module_rank_cost(module_dims[name]) <= target_budget
        ]
        if not candidates:
            break
        winner = max(candidates, key=lambda name: (advantages[name], -module_rank_cost(module_dims[name]), name))
        before = total()
        rank_dict[winner] += 1
        trace.append({"action": "increment", "module": winner, "budget_before": before, "budget_after": total()})

    if total() < lower:
        preferred = dict(rank_dict)
        states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, tuple())}
        for name in names:
            cost = module_rank_cost(module_dims[name])
            next_states: dict[int, tuple[float, tuple[int, ...]]] = {}
            for used, (distance, ranks) in states.items():
                for rank in range(r_min, r_max + 1):
                    new_used = used + rank * cost
                    if new_used > target_budget:
                        continue
                    new_distance = distance + abs(rank - preferred[name]) - 1.0e-6 * advantages[name] * rank
                    new_ranks = ranks + (rank,)
                    existing = next_states.get(new_used)
                    if existing is None or (new_distance, new_ranks) < (existing[0], existing[1]):
                        next_states[new_used] = (new_distance, new_ranks)
            states = next_states
        feasible = [(budget, state) for budget, state in states.items() if lower <= budget <= target_budget]
        if feasible:
            best_budget, (_distance, best_ranks) = max(
                feasible,
                key=lambda item: (-item[1][0], item[0]),
            )
            before = total()
            rank_dict = {name: int(rank) for name, rank in zip(names, best_ranks)}
            trace.append(
                {
                    "action": "window_dp_repair",
                    "module": None,
                    "budget_before": before,
                    "budget_after": best_budget,
                }
            )

    return GoRABWResult(
        rank_dict=rank_dict,
        target_budget=target_budget,
        realized_params=total(),
        r_min=r_min,
        r_max=r_max,
        advantages=advantages,
        trace=trace,
    )
