from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class BudgetInfo:
    budget_mode: str
    target_budget: int
    actual_budget: int
    budget_error: int
    budget_error_ratio: float
    total_active_rank: int
    over_budget: bool = False
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_mode": self.budget_mode,
            "target_budget": self.target_budget,
            "actual_budget": self.actual_budget,
            "budget_error": self.budget_error,
            "budget_error_ratio": self.budget_error_ratio,
            "total_active_rank": self.total_active_rank,
            "over_budget": self.over_budget,
            "warning": self.warning,
        }


@dataclass(frozen=True)
class RepairResult:
    allocation: dict[str, int]
    budget: BudgetInfo


@dataclass(frozen=True)
class WeightedAllocationResult:
    allocation: dict[str, int]
    budget: BudgetInfo
    module_logs: list[dict[str, Any]]


def _dim_value(dims: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if name in dims:
            return int(dims[name])
    raise KeyError(f"Module dims missing one of {names}: {dims}")


def module_rank_cost(module_dims: Mapping[str, Any]) -> int:
    return _dim_value(module_dims, "in_dim", "d_in", "in_features") + _dim_value(
        module_dims, "out_dim", "d_out", "out_features"
    )


def compute_lora_params_for_module(rank: int, in_dim: int, out_dim: int) -> int:
    return int(rank) * (int(in_dim) + int(out_dim))


def compute_total_lora_params(
    rank_allocation: Mapping[str, int],
    module_dims: Mapping[str, Mapping[str, Any]],
) -> int:
    total = 0
    for name, rank in rank_allocation.items():
        if name not in module_dims:
            raise KeyError(f"Missing dims for module {name}")
        total += int(rank) * module_rank_cost(module_dims[name])
    return int(total)


def _warning_for_error(error_ratio: float, threshold: float, over_budget: bool = False) -> str | None:
    if over_budget:
        return "actual_active_lora_params exceeds target_budget; comparison is not fair"
    if error_ratio > threshold:
        return "budget_error_ratio exceeds 1%; comparison may be less fair"
    return None


def _budget_info(
    allocation: Mapping[str, int],
    target_budget: int,
    module_dims: Mapping[str, Mapping[str, Any]],
    budget_mode: str = "equal_trainable_params",
    warning_threshold: float = 0.01,
) -> BudgetInfo:
    actual = compute_total_lora_params(allocation, module_dims)
    error = int(target_budget) - int(actual)
    over_budget = error < 0
    ratio = float(abs(error) / target_budget) if target_budget else (1.0 if actual else 0.0)
    return BudgetInfo(
        budget_mode=budget_mode,
        target_budget=int(target_budget),
        actual_budget=int(actual),
        budget_error=int(error),
        budget_error_ratio=ratio,
        total_active_rank=sum(int(v) for v in allocation.values()),
        over_budget=over_budget,
        warning=_warning_for_error(ratio, warning_threshold, over_budget=over_budget),
    )


def get_uniform_budget(
    rank: int,
    target_modules: list[str],
    module_dims: Mapping[str, Mapping[str, Any]],
    budget_mode: str = "equal_trainable_params",
    warning_threshold: float = 0.01,
) -> BudgetInfo:
    allocation = {name: int(rank) for name in target_modules}
    target = compute_total_lora_params(allocation, module_dims)
    return _budget_info(
        allocation,
        target_budget=target,
        module_dims=module_dims,
        budget_mode=budget_mode,
        warning_threshold=warning_threshold,
    )


def _clip_allocation(
    allocation: Mapping[str, int],
    module_names: list[str],
    r_min: int,
    r_max: int,
) -> dict[str, int]:
    return {
        name: max(int(r_min), min(int(r_max), int(allocation.get(name, r_min))))
        for name in module_names
    }


def repair_allocation_to_budget(
    rank_allocation: Mapping[str, int],
    target_budget: int,
    module_dims: Mapping[str, Mapping[str, Any]],
    r_min: int = 0,
    r_max: int | None = None,
    budget_mode: str = "equal_trainable_params",
    warning_threshold: float = 0.01,
) -> RepairResult:
    """Repair an allocation to the globally closest feasible active-parameter budget.

    When the lower-bound allocation is within target, this bounded integer DP
    maximizes actual_budget subject to actual_budget <= target_budget. Ties are
    deterministic: keep closest to the input allocation, then prefer stable
    module-name order through the fixed module traversal.
    """

    module_names = list(module_dims.keys())
    if r_max is None:
        r_max = max([int(v) for v in rank_allocation.values()] + [int(r_min)])
    allocation = _clip_allocation(rank_allocation, module_names, int(r_min), int(r_max))
    costs = {name: module_rank_cost(module_dims[name]) for name in module_names}

    min_allocation = {name: int(r_min) for name in module_names}
    min_budget = compute_total_lora_params(min_allocation, module_dims)
    if min_budget > target_budget:
        info = _budget_info(
            min_allocation,
            target_budget,
            module_dims,
            budget_mode=budget_mode,
            warning_threshold=warning_threshold,
        )
        return RepairResult(min_allocation, info)

    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, tuple())}
    for name in module_names:
        cost = costs[name]
        preferred = allocation[name]
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for used_budget, (distance, ranks) in states.items():
            for rank in range(int(r_min), int(r_max) + 1):
                new_budget = used_budget + rank * cost
                if new_budget > target_budget:
                    continue
                new_distance = distance + abs(rank - preferred)
                new_ranks = ranks + (rank,)
                existing = next_states.get(new_budget)
                if existing is None or (new_distance, new_ranks) < (existing[0], existing[1]):
                    next_states[new_budget] = (new_distance, new_ranks)
        states = next_states
        if not states:
            break

    if not states:
        info = _budget_info(
            min_allocation,
            target_budget,
            module_dims,
            budget_mode=budget_mode,
            warning_threshold=warning_threshold,
        )
        return RepairResult(min_allocation, info)

    best_budget = max(states)
    _distance, best_ranks = states[best_budget]
    allocation = {name: int(rank) for name, rank in zip(module_names, best_ranks)}

    info = _budget_info(
        allocation,
        target_budget,
        module_dims,
        budget_mode=budget_mode,
        warning_threshold=warning_threshold,
    )
    return RepairResult(allocation, info)


class BudgetManager:
    def __init__(
        self,
        budget_mode: str,
        module_dims: Mapping[str, Mapping[str, Any]],
        warning_threshold: float = 0.01,
    ):
        self.budget_mode = budget_mode
        self.module_dims = dict(module_dims)
        self.warning_threshold = float(warning_threshold)

    def total_params(self, allocation: Mapping[str, int]) -> int:
        return compute_total_lora_params(allocation, self.module_dims)

    def describe(self, allocation: Mapping[str, int], target_budget: int) -> dict[str, Any]:
        return _budget_info(
            allocation,
            target_budget,
            self.module_dims,
            budget_mode=self.budget_mode,
            warning_threshold=self.warning_threshold,
        ).to_dict()

    def repair(
        self,
        allocation: Mapping[str, int],
        target_budget: int,
        r_min: int = 0,
        r_max: int | None = None,
    ) -> RepairResult:
        return repair_allocation_to_budget(
            allocation,
            target_budget,
            self.module_dims,
            r_min=r_min,
            r_max=r_max,
            budget_mode=self.budget_mode,
            warning_threshold=self.warning_threshold,
        )


def allocate_by_weighted_utility(
    module_utilities: Mapping[str, float],
    module_dims: Mapping[str, Mapping[str, Any]],
    total_rank_budget: int,
    target_budget: int,
    r_min: int,
    r_max: int,
    use_cost_aware: bool = True,
    budget_mode: str = "equal_trainable_params",
    warning_threshold: float = 0.01,
) -> WeightedAllocationResult:
    """Allocate integer ranks from continuous module utilities.

    The conversion uses cost-aware scores, continuous ranks, floor + largest
    remainder, a greedy budget-fill pass, then final active-parameter repair.
    """

    module_names = list(module_dims.keys())
    costs = {name: module_rank_cost(module_dims[name]) for name in module_names}
    utilities = {name: max(0.0, float(module_utilities.get(name, 0.0))) for name in module_names}
    scores = {
        name: (utilities[name] / max(costs[name], 1) if use_cost_aware else utilities[name])
        for name in module_names
    }
    score_sum = sum(scores.values())
    if score_sum <= 0:
        scores = {name: 1.0 / max(1, len(module_names)) for name in module_names}
        score_sum = sum(scores.values())

    total_rank_budget = max(int(total_rank_budget), int(r_min) * len(module_names))
    continuous = {name: total_rank_budget * scores[name] / score_sum for name in module_names}
    allocation = {
        name: max(int(r_min), min(int(r_max), int(continuous[name] // 1)))
        for name in module_names
    }
    desired_total = min(int(total_rank_budget), int(r_max) * len(module_names))
    current_total = sum(allocation.values())
    remainders = sorted(
        module_names,
        key=lambda name: (continuous[name] - int(continuous[name]), scores[name]),
        reverse=True,
    )
    while current_total < desired_total:
        changed = False
        for name in remainders:
            if current_total >= desired_total:
                break
            if allocation[name] < r_max:
                allocation[name] += 1
                current_total += 1
                changed = True
        if not changed:
            break

    # Greedily fill any remaining active-parameter budget using marginal utility per cost.
    while True:
        actual = compute_total_lora_params(allocation, module_dims)
        remaining = int(target_budget) - actual
        candidates = [
            name
            for name in module_names
            if allocation[name] < r_max and costs[name] <= remaining
        ]
        if not candidates:
            break
        best = max(
            candidates,
            key=lambda name: (
                scores[name] / ((allocation[name] + 1) ** 0.5),
                utilities[name],
                -costs[name],
            ),
        )
        allocation[best] += 1

    repaired = repair_allocation_to_budget(
        allocation,
        int(target_budget),
        module_dims,
        r_min=int(r_min),
        r_max=int(r_max),
        budget_mode=budget_mode,
        warning_threshold=warning_threshold,
    )
    module_logs = [
        {
            "module_name": name,
            "module_utility": utilities[name],
            "rank_cost": costs[name],
            "cost_aware_score": scores[name],
            "continuous_rank": continuous[name],
            "final_rank": repaired.allocation[name],
        }
        for name in module_names
    ]
    return WeightedAllocationResult(
        allocation=repaired.allocation,
        budget=repaired.budget,
        module_logs=module_logs,
    )


def allocate_by_evidence_aware_utility(
    module_utilities: Mapping[str, float],
    module_dims: Mapping[str, Mapping[str, Any]],
    selected_atom_utilities: Mapping[str, list[float]],
    target_budget: int,
    eta: float,
    lambda_next: float,
    r_min: int,
    r_max: int,
    allow_rank_beyond_selected_evidence: bool = False,
    budget_mode: str = "equal_trainable_params",
    warning_threshold: float = 0.01,
) -> WeightedAllocationResult:
    """Allocate integer ranks while respecting selected evidence counts.

    This is the v0.2.6 path: continuous module demand is computed from selected
    evidence utilities, and rounding may not create ranks unsupported by
    selected atoms unless explicitly configured.
    """

    module_names = list(module_dims.keys())
    costs = {name: module_rank_cost(module_dims[name]) for name in module_names}
    utilities = {name: max(0.0, float(module_utilities.get(name, 0.0))) for name in module_names}
    selected = {
        name: [max(0.0, float(value)) for value in selected_atom_utilities.get(name, [])]
        for name in module_names
    }
    evidence_caps = {
        name: (int(r_max) if allow_rank_beyond_selected_evidence else min(int(r_max), len(selected[name])))
        for name in module_names
    }

    scores = {name: utilities[name] / max(costs[name], 1) for name in module_names}
    score_budget_sum = sum(costs[name] * scores[name] for name in module_names)
    if score_budget_sum <= 0:
        scores = {
            name: (len(selected[name]) / max(costs[name], 1) if selected[name] else 0.0)
            for name in module_names
        }
        score_budget_sum = sum(costs[name] * scores[name] for name in module_names)
    if score_budget_sum <= 0:
        scores = {name: 0.0 for name in module_names}
        continuous = {name: 0.0 for name in module_names}
    else:
        continuous = {
            name: float(target_budget) * scores[name] / score_budget_sum
            for name in module_names
        }

    allocation = {
        name: min(evidence_caps[name], max(0, int(continuous[name] // 1)))
        for name in module_names
    }
    if int(r_min) > 0:
        allocation = {
            name: min(evidence_caps[name], max(int(r_min), allocation[name]))
            for name in module_names
        }

    def total() -> int:
        return compute_total_lora_params(allocation, module_dims)

    while total() > int(target_budget):
        candidates = [name for name in module_names if allocation[name] > 0 and allocation[name] > int(r_min)]
        if not candidates:
            break

        def remove_key(name: str) -> tuple[float, float, int]:
            rank = allocation[name]
            current_utility = selected[name][rank - 1] if rank - 1 < len(selected[name]) else utilities[name]
            loss = (rank - continuous[name] + 1.0) / max(costs[name], 1) / max(current_utility, 1e-12)
            return (loss, -current_utility, costs[name])

        victim = max(candidates, key=remove_key)
        allocation[victim] -= 1

    target_min = int(float(eta) * int(target_budget))
    last_add_gain = {name: 0.0 for name in module_names}
    last_next_utility = {name: None for name in module_names}
    while total() < target_min:
        actual = total()
        candidates = []
        for name in module_names:
            if allocation[name] >= evidence_caps[name]:
                continue
            if actual + costs[name] > int(target_budget):
                continue
            next_utility = selected[name][allocation[name]] if allocation[name] < len(selected[name]) else 0.0
            fractional = max(0.0, continuous[name] - math.floor(continuous[name]))
            add_gain = fractional / max(costs[name], 1) + float(lambda_next) * next_utility / max(costs[name], 1)
            last_add_gain[name] = add_gain
            last_next_utility[name] = next_utility
            candidates.append(name)
        if not candidates:
            break
        best = max(candidates, key=lambda name: (last_add_gain[name], utilities[name], -costs[name], name))
        allocation[best] += 1

    info = _budget_info(
        allocation,
        int(target_budget),
        module_dims,
        budget_mode=budget_mode,
        warning_threshold=warning_threshold,
    )
    if total() < target_min:
        warning = (
            f"selected evidence constraints prevented reaching eta target; "
            f"actual_budget={total()} min_budget={target_min}"
        )
        info = replace(info, warning=warning)

    module_logs = []
    for name in module_names:
        rank = allocation[name]
        next_utility = selected[name][rank] if rank < len(selected[name]) else None
        fractional = max(0.0, continuous[name] - math.floor(continuous[name]))
        add_gain = (
            fractional / max(costs[name], 1)
            + float(lambda_next) * float(next_utility or 0.0) / max(costs[name], 1)
        )
        module_logs.append(
            {
                "module_name": name,
                "module_utility": utilities[name],
                "rank_cost": costs[name],
                "cost_aware_score": scores[name],
                "continuous_rank": continuous[name],
                "r_tilde": continuous[name],
                "floor_rank": int(max(0, math.floor(continuous[name]))),
                "final_rank": rank,
                "selected_atom_count": len(selected[name]),
                "selected_atom_utilities": selected[name],
                "fractional_remainder": fractional,
                "next_atom_utility": next_utility,
                "add_gain": add_gain,
                "allow_rank_beyond_selected_evidence": bool(allow_rank_beyond_selected_evidence),
                "final_parameter_count": int(rank) * costs[name],
            }
        )
    return WeightedAllocationResult(allocation=allocation, budget=info, module_logs=module_logs)
