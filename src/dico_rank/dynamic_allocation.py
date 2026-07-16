from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from dico_rank.lora_masked import MaskedLoRALinear
from dico_rank.rank_budget import BudgetManager, compute_total_lora_params, module_rank_cost


def rank_distance(current: Mapping[str, int], reference: Mapping[str, int]) -> int:
    return sum(abs(int(current.get(name, 0)) - int(reference.get(name, 0))) for name in reference)


class DynamicRankAllocator:
    def __init__(
        self,
        masked_lora_modules: Mapping[str, MaskedLoRALinear],
        module_dims: Mapping[str, Mapping[str, Any]],
        initial_allocation: Mapping[str, int],
        target_budget: int,
        config: Mapping[str, Any],
        base_rank: int,
        preallocation: Mapping[str, int] | None = None,
        budget_manager: BudgetManager | None = None,
    ):
        self.masked_lora_modules = dict(masked_lora_modules)
        self.module_dims = dict(module_dims)
        self.initial_allocation = {name: int(value) for name, value in initial_allocation.items()}
        self.current_allocation = dict(self.initial_allocation)
        self.preallocation = {name: int(value) for name, value in preallocation.items()} if preallocation else None
        self.target_budget = int(target_budget)
        self.config = dict(config)
        self.base_rank = int(base_rank)
        self.r_min = max(0, int(self.base_rank * self.config.get("r_min_multiplier", 0.0)))
        self.r_max = int(self.base_rank * self.config.get("r_max_multiplier", 2.0))
        self.score_smoothing = float(self.config.get("score_smoothing", 1.0e-6))
        self.grad_weight = float(self.config.get("grad_weight", 0.5))
        self.update_weight = float(self.config.get("update_weight", 0.5))
        self.ema_decay = float(self.config.get("score_ema_decay", 0.9))
        self.module_scores = {name: 1.0 for name in self.module_dims}
        self._adjusted_ratios: set[float] = set()
        self.budget_manager = budget_manager or BudgetManager(
            "equal_trainable_params",
            self.module_dims,
            warning_threshold=float(self.config.get("warning_threshold", 0.01)),
        )

    def update_statistics(self) -> None:
        for name, module in self.masked_lora_modules.items():
            channel_scores = module.channel_scores(
                grad_weight=self.grad_weight,
                update_weight=self.update_weight,
            )
            mask = module.get_rank_mask().to(channel_scores.device) > 0
            value = float(channel_scores[mask].mean().item()) if bool(mask.any()) else 0.0
            old = float(self.module_scores.get(name, value))
            self.module_scores[name] = self.ema_decay * old + (1.0 - self.ema_decay) * value

    def should_adjust(self, global_step: int, total_steps: int) -> bool:
        if not self.config.get("enabled", True):
            return False
        if total_steps <= 0:
            return False
        freeze_after = self.config.get("freeze_after_ratio")
        if freeze_after is not None and global_step > math.ceil(float(freeze_after) * total_steps):
            return False
        for ratio in self.config.get("update_ratios", [0.2, 0.4, 0.6]):
            ratio = float(ratio)
            if ratio in self._adjusted_ratios:
                continue
            threshold = max(1, math.ceil(ratio * total_steps))
            if int(global_step) >= threshold:
                self._adjusted_ratios.add(ratio)
                return True
        return False

    def _normalized_scores(self) -> dict[str, float]:
        values = {
            name: max(0.0, float(self.module_scores.get(name, 0.0))) + self.score_smoothing
            for name in self.module_dims
        }
        total = sum(values.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(values))
            return {name: uniform for name in values}
        return {name: value / total for name, value in values.items()}

    def _desired_allocation(self, normalized: Mapping[str, float]) -> dict[str, int]:
        allocation = {name: self.r_min for name in self.module_dims}
        manager = self.budget_manager
        while True:
            actual = manager.total_params(allocation)
            remaining = self.target_budget - actual
            candidates = []
            for name, dims in self.module_dims.items():
                rank = allocation[name]
                cost = int(dims.get("in_dim", dims.get("d_in", 0))) + int(dims.get("out_dim", dims.get("d_out", 0)))
                if rank < self.r_max and cost <= remaining:
                    # Diminishing returns keeps smoothed scores from collapsing all rank into one module.
                    score = float(normalized[name]) / ((rank + 1) ** 0.5) / max(cost, 1)
                    candidates.append((score, name))
            if not candidates:
                break
            _score, name = max(candidates)
            allocation[name] += 1
        return allocation

    def _apply_allocation_to_modules(self) -> None:
        for name, rank in self.current_allocation.items():
            module = self.masked_lora_modules.get(name)
            if module is not None:
                module.set_active_rank(int(rank))

    def adjust_rank(self, global_step: int, total_steps: int | None = None) -> dict[str, Any]:
        before = dict(self.current_allocation)
        budget_before = compute_total_lora_params(before, self.module_dims)
        normalized = self._normalized_scores()
        desired = self._desired_allocation(normalized)
        move_budget = max(1, int(sum(before.values()) * float(self.config.get("move_ratio", 0.2))))
        max_rank_distance = 2 * move_budget
        allocation = dict(before)
        donors_log = []
        receivers_log = []

        def module_cost(module_name: str) -> int:
            return module_rank_cost(self.module_dims[module_name])

        def within_move_limit(candidate: Mapping[str, int]) -> bool:
            return rank_distance(candidate, before) <= max_rank_distance

        def append_donor(module_name: str, rank_before: int) -> None:
            donors_log.append(
                {
                    "module": module_name,
                    "rank_before": rank_before,
                    "rank_after": allocation[module_name],
                    "score": self.module_scores.get(module_name, 0.0),
                }
            )

        def append_receiver(module_name: str, rank_before: int) -> None:
            receivers_log.append(
                {
                    "module": module_name,
                    "rank_before": rank_before,
                    "rank_after": allocation[module_name],
                    "score": self.module_scores.get(module_name, 0.0),
                }
            )

        while within_move_limit(allocation):
            receiver_candidates = [
                name
                for name in allocation
                if allocation[name] < desired.get(name, allocation[name]) and allocation[name] < self.r_max
            ]
            if not receiver_candidates:
                break
            receiver = max(receiver_candidates, key=lambda name: normalized.get(name, 0.0))
            receiver_cost = module_cost(receiver)
            actual = compute_total_lora_params(allocation, self.module_dims)

            while actual + receiver_cost > self.target_budget:
                donor_candidates = [
                    name
                    for name in allocation
                    if name != receiver
                    and allocation[name] > self.r_min
                    and allocation[name] > desired.get(name, self.r_min)
                ]
                if not donor_candidates:
                    donor_candidates = [
                        name for name in allocation if name != receiver and allocation[name] > self.r_min
                    ]
                if not donor_candidates:
                    break
                donor = min(donor_candidates, key=lambda name: normalized.get(name, 0.0))
                candidate = dict(allocation)
                candidate[donor] -= 1
                if not within_move_limit(candidate):
                    break
                rank_before = allocation[donor]
                allocation = candidate
                actual -= module_cost(donor)
                append_donor(donor, rank_before)

            actual = compute_total_lora_params(allocation, self.module_dims)
            candidate = dict(allocation)
            candidate[receiver] += 1
            if (
                actual + receiver_cost <= self.target_budget
                and allocation[receiver] < self.r_max
                and within_move_limit(candidate)
            ):
                rank_before = allocation[receiver]
                allocation = candidate
                append_receiver(receiver, rank_before)
            else:
                break

        while compute_total_lora_params(allocation, self.module_dims) > self.target_budget:
            donor_candidates = [
                name for name in allocation if allocation[name] > self.r_min
            ]
            if not donor_candidates:
                break
            movable = []
            for name in donor_candidates:
                candidate = dict(allocation)
                candidate[name] -= 1
                if within_move_limit(candidate):
                    movable.append(name)
            if not movable:
                break
            donor = min(movable, key=lambda name: (normalized.get(name, 0.0), -module_cost(name), name))
            rank_before = allocation[donor]
            allocation[donor] -= 1
            append_donor(donor, rank_before)

        self.current_allocation = allocation
        self._apply_allocation_to_modules()

        distance_from_initial = rank_distance(self.current_allocation, self.initial_allocation)
        distance_from_preallocation = (
            rank_distance(self.current_allocation, self.preallocation) if self.preallocation is not None else None
        )
        budget = self.budget_manager.describe(self.current_allocation, self.target_budget)
        rank_distance_this_adjustment = rank_distance(self.current_allocation, before)
        return {
            "step": int(global_step),
            "total_steps": int(total_steps) if total_steps is not None else None,
            "move_budget": int(move_budget),
            "rank_distance_this_adjustment": int(rank_distance_this_adjustment),
            "num_moved": int((rank_distance_this_adjustment + 1) // 2),
            "donors": donors_log,
            "receivers": receivers_log,
            "budget_before": int(budget_before),
            "budget_after": int(budget["actual_budget"]),
            "target_budget": int(budget["target_budget"]),
            "budget_error_ratio": float(budget["budget_error_ratio"]),
            "rank_distance_from_initial": int(distance_from_initial),
            "rank_distance_from_preallocation": distance_from_preallocation,
            "score_smoothing": float(self.score_smoothing),
            "allocation": dict(self.current_allocation),
            "warning": budget["warning"],
        }
