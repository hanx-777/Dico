from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from dico.candidates import PhysicalCandidate
from dico.normalize import NormalizationStats, apply_normalized_utility, compute_normalized_utility
from dico.rank_budget import compute_total_lora_params, module_rank_cost

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcurementResult:
    rank_dict: dict[str, int]
    realized_params: int
    target_budget: int
    reserve_filled_ratio: float
    balanced_fill_ratio: float
    zero_rank_module_ratio: float
    budget_gap_ratio: float
    trace: list[dict[str, object]]
    warning: str | None = None
    physical_utility: dict[str, float] = field(default_factory=dict)
    normalized_utility: dict[str, float] = field(default_factory=dict)
    normalization_stats: NormalizationStats | None = None
    module_quota: dict[str, float] = field(default_factory=dict)
    # 3.4.1/3.4.2節: which physical direction(s) actually got tied to each module's
    # granted rank slots (r_min baseline + quota-aware purchase + reserve fallback +
    # balanced fill), in purchase order. Consumed by save_direction_bank so
    # direction-anchored init only anchors to directions procurement actually bought.
    purchased_directions: dict[str, list[str]] = field(default_factory=dict)


def _module_type(module_name: str) -> str:
    return str(module_name).split(".")[-1]


def _quota_pressure(rank: int, r_min: int, quota: float, eps: float) -> float:
    ratio = max(0, rank - int(r_min)) / max(float(quota) - int(r_min), eps)
    return 1.0 + ratio ** 2


def compute_module_quota(
    normalized_utility: Mapping[str, float],
    physical_module_of: Mapping[str, str],
    module_names: Sequence[str],
    costs: Mapping[str, int],
    r_min: int,
    budget_remaining: float,
    eps: float = 1.0e-6,
) -> dict[str, float]:
    """3.4.2节: module soft quota.

    D_m = sum of normalized utility over module m's own (certified) physical
    directions; sqrt-compressed and budget-share-normalized into a soft
    reference rank r_bar_m = r_min + s_m*B_rem/c_m. This is a *reference point*
    for quota pressure, never a hard cap -- callers must not clip r_m to it.
    """
    demand = {name: 0.0 for name in module_names}
    for physical_id, value in normalized_utility.items():
        module_name = physical_module_of.get(physical_id)
        if module_name in demand:
            demand[module_name] += float(value)
    demand_sqrt = {name: math.sqrt(demand[name] + eps) for name in module_names}
    total_demand_sqrt = sum(demand_sqrt.values())
    if total_demand_sqrt > 0:
        share = {name: demand_sqrt[name] / total_demand_sqrt for name in module_names}
    else:
        share = {name: 1.0 / max(len(module_names), 1) for name in module_names}
    budget_remaining = max(0.0, float(budget_remaining))
    return {
        name: float(r_min) + share[name] * budget_remaining / max(costs[name], 1)
        for name in module_names
    }


def _bucket_sorted_desc(
    candidates: Sequence[PhysicalCandidate], w_bar: Mapping[str, float], module_names: Sequence[str]
) -> dict[str, list[PhysicalCandidate]]:
    buckets: dict[str, list[PhysicalCandidate]] = {name: [] for name in module_names}
    for candidate in candidates:
        buckets.setdefault(candidate.module_name, []).append(candidate)
    return {
        name: sorted(rows, key=lambda c: -w_bar.get(c.physical_direction_id, 0.0))
        for name, rows in buckets.items()
    }


def _best_unconsumed_candidate_for_module(
    module_name: str,
    certified: Sequence[PhysicalCandidate],
    reserve: Sequence[PhysicalCandidate],
    w_bar: Mapping[str, float],
    purchased_physical: set[str],
) -> PhysicalCandidate | None:
    pool = [
        c
        for c in (*certified, *reserve)
        if c.module_name == module_name and c.physical_direction_id not in purchased_physical
    ]
    if not pool:
        return None
    return max(pool, key=lambda c: w_bar.get(c.physical_direction_id, 0.0))


def _quota_aware_purchase(
    candidates: Sequence[PhysicalCandidate],
    w_bar: Mapping[str, float],
    rank_dict: dict[str, int],
    costs: Mapping[str, int],
    r_min: int,
    r_max: int,
    quota: Mapping[str, float],
    target_budget: int,
    beta: float,
    purchased_physical: set[str],
    purchased_directions: dict[str, list[str]],
    source: str,
    trace: list[dict[str, object]],
    eps: float = 1.0e-6,
) -> int:
    """3.4.2節: quota-aware direction auction. Recomputes bid_p = w_bar_p /
    (cost^beta * Psi_m(r_m)) for all remaining eligible candidates each round,
    since only the just-bought module's quota pressure changes between rounds.
    """

    def total() -> int:
        return sum(rank_dict[name] * costs[name] for name in rank_dict)

    remaining = [c for c in candidates if c.physical_direction_id not in purchased_physical]
    purchased_count = 0
    while remaining:
        best: PhysicalCandidate | None = None
        best_key: tuple[float, float, str] | None = None
        for candidate in remaining:
            name = candidate.module_name
            if rank_dict[name] >= int(r_max):
                continue
            if total() + costs[name] > int(target_budget):
                continue
            psi = _quota_pressure(rank_dict[name], r_min, quota.get(name, r_min), eps)
            utility = w_bar[candidate.physical_direction_id]
            bid = utility / (max(costs[name], 1) ** float(beta) * psi)
            key = (bid, utility, candidate.physical_direction_id)
            if best_key is None or key > best_key:
                best = candidate
                best_key = key
        if best is None:
            break
        name = best.module_name
        before = total()
        rank_dict[name] += 1
        purchased_physical.add(best.physical_direction_id)
        purchased_directions.setdefault(name, []).append(best.physical_direction_id)
        purchased_count += 1
        remaining.remove(best)
        trace.append(
            {
                "source": source,
                "physical_direction": best.physical_direction_id,
                "module": name,
                "rank_after": rank_dict[name],
                "bid": float(best_key[0]),
                "normalized_utility": float(best_key[1]),
                "budget_before": before,
                "budget_after": total(),
            }
        )
    return purchased_count


def procure_budget_window(
    certified: Sequence[PhysicalCandidate],
    reserve: Sequence[PhysicalCandidate],
    module_dims: Mapping[str, Mapping[str, int]],
    target_budget: int,
    eta: float = 0.98,
    r_min: int = 2,
    r_max: int | None = None,
    beta: float = 0.5,
    reserve_queue_enabled: bool = True,
    balanced_fill_enabled: bool = True,
) -> ProcurementResult:
    module_names = list(module_dims.keys())
    costs = {name: module_rank_cost(module_dims[name]) for name in module_names}
    if r_max is None:
        r_max = max(int(r_min), max([len(module_names), 1]))

    # 3.4.1節: B_base = sum(r_min * c_m) must not exceed the target budget -- if the
    # r_min floor alone is already infeasible, purchasing anything on top of it can
    # only make the overshoot worse, so report the infeasibility immediately with the
    # r_min baseline allocation. Computed directly from r_min (not from rank_dict's
    # runtime init value), since rank_dict now starts at 0 and is built up explicitly.
    base_budget = sum(int(r_min) * costs[name] for name in module_names)
    if base_budget > int(target_budget):
        rank_dict = {name: int(r_min) for name in module_names}
        return ProcurementResult(
            rank_dict=rank_dict,
            realized_params=base_budget,
            target_budget=int(target_budget),
            reserve_filled_ratio=0.0,
            balanced_fill_ratio=0.0,
            zero_rank_module_ratio=sum(1 for rank in rank_dict.values() if rank == 0) / max(len(rank_dict), 1),
            budget_gap_ratio=0.0,
            trace=[],
            warning=(
                f"B_base (r_min={int(r_min)} baseline) = {base_budget} exceeds "
                f"target_budget={int(target_budget)}; allocation is infeasible at "
                f"the requested r_min without lowering r_min or raising the budget"
            ),
        )

    rank_dict = {name: 0 for name in module_names}
    purchased_physical: set[str] = set()
    purchased_directions: dict[str, list[str]] = {name: [] for name in module_names}
    trace: list[dict[str, object]] = []
    reserve_ranks = 0
    balanced_fill_ranks = 0

    def total() -> int:
        return compute_total_lora_params(rank_dict, module_dims)

    # 3.4.1節: type-wise whitening stats are fit on the CERTIFIED pool only, then
    # *reused* (not re-estimated) for the reserve pool via apply_normalized_utility --
    # otherwise reserve's raw energies (never certified, no joint-coverage gain) would
    # pollute the (median, mad) certified directions are scored against.
    module_of: dict[str, str] = {}
    type_of: dict[str, str] = {}
    certified_raw: dict[str, float] = {}
    for candidate in certified:
        certified_raw[candidate.physical_direction_id] = float(candidate.merged_utility)
        module_of[candidate.physical_direction_id] = candidate.module_name
        type_of[candidate.physical_direction_id] = _module_type(candidate.module_name)
    reserve_raw: dict[str, float] = {}
    for candidate in reserve:
        reserve_raw[candidate.physical_direction_id] = float(candidate.raw_energy)
        module_of[candidate.physical_direction_id] = candidate.module_name
        type_of[candidate.physical_direction_id] = _module_type(candidate.module_name)

    if certified_raw:
        w_bar_certified, normalization_stats = compute_normalized_utility(certified_raw, type_of)
    else:
        w_bar_certified, normalization_stats = {}, NormalizationStats()
    w_bar_reserve = (
        apply_normalized_utility(reserve_raw, type_of, normalization_stats) if reserve_raw else {}
    )
    w_bar: dict[str, float] = {**w_bar_certified, **w_bar_reserve}
    raw_utility_by_id: dict[str, float] = {**certified_raw, **reserve_raw}

    # 3.4.1節: r_min-phase direction consumption -- each module's r_min baseline slots
    # consume its own highest-normalized-utility covered directions first (certified,
    # then reserve as fallback), instead of being bare unattributed rank increments.
    certified_by_module = _bucket_sorted_desc(certified, w_bar, module_names)
    reserve_by_module = _bucket_sorted_desc(reserve, w_bar, module_names)
    for name in module_names:
        consumed = 0
        for pool in (certified_by_module.get(name, []), reserve_by_module.get(name, [])):
            for candidate in pool:
                if consumed >= int(r_min):
                    break
                if candidate.physical_direction_id in purchased_physical:
                    continue
                purchased_physical.add(candidate.physical_direction_id)
                purchased_directions[name].append(candidate.physical_direction_id)
                consumed += 1
            if consumed >= int(r_min):
                break
        rank_dict[name] = int(r_min)
        if consumed < int(r_min):
            LOGGER.warning(
                "r_min_direction_shortfall module=%s consumed=%d r_min=%d; remaining "
                "slots will use an orthogonal-random row at init time",
                name,
                consumed,
                int(r_min),
            )

    # 3.4.2節: soft quota is estimated from certified demand only, *before* either
    # certified purchasing or reserve fallback begins, and is not recomputed as
    # purchases proceed. D_m sums over ALL of module m's certified directions (not
    # filtered to not-yet-consumed-by-r_min), matching the doc's definition.
    budget_remaining = int(target_budget) - total()
    quota = compute_module_quota(
        w_bar_certified,
        module_of,
        module_names,
        costs,
        r_min=int(r_min),
        budget_remaining=budget_remaining,
    )

    _quota_aware_purchase(
        certified,
        w_bar,
        rank_dict,
        costs,
        int(r_min),
        int(r_max),
        quota,
        int(target_budget),
        beta,
        purchased_physical,
        purchased_directions,
        source="certified",
        trace=trace,
    )

    lower = int(math.ceil(float(eta) * int(target_budget)))
    if reserve_queue_enabled and total() < lower:
        reserve_ranks = _quota_aware_purchase(
            reserve,
            w_bar,
            rank_dict,
            costs,
            int(r_min),
            int(r_max),
            quota,
            int(target_budget),
            beta,
            purchased_physical,
            purchased_directions,
            source="reserve",
            trace=trace,
        )

    # 3.4.2節: quota-aware balanced fill replaces undifferentiated ("cheapest module
    # first") relaxation -- always give the next rank to whichever eligible module
    # sits furthest *below* its own soft quota, and (finding #7) tie that rank slot to
    # the module's best remaining unconsumed direction when one is available. Gated by
    # balanced_fill_enabled so the ablation ladder can disable this stage and observe
    # the unfilled result directly.
    while balanced_fill_enabled and total() < lower:
        candidates_pool = [
            name
            for name in module_names
            if rank_dict[name] < int(r_max) and total() + costs[name] <= int(target_budget)
        ]
        if not candidates_pool:
            break
        name = min(candidates_pool, key=lambda item: (rank_dict[item] / (quota.get(item, r_min) + 1e-9), item))
        before = total()
        best = _best_unconsumed_candidate_for_module(name, certified, reserve, w_bar, purchased_physical)
        if best is not None:
            purchased_physical.add(best.physical_direction_id)
            purchased_directions.setdefault(name, []).append(best.physical_direction_id)
        rank_dict[name] += 1
        balanced_fill_ranks += 1
        trace.append(
            {
                "source": "balanced_fill",
                "physical_direction": best.physical_direction_id if best else None,
                "module": name,
                "rank_after": rank_dict[name],
                "gain": 0.0,
                "budget_before": before,
                "budget_after": total(),
            }
        )

    for name in module_names:
        assert len(purchased_directions.get(name, [])) <= rank_dict[name], (
            f"purchased_directions[{name}] ({len(purchased_directions.get(name, []))}) exceeds "
            f"rank_dict[{name}] ({rank_dict[name]})"
        )

    final_rank = sum(rank_dict.values())
    final_total = total()
    budget_gap_ratio = max(0.0, float(lower - final_total) / int(target_budget)) if target_budget else 0.0
    return ProcurementResult(
        rank_dict=rank_dict,
        realized_params=final_total,
        target_budget=int(target_budget),
        reserve_filled_ratio=float(reserve_ranks / final_rank) if final_rank else 0.0,
        balanced_fill_ratio=float(balanced_fill_ranks / final_rank) if final_rank else 0.0,
        zero_rank_module_ratio=sum(1 for rank in rank_dict.values() if rank == 0) / max(len(rank_dict), 1),
        budget_gap_ratio=budget_gap_ratio,
        trace=trace,
        physical_utility=raw_utility_by_id,
        normalized_utility=w_bar,
        normalization_stats=normalization_stats,
        module_quota=quota,
        purchased_directions=purchased_directions,
    )
