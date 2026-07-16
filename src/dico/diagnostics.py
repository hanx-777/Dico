from __future__ import annotations

from typing import Mapping, Sequence

from scipy.stats import spearmanr

from dico.path_utils import extract_layer_index
from dico.rank_budget import module_rank_cost


def _module_type(module_name: str) -> str:
    return str(module_name).split(".")[-1]


def gini(values: Sequence[float]) -> float:
    """Standard Gini coefficient over a list of non-negative values. Returns 0.0
    for an empty list or when every value is zero (perfectly "equal" in the
    degenerate sense that there is nothing to be unequal about).
    """
    ordered = sorted(float(value) for value in values)
    n = len(ordered)
    total = sum(ordered)
    if n == 0 or total <= 0.0:
        return 0.0
    weighted_sum = sum((i + 1) * value for i, value in enumerate(ordered))
    return (2.0 * weighted_sum - (n + 1) * total) / (n * total)


def compute_diagnostics(
    rank_dict: Mapping[str, int],
    module_dims: Mapping[str, Mapping[str, int]],
    r_max: int,
    target_budget: int,
    balanced_fill_ratio: float,
    init_summaries: Mapping[str, Mapping[str, object]],
    module_quota: Mapping[str, float] | None = None,
    r_min: int = 0,
    top_k: int = 10,
) -> dict[str, object]:
    """The required allocation diagnostics, computed directly from the final
    rank_dict/module_dims/init summaries -- not re-derived from disk, so it can never
    drift from what was actually written to rank_dict.json.

    `module_quota` (procurement.py's soft reference rank r_bar_m) is optional --
    non-CovRA allocation paths never populate it, in which case `qds` and
    `spearman_corr_r_rbar` (3.4.3節's direction-level-structure-survival diagnostics)
    come out `None` rather than being silently computed from a meaningless empty quota.
    """
    module_names = list(rank_dict.keys())
    costs = {name: module_rank_cost(module_dims[name]) for name in module_names}
    ranks = [int(rank_dict[name]) for name in module_names]
    param_counts = {name: int(rank_dict[name]) * costs[name] for name in module_names}
    total_params = sum(param_counts.values())

    rank_gini = gini(ranks)
    param_share_gini = gini(list(param_counts.values()))
    cap_hit_ratio = sum(1 for rank in ranks if rank >= int(r_max)) / max(len(ranks), 1)
    zero_rank_ratio = sum(1 for rank in ranks if rank == 0) / max(len(ranks), 1)

    type_budget: dict[str, float] = {}
    for name in module_names:
        module_type = _module_type(name)
        type_budget[module_type] = type_budget.get(module_type, 0.0) + param_counts[name]
    type_budget_share = (
        {module_type: value / total_params for module_type, value in type_budget.items()}
        if total_params > 0
        else {module_type: 0.0 for module_type in type_budget}
    )

    sorted_params = sorted(param_counts.values(), reverse=True)
    top10_module_share = (sum(sorted_params[: int(top_k)]) / total_params) if total_params > 0 else 0.0

    by_type_layered: dict[str, list[tuple[int, str]]] = {}
    for name in module_names:
        layer_index = extract_layer_index(name)
        if layer_index is None:
            continue
        by_type_layered.setdefault(_module_type(name), []).append((layer_index, name))
    adjacent_diffs: list[int] = []
    for entries in by_type_layered.values():
        entries.sort(key=lambda item: item[0])
        for i in range(len(entries) - 1):
            adjacent_diffs.append(abs(rank_dict[entries[i][1]] - rank_dict[entries[i + 1][1]]))
    mean_abs_adjacent_rank_diff = sum(adjacent_diffs) / len(adjacent_diffs) if adjacent_diffs else 0.0

    total_certified = sum(int(row.get("certified_rows", 0) or 0) for row in init_summaries.values())
    total_relaxation = sum(int(row.get("relaxation_rows", 0) or 0) for row in init_summaries.values())
    anchored_denominator = total_certified + total_relaxation
    anchored_rank_ratio = (total_certified / anchored_denominator) if anchored_denominator > 0 else 0.0

    budget_realized_ratio = (total_params / int(target_budget)) if target_budget else 0.0

    # 3.4.3節: QDS (quota deviation share) and Spearman corr(r_m, r_bar_m) -- the
    # evidence that final allocation didn't degenerate into a pure module-level quota
    # integerization. Only computed when a real module_quota exists (CovRA paths);
    # other allocation methods never populate module_quota, so these come out None.
    qds: float | None = None
    spearman_corr_r_rbar: float | None = None
    if module_quota:
        clipped_quota = {
            name: min(max(round(module_quota[name]), int(r_min)), int(r_max))
            for name in module_names
            if name in module_quota
        }
        qds = sum(
            abs(int(rank_dict[name]) - clipped_quota.get(name, int(r_min))) * costs[name]
            for name in module_names
        ) / max(int(target_budget), 1)
        if len(module_names) >= 2:
            r_values = [float(rank_dict[name]) for name in module_names]
            rbar_values = [float(module_quota.get(name, r_min)) for name in module_names]
            if len(set(r_values)) > 1 and len(set(rbar_values)) > 1:
                # Indexed access (not `.statistic`) for compatibility across scipy
                # versions: SignificanceResult (>=1.9) vs. the older plain-tuple return.
                correlation = float(spearmanr(r_values, rbar_values)[0])
                spearman_corr_r_rbar = correlation if correlation == correlation else None  # NaN guard

    return {
        "rank_gini": rank_gini,
        "param_share_gini": param_share_gini,
        "cap_hit_ratio": cap_hit_ratio,
        "zero_rank_ratio": zero_rank_ratio,
        "type_budget_share": type_budget_share,
        "top10_module_share": top10_module_share,
        "mean_abs_adjacent_rank_diff": mean_abs_adjacent_rank_diff,
        "anchored_rank_ratio": anchored_rank_ratio,
        "balanced_fill_ratio": float(balanced_fill_ratio),
        "budget_realized_ratio": budget_realized_ratio,
        "qds": qds,
        "spearman_corr_r_rbar": spearman_corr_r_rbar,
    }
