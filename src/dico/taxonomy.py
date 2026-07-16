from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class DirectionStats:
    align: float
    f_stat: float


@dataclass(frozen=True)
class TaxonomyRow:
    index: int
    module_type: str
    align: float
    f_stat: float
    p_align: float
    p_f: float
    fdr_pass_align: bool
    fdr_pass_f: bool
    label: str


def _as_float_vector(profile: torch.Tensor) -> torch.Tensor:
    return profile.detach().flatten().float()


def _group_indices(groups: Sequence[str]) -> dict[str, torch.Tensor]:
    values: dict[str, list[int]] = {}
    for idx, group in enumerate(groups):
        values.setdefault(str(group), []).append(idx)
    return {group: torch.tensor(indices, dtype=torch.long) for group, indices in values.items()}


def direction_statistics(profile: torch.Tensor, groups: Sequence[str], eps: float = 1.0e-12) -> DirectionStats:
    values = _as_float_vector(profile)
    numerator = torch.abs(torch.sum(values))
    denominator = torch.sum(torch.abs(values)) + float(eps)
    align = 0.0 if float(denominator.item()) <= float(eps) else float((numerator / denominator).item())
    grouped = _group_indices(groups)
    if len(grouped) <= 1 or values.numel() <= len(grouped):
        return DirectionStats(align=align, f_stat=0.0)
    overall = torch.mean(values)
    between = torch.tensor(0.0)
    within = torch.tensor(0.0)
    for indices in grouped.values():
        row = values.index_select(0, indices)
        if row.numel() == 0:
            continue
        mean = torch.mean(row)
        between = between + row.numel() * (mean - overall) ** 2
        within = within + torch.sum((row - mean) ** 2)
    between = between / max(len(grouped) - 1, 1)
    within = within / max(values.numel() - len(grouped), 1)
    return DirectionStats(align=align, f_stat=float((between / (within + float(eps))).item()))


def _exact_sign_flip_p(profile: torch.Tensor, observed: float, eps: float) -> float | None:
    values = _as_float_vector(profile)
    n = int(values.numel())
    if n > 16:
        return None
    total = 0
    extreme = 0
    abs_sum = torch.sum(torch.abs(values)) + float(eps)
    for signs in itertools.product((-1.0, 1.0), repeat=n):
        total += 1
        stat = float((torch.abs(torch.sum(values * torch.tensor(signs))) / abs_sum).item())
        if stat >= observed - 1.0e-12:
            extreme += 1
    return extreme / max(total, 1)


def sign_flip_p_value(
    profile: torch.Tensor,
    observed_align: float,
    permutation_count: int,
    seed: int,
    eps: float = 1.0e-12,
) -> float:
    exact = _exact_sign_flip_p(profile, observed_align, eps)
    if exact is not None:
        return float(exact)
    values = _as_float_vector(profile)
    generator = torch.Generator().manual_seed(int(seed))
    abs_sum = torch.sum(torch.abs(values)) + float(eps)
    extreme = 0
    for _ in range(int(permutation_count)):
        signs = torch.randint(0, 2, values.shape, generator=generator, dtype=torch.int64).float()
        signs = signs * 2.0 - 1.0
        stat = float((torch.abs(torch.sum(values * signs)) / abs_sum).item())
        if stat >= observed_align - 1.0e-12:
            extreme += 1
    return (extreme + 1.0) / (int(permutation_count) + 1.0)


def group_label_p_value(
    profile: torch.Tensor,
    groups: Sequence[str],
    observed_f: float,
    permutation_count: int,
    seed: int,
) -> float:
    values = list(groups)
    if len(set(values)) <= 1:
        return 1.0
    generator = torch.Generator().manual_seed(int(seed))
    extreme = 0
    for _ in range(int(permutation_count)):
        order = torch.randperm(len(values), generator=generator).tolist()
        shuffled = [values[idx] for idx in order]
        stat = direction_statistics(profile, shuffled).f_stat
        if stat >= observed_f - 1.0e-12:
            extreme += 1
    return (extreme + 1.0) / (int(permutation_count) + 1.0)


def significant_response_groups(
    profile: torch.Tensor,
    groups: Sequence[str],
    alpha: float = 0.05,
    permutation_count: int = 1000,
    seed: int = 42,
) -> list[str]:
    """4.4节: task-group virtual-split significance test.

    Selects the "significant response group" set T_sig = {t : |mean(π_t)| >
    Q_{1-α}(|mean(π_t)|_null)}, where the null distribution for each group comes
    from permuting the group-label assignment while keeping the profile fixed.
    """
    values = _as_float_vector(profile)
    grouped = _group_indices(groups)
    if len(grouped) <= 1:
        return list(grouped.keys())
    observed = {
        group: float(torch.abs(torch.mean(values.index_select(0, indices))).item())
        for group, indices in grouped.items()
    }
    generator = torch.Generator().manual_seed(int(seed))
    extreme_counts = {group: 0 for group in grouped}
    for _ in range(int(permutation_count)):
        order = torch.randperm(int(values.numel()), generator=generator)
        shuffled = values.index_select(0, order)
        for group, indices in grouped.items():
            stat = float(torch.abs(torch.mean(shuffled.index_select(0, indices))).item())
            if stat >= observed[group] - 1.0e-12:
                extreme_counts[group] += 1
    significant = []
    for group in grouped:
        p_value = (extreme_counts[group] + 1.0) / (int(permutation_count) + 1.0)
        if p_value < float(alpha):
            significant.append(group)
    return significant


def bh_fdr(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    indexed = sorted(enumerate(float(value) for value in p_values), key=lambda item: item[1])
    m = len(indexed)
    selected = -1
    for rank, (_idx, p_value) in enumerate(indexed, start=1):
        if p_value <= float(alpha) * rank / max(m, 1):
            selected = rank
    mask = [False for _ in indexed]
    if selected >= 0:
        threshold = indexed[selected - 1][1]
        for idx, p_value in enumerate(float(value) for value in p_values):
            mask[idx] = p_value <= threshold
    return mask


def classify_profile_matrix(
    profiles: torch.Tensor,
    groups: Sequence[str],
    module_types: Sequence[str],
    alpha: float = 0.05,
    permutation_count: int = 1000,
    seed: int = 42,
    val_mask: Sequence[bool] | None = None,
) -> list[TaxonomyRow]:
    """3.2.3节. `val_mask`, when supplied (pseudo-group fit/val split), restricts the
    F-statistic/permutation test to the val subset only, so the same samples used to
    fit the pseudo-groups (build_pseudo_groups) are never also used to significance-test
    them ("为避免同一批样本同时用于聚类和检验"). The align-test (`p_align`) always uses the
    full sample set -- it doesn't depend on group assignment, so there's no leak risk.
    """
    if profiles.ndim != 2:
        raise ValueError("profiles must have shape [samples, atoms]")
    val_indices = [i for i, flag in enumerate(val_mask) if flag] if val_mask is not None else None
    rows: list[dict[str, object]] = []
    for idx in range(int(profiles.shape[1])):
        profile = profiles[:, idx]
        align_stats = direction_statistics(profile, groups)
        if val_indices is not None:
            index_tensor = torch.tensor(val_indices, dtype=torch.long)
            profile_f = profile.index_select(0, index_tensor)
            groups_f = [groups[i] for i in val_indices]
        else:
            profile_f, groups_f = profile, groups
        f_stat = direction_statistics(profile_f, groups_f).f_stat
        rows.append(
            {
                "index": idx,
                "module_type": str(module_types[idx]),
                "stats": DirectionStats(align=align_stats.align, f_stat=f_stat),
                "p_align": sign_flip_p_value(profile, align_stats.align, permutation_count, seed + idx),
                "p_f": group_label_p_value(profile_f, groups_f, f_stat, permutation_count, seed + 1009 + idx),
            }
        )
    by_type: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_type.setdefault(str(row["module_type"]), []).append(idx)
    fdr_mask_align = [False for _ in rows]
    fdr_mask_f = [False for _ in rows]
    for indices in by_type.values():
        local_align = bh_fdr([float(rows[idx]["p_align"]) for idx in indices], alpha=alpha)
        local_f = bh_fdr([float(rows[idx]["p_f"]) for idx in indices], alpha=alpha)
        for idx, value in zip(indices, local_align):
            fdr_mask_align[idx] = value
        for idx, value in zip(indices, local_f):
            fdr_mask_f[idx] = value
    output: list[TaxonomyRow] = []
    for idx, row in enumerate(rows):
        stats = row["stats"]
        assert isinstance(stats, DirectionStats)
        p_align = float(row["p_align"])
        p_f = float(row["p_f"])
        if fdr_mask_align[idx]:
            label = "consensus"
        elif fdr_mask_f[idx]:
            label = "task_specific"
        else:
            label = "noise"
        output.append(
            TaxonomyRow(
                index=int(row["index"]),
                module_type=str(row["module_type"]),
                align=stats.align,
                f_stat=stats.f_stat,
                p_align=p_align,
                p_f=p_f,
                fdr_pass_align=fdr_mask_align[idx],
                fdr_pass_f=fdr_mask_f[idx],
                label=label,
            )
        )
    return output
