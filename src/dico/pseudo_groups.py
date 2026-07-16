from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _direction_demand_distribution(profiles: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Map each sample to a distribution over direction demand, per 3.2.3节 p_i(a) = |π|^2/(Σ|π'|^2+ε)."""
    magnitude = profiles.detach().float() ** 2
    denom = magnitude.sum(dim=1, keepdim=True) + eps
    return magnitude / denom


@dataclass(frozen=True)
class PseudoGroupResult:
    groups: list[str]
    val_mask: list[bool] = field(default_factory=list)
    k_selected: int = 1
    fit_sample_count: int = 0
    val_sample_count: int = 0


def build_pseudo_groups(
    profiles: torch.Tensor,
    k_range: range = range(2, 6),
    seed: int = 42,
    val_fraction: float = 0.5,
) -> PseudoGroupResult:
    """Construct pseudo task-group labels from signed-profile geometry when the
    calibration set carries no real task-group labels (3.2.3节 "无标签场景下的伪组构造").

    Samples are represented as sqrt-compressed direction-demand distributions
    sqrt(p_i(a)), clustered with k-means; the cluster count is chosen by the
    highest silhouette score over ``k_range``. To avoid using the same samples
    both to fit the clustering and to significance-test the resulting groups
    (3.2.3节: "为避免同一批样本同时用于聚类和检验"), the sample set is split into a
    fit half (clustering only) and a val half (returned via ``val_mask`` for
    callers to restrict the F-statistic/permutation test to); coverage
    accounting elsewhere in the pipeline may still use all samples, per the
    doc's own note.
    """
    num_samples = int(profiles.shape[0])
    if num_samples < 2 * min(k_range):
        return PseudoGroupResult(
            groups=["pseudo_0"] * num_samples,
            val_mask=[True] * num_samples,
            k_selected=1,
            fit_sample_count=0,
            val_sample_count=num_samples,
        )

    distributions = _direction_demand_distribution(profiles)
    sqrt_features = torch.sqrt(distributions.clamp_min(0.0)).numpy()

    rng = np.random.RandomState(seed)
    perm = rng.permutation(num_samples)
    min_fit_needed = 2 * min(k_range)
    val_count = max(1, int(round(num_samples * val_fraction)))
    fit_count = max(num_samples - val_count, min_fit_needed)
    fit_count = min(fit_count, num_samples)
    fit_idx, val_idx = perm[:fit_count], perm[fit_count:]

    best_model = None
    best_k = None
    best_score = -1.0
    for k in k_range:
        if k < 2 or k >= fit_count:
            continue
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        fit_labels = model.fit_predict(sqrt_features[fit_idx])
        if len(set(fit_labels.tolist())) < 2:
            continue
        score = silhouette_score(sqrt_features[fit_idx], fit_labels)
        if score > best_score:
            best_model, best_k, best_score = model, k, score

    if best_model is None:
        return PseudoGroupResult(
            groups=["pseudo_0"] * num_samples,
            val_mask=[True] * num_samples,
            k_selected=1,
            fit_sample_count=fit_count,
            val_sample_count=num_samples - fit_count,
        )

    all_labels = best_model.predict(sqrt_features)
    val_mask = [False] * num_samples
    for idx in val_idx.tolist():
        val_mask[idx] = True
    return PseudoGroupResult(
        groups=[f"pseudo_{int(label)}" for label in all_labels],
        val_mask=val_mask,
        k_selected=int(best_k),
        fit_sample_count=fit_count,
        val_sample_count=len(val_idx),
    )
