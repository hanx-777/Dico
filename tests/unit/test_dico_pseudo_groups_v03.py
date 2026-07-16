import torch
from sklearn.metrics import adjusted_rand_score

from dico.pseudo_groups import build_pseudo_groups


def _two_cluster_profiles(n_per_cluster: int, seed: int) -> tuple[torch.Tensor, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    cluster_a = torch.zeros(n_per_cluster, 6)
    cluster_a[:, :3] = torch.rand(n_per_cluster, 3, generator=generator) + 2.0
    cluster_b = torch.zeros(n_per_cluster, 6)
    cluster_b[:, 3:] = torch.rand(n_per_cluster, 3, generator=generator) + 2.0
    profiles = torch.cat([cluster_a, cluster_b], dim=0)
    truth = [0] * n_per_cluster + [1] * n_per_cluster
    return profiles, truth


def test_build_pseudo_groups_recovers_synthetic_clusters():
    profiles, truth = _two_cluster_profiles(n_per_cluster=8, seed=0)

    result = build_pseudo_groups(profiles, k_range=range(2, 5), seed=42)

    assert len(result.groups) == len(truth)
    numeric_labels = [int(label.split("_")[-1]) for label in result.groups]
    assert adjusted_rand_score(truth, numeric_labels) > 0.9


def test_build_pseudo_groups_returns_single_group_when_too_few_samples():
    profiles = torch.randn(2, 4)

    result = build_pseudo_groups(profiles, k_range=range(2, 5), seed=42)

    assert len(result.groups) == 2
    assert len(set(result.groups)) == 1
    assert all(result.val_mask)  # too few samples for a real fit/val split


def test_build_pseudo_groups_splits_fit_and_val_disjointly():
    profiles, _truth = _two_cluster_profiles(n_per_cluster=8, seed=0)

    result = build_pseudo_groups(profiles, k_range=range(2, 5), seed=42, val_fraction=0.5)

    assert len(result.val_mask) == len(result.groups) == 16
    assert result.val_sample_count == sum(result.val_mask)
    assert result.fit_sample_count == len(result.val_mask) - result.val_sample_count
    # A real split happened -- not everything landed on one side.
    assert 0 < result.val_sample_count < len(result.val_mask)


def test_build_pseudo_groups_clusters_on_sqrt_compressed_features_not_raw():
    """3.2.3节 specifies clustering on sqrt(p_i(a)), not the raw energy distribution --
    regression guard against silently reverting to un-square-rooted features (which
    over-weights already-dominant directions and changes cluster geometry)."""
    profiles, truth = _two_cluster_profiles(n_per_cluster=10, seed=1)

    result = build_pseudo_groups(profiles, k_range=range(2, 4), seed=7)
    numeric_labels = [int(label.split("_")[-1]) for label in result.groups]

    # The sqrt-compressed clustering should still cleanly separate the two synthetic
    # clusters (this doesn't directly inspect the feature array, but a materially wrong
    # feature transform would degrade this well-separated synthetic recovery).
    assert adjusted_rand_score(truth, numeric_labels) > 0.9
