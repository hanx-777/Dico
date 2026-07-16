import torch

from dico.taxonomy import bh_fdr, classify_profile_matrix, direction_statistics, sign_flip_p_value


def _double_dip_trap_profile() -> tuple[torch.Tensor, list[str], list[bool]]:
    """3.2.3节: pseudo-group double-dipping guard. Groups A/B alternate across all 20
    samples. The first half (fit, val_mask=False) has a strong group-correlated signal
    (A=+5, B=-5, no noise) -- exactly the kind of artifact a clustering step fit on
    these same samples could latch onto. The second half (val, val_mask=True) is pure
    noise with no group-dependent mean, so it should show no significant group effect.
    Passing val_mask must restrict the F-test to the (insignificant) val half; omitting
    it lets the fit half's fabricated signal dominate and falsely certify task_specific.
    """
    generator = torch.Generator().manual_seed(11)
    groups = ["A" if i % 2 == 0 else "B" for i in range(20)]
    fit_values = torch.tensor([5.0 if groups[i] == "A" else -5.0 for i in range(10)])
    val_values = torch.randn(10, generator=generator) * 0.1
    profile = torch.cat([fit_values, val_values])
    val_mask = [False] * 10 + [True] * 10
    return profile.unsqueeze(1), groups, val_mask


def test_val_mask_prevents_double_dip_false_task_specific_classification():
    profile, groups, val_mask = _double_dip_trap_profile()

    with_val_mask = classify_profile_matrix(
        profile, groups, ["q_proj"], alpha=0.05, permutation_count=999, seed=7, val_mask=val_mask,
    )
    assert with_val_mask[0].label != "task_specific", (
        "F-test restricted to the (noise-only) val half must not certify a group effect "
        "that only exists in the fit half"
    )

    without_val_mask = classify_profile_matrix(
        profile, groups, ["q_proj"], alpha=0.05, permutation_count=999, seed=7, val_mask=None,
    )
    assert without_val_mask[0].label == "task_specific", (
        "sanity check: without val_mask, the fit half's fabricated signal must dominate "
        "the full-sample F-test and get spuriously certified -- proves val_mask is doing "
        "real work above, not just failing to matter either way"
    )


def test_direction_statistics_alignment_and_group_f():
    profile = torch.tensor([3.0, 2.0, -1.0, -2.0])
    groups = ["math", "math", "code", "code"]

    stats = direction_statistics(profile, groups)

    assert 0.0 <= stats.align <= 1.0
    assert stats.align == torch.abs(profile.sum()).item() / torch.abs(profile).sum().item()
    assert stats.f_stat > 1.0


def test_bh_fdr_is_deterministic_with_expected_mask():
    mask = bh_fdr([0.001, 0.02, 0.04, 0.2], alpha=0.05)

    assert mask == [True, True, False, False]


def test_classification_covers_consensus_task_specific_and_noise():
    profiles = torch.tensor(
        [
            [5.0, 3.0, 0.1],
            [4.0, 2.5, -0.1],
            [5.0, 3.2, 0.05],
            [4.0, 2.7, -0.05],
            [5.0, -3.0, 0.1],
            [4.0, -2.5, -0.1],
            [5.0, -3.2, 0.05],
            [4.0, -2.7, -0.05],
        ]
    )
    groups = ["math", "math", "math", "math", "code", "code", "code", "code"]
    module_types = ["q_proj", "v_proj", "q_proj"]

    rows = classify_profile_matrix(
        profiles,
        groups,
        module_types,
        alpha=0.05,
        permutation_count=99,
        seed=7,
    )

    assert [row.label for row in rows] == ["consensus", "task_specific", "noise"]
    assert all(0.0 <= row.p_align <= 1.0 for row in rows)
    assert all(0.0 <= row.p_f <= 1.0 for row in rows)


def _biased_profile(bias: float, n: int = 20, seed: int = 0, noise_std: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=generator) * noise_std + bias


def test_pure_noise_profiles_are_not_significant_under_either_fdr():
    # v0.5 4.4.1: BH-FDR must be applied to both p_align and p_f, per module type.
    # A batch of pure-noise atoms (no real signal) should almost entirely fail
    # both corrected tests and land as "noise", not be spuriously certified.
    columns = [_biased_profile(0.0, seed=2000 + i) for i in range(12)]
    profiles = torch.stack(columns, dim=1)
    groups = ["math"] * 10 + ["code"] * 10
    module_types = ["q_proj"] * 12

    rows = classify_profile_matrix(profiles, groups, module_types, alpha=0.05, permutation_count=999, seed=7)

    assert all(row.label == "noise" for row in rows)
    assert all(not row.fdr_pass_align for row in rows)
    assert all(not row.fdr_pass_f for row in rows)


def test_bh_fdr_on_p_align_rejects_atom_that_would_pass_raw_threshold():
    # Construct one atom with a moderate signal whose RAW p_align (0.032) is
    # below the uncorrected alpha=0.05 threshold, surrounded by 9 pure-noise
    # atoms in the same module type. Per-type BH-FDR correction over these 10
    # atoms must push the effective threshold below 0.032, so this atom must
    # NOT be labeled "consensus" once FDR correction is applied to p_align
    # (doc 4.4.1: BH-FDR runs on both p_align and p_f, not just p_f).
    signal_profile = _biased_profile(0.75, seed=123)
    noise_profiles = [_biased_profile(0.0, seed=1000 + i) for i in range(9)]
    profiles = torch.stack([signal_profile] + noise_profiles, dim=1)
    groups = ["g"] * 20
    module_types = ["q_proj"] * 10

    rows = classify_profile_matrix(profiles, groups, module_types, alpha=0.05, permutation_count=999, seed=7)

    signal_row = rows[0]
    assert signal_row.p_align < 0.05
    assert not signal_row.fdr_pass_align
    assert signal_row.label != "consensus"
