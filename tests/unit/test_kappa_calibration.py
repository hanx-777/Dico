import torch

from dico.candidates import VirtualCandidate
from dico.kappa_calibration import kappa_calibration_diagnostic


def _candidate(candidate_id: str, module_name: str, u: torch.Tensor, v_tilde: torch.Tensor) -> VirtualCandidate:
    return VirtualCandidate(
        virtual_candidate_id=candidate_id,
        physical_direction_id=candidate_id,
        module_name=module_name,
        atom_index=0,
        profile=torch.zeros(4),
        split_type="positive",
        cost=4,
        u=u,
        v_tilde=v_tilde,
    )


def test_strongly_correlated_cross_layer_directions_do_not_fall_back():
    # All 6 candidates share the exact same (u, v_tilde) direction across 6 different
    # layers -- a strong, real cross-layer kappa signal that must be distinguishable
    # from random noise, so window_h stays at its configured value (no h=0 fallback).
    shared = torch.tensor([1.0, 0.0, 0.0])
    candidates = [_candidate(f"c{i}", f"layers.{i}.q_proj", shared, shared) for i in range(6)]

    result = kappa_calibration_diagnostic(candidates, "q_proj", seed=1)

    assert result.fallback_h0 is False
    assert result.observed_mean_abs_kappa > result.null_mean_abs_kappa
    assert result.ks_pvalue < 0.1


def test_independent_random_directions_fall_back_to_h0():
    torch.manual_seed(0)
    candidates = [
        _candidate(
            f"i{i}",
            f"layers.{i}.q_proj",
            torch.nn.functional.normalize(torch.randn(8), dim=0),
            torch.nn.functional.normalize(torch.randn(8), dim=0),
        )
        for i in range(10)
    ]

    result = kappa_calibration_diagnostic(candidates, "q_proj", seed=1)

    assert result.fallback_h0 is True


def test_too_few_cross_layer_pairs_falls_back_to_h0():
    shared = torch.tensor([1.0, 0.0])
    # Only one candidate -> zero cross-layer pairs at all.
    candidates = [_candidate("solo", "layers.0.q_proj", shared, shared)]

    result = kappa_calibration_diagnostic(candidates, "q_proj", seed=1)

    assert result.fallback_h0 is True
    assert result.num_pairs == 0


def test_same_layer_pairs_are_excluded_from_the_comparison():
    shared = torch.tensor([1.0, 0.0])
    # Two candidates in the SAME layer (different atom indices) must not count as a
    # cross-layer pair -- this diagnostic is specifically about cross-layer kappa.
    candidates = [
        _candidate("a", "layers.0.q_proj", shared, shared),
        _candidate("b", "layers.0.q_proj", shared, shared),
    ]

    result = kappa_calibration_diagnostic(candidates, "q_proj", seed=1)

    assert result.num_pairs == 0
    assert result.fallback_h0 is True


def test_deduplicates_by_physical_direction_id():
    # Two "candidates" sharing the same physical_direction_id (as if they were the
    # sign-splits of one atom) must count as a single unit, not create a self-pair.
    shared = torch.tensor([1.0, 0.0])
    other = torch.tensor([0.0, 1.0])
    same_unit_a = VirtualCandidate(
        virtual_candidate_id="p/positive", physical_direction_id="p", module_name="layers.0.q_proj",
        atom_index=0, profile=torch.zeros(4), split_type="positive", cost=4, u=shared, v_tilde=shared,
    )
    same_unit_b = VirtualCandidate(
        virtual_candidate_id="p/negative", physical_direction_id="p", module_name="layers.0.q_proj",
        atom_index=0, profile=torch.zeros(4), split_type="negative", cost=4, u=shared, v_tilde=shared,
    )
    other_unit = _candidate("q", "layers.5.q_proj", other, other)

    result = kappa_calibration_diagnostic([same_unit_a, same_unit_b, other_unit], "q_proj", seed=1)

    # Only one cross-layer pair should exist: {p, q} -- not {p/positive, q} and
    # {p/negative, q} separately.
    assert result.num_pairs == 1


def test_determinism_same_seed_same_result():
    torch.manual_seed(0)
    candidates = [
        _candidate(
            f"i{i}",
            f"layers.{i}.q_proj",
            torch.nn.functional.normalize(torch.randn(8), dim=0),
            torch.nn.functional.normalize(torch.randn(8), dim=0),
        )
        for i in range(10)
    ]

    result_a = kappa_calibration_diagnostic(candidates, "q_proj", seed=7)
    result_b = kappa_calibration_diagnostic(candidates, "q_proj", seed=7)

    assert result_a == result_b
