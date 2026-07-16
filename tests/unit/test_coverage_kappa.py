"""Regression coverage for finding #5 (CovRA v0.6.2 audit): w_p^joint must be
computed from each candidate's INITIAL (pre-deduction) profile, not the
coverage-mutated `.profile` a candidate is left with after greedy_group_fair_coverage
certifies it -- a candidate certified late in the greedy sequence carries a heavily
residual-deducted `.profile` (reflecting everything selected before it), while one
certified early carries something close to its raw profile. Using `.profile` directly
would leak "when was I selected" into the utility, which is exactly the "选择顺序" the
doc requires w_p^joint to be independent of (3.4节, right before 3.4.1).

Note: reversing the INPUT list order to greedy_group_fair_coverage is not a useful
probe here -- its selection sequence is a deterministic argmax each round and doesn't
depend on input list order (only exact numeric ties would, and floats essentially never
tie by construction), so a fixed candidate set always produces the same certification
order and the same final `.profile` mutation trail regardless of input ordering. The
real invariant under test is "utility must not depend on the mutation trail," not "on
input list order" -- so these tests directly compare the initial_profile-based (fixed)
computation against a simulated profile-based (pre-fix) one for a single coverage run.
"""
from dataclasses import replace

import torch

from dico.candidates import VirtualCandidate
from dico.coverage import greedy_group_fair_coverage
from dico.physical import compute_physical_joint_utility


def _atom_candidates(physical_id: str, direction_vec: torch.Tensor, profile: list[float], module_name: str):
    """Two sign-split virtual candidates for one physical direction, sharing u/v_tilde
    (kappa is a per-unit quantity) and each carrying its own pristine initial_profile.
    """
    base_profile = torch.tensor(profile, dtype=torch.float32)
    pos_profile = torch.clamp(base_profile, min=0.0)
    neg_profile = torch.clamp(-base_profile, min=0.0)
    candidates = []
    for split_type, split_profile in (("positive", pos_profile), ("negative", neg_profile)):
        candidates.append(
            VirtualCandidate(
                virtual_candidate_id=f"{physical_id}/{split_type}",
                physical_direction_id=physical_id,
                module_name=module_name,
                atom_index=0,
                profile=split_profile.clone(),
                split_type=split_type,
                cost=4,
                u=direction_vec,
                v_tilde=direction_vec,
                initial_profile=split_profile.clone(),
            )
        )
    return candidates


def _build_candidates() -> list[VirtualCandidate]:
    # Deliberately correlated (not mutually orthogonal) across physical directions, so
    # cross-unit kappa deduction actually fires and .profile really does end up
    # mutated away from .initial_profile during greedy selection -- an all-orthogonal
    # fixture would never exercise any deduction, making the comparison below vacuous.
    dir_a = torch.tensor([1.0, 0.0, 0.0])
    dir_b = torch.tensor([0.7, 0.7, 0.0])
    dir_c = torch.tensor([0.0, 0.7, 0.7])
    return [
        *_atom_candidates("p_a", dir_a, [5.0, -5.0, 0.0, 0.0], "layers.0.q_proj"),
        *_atom_candidates("p_b", dir_b, [3.0, -3.0, 2.0, -2.0], "layers.1.q_proj"),
        *_atom_candidates("p_c", dir_c, [4.0, 0.0, -4.0, 1.0], "layers.2.q_proj"),
    ]


def test_certification_actually_mutates_profile_away_from_initial_profile():
    """Sanity precondition for the tests below: if nothing gets deducted in this
    fixture, the comparisons that follow would be vacuously true regardless of
    whether the fix is present."""
    groups = ["math", "math", "code", "code"]
    candidates = _build_candidates()

    result = greedy_group_fair_coverage(candidates, groups, max_selected=len(candidates))

    mutated = [c for c in result.selected if not torch.equal(c.profile, c.initial_profile)]
    assert mutated, "fixture must exercise at least one real residual deduction"


def test_physical_joint_utility_uses_initial_profile_not_mutated_profile():
    groups = ["math", "math", "code", "code"]
    candidates = _build_candidates()

    result = greedy_group_fair_coverage(candidates, groups, max_selected=len(candidates))

    fixed_utility = compute_physical_joint_utility(result.selected, groups)

    # Simulate the pre-fix bug: recompute using each candidate's mutated `.profile`
    # directly (by overwriting initial_profile with the mutated value first), instead
    # of the pristine pre-deduction copy.
    buggy_candidates = [replace(c, initial_profile=c.profile) for c in result.selected]
    buggy_utility = compute_physical_joint_utility(buggy_candidates, groups)

    assert fixed_utility != buggy_utility, (
        "using the coverage-mutated .profile instead of .initial_profile must change "
        "the computed joint utility for this fixture -- otherwise this test doesn't "
        "actually distinguish the fix from the bug it guards against"
    )


def test_physical_joint_utility_ignores_profile_mutation_after_certification():
    """Once coverage.selected is produced, compute_physical_joint_utility's result
    must be stable even if something downstream further mutates `.profile` (it should
    only ever read `.initial_profile`)."""
    groups = ["math", "math", "code", "code"]
    candidates = _build_candidates()

    result = greedy_group_fair_coverage(candidates, groups, max_selected=len(candidates))
    utility_before = compute_physical_joint_utility(result.selected, groups)

    for c in result.selected:
        c.profile = torch.zeros_like(c.profile)  # simulate arbitrary downstream mutation

    utility_after = compute_physical_joint_utility(result.selected, groups)

    assert utility_before == utility_after
