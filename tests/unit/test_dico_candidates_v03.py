import torch

from dico.candidates import (
    DirectionAtom,
    create_virtual_candidates,
    merge_physical_candidates,
)


def test_task_specific_atom_splits_into_virtual_candidates_sharing_physical_id():
    atom = DirectionAtom(
        module_name="layers.0.q_proj",
        atom_index=3,
        profile=torch.tensor([2.0, -1.0, 0.0, -3.0]),
        classification="task_specific",
        utility=1.0,
        cost=16,
    )

    candidates = create_virtual_candidates([atom], split_mode="sign")

    assert [candidate.split_type for candidate in candidates] == ["positive", "negative"]
    assert candidates[0].physical_direction_id == candidates[1].physical_direction_id
    assert torch.equal(candidates[0].profile, torch.tensor([2.0, 0.0, 0.0, 0.0]))
    assert torch.equal(candidates[1].profile, torch.tensor([0.0, 1.0, 0.0, 3.0]))


def test_sign_split_shares_u_v_tilde_and_carries_pristine_initial_profile():
    # 3.3.3節: kappa(q',q*) is a per-direction-unit quantity, so every split of the
    # same atom must share that atom's u/v_tilde. initial_profile must be a pristine
    # copy of the split's own profile at construction time (independent object, so
    # later in-place mutation of .profile by coverage.py's greedy loop can't affect it).
    u = torch.tensor([0.6, 0.8])
    v_tilde = torch.tensor([1.0, 0.0, 0.0])
    atom = DirectionAtom(
        module_name="layers.0.q_proj",
        atom_index=3,
        profile=torch.tensor([2.0, -1.0, 0.0, -3.0]),
        classification="task_specific",
        utility=1.0,
        cost=16,
        u=u,
        v_tilde=v_tilde,
    )

    candidates = create_virtual_candidates([atom], split_mode="sign")

    assert len(candidates) == 2
    for candidate in candidates:
        assert torch.equal(candidate.u, u)
        assert torch.equal(candidate.v_tilde, v_tilde)
        assert torch.equal(candidate.initial_profile, candidate.profile)
        assert candidate.initial_profile.data_ptr() != candidate.profile.data_ptr()

    # Mutating one candidate's .profile in place (what coverage.py's greedy loop does)
    # must not affect its initial_profile.
    candidates[0].profile += 100.0
    assert not torch.equal(candidates[0].initial_profile, candidates[0].profile)


def test_group_split_only_generates_candidates_for_significant_response_groups():
    torch.manual_seed(0)
    n = 6
    labels = ["A"] * n + ["B"] * n + ["C"] * n
    # Group A has a strong, consistent response; B and C are noise around zero.
    profile = torch.cat(
        [
            torch.full((n,), 5.0) + torch.randn(n) * 0.1,
            torch.randn(n) * 0.1,
            torch.randn(n) * 0.1,
        ]
    )
    atom = DirectionAtom(
        module_name="layers.0.q_proj",
        atom_index=0,
        profile=profile,
        classification="task_specific",
        utility=1.0,
        cost=16,
    )

    candidates = create_virtual_candidates(
        [atom],
        split_mode="group",
        group_labels=labels,
        significance_alpha=0.05,
        permutation_count=500,
        seed=1,
    )

    assert [candidate.split_type for candidate in candidates] == ["group:A"]
    assert candidates[0].physical_direction_id == atom.physical_direction_id


def test_group_split_generates_no_candidates_when_no_group_is_significant():
    torch.manual_seed(0)
    n = 6
    labels = ["A"] * n + ["B"] * n
    profile = torch.randn(2 * n) * 0.1  # pure noise, no group stands out

    atom = DirectionAtom(
        module_name="layers.0.q_proj",
        atom_index=0,
        profile=profile,
        classification="task_specific",
        utility=1.0,
        cost=16,
    )

    candidates = create_virtual_candidates(
        [atom],
        split_mode="group",
        group_labels=labels,
        significance_alpha=0.05,
        permutation_count=500,
        seed=1,
    )

    assert candidates == []


def test_physical_merge_counts_shared_split_cost_once_regardless_of_utility_source():
    # 4.4.4节: merge_physical_candidates is purely structural now -- cost/virtual-id
    # grouping only. The purchase utility always comes from the caller-supplied
    # mapping (compute_physical_joint_utility in the live pipeline), never a
    # naive sum of certified_gain computed inside this function.
    atom = DirectionAtom(
        module_name="layers.0.q_proj",
        atom_index=3,
        profile=torch.tensor([2.0, -1.0]),
        classification="task_specific",
        utility=1.0,
        cost=16,
    )
    candidates = create_virtual_candidates([atom], split_mode="sign")

    merged = merge_physical_candidates(candidates, utility_by_physical_id={atom.physical_direction_id: 0.42})

    assert len(merged) == 1
    assert merged[0].cost == 16
    assert merged[0].merged_utility == 0.42
    assert len(merged[0].virtual_candidate_ids) == 2
