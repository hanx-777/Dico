import torch

from dico.candidates import VirtualCandidate
from dico.coverage import compute_group_coverage, nsw_objective
from dico.physical import compute_physical_joint_utility


def _candidate(vcid: str, physical_id: str, module: str, profile: list[float]) -> VirtualCandidate:
    return VirtualCandidate(
        virtual_candidate_id=vcid,
        physical_direction_id=physical_id,
        module_name=module,
        atom_index=0,
        profile=torch.tensor(profile),
        split_type="x",
        cost=16,
    )


def test_single_candidate_physical_direction_reduces_to_plain_marginal_gain():
    # 4.4.4节 (doc line 351): when Q_p has only one virtual candidate, w_p^joint
    # must reduce to the ordinary marginal gain F(S_-p U {q}) - F(S_-p).
    groups = ["A", "A", "A", "B", "B", "B"]
    solo = _candidate("m2/atom_0/x", "m2/atom_0", "m2", [0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    other = _candidate("m1/atom_0/x", "m1/atom_0", "m1", [2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    certified = [other, solo]

    joint = compute_physical_joint_utility(certified, groups)

    s_minus = [other]
    f_base = nsw_objective(compute_group_coverage(s_minus, groups))
    f_full = nsw_objective(compute_group_coverage(s_minus + [solo], groups))
    assert abs(joint["m2/atom_0"] - (f_full - f_base)) < 1.0e-9


def test_physical_rank_counted_once_for_conflicting_sign_split():
    # Appendix B checklist item 2: q+ and q- of the same physical direction can
    # both be certified, but the physical direction only shows up once.
    groups = ["A", "A", "A", "B", "B", "B"]
    q_plus = _candidate("m1/atom_0/positive", "m1/atom_0", "m1", [2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    q_minus = _candidate("m1/atom_0/negative", "m1/atom_0", "m1", [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    q_other = _candidate("m2/atom_0/x", "m2/atom_0", "m2", [0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    certified = [q_plus, q_minus, q_other]

    joint = compute_physical_joint_utility(certified, groups)

    assert set(joint.keys()) == {"m1/atom_0", "m2/atom_0"}


def test_joint_utility_does_not_exceed_naive_sum_of_solo_marginal_gains():
    # Appendix B checklist item 3: joint physical utility must not exceed the
    # unconstrained upper bound of simply summing each virtual candidate's own
    # (solo, against the same baseline) marginal gain. When both pieces of a
    # conflicting sign-split compete for the same group's coverage, the log-based
    # Nash-welfare objective's diminishing returns make this strict, not just <=.
    groups = ["A", "A", "A", "B", "B", "B"]
    q_plus = _candidate("m1/atom_0/positive", "m1/atom_0", "m1", [2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    q_minus = _candidate("m1/atom_0/negative", "m1/atom_0", "m1", [1.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    q_other = _candidate("m2/atom_0/x", "m2/atom_0", "m2", [0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    certified = [q_plus, q_minus, q_other]

    joint = compute_physical_joint_utility(certified, groups)

    s_minus_p1 = [q_other]
    f_base = nsw_objective(compute_group_coverage(s_minus_p1, groups))
    solo_plus = nsw_objective(compute_group_coverage(s_minus_p1 + [q_plus], groups)) - f_base
    solo_minus = nsw_objective(compute_group_coverage(s_minus_p1 + [q_minus], groups)) - f_base
    naive_sum_upper_bound = solo_plus + solo_minus

    assert joint["m1/atom_0"] <= naive_sum_upper_bound + 1.0e-9
    assert joint["m1/atom_0"] < naive_sum_upper_bound - 1.0e-6


def test_disjoint_conflict_split_has_no_double_counting_penalty():
    # When q+ and q- of the same physical direction cover disjoint groups (no
    # real competition for the same coverage), the joint utility equals the
    # naive sum exactly -- the sub-additivity above is specific to genuine
    # overlap/conflict, not an artifact of merely having two virtual candidates.
    groups = ["A", "A", "A", "B", "B", "B"]
    q_plus = _candidate("m1/atom_0/positive", "m1/atom_0", "m1", [2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    q_minus = _candidate("m1/atom_0/negative", "m1/atom_0", "m1", [0.0, 0.0, 0.0, 2.0, 1.0, 0.0])
    q_other = _candidate("m2/atom_0/x", "m2/atom_0", "m2", [0.0, 0.0, 0.0, 0.0, 0.0, 3.0])
    certified = [q_plus, q_minus, q_other]

    joint = compute_physical_joint_utility(certified, groups)

    s_minus_p1 = [q_other]
    f_base = nsw_objective(compute_group_coverage(s_minus_p1, groups))
    solo_plus = nsw_objective(compute_group_coverage(s_minus_p1 + [q_plus], groups)) - f_base
    solo_minus = nsw_objective(compute_group_coverage(s_minus_p1 + [q_minus], groups)) - f_base
    naive_sum = solo_plus + solo_minus

    assert abs(joint["m1/atom_0"] - naive_sum) < 1.0e-9
