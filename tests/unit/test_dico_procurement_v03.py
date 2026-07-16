from dico.candidates import PhysicalCandidate
from dico.procurement import procure_budget_window


def test_procurement_uses_certified_then_reserve_then_balanced_fill_to_budget_window():
    dims = {
        "cheap": {"in_dim": 2, "out_dim": 2},
        "expensive": {"in_dim": 8, "out_dim": 8},
    }
    certified = [
        PhysicalCandidate(
            physical_direction_id="cheap/0",
            module_name="cheap",
            atom_index=0,
            virtual_candidate_ids=["cheap/0"],
            merged_utility=8.0,
            cost=4,
        )
    ]
    reserve = [
        PhysicalCandidate(
            physical_direction_id="expensive/0",
            module_name="expensive",
            atom_index=0,
            virtual_candidate_ids=["expensive/0"],
            merged_utility=0.0,
            cost=16,
            raw_energy=32.0,
        )
    ]

    result = procure_budget_window(
        certified,
        reserve,
        dims,
        target_budget=20,
        eta=0.98,
        r_min=0,
        r_max=8,
        beta=1.0,
    )

    assert 0.98 * 20 <= result.realized_params <= 20
    assert result.rank_dict == {"cheap": 1, "expensive": 1}
    assert result.reserve_filled_ratio > 0.0
    assert result.balanced_fill_ratio == 0.0


def test_procurement_balanced_fills_only_after_reserve_is_exhausted():
    dims = {"m": {"in_dim": 5, "out_dim": 5}}

    result = procure_budget_window(
        certified=[],
        reserve=[],
        module_dims=dims,
        target_budget=30,
        eta=0.98,
        r_min=0,
        r_max=3,
    )

    assert result.rank_dict == {"m": 3}
    assert result.reserve_filled_ratio == 0.0
    assert result.balanced_fill_ratio == 1.0


def test_infeasible_r_min_baseline_reports_warning_instead_of_silent_misallocation():
    # 4.6.1节: B_base = sum(r_min*c_m) must be checked against target_budget before
    # any purchasing begins. Here r_min=2 with cost=4/rank means B_base=8, which
    # already exceeds a target_budget of 5 -- this must surface as an explicit
    # warning, not a silently-produced allocation.
    dims = {"m1": {"in_dim": 2, "out_dim": 2}}

    result = procure_budget_window(
        certified=[],
        reserve=[],
        module_dims=dims,
        target_budget=5,
        eta=0.98,
        r_min=2,
        r_max=32,
    )

    assert result.rank_dict == {"m1": 2}
    assert result.warning is not None
    assert "B_base" in result.warning


def test_feasible_allocation_has_no_infeasibility_warning():
    dims = {"m1": {"in_dim": 2, "out_dim": 2}}

    result = procure_budget_window(
        certified=[],
        reserve=[],
        module_dims=dims,
        target_budget=100,
        eta=0.98,
        r_min=2,
        r_max=32,
    )

    assert result.warning is None


def test_rank_bounds_r_min_2_r_max_32_are_enforced_for_every_module():
    # Appendix B checklist item 5: with r_min=2, r_max=32, every module's final
    # rank must fall within [2, 32].
    dims = {
        "cheap": {"in_dim": 2, "out_dim": 2},
        "expensive": {"in_dim": 8, "out_dim": 8},
    }
    certified = [
        PhysicalCandidate(
            physical_direction_id="cheap/0",
            module_name="cheap",
            atom_index=0,
            virtual_candidate_ids=["cheap/0"],
            merged_utility=8.0,
            cost=4,
        )
    ]
    reserve = [
        PhysicalCandidate(
            physical_direction_id="expensive/0",
            module_name="expensive",
            atom_index=0,
            virtual_candidate_ids=["expensive/0"],
            merged_utility=0.0,
            cost=16,
            raw_energy=32.0,
        )
    ]

    result = procure_budget_window(
        certified,
        reserve,
        dims,
        target_budget=1000,
        eta=0.98,
        r_min=2,
        r_max=32,
        beta=0.5,
    )

    assert all(2 <= rank <= 32 for rank in result.rank_dict.values())


def test_quota_aware_bid_is_non_increasing_as_a_module_buys_more_rank():
    # Appendix B checklist item 6: if a module keeps buying rank, its subsequent
    # bid must be monotonically non-increasing. Five identical-utility candidates
    # in one module means the only thing that changes bid-to-bid is the growing
    # quota-pressure term Psi_m(r_m).
    dims = {"m1": {"in_dim": 2, "out_dim": 2}}
    certified = [
        PhysicalCandidate(
            physical_direction_id=f"m1/{i}",
            module_name="m1",
            atom_index=i,
            virtual_candidate_ids=[f"m1/{i}"],
            merged_utility=5.0,
            cost=4,
        )
        for i in range(5)
    ]

    result = procure_budget_window(certified, [], dims, target_budget=1000, eta=0.98, r_min=0, r_max=10, beta=0.5)

    bids = [row["bid"] for row in result.trace if row["source"] == "certified"]
    assert len(bids) == 5
    assert all(bids[i] >= bids[i + 1] for i in range(len(bids) - 1))
    assert bids[0] > bids[-1]


def test_soft_quota_is_pressure_not_a_hard_cap():
    # A high-demand module ("hot", 3 certified candidates) is allowed to exceed
    # its own soft quota r_bar_m -- proving quota is not clipped -- while a
    # lower-demand module ("cold", 1 candidate) still gets purchased rather than
    # being starved outright by hot's larger appetite.
    dims = {"hot": {"in_dim": 2, "out_dim": 2}, "cold": {"in_dim": 2, "out_dim": 2}}
    certified = [
        PhysicalCandidate(
            physical_direction_id=f"hot/{i}",
            module_name="hot",
            atom_index=i,
            virtual_candidate_ids=[f"hot/{i}"],
            merged_utility=5.0,
            cost=4,
        )
        for i in range(3)
    ] + [
        PhysicalCandidate(
            physical_direction_id="cold/0",
            module_name="cold",
            atom_index=0,
            virtual_candidate_ids=["cold/0"],
            merged_utility=5.0,
            cost=4,
        )
    ]

    result = procure_budget_window(certified, [], dims, target_budget=16, eta=0.98, r_min=0, r_max=10, beta=0.5)

    assert result.rank_dict == {"hot": 3, "cold": 1}
    assert result.rank_dict["hot"] > result.module_quota["hot"]
    assert result.rank_dict["cold"] >= 1


def test_balanced_fill_uses_quota_ratio_not_cheapest_absolute_cost():
    # Both modules cost the same per rank, so the old "cheapest module first"
    # relaxation rule would tie-break alphabetically and always pick "alpha".
    # The quota-aware balanced fill instead picks whichever module sits
    # furthest below its own soft quota (argmin r_m/r_bar_m) -- here that is
    # "zulu" (1 certified candidate, small quota, already at its own ceiling),
    # not "alpha" (3 candidates, much larger quota, already ahead of it).
    dims = {"alpha": {"in_dim": 2, "out_dim": 2}, "zulu": {"in_dim": 2, "out_dim": 2}}
    certified = [
        PhysicalCandidate(
            physical_direction_id=f"alpha/{i}",
            module_name="alpha",
            atom_index=i,
            virtual_candidate_ids=[f"alpha/{i}"],
            merged_utility=5.0,
            cost=4,
        )
        for i in range(3)
    ] + [
        PhysicalCandidate(
            physical_direction_id="zulu/0",
            module_name="zulu",
            atom_index=0,
            virtual_candidate_ids=["zulu/0"],
            merged_utility=5.0,
            cost=4,
        )
    ]

    result = procure_budget_window(certified, [], dims, target_budget=20, eta=0.98, r_min=0, r_max=10, beta=0.5)

    balanced_fill_rows = [row for row in result.trace if row["source"] == "balanced_fill"]
    assert len(balanced_fill_rows) == 1
    assert balanced_fill_rows[0]["module"] == "zulu"


def test_budget_gap_ratio_reports_shortfall_when_r_max_caps_every_module():
    # 4.6.6节: when certified+reserve+balanced fill still cannot reach the
    # [eta*B*, B*] window (here every module is capped at r_max=3 well below
    # what the budget could otherwise support), the shortfall must be reported
    # via budget_gap_ratio rather than silently returning an under-filled
    # allocation with no signal.
    dims = {"m1": {"in_dim": 2, "out_dim": 2}}

    result = procure_budget_window(
        certified=[], reserve=[], module_dims=dims, target_budget=100, eta=0.98, r_min=0, r_max=3, beta=0.5
    )

    assert result.rank_dict == {"m1": 3}
    assert result.realized_params == 12
    assert result.budget_gap_ratio > 0.0


def test_budget_gap_ratio_is_zero_when_window_is_reached():
    dims = {"m1": {"in_dim": 1, "out_dim": 1}}

    result = procure_budget_window(
        certified=[], reserve=[], module_dims=dims, target_budget=30, eta=0.98, r_min=0, r_max=30, beta=0.5
    )

    assert result.realized_params >= 0.98 * 30
    assert result.budget_gap_ratio == 0.0


def test_reserve_queue_disabled_skips_reserve_stage_entirely():
    dims = {"cheap": {"in_dim": 2, "out_dim": 2}, "expensive": {"in_dim": 8, "out_dim": 8}}
    reserve = [
        PhysicalCandidate(
            physical_direction_id="expensive/0",
            module_name="expensive",
            atom_index=0,
            virtual_candidate_ids=["expensive/0"],
            merged_utility=0.0,
            cost=16,
            raw_energy=32.0,
        )
    ]

    result = procure_budget_window(
        certified=[],
        reserve=reserve,
        module_dims=dims,
        target_budget=20,
        eta=0.98,
        r_min=0,
        r_max=8,
        beta=0.5,
        reserve_queue_enabled=False,
    )

    assert result.reserve_filled_ratio == 0.0
    assert not any(row["source"] == "reserve" for row in result.trace)
    # since reserve is skipped, balanced fill must pick up the slack instead.
    assert result.balanced_fill_ratio > 0.0


def test_balanced_fill_disabled_leaves_budget_window_unfilled():
    dims = {"m1": {"in_dim": 2, "out_dim": 2}}

    result = procure_budget_window(
        certified=[],
        reserve=[],
        module_dims=dims,
        target_budget=100,
        eta=0.98,
        r_min=0,
        r_max=8,
        beta=0.5,
        balanced_fill_enabled=False,
    )

    assert result.rank_dict == {"m1": 0}
    assert result.balanced_fill_ratio == 0.0
    assert not any(row["source"] == "balanced_fill" for row in result.trace)


def test_r_min_phase_consumes_highest_utility_certified_direction_first():
    # finding #7: r_min slots must be tied to actual directions, highest
    # normalized-utility first, not bare unattributed rank increments.
    dims = {"m1": {"in_dim": 2, "out_dim": 2}}
    certified = [
        PhysicalCandidate(
            physical_direction_id="m1/low", module_name="m1", atom_index=0,
            virtual_candidate_ids=["m1/low"], merged_utility=1.0, cost=4,
        ),
        PhysicalCandidate(
            physical_direction_id="m1/high", module_name="m1", atom_index=1,
            virtual_candidate_ids=["m1/high"], merged_utility=50.0, cost=4,
        ),
    ]

    result = procure_budget_window(certified, [], dims, target_budget=8, eta=0.98, r_min=1, r_max=8, beta=0.5)

    assert result.purchased_directions["m1"][:1] == ["m1/high"]


def test_r_min_phase_falls_back_to_reserve_when_certified_insufficient():
    dims = {"m1": {"in_dim": 2, "out_dim": 2}}
    certified = [
        PhysicalCandidate(
            physical_direction_id="m1/only", module_name="m1", atom_index=0,
            virtual_candidate_ids=["m1/only"], merged_utility=5.0, cost=4,
        ),
    ]
    reserve = [
        PhysicalCandidate(
            physical_direction_id="m1/reserve", module_name="m1", atom_index=1,
            virtual_candidate_ids=["m1/reserve"], merged_utility=0.0, cost=4, raw_energy=1.0,
        ),
    ]

    result = procure_budget_window(
        certified, reserve, dims, target_budget=16, eta=0.98, r_min=2, r_max=8, beta=0.5,
    )

    assert set(result.purchased_directions["m1"][:2]) == {"m1/only", "m1/reserve"}


def test_r_min_phase_logs_warning_when_no_directions_available_to_consume(caplog):
    import logging

    dims = {"m1": {"in_dim": 2, "out_dim": 2}}

    with caplog.at_level(logging.WARNING, logger="dico.procurement"):
        result = procure_budget_window(
            certified=[], reserve=[], module_dims=dims, target_budget=16, eta=0.98, r_min=2, r_max=8, beta=0.5,
        )

    assert result.rank_dict["m1"] >= 2
    # No candidates exist anywhere (certified/reserve both empty), so nothing gets
    # attributed to a direction even after balanced_fill continues past r_min.
    assert result.purchased_directions["m1"] == []
    assert "r_min_direction_shortfall" in caplog.text


def test_balanced_fill_consumes_a_leftover_direction_when_one_exists():
    # finding #7: balanced_fill must tie its rank slot to a real leftover direction
    # (certified or reserve) instead of a bare increment, when one is available.
    # reserve_queue_enabled=False skips the *dedicated* reserve auction phase, but
    # balanced_fill's own direction lookup searches both certified and reserve pools
    # regardless -- so with no certified candidates at all, every rank balanced_fill
    # grants here must come from a reserve direction.
    dims = {"m1": {"in_dim": 2, "out_dim": 2}}
    reserve = [
        PhysicalCandidate(
            physical_direction_id=f"m1/{i}", module_name="m1", atom_index=i,
            virtual_candidate_ids=[f"m1/{i}"], merged_utility=0.0, cost=4, raw_energy=float(i + 1),
        )
        for i in range(4)
    ]

    result = procure_budget_window(
        [], reserve, dims, target_budget=16, eta=0.98, r_min=0, r_max=4, beta=0.5,
        reserve_queue_enabled=False,
    )

    balanced_rows = [row for row in result.trace if row["source"] == "balanced_fill"]
    assert balanced_rows
    assert any(row["physical_direction"] is not None for row in balanced_rows)
    assert len(result.purchased_directions["m1"]) == result.rank_dict["m1"]


def test_reserve_pool_does_not_perturb_certified_normalized_utility():
    # finding #8: reserve-queue normalization must reuse the certified-only
    # (median, mad), not re-estimate over a certified+reserve merged pool -- an
    # extreme reserve outlier must not change certified directions' w_bar at all.
    dims = {"m1": {"in_dim": 2, "out_dim": 2}, "m2": {"in_dim": 2, "out_dim": 2}}
    certified = [
        PhysicalCandidate(
            physical_direction_id=f"m1/{i}", module_name="m1", atom_index=i,
            virtual_candidate_ids=[f"m1/{i}"], merged_utility=float(i + 1), cost=4,
        )
        for i in range(3)
    ]

    result_no_reserve = procure_budget_window(
        certified, [], dims, target_budget=100, eta=0.98, r_min=0, r_max=8, beta=0.5,
    )

    extreme_reserve = [
        PhysicalCandidate(
            physical_direction_id="m2/outlier", module_name="m2", atom_index=0,
            virtual_candidate_ids=["m2/outlier"], merged_utility=0.0, cost=4, raw_energy=1.0e9,
        ),
    ]
    result_with_reserve = procure_budget_window(
        certified, extreme_reserve, dims, target_budget=100, eta=0.98, r_min=0, r_max=8, beta=0.5,
    )

    for candidate in certified:
        pid = candidate.physical_direction_id
        assert result_no_reserve.normalized_utility[pid] == result_with_reserve.normalized_utility[pid]


def test_purchased_directions_never_exceeds_rank_dict():
    dims = {
        "m1": {"in_dim": 2, "out_dim": 2},
        "m2": {"in_dim": 4, "out_dim": 4},
    }
    certified = [
        PhysicalCandidate(
            physical_direction_id=f"m1/{i}", module_name="m1", atom_index=i,
            virtual_candidate_ids=[f"m1/{i}"], merged_utility=float(i + 1), cost=4,
        )
        for i in range(3)
    ] + [
        PhysicalCandidate(
            physical_direction_id=f"m2/{i}", module_name="m2", atom_index=i,
            virtual_candidate_ids=[f"m2/{i}"], merged_utility=float(10 - i), cost=8,
        )
        for i in range(2)
    ]
    reserve = [
        PhysicalCandidate(
            physical_direction_id="m1/reserve", module_name="m1", atom_index=9,
            virtual_candidate_ids=["m1/reserve"], merged_utility=0.0, cost=4, raw_energy=0.5,
        ),
    ]

    for target_budget, r_min, r_max in [(0, 0, 4), (8, 1, 4), (100, 2, 6), (12, 3, 3)]:
        result = procure_budget_window(
            certified, reserve, dims, target_budget=target_budget, eta=0.98, r_min=r_min, r_max=r_max, beta=0.5,
        )
        for name in dims:
            assert len(result.purchased_directions.get(name, [])) <= result.rank_dict[name]
