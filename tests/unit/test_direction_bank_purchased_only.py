import torch

from dico.candidates import PhysicalCandidate
from dico.preallocation import load_direction_bank, save_direction_bank


def test_direction_bank_only_includes_purchased_directions_with_normalized_utility(tmp_path):
    # finding #9: direction_bank must only contain directions procurement actually
    # tied to a granted rank slot (not every certified/reserve candidate that ever
    # existed), and its "utility" field must be the normalized w_bar_p, not the raw
    # unnormalized merged_utility/raw_energy carried on the candidate objects.
    certified = [
        PhysicalCandidate(
            physical_direction_id="m1/bought",
            module_name="m1",
            atom_index=0,
            virtual_candidate_ids=["m1/bought/positive"],
            merged_utility=999.0,  # deliberately different from the normalized utility below
            cost=4,
            full_v=torch.tensor([1.0, 0.0, 0.0]),
        ),
        PhysicalCandidate(
            physical_direction_id="m1/not_bought",
            module_name="m1",
            atom_index=1,
            virtual_candidate_ids=["m1/not_bought/positive"],
            merged_utility=500.0,
            cost=4,
            full_v=torch.tensor([0.0, 1.0, 0.0]),
        ),
    ]
    reserve = [
        PhysicalCandidate(
            physical_direction_id="m1/reserve_bought",
            module_name="m1",
            atom_index=2,
            virtual_candidate_ids=["m1/reserve_bought/reserve"],
            merged_utility=0.0,
            cost=4,
            raw_energy=777.0,
            full_v=torch.tensor([0.0, 0.0, 1.0]),
        ),
    ]
    purchased_directions = {"m1": ["m1/bought", "m1/reserve_bought"]}
    normalized_utility = {"m1/bought": 0.42, "m1/not_bought": 0.99, "m1/reserve_bought": 0.13}

    path = save_direction_bank(
        tmp_path / "direction_bank.pt",
        certified,
        reserve,
        purchased_directions=purchased_directions,
        normalized_utility=normalized_utility,
    )
    bank = load_direction_bank(path)

    assert set(bank.keys()) == {"m1"}
    entries_by_v = {tuple(entry["v"].tolist()): entry for entry in bank["m1"]}
    assert len(bank["m1"]) == 2  # NOT 3 -- "m1/not_bought" must be excluded

    bought_entry = entries_by_v[(1.0, 0.0, 0.0)]
    assert bought_entry["utility"] == 0.42  # normalized, not merged_utility=999.0
    assert bought_entry["source"] == "certified"

    reserve_entry = entries_by_v[(0.0, 0.0, 1.0)]
    assert reserve_entry["utility"] == 0.13  # normalized, not raw_energy=777.0
    assert reserve_entry["source"] == "relaxation"

    assert (0.0, 1.0, 0.0) not in entries_by_v  # the not-purchased direction


def test_direction_bank_skips_modules_with_no_purchased_directions(tmp_path):
    certified = [
        PhysicalCandidate(
            physical_direction_id="m1/only",
            module_name="m1",
            atom_index=0,
            virtual_candidate_ids=["m1/only/positive"],
            merged_utility=1.0,
            cost=4,
            full_v=torch.tensor([1.0, 0.0]),
        ),
    ]
    # m2 has no entry in purchased_directions at all (e.g. rank 0).
    purchased_directions = {"m1": ["m1/only"], "m2": []}
    normalized_utility = {"m1/only": 0.5}

    path = save_direction_bank(
        tmp_path / "direction_bank.pt", certified, [], purchased_directions=purchased_directions,
        normalized_utility=normalized_utility,
    )
    bank = load_direction_bank(path)

    assert set(bank.keys()) == {"m1"}
