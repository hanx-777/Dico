from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from dico.taxonomy import significant_response_groups


@dataclass
class DirectionAtom:
    module_name: str
    atom_index: int
    profile: torch.Tensor
    classification: str
    utility: float
    cost: int
    full_v: torch.Tensor | None = None
    raw_energy: float | None = None
    u: torch.Tensor | None = None
    v_tilde: torch.Tensor | None = None

    @property
    def physical_direction_id(self) -> str:
        return f"{self.module_name}/atom_{self.atom_index}"


@dataclass
class VirtualCandidate:
    virtual_candidate_id: str
    physical_direction_id: str
    module_name: str
    atom_index: int
    profile: torch.Tensor
    split_type: str
    cost: int
    utility: float = 0.0
    certified_gain: float = 0.0
    raw_energy: float = 0.0
    full_v: torch.Tensor | None = None
    # 3.3.3节 kappa(q',q*)=(u_q'.u_q*)(v_tilde_q'.v_tilde_q*): every split of the same
    # atom shares that atom's u/v_tilde, since kappa is a per-direction-unit quantity,
    # not a per-split one. `initial_profile` is a pristine copy of `profile` at
    # construction time, before coverage.py's greedy selection mutates `.profile` into
    # a post-deduction residual -- physical.py's joint-utility recompute must read this
    # instead of `.profile` so it stays independent of selection order (3.4节).
    u: torch.Tensor | None = None
    v_tilde: torch.Tensor | None = None
    initial_profile: torch.Tensor | None = None


@dataclass
class PhysicalCandidate:
    physical_direction_id: str
    module_name: str
    atom_index: int
    virtual_candidate_ids: list[str]
    merged_utility: float
    cost: int
    raw_energy: float = 0.0
    full_v: torch.Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def create_virtual_candidates(
    atoms: Sequence[DirectionAtom],
    split_mode: str = "sign",
    group_labels: Sequence[str] | None = None,
    significance_alpha: float = 0.05,
    permutation_count: int = 1000,
    seed: int = 42,
) -> list[VirtualCandidate]:
    candidates: list[VirtualCandidate] = []
    for atom_index, atom in enumerate(atoms):
        profile = atom.profile.detach().float()
        raw_energy = float(torch.mean(torch.abs(profile) ** 2).item()) if profile.numel() else 0.0
        if atom.classification == "noise":
            continue
        if atom.classification == "task_specific" and split_mode == "sign":
            splits = [
                ("positive", torch.clamp(profile, min=0.0)),
                ("negative", torch.clamp(-profile, min=0.0)),
            ]
        elif atom.classification == "task_specific" and split_mode == "group" and group_labels is not None:
            # 4.4节: only "significant response groups" T_sig get a virtual candidate,
            # not every group that appears in group_labels.
            labels = list(group_labels)
            sig_groups = significant_response_groups(
                profile,
                labels,
                alpha=significance_alpha,
                permutation_count=permutation_count,
                seed=seed + atom_index,
            )
            splits = []
            for group in sorted(sig_groups):
                mask = torch.tensor([label == group for label in labels], dtype=torch.bool)
                group_profile = torch.zeros_like(profile)
                group_profile[mask] = profile[mask]
                splits.append((f"group:{group}", group_profile))
        else:
            splits = [(atom.classification, profile)]
        for split_type, split_profile in splits:
            candidates.append(
                VirtualCandidate(
                    virtual_candidate_id=f"{atom.physical_direction_id}/{split_type}",
                    physical_direction_id=atom.physical_direction_id,
                    module_name=atom.module_name,
                    atom_index=atom.atom_index,
                    profile=split_profile,
                    split_type=split_type,
                    cost=int(atom.cost),
                    utility=float(atom.utility),
                    raw_energy=raw_energy if atom.raw_energy is None else float(atom.raw_energy),
                    full_v=atom.full_v,
                    u=atom.u,
                    v_tilde=atom.v_tilde,
                    initial_profile=split_profile.detach().float().clone(),
                )
            )
    return candidates


def merge_physical_candidates(
    candidates: Sequence[VirtualCandidate],
    utility_by_physical_id: Mapping[str, float],
) -> list[PhysicalCandidate]:
    """4.4.4节: purely structural merge -- groups virtual candidates sharing a physical
    direction and charges its cost once. The purchase utility for each physical
    direction is supplied by the caller (dico.physical.compute_physical_joint_utility),
    not derived here, so that shared structure across sign/group splits of the same
    direction is never double counted.
    """
    grouped: dict[str, list[VirtualCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.physical_direction_id, []).append(candidate)
    merged: list[PhysicalCandidate] = []
    for physical_id, rows in grouped.items():
        first = rows[0]
        utility = float(utility_by_physical_id[physical_id])
        merged.append(
            PhysicalCandidate(
                physical_direction_id=physical_id,
                module_name=first.module_name,
                atom_index=first.atom_index,
                virtual_candidate_ids=[row.virtual_candidate_id for row in rows],
                merged_utility=float(utility),
                cost=int(first.cost),
                raw_energy=max(float(row.raw_energy) for row in rows),
                full_v=first.full_v,
            )
        )
    return sorted(merged, key=lambda row: row.physical_direction_id)
