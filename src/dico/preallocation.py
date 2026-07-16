from __future__ import annotations

import gc
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from dico.atom_svd import (
    aggregate_selected_module_utilities,
    extract_svd_atom_records,
)
from dico.candidates import DirectionAtom, PhysicalCandidate, create_virtual_candidates, merge_physical_candidates
from dico.covra_core import (
    ResponseBlock,
    build_response_block,
    build_type_scaled_utility_curves,
    greedy_conditional_coverage,
    independent_utility_curve,
    module_scalar_utility_curve,
)
from dico.coverage import CoverageResult, greedy_group_fair_coverage
from dico.kappa_calibration import kappa_calibration_diagnostic
from dico.physical import compute_physical_joint_utility
from dico.procurement import procure_budget_window
from dico.pseudo_groups import PseudoGroupResult, build_pseudo_groups
from dico.rank_budget import (
    BudgetInfo,
    WeightedAllocationResult,
    allocate_by_weighted_utility,
    compute_total_lora_params,
    module_rank_cost,
    repair_allocation_to_budget,
    solve_rank_dp,
)
from dico.taxonomy import classify_profile_matrix
from dico.utils import ensure_dir


MODULE_PROXY_LIMITATION = (
    "module_proxy weighted aggregation does not distinguish true SVD/rank-one "
    "atom importance inside a module; it only runs the weighted allocation "
    "pipeline using module-level proxy scores. True atom-level weighting "
    "requires atom_mode=svd."
)


def build_preallocation_cache_context(
    config: Mapping[str, Any],
    module_names: list[str],
    module_dims: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    calibration_cfg = config.get("calibration", {})
    lora_cfg = config.get("lora", {})
    pre_cfg = config.get("preallocation", {})
    dico_cfg = config.get("dico", {})
    evidence_cfg = pre_cfg.get("evidence_selection", {})
    return {
        "seed": config.get("seed", 42),
        "rank": config.get("rank"),
        "model_name_or_path": model_cfg.get("name_or_path"),
        "target_modules": list(lora_cfg.get("target_modules", [])),
        "module_names": list(module_names),
        "module_dims": {
            name: {"in_dim": int(dims["in_dim"]), "out_dim": int(dims["out_dim"])}
            for name, dims in module_dims.items()
        },
        "dataset_name": data_cfg.get("dataset_name"),
        "dataset_config": data_cfg.get("dataset_config"),
        "train_path": data_cfg.get("train_path"),
        "eval_path": data_cfg.get("eval_path"),
        "train_sources": deepcopy(data_cfg.get("train_sources")),
        "max_length": data_cfg.get("max_length"),
        "train_limit": data_cfg.get("train_limit"),
        "shuffle": data_cfg.get("shuffle"),
        "dataset_seed": data_cfg.get("dataset_seed"),
        "group_labels": deepcopy(data_cfg.get("group_labels")),
        "calibration_num_samples": calibration_cfg.get("num_samples"),
        "calibration_seed": calibration_cfg.get("seed", config.get("seed")),
        "calibration_batch_size": calibration_cfg.get("batch_size"),
        "calibration_shuffle": calibration_cfg.get("shuffle"),
        "calibration_group_sampling": calibration_cfg.get("group_sampling"),
        "preallocation": {
            "atom_mode": pre_cfg.get("atom_mode", "svd"),
            "allocation_method": pre_cfg.get("allocation_method", "covra_v05"),
            "eta": pre_cfg.get("eta", 0.98),
            "allow_rank_beyond_selected_evidence": pre_cfg.get("allow_rank_beyond_selected_evidence", True),
            "top_k_atoms": pre_cfg.get("top_k_atoms"),
            "sketch_dim": pre_cfg.get("sketch_dim"),
            "sketch_seed": pre_cfg.get("sketch_seed"),
            "sketch_dtype": pre_cfg.get("sketch_dtype"),
            "sketch_oversample": pre_cfg.get("sketch_oversample"),
            "allocation_device": pre_cfg.get("allocation_device"),
            "compute_device": pre_cfg.get("compute_device"),
            "answer_only": pre_cfg.get("answer_only"),
            "profile_norm_mode": pre_cfg.get("profile_norm_mode"),
            "beta": pre_cfg.get("beta"),
            "lambda_cov": pre_cfg.get("lambda_cov"),
            "response_agg_groups": pre_cfg.get("response_agg_groups"),
            "aggregation_mode": pre_cfg.get("aggregation_mode", "weighted_topk"),
            "evidence_selection.max_selected_atoms": evidence_cfg.get("max_selected_atoms", "auto"),
            "r_min_multiplier": pre_cfg.get("r_min_multiplier", 0.25),
            "r_max_multiplier": pre_cfg.get("r_max_multiplier", 4),
        },
        "dico": {
            section: deepcopy(dico_cfg.get(section, {}))
            for section in ("taxonomy", "pseudo_group", "split", "coverage", "procurement", "init")
        },
}


def _evidence_relaxation_diagnostics(module_logs: list[dict[str, Any]]) -> dict[str, Any]:
    selected_total = 0
    final_total = 0
    beyond_total = 0
    beyond_modules = []
    for row in module_logs:
        selected_count = int(row.get("selected_evidence_count", row.get("selected_atom_count", 0)) or 0)
        final_rank = int(row.get("final_rank", 0) or 0)
        beyond = int(row.get("rank_beyond_selected_evidence", max(0, final_rank - selected_count)) or 0)
        selected_total += selected_count
        final_total += final_rank
        beyond_total += beyond
        if beyond > 0:
            beyond_modules.append(row.get("module_name"))
    return {
        "selected_evidence_count_total": selected_total,
        "final_total_rank": final_total,
        "rank_beyond_selected_evidence_total": beyond_total,
        "modules_with_rank_beyond_selected_evidence": beyond_modules,
    }


@dataclass
class AtomUtility:
    module_name: str
    atom_id: int
    importance: float
    redundancy: float
    utility: float
    selected: bool
    atom_mode: str
    singular_value: float | None = None
    response_norm: float | None = None
    profile_norm: float | None = None
    importance_source: str | None = None
    redundancy_source: str | None = None
    aggregation_mode: str | None = None
    atom_weight_normalization: str | None = None
    atom_mode_limitation: str | None = None


@dataclass
class ModuleUtility:
    module_name: str
    module_utility: float
    rank_cost: int
    cost_aware_score: float
    continuous_rank: float
    final_rank: int


@dataclass
class PreallocationResult:
    rank_allocation: dict[str, int]
    module_scores: dict[str, float]
    atom_logs: list[dict[str, Any]]
    module_logs: list[dict[str, Any]]
    budget: BudgetInfo
    atom_mode: str
    aggregation_mode: str
    weighted_topk_k: int | str
    atom_weight_normalization: str
    use_cost_aware_allocation: bool
    module_names: list[str] = field(default_factory=list)
    module_dims: dict[str, dict[str, int]] = field(default_factory=dict)
    cache_context: dict[str, Any] = field(default_factory=dict)
    preallocation_source: str = "computed"
    atom_mode_limitation: str | None = None
    allocation_method: str | None = None
    profile_norm_mode: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, preallocation_path: str | None = None) -> dict[str, Any]:
        payload = {
            "rank_allocation": self.rank_allocation,
            "module_scores": self.module_scores,
            "atom_logs": self.atom_logs,
            "module_logs": self.module_logs,
            "budget": self.budget.to_dict(),
            "aggregation_mode": self.aggregation_mode,
            "weighted_topk_k": self.weighted_topk_k,
            "atom_weight_normalization": self.atom_weight_normalization,
            "use_cost_aware_allocation": self.use_cost_aware_allocation,
            "atom_mode": self.atom_mode,
            "allocation_method": self.allocation_method,
            "profile_norm_mode": self.profile_norm_mode,
            "module_names": self.module_names,
            "module_dims": self.module_dims,
            "cache_context": self.cache_context,
            "preallocation_source": self.preallocation_source,
            "preallocation_path": preallocation_path,
            "target_budget": self.budget.target_budget,
            "actual_budget": self.budget.actual_budget,
            "target_budget_paramcount": self.budget.to_dict()["target_budget_paramcount"],
            "target_budget_ranksum": self.budget.to_dict()["target_budget_ranksum"],
            "actual_budget_paramcount": self.budget.to_dict()["actual_budget_paramcount"],
            "actual_budget_ranksum": self.budget.to_dict()["actual_budget_ranksum"],
            "budget_ratio_paramcount": self.budget.to_dict()["budget_ratio_paramcount"],
            "budget_ratio_ranksum": self.budget.to_dict()["budget_ratio_ranksum"],
            "budget_error": self.budget.budget_error,
            "budget_error_ratio": self.budget.budget_error_ratio,
        }
        payload.update(self.diagnostics)
        if self.atom_mode_limitation:
            payload["atom_mode_limitation"] = self.atom_mode_limitation
        return payload


class DiCoPreAllocator:
    """DiCo preallocator with auditable weighted atom-to-module aggregation.

    The current implementation can run with module-level proxy atoms. It records
    ``atom_mode=module_proxy`` and an explicit limitation so downstream analysis
    does not confuse proxy atoms with true SVD/rank-one atoms.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Mapping[str, Any],
        module_names: list[str] | None = None,
        module_dims: Mapping[str, Mapping[str, Any]] | None = None,
        module_scores: Mapping[str, float] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = dict(config)
        self.pre_cfg = dict(self.config.get("preallocation", {}))
        self.module_names = list(module_names or [])
        self.module_dims = dict(module_dims or {})
        self.module_scores = {name: float(value) for name, value in (module_scores or {}).items()}
        self.requested_atom_mode = self.pre_cfg.get("atom_mode", self.pre_cfg.get("fallback_atom_mode", "module_proxy"))
        self.atom_mode = str(self.requested_atom_mode)
        self.aggregation_mode = self.pre_cfg.get("aggregation_mode", "weighted_topk")
        self.atom_weight_normalization = self.pre_cfg.get("atom_weight_normalization", "none")
        self.use_cost_aware_allocation = bool(self.pre_cfg.get("use_cost_aware_allocation", True))
        self._calibration_batches: Any = None
        self._svd_diagnostics: dict[str, Any] = {}
        self._validate_candidate_budget_config()

    def _r_max(self) -> int:
        return int(self.config.get("rank", 1) * self.pre_cfg.get("r_max_multiplier", 4))

    def _r_min(self) -> int:
        return max(0, int(self.config.get("rank", 1) * float(self.pre_cfg.get("r_min_multiplier", 0.25))))

    def _weighted_topk_k(self) -> int | str:
        value = self.pre_cfg.get("weighted_topk_k", "auto")
        if value == "auto":
            return self._r_max()
        return int(value)

    def _validate_candidate_budget_config(self) -> None:
        allocation_method = str(self.pre_cfg.get("allocation_method", "covra_v05"))
        if allocation_method not in {"covra_full", "covra_independent", "covra_module_scalar"}:
            return
        top_k = self.pre_cfg.get("top_k_atoms")
        if top_k in (None, "auto"):
            return
        top_k = int(top_k)
        r_max = self._r_max()
        if top_k < r_max:
            raise ValueError(
                "CovRA candidate configuration invalid: top_k_atoms must be >= r_max "
                f"so every feasible rank has candidate evidence; got top_k_atoms={top_k}, r_max={r_max}."
            )

    def collect_calibration_statistics(self, calibration_dataloader: Any) -> dict[str, float]:
        if self.requested_atom_mode == "svd" and self.model is not None:
            self._calibration_batches = list(calibration_dataloader)
            return self.module_scores
        if self.module_scores:
            return self.module_scores
        if self.model is None:
            self.module_scores = {name: 1.0 for name in self.module_names}
            return self.module_scores

        modules = dict(self.model.named_modules())
        records: dict[str, dict[str, torch.Tensor]] = {}
        handles = []

        def make_hook(name: str):
            def hook(_module, inputs, output):
                if torch.is_tensor(output) and output.requires_grad:
                    output.retain_grad()
                    records[name] = {"activation": inputs[0].detach(), "output": output}

            return hook

        for name in self.module_names:
            handles.append(modules[name].register_forward_hook(make_hook(name)))

        scores = {name: 0.0 for name in self.module_names}
        count = 0
        try:
            for batch in calibration_dataloader:
                outputs = None
                try:
                    self.model.zero_grad(set_to_none=True)
                    records.clear()
                    outputs = self.model(**batch)
                    if getattr(outputs, "loss", None) is None:
                        continue
                    outputs.loss.backward()
                    for name in self.module_names:
                        record = records.get(name)
                        if not record or record["output"].grad is None:
                            continue
                        activation = record["activation"].float()
                        grad = record["output"].grad.detach().float()
                        scores[name] += float(torch.linalg.norm(activation).item() * torch.linalg.norm(grad).item())
                    count += 1
                finally:
                    records.clear()
                    if outputs is not None:
                        del outputs
        finally:
            for handle in handles:
                handle.remove()
            records.clear()
            self.model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        denom = max(1, count)
        self.module_scores = {name: value / denom for name, value in scores.items()}
        return self.module_scores

    def build_atom_records(self) -> list[dict[str, Any]]:
        """Build proxy atom records from module-level scores.

        True SVD atoms can be added later by replacing this record builder while
        preserving the utility/aggregation/allocation pipeline below.
        """

        self.atom_mode = "module_proxy"
        proxy_atom_count = max(1, int(self._weighted_topk_k()))
        records = []
        for name in self.module_names:
            score = float(self.module_scores.get(name, 1.0))
            importance = score / proxy_atom_count
            for atom_id in range(proxy_atom_count):
                records.append(
                    {
                        "module_name": name,
                        "atom_id": atom_id,
                        "importance": importance,
                        "redundancy": 0.0,
                        "selected": True,
                        "atom_mode": self.atom_mode,
                        "importance_source": "module_score_div_proxy_atom_count",
                        "redundancy_source": "none",
                        "atom_mode_limitation": MODULE_PROXY_LIMITATION,
                    }
                )
        return records

    def compute_atom_utilities(self, atom_records: list[Mapping[str, Any]]) -> list[AtomUtility]:
        floor = float(self.pre_cfg.get("atom_utility_floor", 0.0))
        coverage_lambda = float(self.pre_cfg.get("coverage_lambda", 0.5))
        utilities = []
        for record in atom_records:
            importance = float(record.get("importance", record.get("utility", 0.0)))
            redundancy = float(record.get("redundancy", 0.0))
            utility = max(floor, importance - coverage_lambda * redundancy)
            atom_mode = str(record.get("atom_mode", self.atom_mode))
            utilities.append(
                AtomUtility(
                    module_name=str(record["module_name"]),
                    atom_id=int(record.get("atom_id", 0)),
                    importance=importance,
                    redundancy=redundancy,
                    utility=utility,
                    selected=bool(record.get("selected", True)),
                    atom_mode=atom_mode,
                    singular_value=record.get("singular_value"),
                    response_norm=record.get("response_norm"),
                    profile_norm=record.get("profile_norm"),
                    importance_source=record.get("importance_source"),
                    redundancy_source=record.get("redundancy_source"),
                    aggregation_mode=self.aggregation_mode,
                    atom_weight_normalization=self.atom_weight_normalization,
                    atom_mode_limitation=record.get("atom_mode_limitation")
                    or (MODULE_PROXY_LIMITATION if atom_mode == "module_proxy" else None),
                )
            )
        return utilities

    def normalize_atom_utilities(
        self,
        atom_utilities: list[AtomUtility],
        mode: str | None = None,
    ) -> list[AtomUtility]:
        mode = mode or self.atom_weight_normalization
        if mode == "none":
            return [replace(atom, atom_weight_normalization=mode) for atom in atom_utilities]
        if mode == "global":
            denom = sum(atom.utility for atom in atom_utilities) or 1.0
            return [
                replace(atom, utility=atom.utility / denom, atom_weight_normalization=mode)
                for atom in atom_utilities
            ]
        if mode == "module":
            module_sums: dict[str, float] = {}
            for atom in atom_utilities:
                module_sums[atom.module_name] = module_sums.get(atom.module_name, 0.0) + atom.utility
            return [
                replace(
                    atom,
                    utility=atom.utility / (module_sums.get(atom.module_name, 0.0) or 1.0),
                    atom_weight_normalization=mode,
                )
                for atom in atom_utilities
            ]
        raise ValueError(f"Unsupported atom_weight_normalization: {mode}")

    def aggregate_module_utilities(
        self,
        atom_utilities: list[AtomUtility],
        aggregation_mode: str | None = None,
    ) -> dict[str, float]:
        mode = aggregation_mode or self.aggregation_mode
        grouped: dict[str, list[AtomUtility]] = {name: [] for name in self.module_names}
        for atom in atom_utilities:
            grouped.setdefault(atom.module_name, []).append(atom)
            atom.selected = False
            atom.aggregation_mode = mode

        module_utilities: dict[str, float] = {}
        for module_name, atoms in grouped.items():
            if mode == "count":
                for atom in atoms:
                    atom.selected = True
                module_utilities[module_name] = float(len(atoms))
            elif mode == "weighted_sum":
                for atom in atoms:
                    atom.selected = True
                module_utilities[module_name] = sum(atom.utility for atom in atoms)
            elif mode == "weighted_log":
                for atom in atoms:
                    atom.selected = True
                module_utilities[module_name] = sum(math.log1p(max(atom.utility, 0.0)) for atom in atoms)
            elif mode == "weighted_topk":
                k = int(self._weighted_topk_k())
                top_atoms = sorted(atoms, key=lambda atom: atom.utility, reverse=True)[:k]
                top_ids = {id(atom) for atom in top_atoms}
                for atom in atoms:
                    atom.selected = id(atom) in top_ids
                module_utilities[module_name] = sum(atom.utility for atom in top_atoms)
            else:
                raise ValueError(f"Unsupported aggregation_mode: {mode}")
        return module_utilities

    def allocate_from_module_utilities(
        self,
        module_utilities: Mapping[str, float],
        rank_budget: int,
    ):
        rank = int(self.config.get("rank", 1))
        total_rank_budget = rank * len(self.module_names)
        r_min = self._r_min()
        r_max = self._r_max()
        return allocate_by_weighted_utility(
            module_utilities=module_utilities,
            module_dims=self.module_dims,
            total_rank_budget=total_rank_budget,
            target_budget=int(rank_budget),
            r_min=r_min,
            r_max=r_max,
            use_cost_aware=self.use_cost_aware_allocation,
            warning_threshold=float(self.config.get("budget", {}).get("warning_threshold", 0.01)),
        )

    def _profile_path(self) -> Path:
        project_root = Path(self.config.get("_project_root", Path.cwd())).resolve()
        save_dir = self.config.get("calibration", {}).get("save_dir", "outputs/preallocations")
        save_dir_path = Path(save_dir)
        if not save_dir_path.is_absolute():
            save_dir_path = project_root / save_dir_path
        rank = int(self.config.get("rank", 1))
        seed = int(self.config.get("calibration", {}).get("seed", self.config.get("seed", 42)))
        return save_dir_path / f"dico_v03_rank{rank}_seed{seed}_profiles.pt"

    def _direction_bank_path(self) -> Path:
        return self._profile_path().with_name(self._profile_path().stem.replace("_profiles", "_direction_bank") + ".pt")

    def _evidence_selected_utilities(self, atom_records: list[Any]) -> dict[str, list[float]]:
        values = {name: [] for name in self.module_names}
        for atom in atom_records:
            if atom.selected:
                values.setdefault(atom.module_name, []).append((atom.atom_index, float(atom.utility)))
        return {
            name: [utility for _idx, utility in sorted(rows, key=lambda item: item[0])]
            for name, rows in values.items()
        }

    def _allocate_v03_from_svd_atoms(
        self,
        atoms: list[Any],
        rank_budget: int,
        base_diagnostics: Mapping[str, Any],
    ) -> WeightedAllocationResult:
        profile_atoms = [atom for atom in atoms if atom.profile is not None]
        if not profile_atoms:
            raise RuntimeError("DiCo v0.3 SVD allocation requires sketch-domain signed profiles.")
        profiles = torch.stack([atom.profile.detach().float() for atom in profile_atoms], dim=1)
        dico_cfg = dict(self.config.get("dico", {}))
        configured_groups = self.config.get("data", {}).get("group_labels")
        pseudo_result: PseudoGroupResult | None = None
        if configured_groups and len(configured_groups) == int(profiles.shape[0]):
            # One label per calibration sample: use the real task-group assignment directly.
            groups = [str(value) for value in configured_groups]
            group_source = "configured"
        else:
            # 3.2.3节: no usable per-sample task-group labels (either none configured, or
            # `data.group_labels` names categories rather than providing one label per
            # sample) -- construct pseudo-groups from profile geometry via k-means
            # clustering instead of silently collapsing every sample into one group.
            pseudo_group_cfg = dict(dico_cfg.get("pseudo_group", {}))
            if bool(pseudo_group_cfg.get("enabled", True)) and int(profiles.shape[0]) >= 2:
                pseudo_result = build_pseudo_groups(
                    profiles,
                    k_range=range(
                        int(pseudo_group_cfg.get("min_k", 2)),
                        int(pseudo_group_cfg.get("max_k", 6)) + 1,
                    ),
                    seed=int(self.config.get("seed", 42)),
                    val_fraction=float(pseudo_group_cfg.get("val_fraction", 0.5)),
                )
                groups = pseudo_result.groups
                group_source = "pseudo"
            else:
                groups = ["calibration" for _ in range(int(profiles.shape[0]))]
                group_source = "single"
        module_types = [str(atom.module_name).split(".")[-1] for atom in profile_atoms]
        taxonomy_cfg = dict(dico_cfg.get("taxonomy", {}))
        taxonomy_rows = classify_profile_matrix(
            profiles,
            groups,
            module_types,
            alpha=float(taxonomy_cfg.get("alpha", 0.05)),
            permutation_count=int(taxonomy_cfg.get("permutation_count", 1000)),
            seed=int(self.config.get("seed", 42)),
            val_mask=pseudo_result.val_mask if pseudo_result is not None else None,
        )
        taxonomy_by_index = {row.index: row for row in taxonomy_rows}
        direction_atoms: list[DirectionAtom] = []
        reserve_atoms: list[DirectionAtom] = []
        for idx, atom in enumerate(profile_atoms):
            row = taxonomy_by_index[idx]
            raw_energy = float(torch.mean(torch.abs(atom.profile.detach().float()) ** 2).item())
            direction = DirectionAtom(
                module_name=atom.module_name,
                atom_index=int(atom.atom_index),
                profile=atom.profile.detach().float(),
                classification=row.label,
                # 4.6.6节 e_p^raw: pre-certification per-atom strength signal, used as
                # DirectionAtom.utility's fallback/tie-break basis (e.g. coverage.py's
                # greedy tie-break, init.py's DA-Init ordering) before any physical
                # joint utility exists yet for this atom.
                utility=raw_energy,
                cost=module_rank_cost(self.module_dims[atom.module_name]),
                full_v=getattr(atom, "v", None),
                raw_energy=raw_energy,
                u=getattr(atom, "u", None),
                v_tilde=getattr(atom, "v_tilde", None),
            )
            if row.label == "noise":
                reserve_atoms.append(direction)
            else:
                direction_atoms.append(direction)
        split_cfg = dict(dico_cfg.get("split", {}))
        virtual = create_virtual_candidates(
            direction_atoms,
            split_mode=str(split_cfg.get("mode", "sign")),
            group_labels=groups,
            significance_alpha=float(taxonomy_cfg.get("alpha", 0.05)),
            permutation_count=int(taxonomy_cfg.get("permutation_count", 1000)),
            seed=int(self.config.get("seed", 42)),
        )
        evidence_cfg = dict(self.pre_cfg.get("evidence_selection", {}))
        max_selected = evidence_cfg.get("max_selected_atoms", "auto")
        if max_selected in {None, "auto"}:
            max_selected_int = int(self.config.get("rank", 1)) * len(self.module_names)
        else:
            max_selected_int = int(max_selected)
        coverage_cfg = dict(dico_cfg.get("coverage", {}))
        # 4.5节: coverage certification is isolated per module type (q/k/v/o_proj, ...) so that
        # functionally heterogeneous modules cannot mask each other's directions.
        type_buckets: dict[str, list] = {}
        for candidate in virtual:
            type_buckets.setdefault(str(candidate.module_name).split(".")[-1], []).append(candidate)
        selected: list = []
        coverage_trace: list[dict[str, object]] = []
        kappa_calibration_cfg = dict(coverage_cfg.get("kappa_calibration", {}))
        kappa_calibration_enabled = bool(kappa_calibration_cfg.get("enabled", True))
        kappa_calibration_by_type: dict[str, Any] = {}
        for module_type, bucket in type_buckets.items():
            bucket_budget = max(1, round(max_selected_int * len(bucket) / len(virtual))) if virtual else 0
            window_h = int(coverage_cfg.get("window_h", 2))
            if kappa_calibration_enabled:
                calibration = kappa_calibration_diagnostic(
                    bucket,
                    module_type,
                    seed=int(self.config.get("seed", 42)),
                    max_pairs=int(kappa_calibration_cfg.get("max_pairs", 2000)),
                    indistinguishable_alpha=float(kappa_calibration_cfg.get("alpha", 0.1)),
                )
                kappa_calibration_by_type[module_type] = asdict(calibration)
                if calibration.fallback_h0:
                    window_h = 0
            bucket_coverage = greedy_group_fair_coverage(
                bucket,
                groups,
                max_selected=bucket_budget,
                eps=float(coverage_cfg.get("eps", 1.0e-6)),
                relative_stop_delta=float(coverage_cfg.get("relative_stop_delta", 1.0e-3)),
                window_h=window_h,
            )
            selected.extend(bucket_coverage.selected)
            coverage_trace.extend({**row, "module_type": module_type} for row in bucket_coverage.trace)
        coverage = CoverageResult(selected=selected, trace=coverage_trace)
        # 4.4.4节: recompute joint physical utility per physical direction *after*
        # certification, instead of summing each virtual candidate's own realized
        # gain -- avoids double-counting shared structure across sign/group splits.
        joint_utilities = compute_physical_joint_utility(coverage.selected, groups)
        certified = merge_physical_candidates(coverage.selected, joint_utilities)
        reserve = [
            PhysicalCandidate(
                physical_direction_id=atom.physical_direction_id,
                module_name=atom.module_name,
                atom_index=atom.atom_index,
                virtual_candidate_ids=[f"{atom.physical_direction_id}/reserve"],
                merged_utility=0.0,
                cost=atom.cost,
                raw_energy=float(atom.raw_energy or 0.0),
                full_v=atom.full_v,
            )
            for atom in reserve_atoms
        ]
        procurement_cfg = dict(dico_cfg.get("procurement", {}))
        procurement_beta = float(procurement_cfg.get("beta", self.pre_cfg.get("beta", 0.5)))
        rank = int(self.config.get("rank", 1))
        r_min = self._r_min()
        procured = procure_budget_window(
            certified,
            reserve,
            self.module_dims,
            target_budget=int(rank_budget),
            eta=float(self.pre_cfg.get("eta", 0.98)),
            r_min=r_min,
            r_max=self._r_max(),
            beta=procurement_beta,
            reserve_queue_enabled=bool(procurement_cfg.get("reserve_queue", True)),
            balanced_fill_enabled=bool(procurement_cfg.get("relaxation_fallback", True)),
        )
        actual = compute_total_lora_params(procured.rank_dict, self.module_dims)
        target_rank_sum = rank * len(self.module_names)
        budget = BudgetInfo(
            budget_mode=self.config.get("budget", {}).get("mode", "equal_trainable_params"),
            target_budget=int(rank_budget),
            actual_budget=actual,
            budget_error=actual - int(rank_budget),
            budget_error_ratio=float((actual - int(rank_budget)) / int(rank_budget)) if int(rank_budget) else 0.0,
            total_active_rank=sum(int(value) for value in procured.rank_dict.values()),
            target_budget_ranksum=target_rank_sum,
            budget_ratio_ranksum=float(sum(int(value) for value in procured.rank_dict.values()) / target_rank_sum)
            if target_rank_sum
            else 0.0,
            over_budget=actual > int(rank_budget),
            warning=None if actual >= float(self.pre_cfg.get("eta", 0.98)) * int(rank_budget) else "covra_v05_allocation_below_budget_window",
        )
        module_logs = []
        for name in self.module_names:
            selected_for_module = [candidate for candidate in certified if candidate.module_name == name]
            module_logs.append(
                {
                    "module_name": name,
                    "module_utility": sum(candidate.merged_utility for candidate in selected_for_module),
                    "rank_cost": module_rank_cost(self.module_dims[name]),
                    "final_rank": int(procured.rank_dict[name]),
                    "selected_atom_count": len(selected_for_module),
                    "selected_evidence_count": len(selected_for_module),
                    "selected_atom_utilities": [candidate.merged_utility for candidate in selected_for_module],
                    "purchased_evidence_rank": min(int(procured.rank_dict[name]), len(selected_for_module)),
                    "evidence_relaxation_rank": max(0, int(procured.rank_dict[name]) - len(selected_for_module)),
                    "rank_beyond_selected_evidence": max(0, int(procured.rank_dict[name]) - len(selected_for_module)),
                    "final_parameter_count": int(procured.rank_dict[name]) * module_rank_cost(self.module_dims[name]),
                    "allocation_method": "covra_v05",
                }
            )
        taxonomy_stats = {
            "consensus": sum(1 for row in taxonomy_rows if row.label == "consensus"),
            "task_specific": sum(1 for row in taxonomy_rows if row.label == "task_specific"),
            "noise": sum(1 for row in taxonomy_rows if row.label == "noise"),
            "fdr_method": "BH",
            "alpha": float(taxonomy_cfg.get("alpha", 0.05)),
            "B_perm": int(taxonomy_cfg.get("permutation_count", 1000)),
            "group_source": group_source,
            "num_groups": len(set(groups)),
            "pseudo_group_k_selected": pseudo_result.k_selected if pseudo_result is not None else None,
            "pseudo_fit_sample_count": pseudo_result.fit_sample_count if pseudo_result is not None else None,
            "pseudo_val_sample_count": pseudo_result.val_sample_count if pseudo_result is not None else None,
        }
        direction_bank_path = save_direction_bank(
            self._direction_bank_path(),
            certified,
            reserve,
            purchased_directions=procured.purchased_directions,
            normalized_utility=procured.normalized_utility,
        )
        diagnostics = {
            **dict(base_diagnostics),
            "allocation_method": "covra_v05",
            "taxonomy_stats": taxonomy_stats,
            "kappa_calibration": kappa_calibration_by_type,
            "coverage_trace": coverage.trace,
            "procurement_trace": procured.trace,
            "reserve_filled_ratio": procured.reserve_filled_ratio,
            "balanced_fill_ratio": procured.balanced_fill_ratio,
            "zero_rank_module_ratio": procured.zero_rank_module_ratio,
            "budget_gap_ratio": procured.budget_gap_ratio,
            "procurement_warning": procured.warning,
            "procurement_beta": procurement_beta,
            "module_quota": procured.module_quota,
            "r_min": r_min,
            "physical_utility": procured.physical_utility,
            "normalized_utility": procured.normalized_utility,
            "normalization_stats": asdict(procured.normalization_stats) if procured.normalization_stats else {},
            "num_atoms": len(profile_atoms),
            "num_selected_atoms": len(certified),
            "direction_bank_path": str(direction_bank_path),
        }
        return WeightedAllocationResult(
            allocation=procured.rank_dict,
            budget=budget,
            module_logs=module_logs,
            diagnostics=diagnostics,
        )

    def _module_type(self, module_name: str) -> str:
        return str(module_name).split(".")[-1]

    def _module_scalar_template(self, r_max: int) -> list[float]:
        template = self.pre_cfg.get("module_scalar_template")
        if template is not None:
            return [float(value) for value in template]
        return [1.0 / float(rank) for rank in range(1, int(r_max) + 1)]

    def _module_scalar_template_metadata(self, r_max: int) -> dict[str, object]:
        return {
            "module_scalar_template": self._module_scalar_template(r_max),
            "module_scalar_template_formula": str(
                self.pre_cfg.get("module_scalar_template_formula", "w_j = 1 / j for j=1..r_max")
            ),
            "module_scalar_template_normalization": str(
                self.pre_cfg.get("module_scalar_template_normalization", "sum_to_module_energy")
            ),
        }

    def _allocate_final_covra_from_svd_atoms(
        self,
        atoms: list[Any],
        rank_budget: int,
        base_diagnostics: Mapping[str, Any],
    ) -> WeightedAllocationResult:
        allocation_method = str(self.pre_cfg.get("allocation_method", "covra_full"))
        if allocation_method not in {"covra_full", "covra_independent", "covra_module_scalar"}:
            raise ValueError(f"Unsupported final CovRA allocation_method: {allocation_method}")

        r_min = self._r_min()
        r_max = self._r_max()
        if "rho" not in self.pre_cfg:
            raise ValueError(
                "Final CovRA allocation requires explicit preallocation.rho; "
                "do not rely on an implicit sign-split threshold."
            )
        rho = float(self.pre_cfg["rho"])
        solver = str(self.pre_cfg.get("solver", "dp"))
        if solver not in {"dp", "proportional_rounding"}:
            raise ValueError(
                f"Unsupported final CovRA solver={solver!r}; "
                "supported values are 'dp' and 'proportional_rounding'."
            )
        sign_split = bool(self.pre_cfg.get("sign_split", True))
        type_scaling = bool(self.pre_cfg.get("type_scaling", True))
        log_compression = bool(self.pre_cfg.get("log_compression", True))
        eps = float(self.config.get("dico", {}).get("profile", {}).get("eps", 1.0e-12))

        blocks_by_module: dict[str, list] = {name: [] for name in self.module_names}
        selected_by_module: dict[str, list[int]] = {name: [] for name in self.module_names}
        marginal_by_module: dict[str, list[float]] = {name: [] for name in self.module_names}
        init_utility_by_module: dict[str, list[float]] = {name: [] for name in self.module_names}
        trace_by_module: dict[str, list[dict[str, object]]] = {name: [] for name in self.module_names}
        candidate_energy_by_module: dict[str, float] = {name: 0.0 for name in self.module_names}
        verify_cpu_reference = bool(self.pre_cfg.get("verify_cpu_reference", True))
        cpu_reference_enabled = verify_cpu_reference and any(
            atom.profile is not None and atom.profile.device.type == "cuda" for atom in atoms
        )
        cpu_raw_curves: dict[str, list[float]] = {}
        cpu_selected_by_module: dict[str, list[int]] = {}
        cpu_marginal_by_module: dict[str, list[float]] = {}
        cpu_trace_by_module: dict[str, list[dict[str, object]]] = {}
        cpu_reference_fallback = False

        for atom in atoms:
            if atom.module_name not in blocks_by_module or atom.profile is None:
                continue
            block = build_response_block(
                module_name=atom.module_name,
                candidate_index=int(atom.atom_index),
                response=atom.profile,
                rho=rho,
                sign_split=sign_split,
            )
            blocks_by_module[atom.module_name].append(block)
            candidate_energy_by_module[atom.module_name] += float(torch.sum(block.matrix * block.matrix).item())

        atoms_by_module_index: dict[str, dict[int, Any]] = {name: {} for name in self.module_names}
        for atom in atoms:
            if atom.module_name in atoms_by_module_index:
                atoms_by_module_index[atom.module_name][int(atom.atom_index)] = atom

        raw_curves: dict[str, list[float]] = {}
        utility_builder = allocation_method.replace("covra_", "")
        for module_name in self.module_names:
            blocks = sorted(blocks_by_module[module_name], key=lambda block: block.candidate_index)
            if len(blocks) < r_max:
                raise ValueError(
                    f"CovRA final path requires at least r_max response blocks per module; "
                    f"module={module_name} blocks={len(blocks)} r_max={r_max}"
                )
            if allocation_method == "covra_full":
                curve = greedy_conditional_coverage(blocks, r_max=r_max, eps=eps)
                raw_curves[module_name] = curve.cumulative_utility
                selected_by_module[module_name] = curve.selected_indices
                marginal_by_module[module_name] = curve.marginal_gains
                init_utility_by_module[module_name] = curve.marginal_gains
                trace_by_module[module_name] = curve.trace
                utility_builder = "conditional"
            elif allocation_method == "covra_independent":
                curve = independent_utility_curve(blocks, r_max=r_max)
                raw_curves[module_name] = curve.cumulative_utility
                selected_by_module[module_name] = curve.selected_indices
                marginal_by_module[module_name] = curve.marginal_gains
                init_utility_by_module[module_name] = curve.marginal_gains
                trace_by_module[module_name] = curve.trace
                utility_builder = "independent"
            else:
                init_curve = independent_utility_curve(blocks, r_max=r_max)
                raw_curves[module_name] = module_scalar_utility_curve(
                    module_energy=candidate_energy_by_module[module_name],
                    r_max=r_max,
                    template=self._module_scalar_template(r_max),
                )
                selected_by_module[module_name] = init_curve.selected_indices
                marginal_by_module[module_name] = [
                    raw_curves[module_name][idx] - raw_curves[module_name][idx - 1]
                    for idx in range(1, len(raw_curves[module_name]))
                ]
                init_utility_by_module[module_name] = init_curve.marginal_gains
                trace_by_module[module_name] = init_curve.trace
                utility_builder = "module_scalar"

            if cpu_reference_enabled:
                cpu_blocks = [
                    ResponseBlock(
                        module_name=block.module_name,
                        candidate_index=block.candidate_index,
                        matrix=block.matrix.detach().cpu(),
                        rank_cost=block.rank_cost,
                        split=block.split,
                        positive_energy_ratio=block.positive_energy_ratio,
                        negative_energy_ratio=block.negative_energy_ratio,
                    )
                    for block in blocks
                ]
                if allocation_method == "covra_full":
                    cpu_curve = greedy_conditional_coverage(cpu_blocks, r_max=r_max, eps=eps)
                    cpu_raw_curves[module_name] = cpu_curve.cumulative_utility
                elif allocation_method == "covra_independent":
                    cpu_curve = independent_utility_curve(cpu_blocks, r_max=r_max)
                    cpu_raw_curves[module_name] = cpu_curve.cumulative_utility
                else:
                    cpu_curve = independent_utility_curve(cpu_blocks, r_max=r_max)
                    cpu_raw_curves[module_name] = list(raw_curves[module_name])
                cpu_selected_by_module[module_name] = cpu_curve.selected_indices
                cpu_marginal_by_module[module_name] = cpu_curve.marginal_gains
                cpu_trace_by_module[module_name] = cpu_curve.trace

        module_types = {name: self._module_type(name) for name in self.module_names}
        utility_curves = build_type_scaled_utility_curves(
            raw_curves,
            module_types,
            type_scaling=type_scaling,
            log_compression=log_compression,
            eps=eps,
        )
        def solve_curves(curves: Mapping[str, Sequence[float]]) -> WeightedAllocationResult:
            if solver == "dp":
                return solve_rank_dp(
                    curves,
                    self.module_dims,
                    target_budget=int(rank_budget),
                    r_min=r_min,
                    r_max=r_max,
                    eta=float(self.pre_cfg.get("eta", 0.98)),
                    budget_mode=self.config.get("budget", {}).get("mode", "equal_trainable_params"),
                    warning_threshold=float(self.config.get("budget", {}).get("warning_threshold", 0.01)),
                )
            result = allocate_by_weighted_utility(
                module_utilities={name: float(curves[name][r_max]) for name in self.module_names},
                module_dims=self.module_dims,
                total_rank_budget=int(self.config.get("rank", 1)) * len(self.module_names),
                target_budget=int(rank_budget),
                r_min=r_min,
                r_max=r_max,
                use_cost_aware=True,
                budget_mode=self.config.get("budget", {}).get("mode", "equal_trainable_params"),
                warning_threshold=float(self.config.get("budget", {}).get("warning_threshold", 0.01)),
            )
            return replace(
                result,
                diagnostics={
                    "solver": "proportional_rounding",
                    "solver_note": (
                        "Ablation baseline: converts each module's final cumulative utility "
                        "to a continuous rank share, then applies rounding and budget repair."
                    ),
                },
            )

        allocation = solve_curves(utility_curves)
        if cpu_reference_enabled:
            cpu_utility_curves = build_type_scaled_utility_curves(
                cpu_raw_curves,
                module_types,
                type_scaling=type_scaling,
                log_compression=log_compression,
                eps=eps,
            )
            cpu_allocation = solve_curves(cpu_utility_curves)
            selections_match = cpu_selected_by_module == selected_by_module
            allocations_match = cpu_allocation.allocation == allocation.allocation
            gains_match = all(
                torch.allclose(
                    torch.tensor(cpu_raw_curves[name]),
                    torch.tensor(raw_curves[name]),
                    atol=1e-6,
                    rtol=1e-5,
                )
                for name in self.module_names
            )
            if not (selections_match and allocations_match and gains_match):
                cpu_reference_fallback = True
                raw_curves = cpu_raw_curves
                utility_curves = cpu_utility_curves
                selected_by_module = cpu_selected_by_module
                marginal_by_module = cpu_marginal_by_module
                init_utility_by_module = cpu_marginal_by_module
                trace_by_module = cpu_trace_by_module
                allocation = cpu_allocation
        rank_override = self.pre_cfg.get("rank_override")
        allocation_before_rank_override = dict(allocation.allocation)
        if rank_override in {"uniform_ref", "uniform_rank"}:
            uniform_rank = int(self.config.get("rank", 1))
            if uniform_rank < r_min or uniform_rank > r_max:
                raise ValueError(
                    f"rank_override={rank_override} requires rank within [r_min, r_max]; "
                    f"rank={uniform_rank}, r_min={r_min}, r_max={r_max}."
                )
            uniform_allocation = {name: uniform_rank for name in self.module_names}
            actual_budget = compute_total_lora_params(uniform_allocation, self.module_dims)
            target_rank_sum = int(self.config.get("rank", 1)) * len(self.module_names)
            allocation = replace(
                allocation,
                allocation=uniform_allocation,
                budget=BudgetInfo(
                    budget_mode=self.config.get("budget", {}).get("mode", "equal_trainable_params"),
                    target_budget=int(rank_budget),
                    actual_budget=actual_budget,
                    budget_error=actual_budget - int(rank_budget),
                    budget_error_ratio=float((actual_budget - int(rank_budget)) / int(rank_budget))
                    if int(rank_budget)
                    else 0.0,
                    total_active_rank=sum(uniform_allocation.values()),
                    target_budget_ranksum=target_rank_sum,
                    budget_ratio_ranksum=float(sum(uniform_allocation.values()) / target_rank_sum)
                    if target_rank_sum
                    else 0.0,
                    over_budget=actual_budget > int(rank_budget),
                    warning="rank_override_uniform_ref_over_budget" if actual_budget > int(rank_budget) else None,
                ),
                diagnostics={
                    **(allocation.diagnostics or {}),
                    "rank_override": "uniform_ref",
                    "allocation_before_rank_override": allocation_before_rank_override,
                },
            )
        elif rank_override not in {None, "none"}:
            raise ValueError(
                f"Unsupported preallocation.rank_override={rank_override!r}; "
                "supported values are 'uniform_ref' and 'none'."
            )

        module_logs = []
        for module_name in self.module_names:
            final_rank = int(allocation.allocation[module_name])
            module_logs.append(
                {
                    "module_name": module_name,
                    "module_utility": float(utility_curves[module_name][final_rank]),
                    "rank_cost": module_rank_cost(self.module_dims[module_name]),
                    "final_rank": final_rank,
                    "selected_atom_count": len(selected_by_module[module_name]),
                    "selected_evidence_count": len(selected_by_module[module_name]),
                    "selected_atom_utilities": marginal_by_module[module_name][:final_rank],
                    "selected_atom_indices": selected_by_module[module_name][:final_rank],
                    "rank_beyond_selected_evidence": max(0, final_rank - len(selected_by_module[module_name])),
                    "final_parameter_count": final_rank * module_rank_cost(self.module_dims[module_name]),
                    "allocation_method": allocation_method,
                }
            )

        direction_bank_path = save_selected_atom_direction_bank(
            self._direction_bank_path(),
            atoms_by_module_index=atoms_by_module_index,
            selected_indices_by_module=selected_by_module,
            selected_utilities_by_module=init_utility_by_module,
            rank_allocation=allocation.allocation,
        )
        diagnostics = {
            **dict(base_diagnostics),
            **(allocation.diagnostics or {}),
            "allocation_method": allocation_method,
            "utility_builder": utility_builder,
            "r_min": r_min,
            "r_max": r_max,
            "rho": rho,
            "sign_split": sign_split,
            "type_scaling": type_scaling,
            "log_compression": log_compression,
            "initialization_selection_builder": "independent"
            if allocation_method == "covra_module_scalar"
            else utility_builder,
            "covra_trace": trace_by_module,
            "raw_utility_curves": raw_curves,
            "scaled_utility_curves": utility_curves,
            "selected_atom_indices": selected_by_module,
            "initialization_selected_atom_utilities": init_utility_by_module,
            "direction_bank_path": str(direction_bank_path),
            "cpu_reference_checked": bool(cpu_reference_enabled),
            "cpu_reference_fallback": bool(cpu_reference_fallback),
        }
        if allocation_method == "covra_module_scalar":
            diagnostics.update(self._module_scalar_template_metadata(r_max))
        return WeightedAllocationResult(
            allocation=allocation.allocation,
            budget=allocation.budget,
            module_logs=module_logs,
            diagnostics=diagnostics,
        )

    def _allocate_from_svd_atoms(
        self,
        atoms: list[Any],
        rank_budget: int,
        diagnostics: Mapping[str, Any],
    ) -> WeightedAllocationResult:
        allocation_method = str(self.pre_cfg.get("allocation_method", "covra_v05"))
        if allocation_method in {"covra_full", "covra_independent", "covra_module_scalar"}:
            return self._allocate_final_covra_from_svd_atoms(atoms, int(rank_budget), diagnostics)
        if allocation_method in {"covra_v05", "dico_v03"} and self.config.get("method") in {
            "dico_cd",
            "dico_cd_da",
        }:
            return self._allocate_v03_from_svd_atoms(atoms, int(rank_budget), diagnostics)
        raise ValueError(
            "Unsupported DiCo v0.3 preallocation method: "
            f"method={self.config.get('method')!r}, allocation_method={allocation_method!r}"
        )

    def _allocate_svd(self, rank_budget: int) -> PreallocationResult:
        self.atom_mode = "svd"
        configured_groups = self.config.get("data", {}).get("group_labels")
        atoms, diagnostics = extract_svd_atom_records(
            self.model,
            self.module_names,
            self.module_dims,
            list(self._calibration_batches or []),
            self.pre_cfg,
            rank=int(self.config.get("rank", 1)),
            profile_path=self._profile_path(),
            group_labels=[str(v) for v in configured_groups] if configured_groups else None,
        )
        self._svd_diagnostics = diagnostics
        atom_logs = [atom.to_log_dict() for atom in atoms]
        rank = int(self.config.get("rank", 1))
        r_max = self._r_max()
        allocation = self._allocate_from_svd_atoms(atoms, int(rank_budget), diagnostics)

        module_log_by_name = {row["module_name"]: row for row in allocation.module_logs}
        module_logs = []
        rank_histogram: dict[str, int] = {}
        for module_name in self.module_names:
            row = dict(module_log_by_name[module_name])
            row["aggregation_mode"] = self.aggregation_mode
            row["atom_weight_normalization"] = self.atom_weight_normalization
            row["use_cost_aware_allocation"] = self.use_cost_aware_allocation
            row["atom_mode"] = self.atom_mode
            row["profile_norm_mode"] = diagnostics.get("profile_norm_mode")
            row["top_selected_atom_utilities"] = sorted(row.get("selected_atom_utilities", []), reverse=True)[:5]
            module_logs.append(row)
            rank_histogram[str(row["final_rank"])] = rank_histogram.get(str(row["final_rank"]), 0) + 1

        final_budget = allocation.budget.actual_budget
        relaxation = _evidence_relaxation_diagnostics(module_logs)
        full_diagnostics = {
            **diagnostics,
            "allocation_method": (allocation.diagnostics or {}).get("allocation_method", "directional_budgeted"),
            **(allocation.diagnostics or {}),
            "num_modules": len(self.module_names),
            "total_params": final_budget,
            "budget_ref": int(rank_budget),
            "budget_target": int(rank_budget),
            "budget_final": final_budget,
            "budget_ratio": float(final_budget / rank_budget) if rank_budget else 0.0,
            "eta": float(self.pre_cfg.get("eta", 0.98)),
            "allow_rank_beyond_selected_evidence": bool(
                self.pre_cfg.get("allow_rank_beyond_selected_evidence", True)
            ),
            "use_soft_tail": bool(self.pre_cfg.get("use_soft_tail", True)),
            "use_cost_aware_allocation": self.use_cost_aware_allocation,
            "rank_histogram": rank_histogram,
            "zero_rank_modules": [name for name, value in allocation.allocation.items() if int(value) == 0],
            **relaxation,
            "relaxation_rank_ratio": float(
                relaxation["rank_beyond_selected_evidence_total"] / relaxation["final_total_rank"]
            )
            if relaxation["final_total_rank"]
            else 0.0,
        }
        return PreallocationResult(
            rank_allocation=allocation.allocation,
            module_scores={
                name: float(row.get("module_utility", 0.0) or 0.0)
                for name, row in module_log_by_name.items()
            },
            atom_logs=atom_logs,
            module_logs=module_logs,
            budget=allocation.budget,
            atom_mode=self.atom_mode,
            aggregation_mode=self.aggregation_mode,
            weighted_topk_k=self.pre_cfg.get("weighted_topk_k", "auto"),
            atom_weight_normalization=self.atom_weight_normalization,
            use_cost_aware_allocation=self.use_cost_aware_allocation,
            module_names=list(self.module_names),
            module_dims={name: {key: int(value) for key, value in dims.items()} for name, dims in self.module_dims.items()},
            cache_context=build_preallocation_cache_context(self.config, self.module_names, self.module_dims),
            allocation_method="directional_budgeted",
            profile_norm_mode=diagnostics.get("profile_norm_mode"),
            diagnostics=full_diagnostics,
        )

    def _allocate_random_at_budget(self, rank_budget: int) -> PreallocationResult:
        seed = int(self.pre_cfg.get("sketch_seed", self.config.get("seed", 42)))
        rng = random.Random(seed)
        rank = int(self.config.get("rank", 1))
        r_min = self._r_min()
        r_max = self._r_max()
        allocation = {name: r_min for name in self.module_names}
        costs = {name: module_rank_cost(self.module_dims[name]) for name in self.module_names}

        def total() -> int:
            return compute_total_lora_params(allocation, self.module_dims)

        while True:
            actual = total()
            candidates = [
                name
                for name in self.module_names
                if allocation[name] < r_max and actual + costs[name] <= int(rank_budget)
            ]
            if not candidates:
                break
            allocation[rng.choice(candidates)] += 1

        repaired = repair_allocation_to_budget(
            allocation,
            int(rank_budget),
            self.module_dims,
            r_min=r_min,
            r_max=r_max,
            budget_mode=self.config.get("budget", {}).get("mode", "equal_trainable_params"),
            warning_threshold=float(self.config.get("budget", {}).get("warning_threshold", 0.01)),
        )
        allocation = repaired.allocation
        module_logs = []
        for name in self.module_names:
            final_rank = int(allocation[name])
            final_budget = final_rank * costs[name]
            module_logs.append(
                {
                    "module_name": name,
                    "module_utility": None,
                    "rank_cost": costs[name],
                    "cost_aware_score": None,
                    "continuous_rank": None,
                    "r_tilde": None,
                    "final_rank": final_rank,
                    "selected_evidence_count": 0,
                    "selected_atom_count": 0,
                    "rank_beyond_selected_evidence": final_rank,
                    "final_budget": final_budget,
                    "final_parameter_count": final_budget,
                    "allocation_method": "random_at_budget",
                    "random_seed": seed,
                }
            )
        final_budget = repaired.budget.actual_budget
        return PreallocationResult(
            rank_allocation=allocation,
            module_scores={name: 0.0 for name in self.module_names},
            atom_logs=[],
            module_logs=module_logs,
            budget=repaired.budget,
            atom_mode=self.atom_mode,
            aggregation_mode=self.aggregation_mode,
            weighted_topk_k=self.pre_cfg.get("weighted_topk_k", "auto"),
            atom_weight_normalization=self.atom_weight_normalization,
            use_cost_aware_allocation=self.use_cost_aware_allocation,
            module_names=list(self.module_names),
            module_dims={name: {key: int(value) for key, value in dims.items()} for name, dims in self.module_dims.items()},
            cache_context=build_preallocation_cache_context(self.config, self.module_names, self.module_dims),
            atom_mode_limitation=MODULE_PROXY_LIMITATION if self.atom_mode == "module_proxy" else None,
            allocation_method="random_at_budget",
            profile_norm_mode=self.pre_cfg.get("profile_norm_mode"),
            diagnostics={
                "allocation_method": "random_at_budget",
                "random_seed": seed,
                "profile_norm_mode": self.pre_cfg.get("profile_norm_mode"),
                "num_modules": len(self.module_names),
                "num_atoms": 0,
                "num_selected_atoms": 0,
                "total_params": final_budget,
                "budget_ref": int(rank_budget),
                "budget_target": int(rank_budget),
                "budget_final": final_budget,
                "budget_ratio": float(final_budget / rank_budget) if rank_budget else 0.0,
                "eta": float(self.pre_cfg.get("eta", 0.98)),
                "allow_rank_beyond_selected_evidence": bool(
                    self.pre_cfg.get("allow_rank_beyond_selected_evidence", True)
                ),
                "use_soft_tail": bool(self.pre_cfg.get("use_soft_tail", True)),
                "use_cost_aware_allocation": self.use_cost_aware_allocation,
                **_evidence_relaxation_diagnostics(module_logs),
            },
        )

    def allocate(self, rank_budget: int) -> PreallocationResult:
        if self.pre_cfg.get("allocation_method") == "random_at_budget":
            return self._allocate_random_at_budget(rank_budget)
        if self.requested_atom_mode == "svd" and self.model is not None and self._calibration_batches:
            return self._allocate_svd(rank_budget)
        if self.requested_atom_mode == "svd":
            self.atom_mode = self.pre_cfg.get("fallback_atom_mode", "module_proxy")
        if not self.module_scores:
            self.module_scores = {name: 1.0 for name in self.module_names}

        atom_records = self.build_atom_records()
        atom_utilities = self.compute_atom_utilities(atom_records)
        atom_utilities = self.normalize_atom_utilities(atom_utilities, self.atom_weight_normalization)
        module_utilities = self.aggregate_module_utilities(atom_utilities, self.aggregation_mode)
        allocation = self.allocate_from_module_utilities(module_utilities, rank_budget)

        module_logs = []
        module_log_by_name = {row["module_name"]: row for row in allocation.module_logs}
        selected_counts = {name: 0 for name in self.module_names}
        for atom in atom_utilities:
            if atom.selected:
                selected_counts[atom.module_name] = selected_counts.get(atom.module_name, 0) + 1
        for module_name in self.module_names:
            row = dict(module_log_by_name[module_name])
            final_rank = int(row.get("final_rank", 0))
            selected_count = int(selected_counts.get(module_name, 0))
            final_budget = final_rank * module_rank_cost(self.module_dims[module_name])
            row["aggregation_mode"] = self.aggregation_mode
            row["atom_weight_normalization"] = self.atom_weight_normalization
            row["use_cost_aware_allocation"] = self.use_cost_aware_allocation
            row["atom_mode"] = self.atom_mode
            row["r_tilde"] = row.get("r_tilde", row.get("continuous_rank", 0.0))
            row["selected_evidence_count"] = selected_count
            row["selected_atom_count"] = selected_count
            row["rank_beyond_selected_evidence"] = max(0, final_rank - selected_count)
            row["final_budget"] = final_budget
            row["final_parameter_count"] = final_budget
            if self.atom_mode == "module_proxy":
                row["atom_mode_limitation"] = MODULE_PROXY_LIMITATION
            module_logs.append(row)

        atom_logs = [asdict(atom) for atom in atom_utilities]
        final_budget = allocation.budget.actual_budget
        return PreallocationResult(
            rank_allocation=allocation.allocation,
            module_scores=dict(self.module_scores),
            atom_logs=atom_logs,
            module_logs=module_logs,
            budget=allocation.budget,
            atom_mode=self.atom_mode,
            aggregation_mode=self.aggregation_mode,
            weighted_topk_k=self.pre_cfg.get("weighted_topk_k", "auto"),
            atom_weight_normalization=self.atom_weight_normalization,
            use_cost_aware_allocation=self.use_cost_aware_allocation,
            module_names=list(self.module_names),
            module_dims={name: {key: int(value) for key, value in dims.items()} for name, dims in self.module_dims.items()},
            cache_context=build_preallocation_cache_context(self.config, self.module_names, self.module_dims),
            atom_mode_limitation=MODULE_PROXY_LIMITATION if self.atom_mode == "module_proxy" else None,
            allocation_method=self.pre_cfg.get("allocation_method"),
            profile_norm_mode=self.pre_cfg.get("profile_norm_mode"),
            diagnostics={
                "allocation_method": self.pre_cfg.get("allocation_method"),
                "profile_norm_mode": self.pre_cfg.get("profile_norm_mode"),
                "num_modules": len(self.module_names),
                "num_atoms": len(atom_logs),
                "num_selected_atoms": sum(1 for row in atom_logs if row.get("selected")),
                "total_params": final_budget,
                "budget_ref": int(rank_budget),
                "budget_target": int(rank_budget),
                "budget_final": final_budget,
                "budget_ratio": float(final_budget / rank_budget) if rank_budget else 0.0,
                "eta": float(self.pre_cfg.get("eta", 0.98)),
                "allow_rank_beyond_selected_evidence": bool(
                    self.pre_cfg.get("allow_rank_beyond_selected_evidence", True)
                ),
                "use_soft_tail": bool(self.pre_cfg.get("use_soft_tail", True)),
                "use_cost_aware_allocation": self.use_cost_aware_allocation,
                **_evidence_relaxation_diagnostics(module_logs),
            },
        )

    def save(self, path: Path | str, result: PreallocationResult) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        payload = result.to_dict(preallocation_path=str(path))
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        for atom_log_path in [path.parent / "atom_logs.jsonl", path.with_name(path.stem + "_atom_logs.jsonl")]:
            with atom_log_path.open("w", encoding="utf-8") as handle:
                for row in result.atom_logs:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_preallocation(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_direction_bank(
    path: Path | str,
    certified: Sequence[PhysicalCandidate],
    reserve: Sequence[PhysicalCandidate],
    purchased_directions: Mapping[str, Sequence[str]],
    normalized_utility: Mapping[str, float],
) -> Path:
    """Persist the full-dimensional input directions behind each *actually purchased*
    rank slot.

    Rank allocation and diagnostics are JSON-serializable and travel through
    ``PreallocationResult``/``load_preallocation``, but the direction vectors
    needed for direction-anchored initialization (3.5节) are Tensors and must be
    saved to a separate sidecar so trainer.py can rebuild {A0, B0} without
    recomputing calibration.

    Only includes directions ``purchased_directions`` (procurement.py's actual
    purchase record) ties to a granted rank slot -- not every certified/reserve
    candidate that existed, which could include far more directions than a module's
    final rank_dict[m] and would let init.py anchor to directions that were never
    actually budgeted for. ``utility`` is the normalized w_bar_p (``normalized_utility``,
    comparable across certified and reserve on the same scale), not the raw
    unnormalized ``merged_utility``/``raw_energy`` fields on the candidate objects
    themselves.
    """
    path = Path(path)
    ensure_dir(path.parent)
    by_id: dict[str, PhysicalCandidate] = {c.physical_direction_id: c for c in certified}
    by_id.update({c.physical_direction_id: c for c in reserve})
    certified_ids = {c.physical_direction_id for c in certified}
    bank: dict[str, list[dict[str, Any]]] = {}
    for module_name, physical_ids in purchased_directions.items():
        rows: list[dict[str, Any]] = []
        for physical_id in physical_ids:
            candidate = by_id.get(physical_id)
            if candidate is None or candidate.full_v is None:
                continue
            rows.append(
                {
                    "v": candidate.full_v.detach().float().cpu(),
                    "utility": float(normalized_utility.get(physical_id, 0.0)),
                    # "relaxation" (not "reserve") matches init.py's existing
                    # certified_rows-vs-relaxation_rows distinction.
                    "source": "certified" if physical_id in certified_ids else "relaxation",
                }
            )
        if rows:
            bank[module_name] = rows
    torch.save(bank, path)
    return path


def save_selected_atom_direction_bank(
    path: Path | str,
    *,
    atoms_by_module_index: Mapping[str, Mapping[int, Any]],
    selected_indices_by_module: Mapping[str, Sequence[int]],
    selected_utilities_by_module: Mapping[str, Sequence[float]],
    rank_allocation: Mapping[str, int],
) -> Path:
    """Persist final CovRA-selected SVD input directions for anchored init.

    The final CovRA path allocates integer ranks directly from utility curves
    rather than from the legacy procurement object.  This helper emits the same
    trainer-facing sidecar schema as :func:`save_direction_bank`, but only for
    the first ``rank_allocation[module]`` physical candidates selected by the
    final CovRA utility builder.
    """

    path = Path(path)
    ensure_dir(path.parent)
    bank: dict[str, list[dict[str, Any]]] = {}
    for module_name, rank_value in rank_allocation.items():
        final_rank = int(rank_value)
        if final_rank <= 0:
            continue
        module_atoms = atoms_by_module_index.get(module_name, {})
        selected_indices = list(selected_indices_by_module.get(module_name, []))[:final_rank]
        selected_utilities = [float(value) for value in selected_utilities_by_module.get(module_name, [])]
        rows: list[dict[str, Any]] = []
        for offset, atom_index in enumerate(selected_indices):
            atom = module_atoms.get(int(atom_index))
            if atom is None:
                continue
            direction = getattr(atom, "v", None)
            if direction is None:
                continue
            utility = selected_utilities[offset] if offset < len(selected_utilities) else 0.0
            rows.append(
                {
                    "atom_index": int(atom_index),
                    "v": torch.as_tensor(direction).detach().float().cpu(),
                    "utility": float(utility),
                    "source": "certified",
                }
            )
        if rows:
            bank[module_name] = rows
    torch.save(bank, path)
    return path


def load_direction_bank(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    return torch.load(Path(path), weights_only=False)
