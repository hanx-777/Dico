from __future__ import annotations

import gc
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from dico_rank.atom_svd import (
    aggregate_selected_module_utilities,
    extract_svd_atom_records,
)
from dico_rank.rank_budget import (
    BudgetInfo,
    allocate_by_rank_allocator,
    allocate_by_weighted_utility,
    compute_total_lora_params,
    module_rank_cost,
    repair_allocation_to_budget,
)
from dico_rank.utils import ensure_dir


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
    evidence_cfg = pre_cfg.get("evidence_selection", {})
    return {
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
        "calibration_num_samples": calibration_cfg.get("num_samples"),
        "calibration_seed": calibration_cfg.get("seed", config.get("seed")),
        "calibration_batch_size": calibration_cfg.get("batch_size"),
        "calibration_shuffle": calibration_cfg.get("shuffle"),
        "preallocation": {
            "atom_mode": pre_cfg.get("atom_mode", pre_cfg.get("fallback_atom_mode", "module_proxy")),
            "allocation_method": pre_cfg.get("allocation_method", "weighted"),
            "rank_allocator": pre_cfg.get("rank_allocator"),
            "eta": pre_cfg.get("eta", 0.98),
            "allow_rank_beyond_selected_evidence": pre_cfg.get("allow_rank_beyond_selected_evidence", True),
            "use_soft_tail": pre_cfg.get("use_soft_tail", True),
            "use_cost_aware_allocation": pre_cfg.get("use_cost_aware_allocation", True),
            "top_k_atoms": pre_cfg.get("top_k_atoms"),
            "sketch_dim": pre_cfg.get("sketch_dim"),
            "sketch_seed": pre_cfg.get("sketch_seed"),
            "sketch_dtype": pre_cfg.get("sketch_dtype"),
            "answer_only": pre_cfg.get("answer_only"),
            "profile_norm_mode": pre_cfg.get("profile_norm_mode"),
            "beta": pre_cfg.get("beta"),
            "gamma": pre_cfg.get("gamma"),
            "delta": pre_cfg.get("delta"),
            "epsilon_cov": pre_cfg.get("epsilon_cov"),
            "aggregation_mode": pre_cfg.get("aggregation_mode", "weighted_topk"),
            "evidence_selection.max_selected_atoms": evidence_cfg.get("max_selected_atoms", "auto"),
            "evidence_selection.coverage_stop_threshold": evidence_cfg.get("coverage_stop_threshold"),
            "r_min_multiplier": pre_cfg.get("r_min_multiplier", 0.0),
            "r_max_multiplier": pre_cfg.get("r_max_multiplier", 2),
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

    def _r_max(self) -> int:
        return int(self.config.get("rank", 1) * self.pre_cfg.get("r_max_multiplier", 2))

    def _weighted_topk_k(self) -> int | str:
        value = self.pre_cfg.get("weighted_topk_k", "auto")
        if value == "auto":
            return self._r_max()
        return int(value)

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
        r_min = max(0, int(rank * float(self.pre_cfg.get("r_min_multiplier", 0.0))))
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
        return save_dir_path / f"dico_pre_rank{rank}_seed{seed}_profiles.pt"

    def _evidence_selected_utilities(self, atom_records: list[Any]) -> dict[str, list[float]]:
        values = {name: [] for name in self.module_names}
        for atom in atom_records:
            if atom.selected:
                values.setdefault(atom.module_name, []).append((atom.atom_index, float(atom.utility)))
        return {
            name: [utility for _idx, utility in sorted(rows, key=lambda item: item[0])]
            for name, rows in values.items()
        }

    def _allocate_svd(self, rank_budget: int) -> PreallocationResult:
        self.atom_mode = "svd"
        atoms, diagnostics = extract_svd_atom_records(
            self.model,
            self.module_names,
            self.module_dims,
            list(self._calibration_batches or []),
            self.pre_cfg,
            rank=int(self.config.get("rank", 1)),
            profile_path=self._profile_path(),
        )
        self._svd_diagnostics = diagnostics
        atom_logs = [atom.to_log_dict() for atom in atoms]
        rank = int(self.config.get("rank", 1))
        r_max = self._r_max()
        allocation = allocate_by_rank_allocator(
            atom_logs=atom_logs,
            module_dims=self.module_dims,
            target_budget=int(rank_budget),
            eta=float(self.pre_cfg.get("eta", 0.98)),
            r_min=max(0, int(rank * float(self.pre_cfg.get("r_min_multiplier", 0.0)))),
            r_max=r_max,
            config=self.pre_cfg.get("rank_allocator"),
            allow_rank_beyond_selected_evidence=bool(self.pre_cfg.get("allow_rank_beyond_selected_evidence", True)),
            budget_mode=self.config.get("budget", {}).get("mode", "equal_trainable_params"),
            warning_threshold=float(self.config.get("budget", {}).get("warning_threshold", 0.01)),
        )

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
            "allocation_method": "directional_budgeted",
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
        r_min = max(0, int(rank * float(self.pre_cfg.get("r_min_multiplier", 0.0))))
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
