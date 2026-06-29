from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from dico_rank.rank_budget import (
    BudgetInfo,
    allocate_by_weighted_utility,
    module_rank_cost,
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
            "module_names": self.module_names,
            "module_dims": self.module_dims,
            "cache_context": self.cache_context,
            "preallocation_source": self.preallocation_source,
            "preallocation_path": preallocation_path,
            "target_budget": self.budget.target_budget,
            "actual_budget": self.budget.actual_budget,
            "budget_error": self.budget.budget_error,
            "budget_error_ratio": self.budget.budget_error_ratio,
        }
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
        self.atom_mode = self.pre_cfg.get("fallback_atom_mode", "module_proxy")
        self.aggregation_mode = self.pre_cfg.get("aggregation_mode", "weighted_topk")
        self.atom_weight_normalization = self.pre_cfg.get("atom_weight_normalization", "none")
        self.use_cost_aware_allocation = bool(self.pre_cfg.get("use_cost_aware_allocation", True))

    def _r_max(self) -> int:
        return int(self.config.get("rank", 1) * self.pre_cfg.get("r_max_multiplier", 2))

    def _weighted_topk_k(self) -> int | str:
        value = self.pre_cfg.get("weighted_topk_k", "auto")
        if value == "auto":
            return self._r_max()
        return int(value)

    def collect_calibration_statistics(self, calibration_dataloader: Any) -> dict[str, float]:
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
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)

        denom = max(1, count)
        self.module_scores = {name: value / denom for name, value in scores.items()}
        return self.module_scores

    def build_atom_records(self) -> list[dict[str, Any]]:
        """Build proxy atom records from module-level scores.

        True SVD atoms can be added later by replacing this record builder while
        preserving the utility/aggregation/allocation pipeline below.
        """

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
        r_min = int(self.pre_cfg.get("r_min", 0))
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

    def allocate(self, rank_budget: int) -> PreallocationResult:
        if not self.module_scores:
            self.module_scores = {name: 1.0 for name in self.module_names}

        atom_records = self.build_atom_records()
        atom_utilities = self.compute_atom_utilities(atom_records)
        atom_utilities = self.normalize_atom_utilities(atom_utilities, self.atom_weight_normalization)
        module_utilities = self.aggregate_module_utilities(atom_utilities, self.aggregation_mode)
        allocation = self.allocate_from_module_utilities(module_utilities, rank_budget)

        module_logs = []
        module_log_by_name = {row["module_name"]: row for row in allocation.module_logs}
        for module_name in self.module_names:
            row = dict(module_log_by_name[module_name])
            row["aggregation_mode"] = self.aggregation_mode
            row["atom_weight_normalization"] = self.atom_weight_normalization
            row["use_cost_aware_allocation"] = self.use_cost_aware_allocation
            row["atom_mode"] = self.atom_mode
            if self.atom_mode == "module_proxy":
                row["atom_mode_limitation"] = MODULE_PROXY_LIMITATION
            module_logs.append(row)

        atom_logs = [asdict(atom) for atom in atom_utilities]
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
