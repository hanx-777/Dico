from __future__ import annotations

import gc
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from dico_rank.rank_budget import module_rank_cost
from dico_rank.sketch import append_gram_schmidt, make_random_projection, orthonormal_basis


LOGGER = logging.getLogger(__name__)


@dataclass
class SvdAtomRecord:
    module_name: str
    atom_index: int
    cost: int
    singular_value: float
    spectral_ratio: float
    profile: torch.Tensor | None
    conflict: float
    coverage: float
    lambda_cov: float
    utility: float
    module_importance: float
    selected: bool = False
    u: torch.Tensor | None = None
    v: torch.Tensor | None = None
    atom_mode: str = "svd"
    prefix_legal: bool = False
    importance_mode: str = "streaming_estimate"
    profile_norm_mode: str = "streaming_estimate"
    profile_path: str | None = None
    profile_index: int | None = None
    profile_norm: float | None = None
    profile_mean: float | None = None
    profile_std: float | None = None
    profile_hash: str | None = None

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "atom_id": self.atom_index,
            "atom_index": self.atom_index,
            "cost": self.cost,
            "singular_value": self.singular_value,
            "spectral_ratio": self.spectral_ratio,
            "conflict": self.conflict,
            "coverage": self.coverage,
            "lambda_cov": self.lambda_cov,
            "utility": self.utility,
            "module_importance": self.module_importance,
            "selected": self.selected,
            "atom_mode": self.atom_mode,
            "prefix_legal": self.prefix_legal,
            "importance_mode": self.importance_mode,
            "profile_norm_mode": self.profile_norm_mode,
            "profile_path": self.profile_path,
            "profile_index": self.profile_index,
            "profile_norm": self.profile_norm,
            "profile_mean": self.profile_mean,
            "profile_std": self.profile_std,
            "profile_hash": self.profile_hash,
        }


@dataclass
class TokenFactorBatch:
    samples: list[tuple[torch.Tensor, torch.Tensor]]
    activation_dim: int
    grad_dim: int

    @property
    def batch_size(self) -> int:
        return len(self.samples)

    def token_slices(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= int(sample_idx) < self.batch_size:
            raise IndexError(f"sample_idx out of range: {sample_idx}")
        return self.samples[int(sample_idx)]


def _as_float_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu()


def extract_svd_atoms_from_response_matrix(
    response: torch.Tensor,
    top_k: int,
    module_name: str,
    cost: int,
    eps: float = 1e-12,
) -> list[SvdAtomRecord]:
    response = response.float()
    u, singular_values, vh = torch.linalg.svd(response, full_matrices=False)
    k = min(int(top_k), int(singular_values.numel()))
    denom = float(torch.sum(singular_values[:k] ** 2).item()) + eps
    atoms: list[SvdAtomRecord] = []
    for idx in range(k):
        sigma = float(singular_values[idx].item())
        atoms.append(
            SvdAtomRecord(
                module_name=module_name,
                atom_index=idx,
                cost=int(cost),
                singular_value=sigma,
                spectral_ratio=float((singular_values[idx] ** 2).item() / denom),
                profile=None,
                conflict=0.0,
                coverage=0.0,
                lambda_cov=1.0,
                utility=0.0,
                module_importance=1.0,
                u=_as_float_tensor(u[:, idx]),
                v=_as_float_tensor(vh[idx, :]),
            )
        )
    return atoms


def signed_projection_from_token_factors(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    if activations.numel() == 0 or gradients.numel() == 0:
        return torch.tensor(0.0)
    left = gradients.float() @ u.float()
    right = activations.float() @ v.float()
    return torch.mean(left * right)


def sample_response_norm_from_token_factors(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    mode: str = "streaming_estimate",
) -> torch.Tensor:
    if activations.numel() == 0 or gradients.numel() == 0:
        return torch.tensor(0.0)
    activations = activations.float()
    gradients = gradients.float()
    if mode == "none":
        return torch.tensor(1.0)
    if mode == "exact_small":
        response = gradients.T @ activations / max(activations.shape[0], 1)
        return torch.linalg.norm(response)
    if mode == "streaming_estimate":
        token_norm_sq = torch.sum(activations * activations, dim=-1) * torch.sum(gradients * gradients, dim=-1)
        return torch.sqrt(torch.sum(token_norm_sq).clamp_min(0.0)) / max(activations.shape[0], 1)
    raise ValueError(f"Unsupported profile_norm_mode: {mode}")


def gradient_conflict(alpha: torch.Tensor) -> float:
    alpha = alpha.detach().flatten()
    total = max(int(alpha.numel()), 1)
    p_pos = float((alpha > 0).sum().item()) / total
    p_neg = float((alpha < 0).sum().item()) / total
    return max(0.0, min(1.0, 4.0 * p_pos * p_neg))


def normalize_signed_profiles(
    alpha: torch.Tensor,
    denominators: torch.Tensor | None,
    mode: str = "streaming_estimate",
    eps: float = 1e-12,
) -> torch.Tensor:
    profiles = alpha.float()
    if mode not in {"exact_small", "streaming_estimate", "none"}:
        raise ValueError(f"Unsupported profile_norm_mode: {mode}")
    if mode != "none":
        if denominators is None:
            raise ValueError("denominators are required unless profile_norm_mode=none")
        profiles = profiles / (denominators.float().reshape(-1, 1) + eps)
    centered = profiles - profiles.mean(dim=0, keepdim=True)
    norms = torch.linalg.norm(centered, dim=0, keepdim=True)
    return centered / (norms + eps)


def coverage_residual(profile: torch.Tensor, basis: torch.Tensor) -> float:
    profile = profile.float()
    if basis.numel() == 0 or basis.shape[1] == 0:
        return float(torch.sum(profile * profile).item())
    residual = profile - basis @ (basis.T @ profile)
    return max(0.0, float(torch.sum(residual * residual).item()))


def atom_utility(
    atom: SvdAtomRecord,
    coverage: float,
    lambda_cov: float,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    floor: float = 0.0,
    cost_aware: bool = True,
) -> float:
    value = (
        max(atom.module_importance, 0.0) ** float(beta)
        * max(atom.spectral_ratio, 0.0) ** float(gamma)
        * max(1.0 - atom.conflict, 0.0) ** float(delta)
        * (float(lambda_cov) * float(coverage) + (1.0 - float(lambda_cov)))
    )
    if cost_aware:
        value /= max(int(atom.cost), 1)
    return max(float(floor), float(value))


def select_coverage_evidence(
    atoms: list[SvdAtomRecord],
    max_selected_atoms: int,
    epsilon_cov: float,
    sparse_stop_by_coverage: bool = True,
    coverage_stop_threshold: float | None = None,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    use_soft_tail: bool = True,
    atom_utility_floor: float = 0.0,
    cost_aware: bool = True,
) -> list[SvdAtomRecord]:
    for atom in atoms:
        atom.selected = False
        atom.prefix_legal = False
    if not atoms or max_selected_atoms <= 0:
        return []

    profile_dim = next((int(atom.profile.numel()) for atom in atoms if atom.profile is not None), 0)
    empty_basis = torch.empty(profile_dim, 0)
    bases: dict[str, torch.Tensor] = {}
    selected_counts: dict[str, int] = {}
    selected: list[SvdAtomRecord] = []
    stop_threshold = float(coverage_stop_threshold if coverage_stop_threshold is not None else epsilon_cov)

    while len(selected) < int(max_selected_atoms):
        candidates = [
            atom
            for atom in atoms
            if not atom.selected and atom.profile is not None and atom.atom_index == selected_counts.get(atom.module_name, 0)
        ]
        if not candidates:
            break
            
        coverages = {}
        for atom in candidates:
            mod_type = atom.module_name.split('.')[-1]
            basis = bases.get(mod_type, empty_basis)
            coverages[id(atom)] = coverage_residual(atom.profile, basis)
            
        max_cov = max(coverages.values()) if coverages else 0.0
        if sparse_stop_by_coverage and max_cov < stop_threshold:
            break
        lambda_cov = min(1.0, max_cov / max(float(epsilon_cov), 1e-12)) if use_soft_tail else 1.0
        for atom in candidates:
            cov = coverages[id(atom)]
            atom.coverage = cov
            atom.lambda_cov = lambda_cov
            atom.prefix_legal = True
            atom.utility = atom_utility(
                atom,
                cov,
                lambda_cov,
                beta=beta,
                gamma=gamma,
                delta=delta,
                floor=atom_utility_floor,
                cost_aware=cost_aware,
            )
        best = max(
            candidates,
            key=lambda atom: (atom.utility, atom.coverage, atom.spectral_ratio, -atom.cost, atom.module_name, -atom.atom_index),
        )
        best.selected = True
        selected.append(best)
        selected_counts[best.module_name] = selected_counts.get(best.module_name, 0) + 1
        
        best_mod_type = best.module_name.split('.')[-1]
        bases[best_mod_type] = append_gram_schmidt(bases.get(best_mod_type, empty_basis), best.profile)
    return selected


def aggregate_selected_module_utilities(
    atoms: list[SvdAtomRecord],
    module_names: list[str],
    aggregation_mode: str = "weighted_log",
) -> dict[str, float]:
    selected = [atom for atom in atoms if atom.selected]
    utilities: dict[str, float] = {name: 0.0 for name in module_names}
    for atom in selected:
        if aggregation_mode == "weighted_log":
            utilities[atom.module_name] = utilities.get(atom.module_name, 0.0) + math.log1p(max(atom.utility, 0.0))
        elif aggregation_mode == "weighted_sum":
            utilities[atom.module_name] = utilities.get(atom.module_name, 0.0) + max(atom.utility, 0.0)
        elif aggregation_mode == "count":
            utilities[atom.module_name] = utilities.get(atom.module_name, 0.0) + 1.0
        elif aggregation_mode == "weighted_topk":
            utilities[atom.module_name] = utilities.get(atom.module_name, 0.0) + max(atom.utility, 0.0)
        else:
            raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")
    return utilities


def _answer_mask(labels: torch.Tensor, seq_len: int, answer_only: bool) -> torch.Tensor:
    if not answer_only:
        return torch.ones(labels.shape[0], seq_len, dtype=torch.bool, device=labels.device)
    shifted = labels[:, 1:] != -100
    if shifted.shape[1] < seq_len:
        pad = torch.zeros(labels.shape[0], seq_len - shifted.shape[1], dtype=torch.bool, device=labels.device)
        shifted = torch.cat([shifted, pad], dim=1)
    return shifted[:, :seq_len]


def _release_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _module_chunks(module_names: list[str], module_chunk_size: int | None) -> list[list[str]]:
    if not module_names:
        return []
    chunk_size = int(module_chunk_size or len(module_names))
    if chunk_size <= 0:
        chunk_size = len(module_names)
    return [module_names[start : start + chunk_size] for start in range(0, len(module_names), chunk_size)]


def _filter_token_factors(
    activation: torch.Tensor,
    grad: torch.Tensor,
    mask: torch.Tensor,
) -> TokenFactorBatch:
    activation_dim = int(activation.shape[-1])
    grad_dim = int(grad.shape[-1])
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for sample_idx in range(int(activation.shape[0])):
        token_mask = mask[sample_idx].bool()
        if bool(token_mask.any()):
            a_tokens = activation[sample_idx, token_mask].detach().float().cpu()
            g_tokens = grad[sample_idx, token_mask].detach().float().cpu()
        else:
            a_tokens = torch.empty(0, activation_dim, dtype=torch.float32)
            g_tokens = torch.empty(0, grad_dim, dtype=torch.float32)
        samples.append((a_tokens, g_tokens))
    return TokenFactorBatch(samples=samples, activation_dim=activation_dim, grad_dim=grad_dim)


def _resolve_compute_device(
    pre_cfg: Mapping[str, Any],
    calibration_batches: list[Mapping[str, torch.Tensor]],
) -> torch.device:
    requested = str(pre_cfg.get("compute_device", "auto")).lower()
    if requested in {"auto", "default", ""}:
        if calibration_batches:
            device = calibration_batches[0]["input_ids"].device
            if device.type == "cuda" and torch.cuda.is_available():
                return device
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("preallocation.compute_device=cuda requested but CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _iter_masked_token_factors(
    activation: torch.Tensor,
    grad: torch.Tensor,
    labels: torch.Tensor | None,
    answer_only: bool,
    compute_device: torch.device,
):
    if labels is not None:
        mask = _answer_mask(labels, activation.shape[1], answer_only).to(device=activation.device)
    else:
        mask = torch.ones(activation.shape[:2], dtype=torch.bool, device=activation.device)
    for sample_idx in range(int(activation.shape[0])):
        token_mask = mask[sample_idx].bool()
        if bool(token_mask.any()):
            a_tokens = activation[sample_idx, token_mask].detach().to(device=compute_device, dtype=torch.float32)
            g_tokens = grad[sample_idx, token_mask].detach().to(device=compute_device, dtype=torch.float32)
        else:
            a_tokens = torch.empty(0, int(activation.shape[-1]), dtype=torch.float32, device=compute_device)
            g_tokens = torch.empty(0, int(grad.shape[-1]), dtype=torch.float32, device=compute_device)
        yield sample_idx, a_tokens, g_tokens


def _run_backward_and_collect(
    model: Any,
    modules: Mapping[str, Any],
    module_names: list[str],
    batch: Mapping[str, torch.Tensor],
    answer_only: bool,
    module_chunk_size: int | None = None,
    pass_name: str | None = None,
    batch_index: int | None = None,
    total_batches: int | None = None,
    progress_logging_steps: int = 1,
) -> dict[str, TokenFactorBatch]:
    chunks = _module_chunks(module_names, module_chunk_size)
    collected: dict[str, TokenFactorBatch] = {}
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for chunk_index, chunk_names in enumerate(chunks, start=1):
            records: dict[str, dict[str, torch.Tensor]] = {}
            handles = []

            def make_hook(name: str):
                def hook(_module, inputs, output):
                    if torch.is_tensor(output) and output.requires_grad:
                        output.retain_grad()
                        records[name] = {"activation": inputs[0].detach(), "output": output}

                return hook

            for name in chunk_names:
                handles.append(modules[name].register_forward_hook(make_hook(name)))
            outputs = None
            try:
                model.zero_grad(set_to_none=True)
                outputs = model(**batch)
                if getattr(outputs, "loss", None) is None:
                    continue
                outputs.loss.backward()
                labels = batch.get("labels")
                for name in chunk_names:
                    row = records.get(name)
                    if not row or row["output"].grad is None:
                        continue
                    activation = row["activation"]
                    grad = row["output"].grad
                    if labels is not None:
                        mask = _answer_mask(labels, activation.shape[1], answer_only).to(device=activation.device)
                    else:
                        mask = torch.ones(activation.shape[:2], dtype=torch.bool, device=activation.device)
                    collected[name] = _filter_token_factors(activation, grad, mask)
            finally:
                for handle in handles:
                    handle.remove()
                records.clear()
                if outputs is not None:
                    del outputs
                model.zero_grad(set_to_none=True)
                _release_torch_memory()

            interval = max(1, int(progress_logging_steps))
            should_log = (
                pass_name is not None
                and (
                    chunk_index == 1
                    or chunk_index == len(chunks)
                    or chunk_index % interval == 0
                )
            )
            if should_log:
                LOGGER.info(
                    "svd_preallocation_progress pass=%s batch=%s/%s chunk=%d/%d modules=%d",
                    pass_name,
                    batch_index if batch_index is not None else "?",
                    total_batches if total_batches is not None else "?",
                    chunk_index,
                    len(chunks),
                    len(chunk_names),
                )
        return collected
    finally:
        if was_training:
            model.train()


def _run_backward_and_stream(
    model: Any,
    modules: Mapping[str, Any],
    module_names: list[str],
    batch: Mapping[str, torch.Tensor],
    answer_only: bool,
    compute_device: torch.device,
    handle_sample: Any,
    module_chunk_size: int | None = None,
    pass_name: str | None = None,
    batch_index: int | None = None,
    total_batches: int | None = None,
    progress_logging_steps: int = 1,
) -> None:
    chunks = _module_chunks(module_names, module_chunk_size)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for chunk_index, chunk_names in enumerate(chunks, start=1):
            records: dict[str, dict[str, torch.Tensor]] = {}
            handles = []

            def make_hook(name: str):
                def hook(_module, inputs, output):
                    if torch.is_tensor(output) and output.requires_grad:
                        output.retain_grad()
                        records[name] = {"activation": inputs[0].detach(), "output": output}

                return hook

            for name in chunk_names:
                handles.append(modules[name].register_forward_hook(make_hook(name)))
            outputs = None
            try:
                model.zero_grad(set_to_none=True)
                outputs = model(**batch)
                if getattr(outputs, "loss", None) is None:
                    continue
                outputs.loss.backward()
                labels = batch.get("labels")
                for name in chunk_names:
                    row = records.get(name)
                    if not row or row["output"].grad is None:
                        continue
                    activation = row["activation"]
                    grad = row["output"].grad
                    for sample_idx, a_tokens, g_tokens in _iter_masked_token_factors(
                        activation,
                        grad,
                        labels,
                        answer_only,
                        compute_device,
                    ):
                        handle_sample(name, sample_idx, a_tokens, g_tokens)
            finally:
                for handle in handles:
                    handle.remove()
                records.clear()
                if outputs is not None:
                    del outputs
                model.zero_grad(set_to_none=True)
                _release_torch_memory()

            interval = max(1, int(progress_logging_steps))
            should_log = (
                pass_name is not None
                and (
                    chunk_index == 1
                    or chunk_index == len(chunks)
                    or chunk_index % interval == 0
                )
            )
            if should_log:
                LOGGER.info(
                    "svd_preallocation_progress pass=%s batch=%s/%s chunk=%d/%d modules=%d",
                    pass_name,
                    batch_index if batch_index is not None else "?",
                    total_batches if total_batches is not None else "?",
                    chunk_index,
                    len(chunks),
                    len(chunk_names),
                )
    finally:
        if was_training:
            model.train()


def _token_slices(factors: TokenFactorBatch, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return factors.token_slices(sample_idx)


def _profile_hash(profile: torch.Tensor) -> str:
    data = profile.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _resolve_max_selected(value: Any, rank: int, num_modules: int, top_k: int) -> int:
    max_possible = int(num_modules) * int(top_k)
    if value in {None, "auto"}:
        return min(int(rank) * int(num_modules), max_possible)
    return min(int(value), max_possible)


def extract_svd_atom_records(
    model: Any,
    module_names: list[str],
    module_dims: Mapping[str, Mapping[str, Any]],
    calibration_batches: list[Mapping[str, torch.Tensor]],
    pre_cfg: Mapping[str, Any],
    rank: int,
    profile_path: Path,
) -> tuple[list[SvdAtomRecord], dict[str, Any]]:
    modules = dict(model.named_modules())
    top_k = int(pre_cfg.get("top_k_atoms", pre_cfg.get("weighted_topk_k", rank)))
    oversample = int(pre_cfg.get("sketch_oversample", 0))
    sketch_dim = int(pre_cfg.get("sketch_dim", top_k + oversample))
    sketch_dim = max(top_k, sketch_dim)
    sketch_seed = int(pre_cfg.get("sketch_seed", 42))
    sketch_dtype = torch.float32
    answer_only = bool(pre_cfg.get("answer_only", True))
    profile_norm_mode = str(pre_cfg.get("profile_norm_mode", "streaming_estimate"))
    module_chunk_size = int(pre_cfg.get("module_chunk_size", len(module_names)))
    progress_logging_steps = int(pre_cfg.get("progress_logging_steps", 1))
    module_chunks = _module_chunks(module_names, module_chunk_size)
    compute_device = _resolve_compute_device(pre_cfg, calibration_batches)
    timings: dict[str, float] = {}

    omegas = {}
    y_states = {}
    for offset, name in enumerate(module_names):
        dims = module_dims[name]
        in_dim = int(dims["in_dim"])
        out_dim = int(dims["out_dim"])
        s = min(sketch_dim, in_dim)
        omegas[name] = make_random_projection(in_dim, s, sketch_seed + offset, dtype=sketch_dtype, device=compute_device)
        y_states[name] = torch.zeros(out_dim, s, dtype=torch.float32, device=compute_device)

    LOGGER.info(
        "svd_preallocation_start batches=%d modules=%d module_chunk_size=%d module_chunks=%d top_k=%d sketch_dim=%d compute_device=%s",
        len(calibration_batches),
        len(module_names),
        module_chunk_size,
        len(module_chunks),
        top_k,
        sketch_dim,
        compute_device,
    )

    started_at = time.perf_counter()
    for batch_index, batch in enumerate(calibration_batches, start=1):
        def sketch_update(name: str, _sample_idx: int, a_tokens: torch.Tensor, g_tokens: torch.Tensor) -> None:
            if a_tokens.numel() == 0:
                return
            omega = omegas[name]
            y_states[name] += g_tokens.T @ (a_tokens @ omega) / max(a_tokens.shape[0], 1)

        _run_backward_and_stream(
            model,
            modules,
            module_names,
            batch,
            answer_only,
            compute_device,
            sketch_update,
            module_chunk_size=module_chunk_size,
            pass_name="sketch_pass",
            batch_index=batch_index,
            total_batches=len(calibration_batches),
            progress_logging_steps=progress_logging_steps,
        )
    timings["sketch_pass_sec"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    q_states = {name: orthonormal_basis(y_states[name]) for name in module_names}
    b_states = {
        name: torch.zeros(q_states[name].shape[1], int(module_dims[name]["in_dim"]), dtype=torch.float32, device=compute_device)
        for name in module_names
    }
    for batch_index, batch in enumerate(calibration_batches, start=1):
        def basis_update(name: str, _sample_idx: int, a_tokens: torch.Tensor, g_tokens: torch.Tensor) -> None:
            q = q_states[name]
            if q.numel() == 0 or a_tokens.numel() == 0:
                return
            b_states[name] += (g_tokens @ q).T @ a_tokens / max(a_tokens.shape[0], 1)

        _run_backward_and_stream(
            model,
            modules,
            module_names,
            batch,
            answer_only,
            compute_device,
            basis_update,
            module_chunk_size=module_chunk_size,
            pass_name="basis_pass",
            batch_index=batch_index,
            total_batches=len(calibration_batches),
            progress_logging_steps=progress_logging_steps,
        )
    timings["basis_pass_sec"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    atoms: list[SvdAtomRecord] = []
    atoms_by_module: dict[str, list[SvdAtomRecord]] = {}
    for name in module_names:
        b_matrix = b_states[name]
        q = q_states[name]
        if b_matrix.numel() == 0 or q.numel() == 0:
            response = torch.zeros(int(module_dims[name]["out_dim"]), int(module_dims[name]["in_dim"]), device=compute_device)
            module_atoms = extract_svd_atoms_from_response_matrix(response, top_k, name, module_rank_cost(module_dims[name]))
        else:
            u_tilde, singular, vh = torch.linalg.svd(b_matrix, full_matrices=False)
            k = min(top_k, int(singular.numel()))
            denom = float(torch.sum(singular[:k] ** 2).item()) + 1e-12
            module_atoms = []
            for idx in range(k):
                module_atoms.append(
                    SvdAtomRecord(
                        module_name=name,
                        atom_index=idx,
                        cost=module_rank_cost(module_dims[name]),
                        singular_value=float(singular[idx].item()),
                        spectral_ratio=float((singular[idx] ** 2).item() / denom),
                        profile=None,
                        conflict=0.0,
                        coverage=0.0,
                        lambda_cov=1.0,
                        utility=0.0,
                        module_importance=1.0,
                        u=_as_float_tensor(q @ u_tilde[:, idx]),
                        v=_as_float_tensor(vh[idx, :]),
                        profile_norm_mode=profile_norm_mode,
                    )
                )
        atoms_by_module[name] = module_atoms
        atoms.extend(module_atoms)
    y_states.clear()
    q_states.clear()
    b_states.clear()
    _release_torch_memory()

    total_samples = sum(int(batch["input_ids"].shape[0]) for batch in calibration_batches)
    alpha = torch.zeros(total_samples, len(atoms), dtype=torch.float32, device=compute_device)
    denominators = torch.zeros(total_samples, dtype=torch.float32, device=compute_device)
    module_norms = {name: torch.zeros(total_samples, dtype=torch.float32, device=compute_device) for name in module_names}
    atom_offsets = {(atom.module_name, atom.atom_index): idx for idx, atom in enumerate(atoms)}
    atom_factor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, list[int]]] = {}
    for name, module_atoms in atoms_by_module.items():
        if not module_atoms:
            continue
        u_stack = torch.stack([atom.u for atom in module_atoms], dim=1).to(device=compute_device, dtype=torch.float32)
        v_stack = torch.stack([atom.v for atom in module_atoms], dim=1).to(device=compute_device, dtype=torch.float32)
        offsets = [atom_offsets[(atom.module_name, atom.atom_index)] for atom in module_atoms]
        atom_factor_cache[name] = (u_stack, v_stack, offsets)

    sample_base = 0
    for batch_index, batch in enumerate(calibration_batches, start=1):
        def profile_update(name: str, sample_idx: int, a_tokens: torch.Tensor, g_tokens: torch.Tensor) -> None:
            global_idx = sample_base + sample_idx
            norm = sample_response_norm_from_token_factors(a_tokens, g_tokens, mode=profile_norm_mode).to(device=compute_device)
            module_norms[name][global_idx] = norm
            denominators[global_idx] += norm
            if a_tokens.numel() == 0 or name not in atom_factor_cache:
                return
            u_stack, v_stack, offsets = atom_factor_cache[name]
            projections = torch.mean((g_tokens @ u_stack) * (a_tokens @ v_stack), dim=0)
            alpha[global_idx, offsets] = projections

        _run_backward_and_stream(
            model,
            modules,
            module_names,
            batch,
            answer_only,
            compute_device,
            profile_update,
            module_chunk_size=module_chunk_size,
            pass_name="profile_pass",
            batch_index=batch_index,
            total_batches=len(calibration_batches),
            progress_logging_steps=progress_logging_steps,
        )
        batch_size = int(batch["input_ids"].shape[0])
        sample_base += batch_size
    timings["profile_pass_sec"] = time.perf_counter() - started_at

    profiles = normalize_signed_profiles(alpha, denominators, mode=profile_norm_mode)
    profile_path = Path(profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "profiles": profiles.T.cpu(),
            "module_names": [atom.module_name for atom in atoms],
            "atom_indices": [atom.atom_index for atom in atoms],
            "profile_norm_mode": profile_norm_mode,
        },
        profile_path,
    )
    for idx, atom in enumerate(atoms):
        profile = profiles[:, idx].detach().cpu()
        atom.profile = profile
        atom.profile_path = str(profile_path)
        atom.profile_index = idx
        atom.profile_norm = float(torch.linalg.norm(profile).item())
        atom.profile_mean = float(profile.mean().item())
        atom.profile_std = float(profile.std(unbiased=False).item())
        atom.profile_hash = _profile_hash(profile)
        atom.conflict = gradient_conflict(alpha[:, idx])
        norms = module_norms[atom.module_name]
        valid_norms = norms[norms > 1e-8]
        if len(valid_norms) > 0:
            atom.module_importance = float(torch.exp(torch.mean(torch.log(valid_norms))).item())
        else:
            atom.module_importance = 0.0
        atom.importance_mode = profile_norm_mode

    evidence_cfg = dict(pre_cfg.get("evidence_selection", {}))
    max_selected = _resolve_max_selected(
        evidence_cfg.get("max_selected_atoms", "auto"),
        rank=rank,
        num_modules=len(module_names),
        top_k=top_k,
    )
    selected = select_coverage_evidence(
        atoms,
        max_selected_atoms=max_selected,
        epsilon_cov=float(pre_cfg.get("epsilon_cov", 0.05)),
        sparse_stop_by_coverage=bool(evidence_cfg.get("sparse_stop_by_coverage", True)),
        coverage_stop_threshold=float(evidence_cfg.get("coverage_stop_threshold", pre_cfg.get("epsilon_cov", 0.05))),
        beta=float(pre_cfg.get("beta", 1.0)),
        gamma=float(pre_cfg.get("gamma", 1.0)),
        delta=float(pre_cfg.get("delta", 1.0)),
        use_soft_tail=bool(pre_cfg.get("use_soft_tail", True)),
        atom_utility_floor=float(pre_cfg.get("atom_utility_floor", 0.0)),
        cost_aware=bool(pre_cfg.get("use_cost_aware_allocation", True)),
    )
    diagnostics = {
        "num_atoms": len(atoms),
        "num_selected_atoms": len(selected),
        "top_k_atoms": top_k,
        "sketch_dim": sketch_dim,
        "sketch_seed": sketch_seed,
        "sketch_dtype": str(pre_cfg.get("sketch_dtype", "float32")),
        "compute_device": str(compute_device),
        "module_chunk_size": module_chunk_size,
        "num_module_chunks": len(module_chunks),
        "answer_only": answer_only,
        "profile_norm_mode": profile_norm_mode,
        **timings,
        "evidence_selection": {
            "max_selected_atoms": max_selected,
            "sparse_stop_by_coverage": bool(evidence_cfg.get("sparse_stop_by_coverage", True)),
            "coverage_stop_threshold": float(evidence_cfg.get("coverage_stop_threshold", pre_cfg.get("epsilon_cov", 0.05))),
        },
    }
    return atoms, diagnostics
