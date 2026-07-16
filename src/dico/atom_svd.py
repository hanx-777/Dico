from __future__ import annotations

import gc
import hashlib
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from dico.profiles import compute_sketch_signed_profile
from dico.rank_budget import module_rank_cost
from dico.sketch import make_random_projection


LOGGER = logging.getLogger(__name__)
EPS = 1e-12


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
    sample_weights: torch.Tensor | None = None
    signed_response_sum: float = 0.0
    signed_response_abs_sum: float = 0.0
    alignment: float = 0.0
    selected_coverage_gain: float = 0.0
    selected: bool = False
    u: torch.Tensor | None = None
    v: torch.Tensor | None = None
    v_tilde: torch.Tensor | None = None
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
            "signed_response_sum": self.signed_response_sum,
            "signed_response_abs_sum": self.signed_response_abs_sum,
            "alignment": self.alignment,
            "selected_coverage_gain": self.selected_coverage_gain,
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
            "profile_domain": "sketch" if self.v_tilde is not None else "full",
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


@torch.no_grad()
def rpca_ialm(
    Y: torch.Tensor,
    lam: float | None = None,
    tol: float = 1.0e-6,
    max_iter: int = 200,
    lambda_scale: float = 1.0,
    fallback_on_error: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Decompose a sketch matrix Y = L + S with inexact ALM RPCA."""

    original = Y.detach()
    diag: dict[str, Any] = {
        "rpca_iter": 0,
        "rpca_converged": False,
        "rpca_residual_norm": 0.0,
        "rpca_sparse_fraction": 0.0,
        "rpca_fallback_used": False,
    }

    def fallback(reason: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        if not fallback_on_error:
            raise RuntimeError(f"RPCA sketch denoising failed: {reason}")
        LOGGER.warning("rpca_sketch_fallback reason=%s shape=%s", reason, tuple(original.shape))
        diag["rpca_fallback_used"] = True
        return original.clone(), torch.zeros_like(original), dict(diag)

    if original.ndim != 2 or min(original.shape) < 2:
        return fallback("matrix_too_small")
    if not bool(torch.isfinite(original).all()):
        return fallback("non_finite_input")

    matrix = original.to(dtype=torch.float32)
    if float(torch.count_nonzero(matrix).item()) == 0.0:
        return fallback("all_zero_input")

    try:
        m, n = int(matrix.shape[0]), int(matrix.shape[1])
        lam_value = float(lam) if lam is not None else float(lambda_scale) / math.sqrt(max(m, n))
        lam_value = max(lam_value, EPS)
        norm_y = torch.linalg.norm(matrix, ord="fro").clamp_min(EPS)
        spectral_norm = torch.linalg.svdvals(matrix)[0].clamp_min(EPS)
        dual_norm = max(float(spectral_norm.item()), float((torch.max(torch.abs(matrix)) / lam_value).item()), EPS)
        dual = matrix / dual_norm
        mu = 1.25 / float(spectral_norm.item())
        mu_bar = mu * 1.0e7
        rho = 1.5
        low_rank = torch.zeros_like(matrix)
        sparse = torch.zeros_like(matrix)
        residual = matrix.clone()
        rel_residual = float("inf")
        converged = False
        iterations = 0
        for iteration in range(1, int(max_iter) + 1):
            u, singular, vh = torch.linalg.svd(matrix - sparse + dual / mu, full_matrices=False)
            shrunk = torch.clamp(singular - 1.0 / mu, min=0.0)
            low_rank = (u * shrunk.reshape(1, -1)) @ vh
            sparse_input = matrix - low_rank + dual / mu
            sparse = torch.sign(sparse_input) * torch.clamp(torch.abs(sparse_input) - lam_value / mu, min=0.0)
            residual = matrix - low_rank - sparse
            rel_residual = float((torch.linalg.norm(residual, ord="fro") / norm_y).item())
            iterations = iteration
            if not math.isfinite(rel_residual) or not bool(torch.isfinite(low_rank).all()) or not bool(torch.isfinite(sparse).all()):
                return fallback("non_finite_iteration")
            if rel_residual <= float(tol):
                converged = True
                break
            dual = dual + mu * residual
            mu = min(mu * rho, mu_bar)
        if not bool(torch.isfinite(low_rank).all()):
            return fallback("unsafe_low_rank_output")
        diag.update(
            {
                "rpca_iter": iterations,
                "rpca_converged": converged,
                "rpca_residual_norm": rel_residual if math.isfinite(rel_residual) else float("inf"),
                "rpca_sparse_fraction": float((torch.abs(sparse) > 1.0e-6).float().mean().item()),
                "rpca_rank_est": int((torch.linalg.svdvals(low_rank) > 1.0e-6).sum().item()),
                "rpca_sparse_energy_ratio": float(
                    (
                        torch.linalg.norm(sparse, ord="fro")
                        / (torch.linalg.norm(matrix, ord="fro").clamp_min(EPS))
                    ).item()
                ),
            }
        )
        return low_rank.to(device=original.device, dtype=original.dtype), sparse.to(device=original.device, dtype=original.dtype), dict(diag)
    except Exception as exc:
        return fallback(type(exc).__name__)


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
    return torch.sum(left * right)


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
        response = gradients.T @ activations
        return torch.linalg.norm(response)
    if mode == "streaming_estimate":
        token_norm_sq = torch.sum(activations * activations, dim=-1) * torch.sum(gradients * gradients, dim=-1)
        return torch.sqrt(torch.sum(token_norm_sq).clamp_min(0.0))
    raise ValueError(f"Unsupported profile_norm_mode: {mode}")


def gradient_conflict(alpha: torch.Tensor) -> float:
    alpha = alpha.detach().flatten()
    total = max(int(alpha.numel()), 1)
    p_pos = float((alpha > 0).sum().item()) / total
    p_neg = float((alpha < 0).sum().item()) / total
    return max(0.0, min(1.0, 4.0 * p_pos * p_neg))


def direction_alignment(alpha: torch.Tensor, eps: float = 1e-12) -> float:
    alpha = alpha.detach().flatten().float()
    numerator = torch.abs(torch.sum(alpha))
    denominator = torch.sum(torch.abs(alpha)) + float(eps)
    if float(denominator.item()) <= float(eps):
        return 0.0
    return max(0.0, min(1.0, float((numerator / denominator).item())))


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


def _release_torch_memory(*, phase_boundary: bool = False) -> None:
    # Python GC and CUDA allocator flushing synchronize the device and are very
    # expensive inside every module-chunk backward.  Tensor references are
    # released by callers; force collection only at coarse phase boundaries.
    if phase_boundary:
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
        raise RuntimeError(
            "preallocation.compute_device=cuda requested but CUDA is unavailable; "
            "formal CovRA must not silently fall back to CPU"
        )
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


def extract_svd_atom_records(
    model: Any,
    module_names: list[str],
    module_dims: Mapping[str, Mapping[str, Any]],
    calibration_batches: list[Mapping[str, torch.Tensor]],
    pre_cfg: Mapping[str, Any],
    rank: int,
    profile_path: Path,
    group_labels: Sequence[str] | None = None,
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
    rpca_cfg = dict(pre_cfg.get("rpca", {}))
    rpca_enabled = bool(rpca_cfg.get("enabled", False))
    lambda_cov = float(pre_cfg.get("lambda_cov", 1.0))
    sketch_block_mode = str(pre_cfg.get("sketch_block_mode", "hybrid"))
    if sketch_block_mode not in {"hybrid", "global_only", "grouped_only"}:
        raise ValueError(
            f"Unsupported preallocation.sketch_block_mode={sketch_block_mode!r}; "
            "expected 'hybrid', 'global_only', or 'grouped_only'."
        )

    # 3.2.1节: hybrid grouped sketch. Partition calibration samples into response
    # aggregation groups C -- real task-group labels when one is supplied per sample,
    # else C random seeded blocks (default 4) -- so the SVD that produces u_{m,k} runs
    # on a group-aware hybrid matrix [λ·Y_agg | Y^(1) | ... | Y^(C)] instead of a single
    # pooled/aggregate response matrix, which can silently cancel opposing-sign
    # direction demand across groups (the doc's motivating failure mode for "纯聚合草图").
    total_samples = sum(int(batch["input_ids"].shape[0]) for batch in calibration_batches)
    if group_labels is not None and len(group_labels) == total_samples:
        sketch_groups = [str(value) for value in group_labels]
        sketch_group_source = "configured"
    else:
        if group_labels is not None:
            LOGGER.warning(
                "response_agg_group_label_length_mismatch got=%d expected=%d; "
                "falling back to random response-aggregation blocks",
                len(group_labels),
                total_samples,
            )
        c = max(2, int(pre_cfg.get("response_agg_groups", 4)))
        rng = random.Random(sketch_seed)
        order = list(range(total_samples))
        rng.shuffle(order)
        sketch_groups = [""] * total_samples
        for position, sample_idx in enumerate(order):
            sketch_groups[sample_idx] = f"sketch_block_{position % c}"
        sketch_group_source = "random_block"
    unique_sketch_groups = sorted(set(sketch_groups))

    omegas = {}
    y_states = {}
    y_group_states: dict[str, dict[str, torch.Tensor]] = {}
    for offset, name in enumerate(module_names):
        dims = module_dims[name]
        in_dim = int(dims["in_dim"])
        out_dim = int(dims["out_dim"])
        s = min(sketch_dim, in_dim)
        omegas[name] = make_random_projection(in_dim, s, sketch_seed + offset, dtype=sketch_dtype, device=compute_device)
        y_states[name] = torch.zeros(out_dim, s, dtype=torch.float32, device=compute_device)
        y_group_states[name] = {
            group: torch.zeros(out_dim, s, dtype=torch.float32, device=compute_device)
            for group in unique_sketch_groups
        }

    LOGGER.info(
        "svd_preallocation_start batches=%d modules=%d module_chunk_size=%d module_chunks=%d top_k=%d "
        "sketch_dim=%d compute_device=%s response_agg_groups=%d sketch_group_source=%s",
        len(calibration_batches),
        len(module_names),
        module_chunk_size,
        len(module_chunks),
        top_k,
        sketch_dim,
        compute_device,
        len(unique_sketch_groups),
        sketch_group_source,
    )

    started_at = time.perf_counter()
    sample_base = 0
    for batch_index, batch in enumerate(calibration_batches, start=1):
        def sketch_update(name: str, sample_idx: int, a_tokens: torch.Tensor, g_tokens: torch.Tensor) -> None:
            if a_tokens.numel() == 0:
                return
            omega = omegas[name]
            contrib = g_tokens.T @ (a_tokens @ omega)
            y_states[name] += contrib
            group = sketch_groups[sample_base + sample_idx]
            y_group_states[name][group] += contrib

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
        sample_base += int(batch["input_ids"].shape[0])
    timings["sketch_pass_sec"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    rpca_rows: list[dict[str, Any]] = []
    def hybrid_blocks(name: str, aggregate: torch.Tensor, grouped: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if sketch_block_mode == "global_only":
            return [lambda_cov * aggregate]
        if sketch_block_mode == "grouped_only":
            return list(grouped)
        return [lambda_cov * aggregate] + list(grouped)

    y_hat_states: dict[str, torch.Tensor] = {
        name: torch.cat(
            hybrid_blocks(name, y_states[name], [y_group_states[name][group] for group in unique_sketch_groups]),
            dim=1,
        )
        for name in module_names
    }
    # RPCA denoising (a repo-local feature the doc doesn't mention) is applied to the
    # hybrid-concatenated matrix, not the old pooled y_states, preserving its intent of
    # denoising immediately before the SVD that actually produces u_{m,k}.
    y_for_basis = dict(y_hat_states)
    if rpca_enabled:
        y_for_basis = {}
        for name in module_names:
            low_rank, _sparse, rpca_diag = rpca_ialm(
                y_hat_states[name],
                tol=float(rpca_cfg.get("tol", 1.0e-6)),
                max_iter=int(rpca_cfg.get("max_iter", 200)),
                lambda_scale=float(rpca_cfg.get("lambda_scale", 1.0)),
                fallback_on_error=bool(rpca_cfg.get("fallback_on_error", True)),
            )
            y_for_basis[name] = low_rank.to(device=compute_device, dtype=torch.float32)
            rpca_rows.append(rpca_diag)
    u_states: dict[str, torch.Tensor] = {}
    singular_states: dict[str, torch.Tensor] = {}
    v_tilde_states: dict[str, torch.Tensor] = {}
    for name in module_names:
        if y_for_basis[name].numel() == 0:
            u_states[name] = torch.empty(int(module_dims[name]["out_dim"]), 0, dtype=torch.float32, device=compute_device)
            singular_states[name] = torch.empty(0, dtype=torch.float32, device=compute_device)
            v_tilde_states[name] = torch.empty(int(omegas[name].shape[1]), 0, dtype=torch.float32, device=compute_device)
            continue
        u_matrix, singular, _vh = torch.linalg.svd(y_for_basis[name].float(), full_matrices=False)
        k = min(top_k, int(singular.numel()))
        u_states[name] = u_matrix[:, :k].to(device=compute_device, dtype=torch.float32)
        singular_states[name] = singular[:k].to(device=compute_device, dtype=torch.float32)

        # 3.2.2节: ṽ_{m,k} is the top eigenvector of M_{m,k} = λ²·y_agg,k y_agg,k^T +
        # Σ_c y^(c)_k y^(c)_k^T -- the same hybrid-grouped principle applied per-atom to
        # its own sketch-domain input response. This is *not* `vh` from the y_hat SVD
        # above: y_hat's right-singular space is block-structured over (1+C)*s columns,
        # unrelated to the s-dimensional space each per-atom M_{m,k} lives in.
        v_tilde_cols = []
        for idx in range(k):
            u_k = u_matrix[:, idx]
            y_agg_k = y_states[name].T @ u_k
            y_group_k = [y_group_states[name][group].T @ u_k for group in unique_sketch_groups]
            x_k = torch.stack(hybrid_blocks(name, y_agg_k, y_group_k), dim=1)
            u_x, _s_x, _vh_x = torch.linalg.svd(x_k, full_matrices=False)
            v_tilde_cols.append(u_x[:, 0])
        v_tilde_states[name] = (
            torch.stack(v_tilde_cols, dim=1).to(device=compute_device, dtype=torch.float32)
            if v_tilde_cols
            else torch.empty(int(omegas[name].shape[1]), 0, dtype=torch.float32, device=compute_device)
        )
    b_states = {
        name: torch.zeros(u_states[name].shape[1], int(module_dims[name]["in_dim"]), dtype=torch.float32, device=compute_device)
        for name in module_names
    }
    b_group_states: dict[str, dict[str, torch.Tensor]] = {
        name: {
            group: torch.zeros(u_states[name].shape[1], int(module_dims[name]["in_dim"]), dtype=torch.float32, device=compute_device)
            for group in unique_sketch_groups
        }
        for name in module_names
    }
    sample_base = 0
    for batch_index, batch in enumerate(calibration_batches, start=1):
        def basis_update(name: str, sample_idx: int, a_tokens: torch.Tensor, g_tokens: torch.Tensor) -> None:
            u_matrix = u_states[name]
            if u_matrix.numel() == 0 or a_tokens.numel() == 0:
                return
            contrib = (g_tokens @ u_matrix).T @ a_tokens
            b_states[name] += contrib
            group = sketch_groups[sample_base + sample_idx]
            b_group_states[name][group] += contrib

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
        sample_base += int(batch["input_ids"].shape[0])
    timings["basis_pass_sec"] = time.perf_counter() - started_at

    started_at = time.perf_counter()
    atoms: list[SvdAtomRecord] = []
    atoms_by_module: dict[str, list[SvdAtomRecord]] = {}
    for name in module_names:
        b_matrix = b_states[name]
        u_matrix = u_states[name]
        singular = singular_states[name]
        v_tilde = v_tilde_states[name]
        if b_matrix.numel() == 0 or u_matrix.numel() == 0:
            response = torch.zeros(int(module_dims[name]["out_dim"]), int(module_dims[name]["in_dim"]), device=compute_device)
            module_atoms = extract_svd_atoms_from_response_matrix(response, top_k, name, module_rank_cost(module_dims[name]))
        else:
            k = min(top_k, int(singular.numel()))
            denom = float(torch.sum(singular[:k] ** 2).item()) + 1e-12
            module_atoms = []
            for idx in range(k):
                # 3.2.2节: v_{m,k} (full-dim input direction) is the top left-singular
                # vector of the hybrid Z_{m,k} = [λ·z_agg,k | z^(1)_k | ... | z^(C)_k],
                # not a bare normalization of the pooled accumulator.
                z_agg_k = b_matrix[idx, :]
                z_group_k = [b_group_states[name][group][idx, :] for group in unique_sketch_groups]
                z_k = torch.stack(hybrid_blocks(name, z_agg_k, z_group_k), dim=1)
                u_z, _s_z, _vh_z = torch.linalg.svd(z_k, full_matrices=False)
                full_v = u_z[:, 0]
                # SVD determines direction up to sign; pick the sign that agrees with
                # the raw aggregate accumulator, for determinism/testability.
                if torch.dot(full_v, z_agg_k) < 0:
                    full_v = -full_v
                full_v = full_v / torch.linalg.norm(full_v).clamp_min(1e-12)
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
                        lambda_cov=lambda_cov,
                        utility=0.0,
                        module_importance=1.0,
                        u=_as_float_tensor(u_matrix[:, idx]),
                        v=_as_float_tensor(full_v),
                        v_tilde=_as_float_tensor(v_tilde[:, idx]),
                        profile_norm_mode=profile_norm_mode,
                    )
                )
        atoms_by_module[name] = module_atoms
        atoms.extend(module_atoms)
    y_states.clear()
    y_group_states.clear()
    y_hat_states.clear()
    y_for_basis.clear()
    u_states.clear()
    b_states.clear()
    b_group_states.clear()
    _release_torch_memory(phase_boundary=True)

    total_samples = sum(int(batch["input_ids"].shape[0]) for batch in calibration_batches)
    alpha = torch.zeros(total_samples, len(atoms), dtype=torch.float32, device=compute_device)
    denominators = torch.zeros(total_samples, dtype=torch.float32, device=compute_device)
    token_counts = torch.zeros(total_samples, dtype=torch.float32, device=compute_device)
    module_norms = {name: torch.zeros(total_samples, dtype=torch.float32, device=compute_device) for name in module_names}
    atom_offsets = {(atom.module_name, atom.atom_index): idx for idx, atom in enumerate(atoms)}
    atom_factor_cache: dict[str, tuple[torch.Tensor, torch.Tensor, list[int]]] = {}
    for name, module_atoms in atoms_by_module.items():
        module_atoms = [atom for atom in module_atoms if atom.u is not None and atom.v_tilde is not None]
        if not module_atoms:
            continue
        u_stack = torch.stack([atom.u for atom in module_atoms], dim=1).to(device=compute_device, dtype=torch.float32)
        v_tilde_stack = torch.stack([atom.v_tilde for atom in module_atoms], dim=1).to(device=compute_device, dtype=torch.float32)
        offsets = [atom_offsets[(atom.module_name, atom.atom_index)] for atom in module_atoms]
        atom_factor_cache[name] = (u_stack, v_tilde_stack, offsets)

    sample_base = 0
    for batch_index, batch in enumerate(calibration_batches, start=1):
        def profile_update(name: str, sample_idx: int, a_tokens: torch.Tensor, g_tokens: torch.Tensor) -> None:
            global_idx = sample_base + sample_idx
            if token_counts[global_idx] <= 0:
                token_counts[global_idx] = max(int(a_tokens.shape[0]), 1)
            norm = sample_response_norm_from_token_factors(a_tokens, g_tokens, mode=profile_norm_mode).to(device=compute_device)
            module_norms[name][global_idx] = norm
            denominators[global_idx] += norm
            if a_tokens.numel() == 0 or name not in atom_factor_cache:
                return
            u_stack, v_tilde_stack, offsets = atom_factor_cache[name]
            # profiles.py::compute_sketch_signed_profile generalizes to stacked
            # multi-atom u/v_tilde via matmul broadcasting, so this reuses the
            # same 4.3节 formula rather than re-deriving it inline.
            projections = compute_sketch_signed_profile(
                g_tokens.unsqueeze(0), a_tokens.unsqueeze(0), u_stack, v_tilde_stack, omegas[name]
            ).squeeze(0)
            # 4.3节: signed profile is a per-sample token average, π_{m,k}^{(i)} = (1/T_i) Σ_t (...).
            alpha[global_idx, offsets] = projections / max(int(a_tokens.shape[0]), 1)

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

    profiles = alpha
    sample_weights = torch.where(token_counts > 0, 1.0 / token_counts.clamp_min(1.0), torch.zeros_like(token_counts))
    profile_path = Path(profile_path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "profiles": profiles.T.cpu(),
            "module_names": [atom.module_name for atom in atoms],
            "atom_indices": [atom.atom_index for atom in atoms],
            "profile_norm_mode": profile_norm_mode,
            "profile_weight_mode": "inverse_token_count",
        },
        profile_path,
    )
    allocation_requested = str(pre_cfg.get("allocation_device", str(compute_device))).lower()
    if allocation_requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "preallocation.allocation_device=cuda requested but CUDA is unavailable; "
            "formal CovRA must not silently move conditional coverage to CPU"
        )
    allocation_device = compute_device if allocation_requested in {"auto", "default", ""} else torch.device(allocation_requested)
    for idx, atom in enumerate(atoms):
        profile = profiles[:, idx].detach().to(device=allocation_device, dtype=torch.float32)
        atom.profile = profile
        atom.sample_weights = sample_weights.detach().to(device=allocation_device, dtype=torch.float32)
        atom.profile_path = str(profile_path)
        atom.profile_index = idx
        atom.profile_norm = float(torch.linalg.norm(profile).item())
        atom.profile_mean = float(profile.mean().item())
        atom.profile_std = float(profile.std(unbiased=False).item())
        atom.profile_hash = _profile_hash(profile)
        atom.conflict = gradient_conflict(alpha[:, idx])
        atom.signed_response_sum = float(torch.sum(alpha[:, idx]).item())
        atom.signed_response_abs_sum = float(torch.sum(torch.abs(alpha[:, idx])).item())
        atom.alignment = direction_alignment(alpha[:, idx])
        norms = module_norms[atom.module_name]
        valid_norms = norms[norms > 1e-8]
        if len(valid_norms) > 0:
            atom.module_importance = float(torch.exp(torch.mean(torch.log(valid_norms))).item())
        else:
            atom.module_importance = 0.0
        atom.importance_mode = profile_norm_mode

    diagnostics = {
        "num_atoms": len(atoms),
        "top_k_atoms": top_k,
        "sketch_dim": sketch_dim,
        "sketch_seed": sketch_seed,
        "sketch_dtype": str(pre_cfg.get("sketch_dtype", "float32")),
        "compute_device": str(compute_device),
        "allocation_device": str(allocation_device),
        "module_chunk_size": module_chunk_size,
        "num_module_chunks": len(module_chunks),
        "answer_only": answer_only,
        "profile_norm_mode": profile_norm_mode,
        "hybrid_lambda": lambda_cov,
        "sketch_block_mode": sketch_block_mode,
        "response_agg_group_count": len(unique_sketch_groups),
        "sketch_group_source": sketch_group_source,
        "rpca_enabled": rpca_enabled,
        "rpca_iter": max([int(row.get("rpca_iter", 0)) for row in rpca_rows] + [0]),
        "rpca_converged": all(bool(row.get("rpca_converged", False)) for row in rpca_rows) if rpca_rows else False,
        "rpca_residual_norm": max([float(row.get("rpca_residual_norm", 0.0)) for row in rpca_rows] + [0.0]),
        "rpca_sparse_fraction": (
            sum(float(row.get("rpca_sparse_fraction", 0.0)) for row in rpca_rows) / len(rpca_rows)
            if rpca_rows
            else 0.0
        ),
        "rpca_fallback_used": sum(1 for row in rpca_rows if bool(row.get("rpca_fallback_used", False))),
        **timings,
    }
    return atoms, diagnostics
