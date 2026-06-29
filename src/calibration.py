import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from src.data_gsm8k import build_sft_example
from src.svd_utils import randomized_svd_topk
from src.utils import ensure_dir


LOGGER = logging.getLogger(__name__)


@dataclass
class CalibrationStats:
    module_names: List[str]
    R: Dict[str, torch.Tensor]
    norm_matrix: torch.Tensor
    module_dims: Dict[str, Dict[str, int]]
    sample_ids: List[int]
    calibration_config: Dict[str, Any]
    importance: Dict[str, float]


@dataclass
class DicoAtoms:
    U: Dict[str, torch.Tensor]
    S: Dict[str, torch.Tensor]
    V: Dict[str, torch.Tensor]
    rho: Dict[str, torch.Tensor]


@dataclass
class CalibrationProfiles:
    psi_profiles: torch.Tensor
    normalized_profiles: torch.Tensor
    module_profiles: torch.Tensor


def build_shifted_answer_mask(labels: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if labels.shape != attention_mask.shape:
        raise ValueError("labels and attention_mask must have the same shape")
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, :-1] = labels[:, 1:] != -100
    mask = mask & attention_mask.bool()
    mask[:, -1] = False
    return mask


def _module_map(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _install_hooks(
    model: nn.Module,
    target_module_names: List[str],
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[Any]]:
    modules = _module_map(model)
    records: Dict[str, Dict[str, torch.Tensor]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs, output):
            if not torch.is_tensor(output):
                raise RuntimeError("Target module %s did not return a tensor" % name)
            if output.requires_grad:
                output.retain_grad()
            records[name] = {"activation": inputs[0], "output": output}

        return hook

    for name in target_module_names:
        if name not in modules:
            raise KeyError("Target module %s not found in model" % name)
        handles.append(modules[name].register_forward_hook(make_hook(name)))
    return records, handles


def _remove_hooks(handles: List[Any]) -> None:
    for handle in handles:
        handle.remove()


def _module_dims(model: nn.Module, module_names: List[str]) -> Dict[str, Dict[str, int]]:
    modules = _module_map(model)
    dims: Dict[str, Dict[str, int]] = {}
    for name in module_names:
        module = modules[name]
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            raise TypeError("Target module %s is not nn.Linear-like" % name)
        d_in = int(module.in_features)
        d_out = int(module.out_features)
        dims[name] = {"d_in": d_in, "d_out": d_out, "cost": d_in + d_out}
    return dims


def _prepare_batch(example: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": example["input_ids_tensor"].unsqueeze(0).to(device),
        "attention_mask": example["attention_mask_tensor"].unsqueeze(0).to(device),
        "labels": example["labels_tensor"].unsqueeze(0).to(device),
    }


def _save_mask_debug(
    output_dir: Path,
    sample_index: int,
    tokenizer: Any,
    example: Dict[str, Any],
    shifted_mask: torch.Tensor,
) -> None:
    debug_dir = ensure_dir(output_dir / "mask_debug")
    labels = example["labels_tensor"]
    predicted_label_positions = torch.zeros_like(labels, dtype=torch.bool)
    predicted_label_positions[:-1] = shifted_mask[0, :-1].cpu()
    loss_token_ids = labels[1:][shifted_mask[0, :-1].cpu()].tolist()
    decoded = tokenizer.decode(loss_token_ids, skip_special_tokens=False)
    payload = {
        "sample_index": sample_index,
        "prompt_text": example["prompt_text"],
        "answer_text": example["answer_text"],
        "decoded_loss_bearing_tokens": decoded,
        "shifted_answer_token_count": int(shifted_mask.sum().item()),
        "first_loss_token_id": int(loss_token_ids[0]) if loss_token_ids else None,
        "last_loss_token_id": int(loss_token_ids[-1]) if loss_token_ids else None,
        "input_length": int(len(example["input_ids"])),
        "num_label_tokens": int(sum(x != -100 for x in example["labels"])),
        "mask_positions": [int(i) for i in torch.where(shifted_mask[0].cpu())[0].tolist()],
        "predicted_label_positions": [
            int(i) for i in torch.where(predicted_label_positions.cpu())[0].tolist()
        ],
    }
    (debug_dir / ("sample_%03d.json" % sample_index)).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _sample_module_matrices(
    records: Dict[str, Dict[str, torch.Tensor]],
    module_names: List[str],
    shifted_mask: torch.Tensor,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    result: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    mask = shifted_mask[0].detach().cpu()
    for module_name in module_names:
        activation = records[module_name]["activation"].detach()[0].cpu().float()
        output = records[module_name]["output"]
        if output.grad is None:
            raise RuntimeError("Missing retained output gradient for %s" % module_name)
        grad = output.grad.detach()[0].cpu().float()
        seq_len = min(activation.shape[0], grad.shape[0], mask.shape[0])
        local_mask = mask[:seq_len]
        A = activation[:seq_len][local_mask]
        Gout = grad[:seq_len][local_mask]
        if A.ndim != 2 or Gout.ndim != 2:
            raise AssertionError("Expected A and Gout to be rank-2")
        result[module_name] = (A.contiguous(), Gout.contiguous())
    return result


def collect_dico_calibration_stats(
    model: nn.Module,
    tokenizer: Any,
    dataset: List[Dict[str, str]],
    target_module_names: List[str],
    calibration_size: int,
    max_length: int,
    output_dir: Path,
    device: torch.device,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
    eps: float = 1e-8,
) -> CalibrationStats:
    output_dir = ensure_dir(Path(output_dir))
    model.to(device)
    model.eval()
    module_dims = _module_dims(model, target_module_names)
    sample_count = min(calibration_size, len(dataset))
    R = {
        name: torch.zeros(module_dims[name]["d_out"], module_dims[name]["d_in"], dtype=torch.float32)
        for name in target_module_names
    }
    norm_matrix = torch.zeros(sample_count, len(target_module_names), dtype=torch.float32)
    sample_ids = list(range(sample_count))
    records, handles = _install_hooks(model, target_module_names)
    nonempty_masks = 0

    try:
        for row_index in range(sample_count):
            raw = dataset[row_index]
            example = build_sft_example(
                raw["question"],
                raw["answer"],
                tokenizer,
                max_length=max_length,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            batch = _prepare_batch(example, device)
            shifted_mask = build_shifted_answer_mask(batch["labels"], batch["attention_mask"])
            if int(shifted_mask.sum().item()) > 0:
                nonempty_masks += 1
            if row_index < 3:
                _save_mask_debug(output_dir, row_index, tokenizer, example, shifted_mask)

            model.zero_grad(set_to_none=True)
            records.clear()
            outputs = model(**batch)
            if outputs.loss is None:
                raise RuntimeError("Model did not return loss during calibration")
            outputs.loss.backward()
            matrices = _sample_module_matrices(records, target_module_names, shifted_mask)
            for module_index, module_name in enumerate(target_module_names):
                A, Gout = matrices[module_name]
                if A.shape[0] == 0:
                    continue
                d_in = module_dims[module_name]["d_in"]
                d_out = module_dims[module_name]["d_out"]
                assert A.shape[1] == d_in
                assert Gout.shape[1] == d_out
                G_i = (Gout.T @ A) / float(A.shape[0])
                assert G_i.shape == (d_out, d_in)
                R[module_name] += G_i
                norm_matrix[row_index, module_index] = torch.linalg.norm(G_i, ord="fro")
            model.zero_grad(set_to_none=True)
    finally:
        _remove_hooks(handles)

    if nonempty_masks == 0:
        raise RuntimeError("All calibration samples had empty answer-token masks")
    if not torch.isfinite(norm_matrix).all():
        raise RuntimeError("Calibration norm_matrix contains non-finite values")

    importance = {}
    for module_index, module_name in enumerate(target_module_names):
        importance[module_name] = float(torch.log(eps + norm_matrix[:, module_index]).mean().item())

    stats = CalibrationStats(
        module_names=list(target_module_names),
        R=R,
        norm_matrix=norm_matrix,
        module_dims=module_dims,
        sample_ids=sample_ids,
        calibration_config={
            "calibration_size": calibration_size,
            "max_length": max_length,
            "use_chat_template": use_chat_template,
            "enable_thinking": enable_thinking,
        },
        importance=importance,
    )
    torch.save(
        {
            "module_names": stats.module_names,
            "R": stats.R,
            "norm_matrix": stats.norm_matrix,
            "module_dims": stats.module_dims,
            "sample_ids": stats.sample_ids,
            "calibration_config": stats.calibration_config,
            "importance": stats.importance,
        },
        output_dir / "calibration_pass1.pt",
    )
    return stats


def compute_dico_atoms(
    stats: CalibrationStats,
    top_k_atoms: int,
    output_dir: Path,
    exact_svd: bool = False,
) -> DicoAtoms:
    output_dir = ensure_dir(Path(output_dir))
    U: Dict[str, torch.Tensor] = {}
    S: Dict[str, torch.Tensor] = {}
    V: Dict[str, torch.Tensor] = {}
    rho: Dict[str, torch.Tensor] = {}
    for module_name in stats.module_names:
        matrix = stats.R[module_name].float().cpu()
        if exact_svd:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            u = u[:, :top_k_atoms].contiguous()
            s = s[:top_k_atoms].contiguous()
            v = vh[:top_k_atoms, :].T.contiguous()
        else:
            u, s, v = randomized_svd_topk(matrix, top_k_atoms)
        denom = torch.sum(s * s)
        rho[module_name] = (s * s / denom) if float(denom) > 0 else torch.zeros_like(s)
        U[module_name] = u.cpu().float()
        S[module_name] = s.cpu().float()
        V[module_name] = v.cpu().float()
    atoms = DicoAtoms(U=U, S=S, V=V, rho=rho)
    torch.save({"U": U, "S": S, "V": V, "rho": rho}, output_dir / "dico_atoms.pt")
    return atoms


def _center_l2_normalize(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    centered = matrix - matrix.mean(dim=-1, keepdim=True)
    norms = torch.linalg.norm(centered, dim=-1, keepdim=True)
    return torch.where(norms > eps, centered / torch.clamp(norms, min=eps), torch.zeros_like(centered))


def collect_dico_profiles(
    model: nn.Module,
    tokenizer: Any,
    dataset: List[Dict[str, str]],
    stats: CalibrationStats,
    atoms: DicoAtoms,
    max_length: int,
    output_dir: Path,
    device: torch.device,
    use_chat_template: bool = False,
    enable_thinking: bool = False,
    eps: float = 1e-8,
) -> CalibrationProfiles:
    output_dir = ensure_dir(Path(output_dir))
    model.to(device)
    model.eval()
    sample_count = len(stats.sample_ids)
    module_count = len(stats.module_names)
    max_k = max(atoms.U[name].shape[1] for name in stats.module_names)
    psi_profiles = torch.zeros(module_count, max_k, sample_count, dtype=torch.float32)
    records, handles = _install_hooks(model, stats.module_names)

    try:
        for row_index in range(sample_count):
            raw = dataset[stats.sample_ids[row_index]]
            example = build_sft_example(
                raw["question"],
                raw["answer"],
                tokenizer,
                max_length=max_length,
                use_chat_template=use_chat_template,
                enable_thinking=enable_thinking,
            )
            batch = _prepare_batch(example, device)
            shifted_mask = build_shifted_answer_mask(batch["labels"], batch["attention_mask"])
            model.zero_grad(set_to_none=True)
            records.clear()
            outputs = model(**batch)
            if outputs.loss is None:
                raise RuntimeError("Model did not return loss during profile pass")
            outputs.loss.backward()
            matrices = _sample_module_matrices(records, stats.module_names, shifted_mask)
            for module_index, module_name in enumerate(stats.module_names):
                A, Gout = matrices[module_name]
                if A.shape[0] == 0:
                    continue
                U = atoms.U[module_name].float()
                V = atoms.V[module_name].float()
                T = float(A.shape[0])
                H = Gout @ U
                term1 = A.T @ H / T
                S2 = A @ V
                term2 = Gout.T @ S2 / T
                psi = torch.linalg.norm(term1, dim=0) + torch.linalg.norm(term2, dim=0)
                psi_profiles[module_index, : psi.shape[0], row_index] = psi
            model.zero_grad(set_to_none=True)
    finally:
        _remove_hooks(handles)

    denom = stats.norm_matrix.sum(dim=1).clamp_min(eps)
    normalized = psi_profiles / denom.view(1, 1, -1)
    normalized = _center_l2_normalize(normalized, eps)
    module_profiles = stats.norm_matrix.T / denom.view(1, -1)
    module_profiles = _center_l2_normalize(module_profiles, eps)
    if not torch.isfinite(normalized).all() or not torch.isfinite(module_profiles).all():
        raise RuntimeError("Profile tensors contain non-finite values")

    profiles = CalibrationProfiles(
        psi_profiles=psi_profiles,
        normalized_profiles=normalized,
        module_profiles=module_profiles,
    )
    torch.save(
        {
            "psi_profiles": psi_profiles,
            "normalized_profiles": normalized,
            "module_profiles": module_profiles,
        },
        output_dir / "calibration_profiles.pt",
    )
    summary = {
        "num_modules": module_count,
        "num_samples": sample_count,
        "top_k_atoms": max_k,
        "norm_matrix_min": float(stats.norm_matrix.min().item()),
        "norm_matrix_max": float(stats.norm_matrix.max().item()),
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return profiles
