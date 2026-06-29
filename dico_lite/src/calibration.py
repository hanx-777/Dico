import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from tqdm import tqdm

from src.data import build_sft_example

LOGGER = logging.getLogger(__name__)


@dataclass
class ModuleProfiles:
    method: str
    module_names: List[str]
    costs: torch.Tensor
    response_norms: torch.Tensor  # [N, M]
    sample_ids: List[int]
    num_samples: int
    num_modules: int


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


def _compute_module_scalar(A: torch.Tensor, Gout: torch.Tensor) -> float:
    """
    Computes s_{i,m} = || Gout.T @ A ||_F / max(T, 1) natively on GPU using Gram trick
    """
    T = A.shape[0]
    if T == 0:
        return 0.0

    K_a = A @ A.T
    K_g = Gout @ Gout.T
    score_sq = (K_a * K_g).sum()

    # ensure score_sq is non-negative before sqrt
    score_sq = max(float(score_sq.item()), 0.0)
    return math.sqrt(score_sq) / max(T, 1)


def _sample_module_scalars(
    records: Dict[str, Dict[str, torch.Tensor]],
    module_names: List[str],
    shifted_mask: torch.Tensor,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    mask = shifted_mask[0].detach()
    for module_name in module_names:
        activation = records[module_name]["activation"].detach()[0].float()
        output = records[module_name]["output"]
        if output.grad is None:
            raise RuntimeError("Missing retained output gradient for %s" % module_name)
        grad = output.grad.detach()[0].float()
        
        seq_len = min(activation.shape[0], grad.shape[0], mask.shape[0])
        local_mask = mask[:seq_len]
        
        A = activation[:seq_len][local_mask].contiguous()
        Gout = grad[:seq_len][local_mask].contiguous()
        
        if A.ndim != 2 or Gout.ndim != 2:
            raise AssertionError("Expected A and Gout to be rank-2")
            
        result[module_name] = _compute_module_scalar(A, Gout)
    return result


def collect_module_dico_profiles(
    model: nn.Module,
    tokenizer: Any,
    dataset: List[Dict[str, str]],
    target_module_names: List[str],
    calibration_size: int,
    max_length: int,
    device: torch.device,
) -> ModuleProfiles:
    
    model.eval()
    module_dims = _module_dims(model, target_module_names)
    sample_count = min(calibration_size, len(dataset))
    
    # response_norms: [N, M]
    response_norms = torch.zeros(sample_count, len(target_module_names), dtype=torch.float32, device=device)
    costs = torch.tensor([module_dims[name]["cost"] for name in target_module_names], dtype=torch.float32)
    sample_ids = list(range(sample_count))
    records, handles = _install_hooks(model, target_module_names)
    nonempty_masks = 0

    try:
        for row_index in tqdm(range(sample_count), desc="Module-DiCo-lite Calibration", unit="sample"):
            raw = dataset[row_index]
            example = build_sft_example(
                raw["question"],
                raw["answer"],
                tokenizer,
                max_length=max_length,
                use_chat_template=False,
                enable_thinking=False,
            )
            batch = _prepare_batch(example, device)
            shifted_mask = build_shifted_answer_mask(batch["labels"], batch["attention_mask"])
            if int(shifted_mask.sum().item()) > 0:
                nonempty_masks += 1

            model.zero_grad(set_to_none=True)
            records.clear()
            outputs = model(**batch)
            if outputs.loss is None:
                raise RuntimeError("Model did not return loss during calibration")
            outputs.loss.backward()
            del outputs
            
            scalars = _sample_module_scalars(records, target_module_names, shifted_mask)
            for module_index, module_name in enumerate(target_module_names):
                response_norms[row_index, module_index] = scalars[module_name]
            
            model.zero_grad(set_to_none=True)
            
    finally:
        _remove_hooks(handles)

    response_norms = response_norms.cpu()

    if nonempty_masks == 0:
        raise RuntimeError("All calibration samples had empty answer-token masks")
    if not torch.isfinite(response_norms).all():
        raise RuntimeError("Calibration response_norms contains non-finite values")

    return ModuleProfiles(
        method="module_dico",
        module_names=list(target_module_names),
        costs=costs,
        response_norms=response_norms,
        sample_ids=sample_ids,
        num_samples=sample_count,
        num_modules=len(target_module_names),
    )


def normalize_profiles(
    response_norms: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        module_profiles_normalized: [M, N]
        importance: [M]
    """
    sample_sums = response_norms.sum(dim=1, keepdim=True).clamp_min(eps)
    s_tilde = response_norms / sample_sums
    p = s_tilde.T  # [M, N]
    
    p_mean = p.mean(dim=1, keepdim=True)
    p_centered = p - p_mean
    
    p_l2 = torch.linalg.norm(p_centered, dim=1, keepdim=True).clamp_min(eps)
    p_normalized = p_centered / p_l2
    
    importance = torch.exp(torch.mean(torch.log(eps + response_norms), dim=0))
    return p_normalized, importance
