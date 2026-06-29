import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from src.utils import ensure_dir, read_version


LOGGER = logging.getLogger(__name__)


def select_torch_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("Unsupported torch dtype: %s" % name)


def load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_name_or_path: str,
    torch_dtype: str = "bf16",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = False,
):
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            "Could not load model config for %s. If this is Qwen3, upgrade transformers "
            "to a recent version. Original error: %s" % (model_name_or_path, exc)
        ) from exc

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": select_torch_dtype(torch_dtype),
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, **kwargs)
    # NOTE: For 4-bit/8-bit models, gradient_checkpointing_enable() is called here
    # but will be re-applied correctly AFTER prepare_model_for_kbit_training() in
    # apply_lora_adapters(). The order matters: kbit prep disables some hooks that
    # gradient checkpointing relies on.
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, config


def find_target_linear_modules(
    model: nn.Module,
    target_suffixes: Iterable[str],
) -> List[Tuple[str, nn.Linear]]:
    suffixes = tuple(s.strip() for s in target_suffixes if s.strip())
    found: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(suffix) for suffix in suffixes):
            found.append((name, module))
    return found


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


class CustomLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("CustomLoRALinear rank must be positive")
        self.base_layer = base
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scaling = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.in_features = base.in_features
        self.out_features = base.out_features

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def _apply_custom_lora(
    model: nn.Module,
    rank_pattern: Dict[str, int],
    lora_alpha: float,
) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False
    for name, rank in rank_pattern.items():
        if int(rank) <= 0:
            continue
        parent, attr = _get_parent_module(model, name)
        base = getattr(parent, attr)
        if not isinstance(base, nn.Linear):
            raise TypeError("Custom LoRA target %s is not nn.Linear" % name)
        setattr(parent, attr, CustomLoRALinear(base, int(rank), lora_alpha))
    return model


def _regex_for_exact_modules(module_names: List[str]) -> str:
    return r"^(%s)$" % "|".join(re.escape(name) for name in module_names)


def apply_lora_adapters(
    model: nn.Module,
    rank_pattern: Dict[str, int],
    target_module_names: List[str],
    lora_alpha: float = 16,
    use_peft: bool = True,
) -> nn.Module:
    positive = {name: int(rank) for name, rank in rank_pattern.items() if int(rank) > 0}
    if not positive:
        raise ValueError("No positive-rank modules were selected for LoRA")
    unknown = set(positive) - set(target_module_names)
    if unknown:
        raise ValueError("rank_pattern contains unknown modules: %s" % sorted(unknown))

    if not use_peft:
        return _apply_custom_lora(model, positive, lora_alpha=lora_alpha)

    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:
        raise RuntimeError("PEFT is required for real LoRA injection: %s" % exc) from exc

    if any(getattr(model, attr, False) for attr in ["is_loaded_in_4bit", "is_loaded_in_8bit"]):
        model = prepare_model_for_kbit_training(model)
        # Re-enable gradient checkpointing AFTER kbit prep (which resets internal hooks)
        if getattr(model, "gradient_checkpointing", False) or any(
            getattr(model, a, False) for a in ["is_gradient_checkpointing"]
        ):
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

    target_regex = _regex_for_exact_modules(list(positive))
    base_rank = max(positive.values())
    alpha_pattern = {name: float(lora_alpha) for name in positive}
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=base_rank,
        lora_alpha=lora_alpha,
        target_modules=target_regex,
        rank_pattern=positive,
        alpha_pattern=alpha_pattern,
        bias="none",
    )
    try:
        return get_peft_model(model, config)
    except Exception as exc:
        raise RuntimeError(
            "PEFT could not inject exact per-module LoRA adapters. This MVP refuses to "
            "fall back to suffix-wide targeting because that can silently inject rank-0 "
            "modules. PEFT version=%s, target_regex=%s, rank_pattern=%s. Error: %s"
            % (read_version("peft"), target_regex, positive, exc)
        ) from exc


def _actual_custom_rank(module: nn.Module) -> Optional[int]:
    if isinstance(module, CustomLoRALinear):
        return int(module.lora_A.out_features)
    return None


def _actual_peft_rank(module: nn.Module) -> Optional[int]:
    if not hasattr(module, "lora_A"):
        return None
    lora_A = getattr(module, "lora_A")
    try:
        if isinstance(lora_A, nn.ModuleDict):
            if "default" not in lora_A:
                return None
            return int(lora_A["default"].out_features)
        if isinstance(lora_A, nn.Linear):
            return int(lora_A.out_features)
    except Exception:
        return None
    return None


def verify_lora_ranks(
    model: nn.Module,
    requested_rank_pattern: Dict[str, int],
    all_target_module_names: List[str],
    output_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    modules = dict(model.named_modules())
    verification: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []
    for module_name in all_target_module_names:
        requested = int(requested_rank_pattern.get(module_name, 0))
        module = modules.get(module_name)
        if module is None:
            suffix = "." + module_name
            matches = [(name, value) for name, value in modules.items() if name.endswith(suffix)]
            if len(matches) == 1:
                module = matches[0][1]
            elif len(matches) > 1:
                errors.append("%s matched multiple wrapped modules: %s" % (module_name, [m[0] for m in matches]))
        actual = None if module is None else _actual_custom_rank(module)
        if actual is None and module is not None:
            actual = _actual_peft_rank(module)
        actual_rank = int(actual) if actual is not None else 0
        verification[module_name] = {
            "requested_rank": requested,
            "actual_rank": actual_rank,
            "has_lora": actual_rank > 0,
        }
        if requested != actual_rank:
            errors.append("%s requested=%d actual=%d" % (module_name, requested, actual_rank))
        if requested == 0 and actual_rank != 0:
            errors.append("%s received LoRA despite requested rank 0" % module_name)

    ensure_dir(Path(output_dir))
    path = Path(output_dir) / "lora_rank_verification.json"
    path.write_text(json.dumps(verification, indent=2, sort_keys=True), encoding="utf-8")
    if errors:
        raise RuntimeError("LoRA rank verification failed: " + "; ".join(errors))
    return verification


def module_names_from_rank_pattern(rank_pattern: Dict[str, int]) -> List[str]:
    return [name for name, rank in rank_pattern.items() if int(rank) > 0]
