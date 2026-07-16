from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

from dico.data import TinyTokenizer


@dataclass
class TinyConfig:
    vocab_size: int = 512
    hidden_size: int = 16
    model_type: str = "tiny-dico-rank"


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(torch.tanh(q + k + v))


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.tanh(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = TinyAttention(hidden_size)
        self.mlp = TinyMLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.self_attn(x) + self.mlp(x)


class TinyDecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size: int = 512, hidden_size: int = 16):
        super().__init__()
        self.config = TinyConfig(vocab_size=vocab_size, hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TinyBlock(hidden_size)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.gradient_checkpointing = False
        self._input_require_grads_handle = None

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def get_input_embeddings(self):
        return self.embed

    def enable_input_require_grads(self):
        if self._input_require_grads_handle is not None:
            return

        def require_grads(_module, _inputs, output):
            output.requires_grad_(True)

        self._input_require_grads_handle = self.embed.register_forward_hook(require_grads)

    def forward(self, input_ids, attention_mask=None, labels=None):
        del attention_mask
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return types.SimpleNamespace(loss=loss, logits=logits)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, do_sample=False, **kwargs):
        del attention_mask, do_sample, kwargs
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(cur).logits
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur = torch.cat([cur, next_id], dim=1)
        return cur


def select_torch_dtype(name: str) -> torch.dtype:
    lowered = str(name).lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def is_linear_like_module(module: nn.Module) -> bool:
    """Return true for nn.Linear and quantized linear layers with compatible shape metadata."""
    return (
        hasattr(module, "in_features")
        and hasattr(module, "out_features")
        and hasattr(module, "weight")
        and callable(getattr(module, "forward", None))
    )


def load_tokenizer_and_model(config: dict[str, Any]):
    model_cfg = config.get("model", {})
    if model_cfg.get("type") == "tiny" or model_cfg.get("name_or_path") == "tiny":
        tokenizer = TinyTokenizer()
        model = TinyDecoderOnlyLM(
            vocab_size=int(model_cfg.get("vocab_size", 512)),
            hidden_size=int(model_cfg.get("hidden_size", 16)),
        )
        model.enable_input_require_grads()
        return tokenizer, model

    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = model_cfg["name_or_path"]
    if bool(config.get("runtime", {}).get("require_flash_attention_2", False)) and str(
        model_cfg.get("attn_implementation")
    ) != "flash_attention_2":
        raise ValueError(
            "runtime.require_flash_attention_2=true requires "
            "model.attn_implementation=flash_attention_2"
        )
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
    tokenizer_revision = model_cfg.get("tokenizer_revision", model_cfg.get("revision"))
    if tokenizer_revision:
        tokenizer_kwargs["revision"] = tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = select_torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))
    load_in_8bit = bool(model_cfg.get("load_in_8bit", False))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", False))
    if load_in_8bit and load_in_4bit:
        raise ValueError("Only one of model.load_in_8bit and model.load_in_4bit can be true")
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch_dtype,
    }
    if model_cfg.get("revision"):
        kwargs["revision"] = model_cfg["revision"]
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = str(model_cfg["attn_implementation"])
    if load_in_8bit or load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "model.load_in_8bit/load_in_4bit requires a transformers version with BitsAndBytesConfig "
                "and the bitsandbytes package installed."
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
        )
    if model_cfg.get("device_map"):
        kwargs["device_map"] = model_cfg["device_map"]
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if config.get("training", {}).get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return tokenizer, model


def find_target_linear_modules(
    model: nn.Module,
    target_suffixes: Iterable[str],
) -> list[tuple[str, nn.Module]]:
    suffixes = tuple(str(s).strip() for s in target_suffixes if str(s).strip())
    found = []
    for name, module in model.named_modules():
        if is_linear_like_module(module) and any(name.endswith(suffix) for suffix in suffixes):
            found.append((name, module))
    return found


def collect_module_dims(target_modules: list[tuple[str, nn.Module]]) -> dict[str, dict[str, int]]:
    return {
        name: {"in_dim": int(getattr(module, "in_features")), "out_dim": int(getattr(module, "out_features"))}
        for name, module in target_modules
    }


def model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def model_input_device(model: nn.Module) -> torch.device:
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return next(embeddings.parameters()).device
    except (AttributeError, StopIteration):
        pass
    return model_device(model)
