from __future__ import annotations

import sys
import types

import torch

from dico.model_loader import TinyDecoderOnlyLM, load_tokenizer_and_model


def test_model_loader_passes_flash_attention_implementation(monkeypatch):
    observed = {}

    class Tokenizer:
        pad_token = None
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            observed["tokenizer"] = (name, kwargs)
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            observed["model"] = (name, kwargs)
            return cls()

        def enable_input_require_grads(self):
            pass

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=Tokenizer,
        AutoModelForCausalLM=Model,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    load_tokenizer_and_model(
        {
            "model": {
                "name_or_path": "local-model",
                "torch_dtype": "bfloat16",
                "attn_implementation": "flash_attention_2",
            },
            "runtime": {"require_flash_attention_2": True},
            "training": {"gradient_checkpointing": False},
        }
    )

    assert observed["model"][1]["attn_implementation"] == "flash_attention_2"


def test_model_loader_passes_sdpa_without_requiring_flash_attention(monkeypatch):
    observed = {}

    class Tokenizer:
        pad_token = None
        eos_token = "<eos>"

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

    class Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            observed["kwargs"] = kwargs
            return cls()

        def enable_input_require_grads(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=Tokenizer, AutoModelForCausalLM=Model),
    )

    load_tokenizer_and_model(
        {
            "model": {
                "name_or_path": "local-model",
                "torch_dtype": "bfloat16",
                "attn_implementation": "sdpa",
            },
            "runtime": {"require_flash_attention_2": False},
            "training": {"gradient_checkpointing": False},
        }
    )

    assert observed["kwargs"]["attn_implementation"] == "sdpa"


def test_tiny_model_can_backpropagate_activation_grads_with_all_weights_frozen():
    model = TinyDecoderOnlyLM(vocab_size=32, hidden_size=8)
    model.enable_input_require_grads()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    observed = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        observed["output"] = output

    handle = model.layers[0].self_attn.q_proj.register_forward_hook(hook)
    try:
        output = model(torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]]))
        output.loss.backward()
    finally:
        handle.remove()

    assert observed["output"].grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())
