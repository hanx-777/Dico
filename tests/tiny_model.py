import types
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class TinyTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._token_to_id = {self.pad_token: 0, self.eos_token: 1}
        self._id_to_token = {0: self.pad_token, 1: self.eos_token}

    def _tokenize(self, text: str):
        return text.replace("\n", " \n ").split()

    def _id_for_token(self, token: str) -> int:
        if token not in self._token_to_id:
            idx = len(self._token_to_id)
            self._token_to_id[token] = idx
            self._id_to_token[idx] = token
        return self._token_to_id[token]

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def encode(self, text: str, add_special_tokens: bool = False):
        ids = [self._id_for_token(tok) for tok in self._tokenize(text)]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        toks = []
        for idx in ids:
            idx = int(idx)
            if skip_special_tokens and idx in {self.pad_token_id, self.eos_token_id}:
                continue
            toks.append(self._id_to_token.get(idx, f"<unk{idx}>"))
        return " ".join(toks).replace(" \n ", "\n")

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]


@dataclass
class TinyConfig:
    vocab_size: int = 128
    hidden_size: int = 16
    model_type: str = "tiny-dico"

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "model_type": self.model_type,
        }


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.o_proj(torch.tanh(self.q_proj(x)) + self.v_proj(x))


class TinyDecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 16):
        super().__init__()
        self.config = TinyConfig(vocab_size=vocab_size, hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([TinyAttention(hidden_size)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = x + layer(x)
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
        del do_sample, kwargs
        cur = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(cur).logits
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur = torch.cat([cur, next_id], dim=1)
        return cur
