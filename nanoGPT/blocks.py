"""
"Blocks" we may want to use
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import layers
from .config import NanoGPTConfig


class GPTBlock(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.ln_1 = layers.LayerNorm(config.n_embed, bias=config.bias)
        self.attn = layers.CausalSelfAttention(config)
        self.ln_2 = layers.LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = layers.FannedGeLU(config.n_embed, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
