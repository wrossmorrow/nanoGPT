"""
"Blocks" we may want to use
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nanoGPT import layers
from nanoGPT.config import NanoGPTConfig


class NanoGPTBlock(nn.Module):
    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.ln_1 = (
            layers.LinearLayerNorm(config.n_embed, bias=config.ln_bias)
            if config.linear_layernorms
            else layers.LayerNorm()
        )
        self.attn = (
            layers.BatchedCausalSelfAttention(
                config.n_block,
                config.n_embed,
                config.n_heads,
                scale=config.attn_scale,
                dropout=config.attn_dropout,
                bias=config.attn_bias,
            )
            if config.batched_qkv
            else layers.SplitCausalSelfAttention(
                config.n_block,
                config.n_embed,
                config.n_heads,
                scale=config.attn_scale,
                dropout=config.attn_dropout,
                q_bias=config.q_bias,
                k_bias=config.k_bias,
                v_bias=config.v_bias,
                o_bias=config.o_bias,
            )
        )
        if config.attention_only:
            self.ln_2, self.mlp = nn.Identity(), nn.Identity()
        else:
            self.ln_2 = (
                layers.LinearLayerNorm(config.n_embed, bias=config.ln_bias)
                if config.linear_layernorms
                else layers.LayerNorm()
            )
            self.mlp = layers.FannedGeLU(config.n_embed, fanout=config.ll_fanout, bias=config.ll_bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X + self.attn(self.ln_1(X))
        return self.mlp(self.ln_2(X))
