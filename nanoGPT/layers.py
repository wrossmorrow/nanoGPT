"""
Layers we might use
"""
from __future__ import annotations

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .activations import new_gelu
from .config import CausalSelfAttentionConfig


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1.0e-5)


class QuadraticForm(nn.Module):
    """A "quadratic form" layer, like X' W X / s. This is a basic
    unit of what appears in a self-attention head like (in math
    not "transposed" ML/DL notation)

        (W_K X)'(W_Q X) = X' W_K' W_Q X = X' W_{K,Q} X

    for X ~ T x C, W_Q, W_K ~ T x C and W_{K,Q} ~ C x C. Training the
    product W_{Q,K} explicitly saves

        2TC - CC = (2T-C)C = 2(T-C/2)C

    parameters, which is positive when T > C/2 (the token sequence length
    is larger than half the embedding dimension).
    """

    def __init__(self, ndim: int, scale: Optional[float] = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((ndim, ndim)))
        self.scale = sqrt(ndim) if scale is None else scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic torch.nn.Linear parameter initialization:
        # https://github.com/pytorch/pytorch/blob/7324aef9a86babd43b037b14b4cfef234e4c5db2/torch/nn/modules/linear.py#L107
        #
        # Based on method described by Kaiming, et al. (2015):
        # https://arxiv.org/abs/1502.01852
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        # TODO: stronger justification for using this approach

    def extra_repr(self) -> str:
        in_f, out_f = self.weight.shape
        return f"in_features={in_f}, out_features={out_f}, scale={self.scale}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Y = F.linear(x, self.weight, None)
        return Y @ x.transpose(-2, -1) / self.scale


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: CausalSelfAttentionConfig,
    ) -> None:
        super().__init__()

        # data copied from config (TODO: just pass params?)
        self.quadf_scale = config.scale
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.dropout = config.dropout

        # key, query, value projections for all heads, but in a batch
        #
        #   self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        #
        # NOTE: if we batch we can't test out invariance to some biases
        self.QLL = nn.Linear(config.n_embed, config.n_embed, bias=config.Q_bias)
        self.KLL = nn.Linear(config.n_embed, config.n_embed, bias=config.K_bias)
        self.VLL = nn.Linear(config.n_embed, config.n_embed, bias=config.V_bias)

        # output projection
        self.OLL = nn.Linear(config.n_embed, config.n_embed, bias=config.O_bias)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # NOTE: ignoring flash attention to manipulate better; often unusable anyway
        mask = torch.tril(torch.ones(config.n_block, config.n_block))
        mask = mask.view(1, 1, config.n_block, config.n_block)
        self.register_buffer("bias", mask)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, C = X.size()  # batch size, sequence length/block size, embedding dim
        H = self.n_heads
        D = C // H  # also config.n_attno...

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        Q = self.QLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D
        K = self.KLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D
        V = self.VLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D

        # scaled "quadratic form"
        A = (Q @ K.transpose(-2, -1)) / self.quadf_scale

        # mask using tril buffer from above, setting -> -inf for softmax -> 0
        #
        #   A[0:B,0:H,t:,t:] = -inf for all t = 1,...,T
        #
        A = A.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore

        # B x H x T x T -> B x H x T x Softmax(T)
        A = F.softmax(A, dim=-1)

        # attention regularization?
        A = self.attn_dropout(A)

        # B x H x T x T @ B x H x T x D = B x H x T x D
        Y = A @ V

        # re-assemble all head outputs side by side (does contiguous copy?)
        #
        #   B x H x T x (C/H) -> B x T x H x (C/H) -> B x T x C
        #
        Y = Y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        #
        #   B x T x C @ C x C = B x T x C
        #
        Y = self.OLL(Y)

        # output regularization
        Y = self.resid_dropout(Y)

        return Y


class FannedGeLU(nn.Module):
    def __init__(
        self,
        n_embed: int,
        fanout: int = 4,
        bias: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        assert fanout > 0, f"fanout <= 0 ({fanout}) is invalid"
        self.c_fc = nn.Linear(n_embed, fanout * n_embed, bias=bias)
        self.c_proj = nn.Linear(fanout * n_embed, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
