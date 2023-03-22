"""
Layers we might use
"""
from __future__ import annotations

from math import sqrt
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from nanoGPT.activations import new_gelu


class LayerNorm(nn.Module):
    """LayerNorm without any weights or biases."""

    def __init__(self, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(X, (X.shape[-1],), eps=self.eps)


class LinearLayerNorm(nn.Module):
    """
    LayerNorm but with weights and (optional) bias. PyTorch doesn't
    support simply `bias=False`.
    """

    def __init__(self, ndim: int, bias: bool, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(X, self.weight.shape, self.weight, self.bias, self.eps)


class QuadraticForm(nn.Module):
    """
    A "quadratic form" layer, like X' W X / s. This is a basic
    unit of what appears in a self-attention head like (in math
    not "transposed" ML/DL notation)

        (W_K X)'(W_Q X) = X' W_K' W_Q X = X' W_{K,Q} X

    This is more expensive than in multihead attentions where the weight
    matrices are "short"/"wide", as opposed to "thin"/"tall".

    Note W ~ X.shape[0] x X.shape[0] and 

        (X' W X)[j, k] = X[:, j]' W X[:, k]

    which can appear to have a "causal mask" iff upper triangular, i.e.

        X[:, j]' W X[:, k] = 0 when j > k

    we can impose this explicitly but that seems hacky and would suggest 
    wasted parameters? (Seems that would apply to attention as well?)

    Composable with a residual connection with
    
        X + W_V X' W_{K,Q} X

    when W_V has the same shape as X[b, :, :]. Or we can just "replicate"
    attention as 

        X + W_V X F(X)

    for any (matrix) function F that is upper triangular for causal models
    """

    def __init__(self, ndim: int, scale: Optional[float] = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((ndim, ndim)))
        self.scale = sqrt(ndim) if scale is None else scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Mimic torch.nn.Linear parameter initialization:
        #
        #   https://github.com/pytorch/pytorch/blob/ \
        #       7324aef9a86babd43b037b14b4cfef234e4c5db2/\
        #       torch/nn/modules/linear.py#L107
        #
        # Based on method described by Kaiming, et al. (2015):
        #
        #   https://arxiv.org/abs/1502.01852
        #
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        # TODO: stronger justification for using this approach

    def extra_repr(self) -> str:
        in_f, out_f = self.weight.shape
        return f"in_features={in_f}, out_features={out_f}, scale={self.scale}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Y = F.linear(x, self.weight, None)
        return Y @ x.transpose(-2, -1) / self.scale


class SplitQKV:
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        q_bias: bool = False,
        k_bias: bool = False,
        v_bias: bool = False,
    ) -> None:
        self.n_heads = n_heads
        self.QLL = nn.Linear(n_embed, n_embed, bias=q_bias)
        self.KLL = nn.Linear(n_embed, n_embed, bias=k_bias)
        self.VLL = nn.Linear(n_embed, n_embed, bias=v_bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T, C = X.size()  # batch size, sequence length/block size, embedding dim
        H = self.n_heads
        D = C // H  # also config.n_attno...

        Q = self.QLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D
        K = self.KLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D
        V = self.VLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D

        # now stack... but then to just split?
        return torch.cat([Q, K, V])


class FFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(X, norm="ortho").real


class SplitCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_block: int,
        n_embed: int,
        n_heads: int,
        scale: Optional[float] = None,
        dropout: float = 0.2,
        q_bias: bool = False,
        k_bias: bool = False,
        v_bias: bool = False,
        o_bias: bool = True,
    ) -> None:
        super().__init__()

        # data copied from config (TODO: just pass params?)
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.dropout = dropout
        self.quadf_scale = sqrt(n_embed / n_heads) if scale is None else scale

        # split Q/K/V layers (to test out invariance to biases)
        self.QLL = nn.Linear(n_embed, n_embed, bias=q_bias)
        self.KLL = nn.Linear(n_embed, n_embed, bias=k_bias)
        self.VLL = nn.Linear(n_embed, n_embed, bias=v_bias)

        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=o_bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # NOTE: ignoring flash attention to manipulate better; often unusable anyway
        # NOTE: can probably add `diagonal=1` to tril to ignore self-correlatiions?
        self.mask = torch.tril(torch.ones(n_block, n_block))
        self.mask = self.mask.view(1, 1, n_block, n_block)
        self.register_buffer("bias", self.mask)

    def _get_dims(self, X: torch.Tensor) -> Tuple[int, int, int, int, int]:
        B, T, C = X.size()  # batch size, sequence length/block size, embedding dim
        H = self.n_heads
        D = C // H  # also config.n_attno...
        return B, H, T, C, D

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, H, T, C, D = self._get_dims(X)

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        Q = self.QLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D
        K = self.KLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D
        V = self.VLL(X).view(B, T, H, D).transpose(1, 2)  # B x H x T x D

        A = (Q @ K.transpose(-2, -1)) / self.quadf_scale
        A = A.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
        A = F.softmax(A, dim=-1)
        A = self.attn_dropout(A)

        Y = A @ V
        Y = Y.transpose(1, 2).contiguous().view(B, T, C)
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)

        return Y


class BatchedCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_block: int,
        n_embed: int,
        n_heads: int,
        scale: Optional[float] = None,
        dropout: float = 0.2,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.n_embed = n_embed
        self.n_heads = n_heads
        self.dropout = dropout
        self.quadf_scale = sqrt(n_embed / n_heads) if scale is None else scale

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)

        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask
        self.mask = torch.tril(torch.ones(n_block, n_block))
        self.mask = self.mask.view(1, 1, n_block, n_block)
        self.register_buffer("bias", self.mask)

    def _get_dims(self, X: torch.Tensor) -> Tuple[int, int, int, int, int]:
        B, T, C = X.size()  # batch size, sequence length/block size, embedding dim
        H = self.n_heads
        D = C // H  # also config.n_attno...
        return B, H, T, C, D

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, H, T, C, D = self._get_dims(X)

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        Q, K, V = self.c_attn(X).split(self.n_embed, dim=2)
        Q = Q.view(B, T, H, D).transpose(1, 2)  # (B, nh, T, hs)
        K = K.view(B, T, H, D).transpose(1, 2)  # (B, nh, T, hs)
        V = V.view(B, T, H, D).transpose(1, 2)  # (B, nh, T, hs)

        A = (Q @ K.transpose(-2, -1)) / self.quadf_scale
        A = A.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))  # type: ignore
        A = F.softmax(A, dim=-1)
        A = self.attn_dropout(A)

        Y = A @ V
        Y = Y.transpose(1, 2).contiguous().view(B, T, C)
        Y = self.c_proj(Y)
        Y = self.resid_dropout(Y)

        return Y


# TODO: subclass from Batched?
class FlashCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_block: int,
        n_embed: int,
        n_heads: int,
        scale: Optional[float] = None,
        dropout: float = 0.2,
        bias: bool = False,
    ) -> None:

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        assert hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ), "Current environment does not appear to have access to flash attention"

        super().__init__()

        self.n_embed = n_embed
        self.n_heads = n_heads
        self.dropout = dropout
        self.quadf_scale = sqrt(n_embed / n_heads) if scale is None else scale

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=bias)

        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _get_dims(self, X: torch.Tensor) -> Tuple[int, int, int, int, int]:
        B, T, C = X.size()  # batch size, sequence length/block size, embedding dim
        H = self.n_heads
        D = C // H  # also config.n_attno...
        return B, H, T, C, D

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, H, T, C, D = self._get_dims(X)

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        Q, K, V = self.c_attn(X).split(self.n_embed, dim=2)
        Q = Q.view(B, T, H, D).transpose(1, 2)  # (B, nh, T, hs)
        K = K.view(B, T, H, D).transpose(1, 2)  # (B, nh, T, hs)
        V = V.view(B, T, H, D).transpose(1, 2)  # (B, nh, T, hs)

        Y = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.dropout, is_causal=True)

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # include residual connection here so we can use Identity 
        # in the block when excluding the feedforward entirely
        return X + self.dropout(self.c_proj(new_gelu(self.c_fc(X))))
