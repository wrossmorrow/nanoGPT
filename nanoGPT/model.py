"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
from __future__ import annotations

import os.path

from math import sqrt, log10
from typing import Any, Optional, Tuple
from warnings import warn

import torch
import torch.nn as nn
from torch.nn import functional as F

from nanoGPT import layers
from nanoGPT import blocks
from nanoGPT.config import CheckpointConfig, GenerateConfig, NanoGPTConfig, NanoGPTContext


class NanoGPT(nn.Module):
    def __init__(self, config: NanoGPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.n_block is not None

        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embed),
                "wpe": nn.Embedding(config.n_block, config.n_embed),
                "drop": nn.Dropout(config.dropout),
                "heads": nn.ModuleList([blocks.NanoGPTBlock(config) for _ in range(config.n_layer)]),
                "ln_f": (
                    layers.LinearLayerNorm(config.n_embed, bias=config.ln_bias)
                    if config.linear_layernorms
                    else layers.LayerNorm()
                ),
            }
        )

        if config.linear_layernorms:
            warn("NOTE: composing a Linear after a LinearLayerNorm composes parameters")
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # TODO: investigate
        #
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless.
        #
        # Reference: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight  # type: ignore

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / sqrt(2 * config.n_layer))

        # report number of parameters (flexibly, if verbosely)
        num_params, num_params_scale, num_params_sunit = self.get_verbose_num_params()
        print(f"number of parameters: {num_params/num_params_scale:.2f}{num_params_sunit} ({num_params:,})")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()  # type: ignore
        return n_params

    def get_verbose_num_params(self, non_embedding: bool = True) -> Tuple[int, int, str]:
        num_params = self.get_num_params(non_embedding=non_embedding)
        num_params_log10 = int(log10(num_params))  # num_params = 10^(num_params_log10)
        num_params_order = (num_params_log10 - num_params_log10 % 3) // 3
        num_params_scale = 10 ** (3 * int(num_params_order))
        num_params_sunit: str = {0: "", 1: "k", 2: "M", 3: "B", 4: "T"}[num_params_order]
        return num_params, num_params_scale, num_params_sunit

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[Any] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        B, T = idx.size()
        assert (
            T <= self.config.n_block
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.n_block}"

        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, T)

        # forward the GPT model itself
        # * Token embeddings of shape B x T x C (type issue: "Tensor" is not callable)
        # * Position embeddings of shape 1 x T x C (type issue: "Tensor" is not callable)
        # * Dropout (type issue: "Tensor" is not callable)
        # * All the attention blocks (type issues: Module has no attr __iter__, "Tensor" is not callable)
        # * LayerNorm (type issue: "Tensor" is not callable)
        tok_emb = self.transformer.wte(idx)  # type: ignore [operator]
        pos_emb = self.transformer.wpe(pos)  # type: ignore [operator]
        X = self.transformer.drop(tok_emb + pos_emb)  # type: ignore [operator]
        for block in self.transformer.heads:  # type: ignore [union-attr]
            X = block(X)  # type: ignore [operator]
        X = self.transformer.ln_f(X)  # type: ignore [operator]

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(X[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        config: GenerateConfig,
        context: NanoGPTContext,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape
        (B, T)) and complete the sequence max_new_tokens times, feeding the
        predictions back into the model each time. Most likely you'll want
        to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(config.max_new_tokens):

            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.n_block else (idx[:, -self.config.n_block :])

            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / config.temperature

            # optionally crop the logits to only the top k options
            if config.top_k is not None:
                v, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def to_checkpoint(self, iter_num: int, best_val_loss: float, config: CheckpointConfig) -> None:
        torch.save(
            {
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "model_config": self.config.dict(),
                "model_state": self.state_dict(),
            },
            config.checkpoint_filename("model"),
        )

    @staticmethod
    def from_checkpoint(config: CheckpointConfig, device: str) -> NanoGPT:

        # load model checkpoint
        filename = os.path.join(config.checkpoint_dir, config.model_checkpoint)
        checkpoint = torch.load(filename, map_location=device)

        # create the model
        model_config = NanoGPTConfig(**checkpoint["model_config"])
        model = NanoGPT(model_config)

        # get model's state dict stored in checkpoint
        state_dict = checkpoint["model_state"]

        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        # load that state dict in this model
        model.load_state_dict(state_dict)

        return model
