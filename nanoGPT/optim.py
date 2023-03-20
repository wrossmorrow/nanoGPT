from __future__ import annotations

import os.path

from inspect import signature
from typing import Any
from warnings import warn

import torch
from torch import nn

from nanoGPT import layers
from nanoGPT.config import CheckpointConfig, TrainingConfig


def configure_optimizer(model: nn.Module, config: TrainingConfig, device: str) -> torch.optim.Optimizer:
    """
    This long function is unfortunately doing something very simple and
    is being very defensive. We are separating out all parameters of the
    model into two buckets: those that will experience weight decay for
    regularization and those that won't (biases, and layernorm/embedding
    weights). We are then returning the PyTorch optimizer object.

    Effective arguments:

        named_modules: Iterator[Any],
        named_parameters: Iterator[Any],
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,

    TODO: admit layer rules for decay via configuration
    """

    # separate out all parameters to those that will and won't
    # experience regularizing weight decay
    decay, no_decay = set(), set()
    do_decay = (nn.Linear,)
    dont_decay = (nn.LayerNorm, layers.LinearLayerNorm, layers.LayerNorm, nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight"):
                if isinstance(m, do_decay):
                    decay.add(fpn)  # whitelist modules WILL be weight decayed
                elif isinstance(m, dont_decay):
                    no_decay.add(fpn)  # blacklist modules will NOT be weight decayed

    # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
    # will appear in the no_decay and decay sets respectively after the above.
    # In addition, because named_parameters() doesn't return duplicates, it
    # will only return the first occurence, key'd by 'transformer.wte.weight', below.
    # so let's manually remove 'lm_head.weight' from decay set. This will include
    # this tensor into optimization via transformer.wte.weight only, and not decayed.
    decay.remove("lm_head.weight")

    # validate that we considered every parameter, and "partition" in the
    # sense that we either decay or don't
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params, union_params = decay & no_decay, decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} in both decay/no_decay sets"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} not separated into either decay/no_decay set"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    extra_args = {}
    if ("cuda" in device) and ("fused" in signature(torch.optim.AdamW).parameters):
        warn("using fused AdamW")
        extra_args["fused"] = True

    # return optimizer object
    learning_rate = config.learning_rate
    beta1, beta2 = config.beta1, config.beta2
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)


def save_checkpoint(
    iter_num: int,
    best_val_loss: float,
    config: CheckpointConfig,
    optimizer: torch.optim.Optimizer,
) -> None:
    torch.save(
        {
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            "optim_state": optimizer.state_dict(),
        },
        config.checkpoint_filename("optim"),
    )


def load_checkpoint(
    config: CheckpointConfig,
    device: str,
    optimizer: torch.optim.Optimizer,
) -> None:
    filename = os.path.join(config.checkpoint_dir, config.optim_checkpoint)
    checkpoint = torch.load(filename, map_location=device)
    optimizer.load_state_dict(checkpoint["optim_state"])


class WrappedTorchOptimizer:
    def __init__(self, model: nn.Module, config: TrainingConfig, device: str) -> None:
        self._optimizer = configure_optimizer(model, config, device)

    def __getattr__(self, attr: str) -> Any:
        if hasattr(self._optimizer, attr):
            return getattr(self._optimizer, attr)
        raise AttributeError(f"{self.__class__.__name__} has not attribute {attr}")

    def to_checkpoint(self, iter_num: int, best_val_loss: float, config: CheckpointConfig) -> None:
        torch.save(
            {
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "optim_state": self._optimizer.state_dict(),
            },
            config.checkpoint_filename("optim"),
        )

    @staticmethod
    def from_checkpoint(self, config: CheckpointConfig, device: str) -> WrappedTorchOptimizer:
        # filename = config.checkpoint_filename("optim")
        # checkpoint = torch.load(filename, map_location=device)
        # need to initialize ...
        # optimizer.load_state_dict(checkpoint["optim_state"])
        pass
