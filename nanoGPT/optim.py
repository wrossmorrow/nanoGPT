from inspect import signature
from typing import Any, Iterator, Tuple

import torch
from torch import nn

from . import layers


def configure_optimizers(
    named_modules: Iterator[Any],
    named_parameters: Iterator[Any],
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str,
) -> torch.optim.Optimizer:
    """
    This long function is unfortunately doing something very simple and
    is being very defensive. We are separating out all parameters of the
    model into two buckets: those that will experience weight decay for
    regularization and those that won't (biases, and layernorm/embedding
    weights). We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't
    # experience regularizing weight decay
    decay, no_decay = set(), set()
    do_decay = (nn.Linear,)
    dont_decay = (nn.LayerNorm, layers.LayerNorm, nn.Embedding)
    for mn, m in named_modules:
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
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif isinstance(m, dont_decay):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

    # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
    # will appear in the no_decay and decay sets respectively after the above.
    # In addition, because named_parameters() doesn't return duplicates, it
    # will only return the first occurence, key'd by 'transformer.wte.weight', below.
    # so let's manually remove 'lm_head.weight' from decay set. This will include
    # this tensor into optimization via transformer.wte.weight only, and not decayed.
    decay.remove("lm_head.weight")

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in named_parameters}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    extra_args = {}
    if (device_type == "cuda") and ("fused" in signature(torch.optim.AdamW).parameters):
        extra_args["fused"] = True
        print("using fused AdamW")

    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
