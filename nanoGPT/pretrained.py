"""
For posterity
"""

from typing import Dict, Optional

import torch
from transformers import GPT2LMHeadModel  # huggingface model

from .config import NanoGPTConfig
from .model import NanoGPT


def from_pretrained_gpt2(size: str = "default", override_args: Optional[Dict] = None) -> NanoGPT:

    override_args = override_args or {}  # default to empty dict

    # only dropout can be overridden see more notes below
    # TODO: if that's the case, we shouldn't admit a whole Dict
    # of override arguments; misleading API.
    assert all(k == "dropout" for k in override_args)

    print(f"loading weights from pretrained gpt2 ({size})")

    # init a huggingface/transformers model
    hf_model_type = "gpt2" + (f"-{size}" if size != "default" else "")
    model_hf = GPT2LMHeadModel.from_pretrained(hf_model_type)
    sd_hf = model_hf.state_dict()

    # create a from-scratch initialized NanoGPT model

    config = NanoGPTConfig.gpt2(size, override_args.get("dropout", None))
    model = NanoGPT(config)

    # get the "state dict" for storing model parameter copies
    sd = model.state_dict()

    # discard this mask / buffer, not a param
    sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

    # Copy while ensuring all of the parameters are aligned and match in names
    # and shapes
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]  # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]  # same, just the mask (buffer)
    transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

    # Basically the openai checkpoints use a "Conv1D" module, but we only want
    # to use a vanilla Linear layer. This means that we have to transpose these
    # weights when we import them.
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])

    return model
