"""
Activation functions we may want to use.
"""
import torch


# good to enable when not using torch.compile, disable when using (our default)
# @torch.jit.script
def new_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the GELU activation function currently in Google
    BERT repo (identical to OpenAI GPT).

    Reference: https://arxiv.org/abs/1606.08415
    """
    # Note: math.sqrt(2.0 / math.pi) ~ 0.7978845608028654
    errf_approx = torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3.0)))
    return 0.5 * x * (1.0 + errf_approx)
