"""
Activation functions we may want to use.
"""
import torch


SQRT_2_OVER_PI = 0.7978845608028654
OTHER_CONST = 0.044715


# good to enable when not using torch.compile, disable when using (our default)
# @torch.jit.script
def new_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Implementation of the GELU activation function currently in Google
    BERT repo (identical to OpenAI GPT).

    Reference: https://arxiv.org/abs/1606.08415
    """
    # Note: math.sqrt(2.0 / math.pi) ~ 0.7978845608028654
    erf_approx = torch.tanh(SQRT_2_OVER_PI * (x + OTHER_CONST * torch.pow(x, 3.0)))
    return 0.5 * x * (1.0 + erf_approx)
