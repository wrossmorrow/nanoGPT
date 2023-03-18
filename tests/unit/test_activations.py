import pytest

import torch
import torch.linalg as la

from nanoGPT.activations import new_gelu


@pytest.mark.parametrize(
    "B, M, N",
    (
        (32, 65, 65),
        (32, 768, 1024),
    ),
)
def test_gelu(B: int, M: int, N: int) -> None:

    x = torch.zeros(M, N)
    y = new_gelu(x)
    assert y.shape == x.shape
    assert la.norm(y, ord="fro") <= 1.0e-6

    x = torch.rand(M, N)
    y = new_gelu(x)
    assert y.shape == x.shape
    # what can we assert about values?
