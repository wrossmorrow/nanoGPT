import pytest

import torch


@pytest.fixture
def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
