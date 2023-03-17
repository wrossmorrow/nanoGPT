"""
nanoGPT model configuration
"""
from __future__ import annotations

import json

from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Optional

import yaml


class Loadable:
    @classmethod
    def from_file(cls, filename: str) -> Any:
        if filename.lower().endswith(".yaml") or filename.lower().endswith(".yml"):
            return cls.from_yaml_file(filename)
        if filename.lower().endswith(".json"):
            return cls.from_json_file(filename)

    @classmethod
    def from_yaml_file(cls, filename: str) -> Any:
        with open(filename, "r") as f:
            return cls.from_yaml(f.read())

    @classmethod
    def from_json_file(cls, filename: str) -> Any:
        with open(filename, "r") as f:
            return cls.from_json(f.read())

    @classmethod
    def from_yaml(cls, config: str) -> Any:
        return cls(**yaml.load(config, Loader=yaml.SafeLoader))

    @classmethod
    def from_json(cls, config: str) -> Any:
        return cls(**json.loads(config))


@dataclass
class NanoGPTConfig(Loadable):
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12  # number of layers
    n_block: int = 1024  # sequence length; formerly "block_size"
    n_heads: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


@dataclass
class CausalSelfAttentionConfig(Loadable):
    n_block: int
    n_embed: int
    n_heads: int
    n_attnd: int = field(init=False)
    dropout: float = 0.2
    scale: Optional[float] = None
    Q_bias: bool = False
    K_bias: bool = False
    V_bias: bool = False
    O_bias: bool = False

    def __post_init__(self) -> None:
        assert self.n_embed % self.n_heads == 0
        self.n_attnd = self.n_embed // self.n_heads
        if self.scale is None:
            self.scale = sqrt(self.n_attnd)  # TODO: try 2 * self.n_embed
