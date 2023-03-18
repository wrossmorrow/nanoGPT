"""
nanoGPT model configuration
"""
from __future__ import annotations

import json
import os.path
from os import environ

from contextlib import nullcontext
from dataclasses import dataclass, field
from math import sqrt, cos, pi
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import yaml


TorchTypes = Union[torch.float32, torch.bfloat16, torch.float16]

TORCH_TYPES: Dict[str, Any] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class Loadable:

    __conf_name__: Optional[str] = None

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
        return cls(**(yaml.load(config, Loader=yaml.SafeLoader)[cls.__conf_name__]))

    @classmethod
    def from_json(cls, config: str) -> Any:
        return cls(**(json.loads(config)[cls.__conf_name__]))


@dataclass
class CausalSelfAttentionConfig(Loadable):

    __conf_name__ = "attention"

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


@dataclass
class NanoGPTConfig(Loadable):

    __conf_name__ = "model"

    # model definition
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_block: int = 1024  # sequence length; formerly "block_size"
    n_layer: int = 12  # number of layers
    n_heads: int = 12  # number of heads in the multiattention layers
    n_embed: int = 768  # embedding dimension
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?


@dataclass
class TrainingConfig(Loadable):

    __conf_name__ = "training"

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster

    always_save_checkpoint: bool = True  # if True, always save a checkpoint after each eval
    checkpoint_filename: str = "ckpt.pt"

    # TODO: probably remove, in favor of a call setting?
    eval_only: int = False  # if True, script exits right after the first eval
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

    #
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200

    # directories
    data_dir: str = "data"
    out_dir: str = "out"
    log_dir: str = "log"

    # wandb logging
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"  # 'run' + str(time.time())

    # training
    gradient_accumulation_steps: int = 5  # used to simulate larger batch sizes
    n_batch: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size

    # adamw optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    def __post_init__(self) -> None:

        assert self.device in ["cpu", "mps"] or self.device.startswith("cuda"), f'Invalid device = "{self.device}"'
        assert self.dtype in TORCH_TYPES, f'Invalid dtype = "{self.dtype}"'

        # assert (
        #     self.init_from in ["scratch", "resume"] or self.init_from.startswith("gpt")
        # ), f"Invalid init_from = \"{self.init_from}\""

    def checkpoint_file(self) -> str:
        return os.path.join(self.out_dir, self.checkpoint_filename)

    def device_type(self) -> str:
        return "cuda" if "cuda" in self.device else "cpu"  # used for torch.autocast

    def ptdtype(self) -> TorchTypes:
        return TORCH_TYPES[self.dtype]

    def context(self) -> Any:
        device_type, ptdtype = self.device_type(), self.ptdtype()
        if device_type == "cpu":
            return nullcontext()
        return torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    def scaler(self) -> torch.cuda.amp.GradScaler:
        return torch.cuda.amp.GradScaler(enabled=(self.dtype == "float16"))

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it: int) -> float:

        # 0) using?
        if not self.decay_lr:
            return self.learning_rate

        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + cos(pi * self.decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


@dataclass
class DDPConfig:
    rank: int
    local_rank: int
    seed_offset: int
    device: str = field(init=False)
    master_process: bool = field(init=False)

    def __post_init__(self) -> None:
        self.master_process = self.rank == 0
        self.device = f"cuda:{self.local_rank}"

    def wrap(self, model: nn.Module) -> DDP:
        return DDP(model, device_ids=[self.local_rank])

    @staticmethod
    def from_env() -> Optional[DDPConfig]:
        rank = int(environ.get("RANK", -1))
        if rank == -1:  # not a ddp run?
            return None

        local_rank = int(environ["LOCAL_RANK"])
        config = DDPConfig(rank=rank, ddp_local_rank=local_rank, seed_offset=rank)
        torch.cuda.set_device(config.device)
