"""
nanoGPT model configuration
"""
from __future__ import annotations

import json

from contextlib import nullcontext
from datetime import datetime as dt
from dataclasses import asdict, dataclass, field, fields
from math import sqrt
from os import environ, path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import yaml


TORCH_TYPES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

SHARED_GPT2_CONF: Dict[str, Union[bool, int, str, float]] = {
    "vocab_size": 50257,
    "n_block": 1024,
    "linear_layernorms": True,
    "ln_bias": True,
    "ll_bias": True,
    "batched_qkv": True,
    "attn_bias": True,
}

GPT2_CONF: Dict[str, Dict[str, Union[bool, int, str, float]]] = {
    "gpt2-default": {"n_layer": 12, "n_heads": 12, "n_embed": 768, **SHARED_GPT2_CONF},  # 124M params
    "gpt2-medium": {"n_layer": 24, "n_heads": 16, "n_embed": 1024, **SHARED_GPT2_CONF},  # 350M params
    "gpt2-large": {"n_layer": 36, "n_heads": 20, "n_embed": 1280, **SHARED_GPT2_CONF},  # 774M params
    "gpt2-xl": {"n_layer": 48, "n_heads": 25, "n_embed": 1600, **SHARED_GPT2_CONF},  # 1.556B params
}


def simpleiso() -> str:
    return dt.now().isoformat().split(".")[0].replace(":", "-")


class Loadable:
    """Logic to read/write to/from YAML and JSON formats"""

    __conf_name__: Optional[str] = None

    @classmethod
    def from_file(cls, filename: str, **kwargs) -> Any:
        extension = filename.lower().split(".")[-1]
        if extension in ["yml", "yaml"]:
            return cls.from_yaml_file(filename, **kwargs)
        if extension == ".json":
            return cls.from_json_file(filename, **kwargs)
        raise ValueError(f'Unknown config file extension "{extension}"')

    @classmethod
    def from_yaml_file(cls, filename: str, **kwargs) -> Any:
        with open(filename, "r") as f:
            return cls.from_yaml(f.read(), **kwargs)

    @classmethod
    def from_json_file(cls, filename: str, **kwargs) -> Any:
        with open(filename, "r") as f:
            return cls.from_json(f.read(), **kwargs)

    @classmethod
    def from_yaml(cls, config: str, **kwargs) -> Any:
        data = yaml.load(config, Loader=yaml.SafeLoader).get(cls.__conf_name__, {})
        return cls.from_data(data, **kwargs)

    @classmethod
    def from_json(cls, config: str, **kwargs) -> Any:
        data = json.loads(config)[cls.__conf_name__]
        return cls.from_data(data, **kwargs)

    @classmethod
    def from_data(cls, data: Dict, **kwargs) -> Any:
        data.update({k: v for k, v in kwargs.items() if k in [f.name for f in fields(cls)]})
        return cls(**data)

    def to_yaml(self) -> str:
        return yaml.dump({self.__conf_name__: asdict(self)})

    def to_json(self) -> str:
        return json.dumps({self.__conf_name__: asdict(self)})

    def to_yaml_file(self, filename: str, append: bool = False) -> None:
        mode = "a" if append else "w"
        with open(filename, mode) as f:
            f.write(self.to_yaml())

    def to_json_file(self, filename: str, append: bool = False) -> None:
        mode = "a" if append else "w"
        with open(filename, mode) as f:
            f.write(self.to_json())


class Dictable:
    """Logic to convert to dict with a simple owned method call"""

    def dict(self) -> Dict:
        return asdict(self)


# TODO: implement
class Overridable:
    """Logic to accept kwargs and override any _non default_ value"""

    pass


@dataclass
class NanoGPTConfig(Loadable, Dictable):

    __conf_name__ = "model"

    # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size: int = field(default=50304, metadata={"help": "Vocabulary size"})

    n_block: int = field(default=1024, metadata={"help": "Sequence length (aka block_size)"})
    n_layer: int = field(default=12, metadata={"help": "Number of layers"})
    n_heads: int = field(default=12, metadata={"help": "Number of heads in the multiattention layers"})
    n_embed: int = field(default=768, metadata={"help": "Embedding dimension"})
    # n_attnd: int = field(init=False, metadata={"cli": False})

    batched_qkv: bool = field(
        default=False, metadata={"help": "batch queries, keys, and values (all with or without bias)"}
    )

    # for pretraining 0 is good, for finetuning try 0.1+
    dropout: float = field(default=0.0, metadata={"help": "Dropout fraction"})

    linear_layernorms: bool = field(
        default=False, metadata={"help": 'Use "linear" layernorms with weight and (maybe) bias'}
    )
    ln_bias: bool = field(default=False, metadata={"help": "Use a bias inside Linear layers (not in attention heads)"})
    ll_bias: bool = field(default=False, metadata={"help": "Use a bias inside LayerNorm modules"})

    attn_scale: Optional[float] = field(
        default=None, metadata={"help": "Scale factor, to divide the key-query product by"}
    )
    attn_dropout: float = field(default=0.2, metadata={"help": "Distinct attention dropout"})
    attn_bias: bool = field(default=False, metadata={"help": "Include a bias term in all attention head linear layers"})
    q_bias: bool = field(default=False, metadata={"help": "Include a bias term in the queries"})
    k_bias: bool = field(default=False, metadata={"help": "Include a bias term in the keys"})
    v_bias: bool = field(default=False, metadata={"help": "Include a bias term in the values"})
    o_bias: bool = field(default=False, metadata={"help": "Include a bias term in the concatenate-project step"})

    def __post_init__(self) -> None:
        assert self.n_embed % self.n_heads == 0
        # self.n_attnd = self.n_embed // self.n_heads
        if self.attn_scale is None:  # TODO: try 2 * self.n_embed
            self.attn_scale = sqrt(self.n_embed // self.n_heads)
            # self.attn_scale = sqrt(self.n_attnd)

    @staticmethod
    def gpt2(size: str = "default") -> NanoGPTConfig:
        """
        Return a GPT-2 like configuration, in our language.

        NOTE: This only returns _configuration_ roughly equivalent to what
        GPT-2 would use, NOT any parameters unlike the setup in Karpathy's
        nanoGPT. (TODO: may reinstate that, but it seems unecessary.)

        TODO: typing is picky here; ignore for now
        """
        assert size in ["default", "medium", "large", "xl"], f'Unknown GPT size "{size}"'
        return NanoGPTConfig(**(GPT2_CONF[f"gpt2-{size}"]))  # type: ignore


@dataclass
class TrainingConfig(Loadable, Dictable):

    __conf_name__ = "training"

    # datatype
    dtype: str = field(
        default="bfloat16",
        metadata={
            "help": "torch datatype to use (f16 will use a GradScaler)",
            "choices": ["float32", "bfloat16", "float16"],
        },
    )  # the latter will auto implement a GradScaler

    # "compile" the model or not
    torch_compile: bool = field(default=True, metadata={"help": "Compile the model to be faster (requires torch v2.0)"})

    estimate_mfu: bool = field(default=True, metadata={"help": "Keep up a running estimate of MFU"})

    # evaluation/logging intervals and associated data
    eval_interval: int = field(default=2000, metadata={"help": "Iteration interval for train/test evaluations"})
    eval_iters: int = field(
        default=200, metadata={"help": "Number of iterations to use in evaluations of train/test loss"}
    )
    always_save_checkpoint: bool = field(default=True, metadata={"help": "Always save a checkpoint after each eval"})

    # directories

    # logging/wandb logging
    log_interval: int = field(default=1, metadata={"help": "Iteration interval for logging"})
    log_dir: str = field(default="log", metadata={"help": "Directory where  logs are stored"})
    wandb_log: bool = field(default=False, metadata={"help": "Use weights-and-biases logging"})  # disabled by default
    wandb_project: str = field(default="owt", metadata={"help": "Project for weights-and-biases logging"})
    wandb_run_name: str = field(
        default=f"nanogpt-run-{simpleiso()}", metadata={"help": "Run name for weights-and-biases loggin"}
    )

    # training
    gradient_accumulation_steps: int = field(default=5, metadata={"help": "simulate larger batch sizes"})
    n_batch: int = field(default=12, metadata={"help": "Number of elements in a batch"})
    # Note: if gradient_accumulation_steps > 1, this is the micro-batch size

    # adamw optimizer
    learning_rate: float = field(default=6e-4, metadata={"help": "max learning rate"})
    max_iters: int = field(default=600000, metadata={"help": "total number of training iterations"})
    weight_decay: float = field(default=1e-1, metadata={"help": "TBD"})
    beta1: float = field(default=0.9, metadata={"help": "TBD"})
    beta2: float = field(default=0.95, metadata={"help": "TBD"})
    grad_clip: float = field(default=1.0, metadata={"help": "clip gradients at this value, or disable if == 0.0"})

    # learning rate decay settings
    decay_lr: bool = field(default=True, metadata={"help": "Whether to decay the learning rate"})
    warmup_iters: int = field(default=2000, metadata={"help": 'How many steps to "warm up" for'})
    lr_decay_iters: int = field(default=600000, metadata={"help": "TBD"})  # should be ~= max_iters per Chinchilla
    min_lr: float = field(
        default=6e-5, metadata={"help": "Minimum learning rate"}
    )  # should be ~= learning_rate/10 per Chinchilla

    def __post_init__(self) -> None:
        assert self.dtype in TORCH_TYPES, f'Invalid dtype = "{self.dtype}"'

    def torch_type(self) -> torch.dtype:
        return TORCH_TYPES[self.dtype]


@dataclass
class LoggingConfig(Loadable, Dictable):

    __conf_name__ = "dataset"

    log_dir: str = field(default="log", metadata={"help": "Directory where  logs are stored"})
    log_interval: int = field(default=1, metadata={"help": "Iteration interval for logging"})

    # wandb logging
    wandb_log: bool = field(default=False, metadata={"help": "Use weights-and-biases logging"})  # disabled by default
    wandb_project: str = field(default="owt", metadata={"help": "Project for weights-and-biases logging"})
    wandb_run_name: str = field(
        default=f"nanogpt-run-{simpleiso()}", metadata={"help": "Run name for weights-and-biases loggin"}
    )


@dataclass
class DatasetConfig(Loadable, Dictable):

    __conf_name__ = "dataset"

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Internal name of this dataset"})
    dataset_dir: str = field(default="data", metadata={"help": "Directory dataset files are in"})
    train_filename: str = field(default="train.bin", metadata={"help": "Training data filename (in dataset_dir)"})
    val_filename: str = field(default="val.bin", metadata={"help": "Test/Validate data filename (in dataset_dir)"})


@dataclass
class CheckpointConfig(Loadable, Dictable):

    __conf_name__ = "checkpoint"

    checkpoint_dir: str = field(default="out", metadata={"help": "Directory to find files in"})
    train_checkpoint: str = field(
        default="train.pt", metadata={"help": "Training checkpoint filename (in checkpoint_dir)"}
    )
    model_checkpoint: str = field(
        default="model.pt", metadata={"help": "Model checkpoint filename (in checkpoint_dir)"}
    )
    optim_checkpoint: str = field(
        default="optim.pt", metadata={"help": "Optimizer checkpoint filename (in checkpoint_dir)"}
    )

    def checkpoint_filename(self, unit: str) -> str:
        if unit in ["model", "train", "optim"]:
            return path.join(self.checkpoint_dir, getattr(self, f"{unit}_checkpoint"))
        raise ValueError(f'Unknown checkpoint unit "{unit}"')

    def save(
        self,
        iter_num: int,
        best_val_loss: float,
        model: nn.Module,
        model_config: NanoGPTConfig,
        optimizer: torch.optim.Optimizer,
        train_config: TrainingConfig,
    ) -> None:
        # splitting up into train_ckpt, model_ckpt, optim_ckpt
        # will help organize as well as be efficient in "rehydrating"
        # training, model, and optimizer separately
        print("saving checkpoint(s)")
        torch.save(
            {
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "train_config": train_config.dict(),
            },
            path.join(self.checkpoint_dir, self.train_checkpoint),
        )
        torch.save(
            {
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "model_config": model_config.dict(),
                "model_state": model.state_dict(),
            },
            path.join(self.checkpoint_dir, self.model_checkpoint),
        )
        torch.save(
            {
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "optim_state": optimizer.state_dict(),
            },
            path.join(self.checkpoint_dir, self.optim_checkpoint),
        )


@dataclass
class EvaluateConfig(Loadable, Dictable):

    __conf_name__ = "evaluate"

    dtype: str = field(
        default="bfloat16",
        metadata={"help": "torch datatype to use", "choices": ["float32", "bfloat16", "float16"]},
    )
    n_batch: int = field(default=32, metadata={"help": "Batch size to use in evaluation iterations"})
    eval_iters: int = field(
        default=200, metadata={"help": "Number of iterations to use in evaluations of train/test loss"}
    )


@dataclass
class GenerateConfig(Loadable, Dictable):

    __conf_name__ = "generate"

    dtype: str = field(
        default="bfloat16",
        metadata={"help": "torch datatype to use", "choices": ["float32", "bfloat16", "float16"]},
    )

    prompt: str = field(default="\n", metadata={"help": 'Text prompt for generation ("document completion")'})
    max_new_tokens: int = field(default=500, metadata={"help": "Maximum new tokens to generate"})
    temperature: float = field(default=1.0, metadata={"help": "'Temperature'"})
    top_k: Optional[int] = field(default=None, metadata={"help": "'Top k'"})


@dataclass
class DDPConfig:

    __conf_name__ = "ddp"

    rank: int
    local_rank: int
    seed_offset: int
    enabled: bool = False
    device: str = field(init=False)
    main_process: bool = field(init=False)

    def __post_init__(self) -> None:
        self.main_process = self.rank == 0
        self.device = f"cuda:{self.local_rank}"

    def wrap(self, model: nn.Module) -> Union[DDP, nn.Module]:
        if self.enabled:
            return DDP(model, device_ids=[self.local_rank])
        return model

    def __enter__(self, backend: str) -> DDPConfig:
        if self.enabled:
            init_process_group(backend=backend)
            torch.cuda.set_device(self.device)
            # ...
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.enabled:
            destroy_process_group()

    @staticmethod
    def from_env() -> DDPConfig:
        enabled = False
        rank = int(environ.get("RANK", -1))
        if rank == -1:  # not a ddp run?
            return DDPConfig(enabled=enabled, rank=1, local_rank=0, seed_offset=0)

        local_rank = int(environ["LOCAL_RANK"])
        config = DDPConfig(enabled=enabled, rank=rank, local_rank=local_rank, seed_offset=rank)
        return config


@dataclass
class NanoGPTContext:
    ddp_enabled: bool = False
    main_process: bool = False
    amp_context: Union[nullcontext, torch.amp.autocast] = field(default_factory=nullcontext)

    @staticmethod
    def from_env() -> DDPConfig:
        enabled = False
        rank = int(environ.get("RANK", -1))
        if rank == -1:  # not a ddp run?
            return DDPConfig(enabled=enabled, rank=1, local_rank=0, seed_offset=0)

        local_rank = int(environ["LOCAL_RANK"])
        config = DDPConfig(enabled=enabled, rank=rank, local_rank=local_rank, seed_offset=rank)
        return config
