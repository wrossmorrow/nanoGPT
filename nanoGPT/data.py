"""
Utility for loading data, getting batches, and estimating losses
for train/test splits for an arbitrary model
"""
from __future__ import annotations

import os.path

from dataclasses import dataclass
from time import time
from typing import Tuple

import torch
from torch.nn import functional as F
import numpy as np

from nanoGPT.config import DatasetConfig, NanoGPTContext


@dataclass
class Statistic:
    mean: float
    std: float

    def __repr__(self) -> str:
        return f"{self.mean:.4f} +/- {self.std:.4f}"

    def to_csv(self) -> str:
        return f"{self.mean:.4f},{self.std:.4f}"

    @staticmethod
    def from_tensor(x: torch.Tensor) -> Statistic:
        return Statistic(mean=x.mean().item(), std=x.std().item())


@dataclass
class EstimatedLoss:
    full: Statistic
    last: Statistic

    def __repr__(self) -> str:
        return f"{self.full} | {self.last}"

    def to_csv(self) -> str:
        return f"{self.full.to_csv()},{self.last.to_csv()}"


@dataclass
class EstimatedLosses:
    train: EstimatedLoss
    val: EstimatedLoss

    def __repr__(self) -> str:
        return f"train: {self.train}, test: {self.val}"

    def to_csv(self) -> str:
        return f"{self.train.to_csv()},{self.val.to_csv()}"


class DataLoader:
    def __init__(
        self,
        config: DatasetConfig,
        n_block: int,
        n_batch: int,
        device: str,
    ) -> None:

        self.name = config.dataset_name

        train_data_fn = os.path.join(config.dataset_dir, config.train_filename)
        val_data_fn = os.path.join(config.dataset_dir, config.val_filename)

        self.train_data = np.memmap(train_data_fn, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(val_data_fn, dtype=np.uint16, mode="r")

        self.n_block = n_block
        self.n_batch = n_batch
        self.device = device

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:

        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(data) - self.n_block, (self.n_batch,))
        x = torch.stack([torch.from_numpy((data[i : i + self.n_block]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + self.n_block]).astype(np.int64)) for i in ix])

        if "cuda" in self.device:
            # pin arrays x,y, which allows us to move them to GPU asynchronously
            # (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
            return x, y

        return x.to(self.device), y.to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        context: NanoGPTContext,
        eval_iters: int,
    ) -> Tuple[float, float, float]:
        ts = time()
        tl, vl = self.estimate_loss(model, context, eval_iters)
        dt = time() - ts
        return tl, vl, dt

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(
        self,
        model: torch.nn.Module,
        context: NanoGPTContext,
        eval_iters: int,
    ) -> EstimatedLosses:  # Tuple[float, float]:
        out, new_out = {}, {}
        losses, last_losses = torch.zeros(eval_iters), torch.zeros(eval_iters)
        model.eval()
        for split in ["train", "val"]:
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                with context.amp_context:
                    logits, loss = model(X, Y)
                    logits = logits[:, -1, :]
                    last_Y = Y[:, -1]
                    last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), last_Y.view(-1), ignore_index=-1)
                losses[k] = loss.item()
                last_losses[k] = last_loss.item()
            new_out[split] = EstimatedLoss(
                full=Statistic.from_tensor(losses),
                last=Statistic.from_tensor(last_losses),
            )
            out[split] = losses.mean().item()

        model.train()
        # return out["train"], out["val"]
        return EstimatedLosses(**new_out)
        # NOTE: CPU/GPU sync point; but honestly what else should
        # we expect when estimating loss? Like, if we use these
        # to print or basically do anything orthogonal to training
        # we're going to sync CPU and GPU data.
