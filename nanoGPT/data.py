"""
Utility for loading data, getting batches, and estimating losses
for train/test splits for an arbitrary model
"""
import os.path

from typing import Any, Tuple

import torch
import numpy as np


class DataLoader:
    def __init__(
        self,
        dataset: str,
        n_block: int,
        n_batch: int,
        device: str,
        train_fn: str = "train.bin",
        val_fn: str = "val.bin",
    ) -> None:

        self.data_dir = os.path.join("data", dataset)
        self.train_data = np.memmap(
            os.path.join(self.data_dir, train_fn),
            dtype=np.uint16,
            mode="r",
        )
        self.val_data = np.memmap(
            os.path.join(self.data_dir, val_fn),
            dtype=np.uint16,
            mode="r",
        )
        self.n_block = n_block
        self.n_batch = n_batch
        self.device = device

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:

        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(data) - self.n_block, (self.n_batch,))
        x = torch.stack([torch.from_numpy((data[i : i + self.n_block]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + self.n_block]).astype(np.int64)) for i in ix])

        if self.device == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously
            # (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
            return x, y

        return x.to(self.device), y.to(self.device)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(
        self,
        model: torch.nn.Module,
        eval_iters: int,
        ctx: Any,  # TODO: type for a context?
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()

        model.train()
        return out["train"], out["val"]
