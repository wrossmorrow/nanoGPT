from time import time
from typing import Any, Tuple, Optional

import torch
from torch import nn

from .config import TrainingConfig
from .data import DataLoader


def evaluate(
    model: nn.Module,  # expect a NanoGPT, but not important actually
    data: DataLoader,
    context: Any,
    eval_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    X, Y = data.get_batch("train")  # fetch the very first batch
    t0 = time()
    tl, vl = data.estimate_loss(model, eval_iters, context)
    dt = time() - t0
    return tl, vl, dt


def train(
    model: nn.Module,  # expect a NanoGPT, but not important actually
    data: DataLoader,
    optimizer: torch.optim.Optimizer,
    context: Any,
    config: TrainingConfig,
    ddp: bool = False,
    master_process: bool = True,
) -> Any:

    bvl: float = 1.0e9  # best validation loss
    gas: int = config.gradient_accumulation_steps

    # training loop
    X, Y = data.get_batch("train")  # fetch the very first batch

    # get "scaler" (WTF)
    scaler = config.scaler()

    t0 = time()
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    mfu: Optional[float] = None

    for it in range(config.max_iters):

        # determine and set the learning rate for this iteration
        lr = config.get_lr(it)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if it % config.eval_interval == 0 and master_process:
            tl, vl = data.estimate_loss(model, config.eval_iters, context)
            print(f"step {it}: train loss {tl:.4f}, val loss {vl:.4f}")

            if config.wandb_log:
                wandb.log(
                    {
                        "iter": it,
                        "train/loss": tl,
                        "val/loss": vl,
                        "lr": lr,
                        "mfu": 0.0 if mfu is None else 100 * mfu,  # convert to percentage
                    }
                )

            if vl < bvl or config.always_save_checkpoint:
                bvl = vl
                if it > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": it,
                        "best_val_loss": bvl,
                        "config": config,
                    }
                    filename = config.checkpoint_file()
                    print(f'saving checkpoint to "{filename}"')
                    torch.save(checkpoint, filename)

        # forward backward update, with optional gradient accumulation to
        # simulate larger batch size and using the GradScaler if data type
        # is float16
        for ms in range(gas):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = ms == gas - 1

            with context:
                logits, loss = model(X, Y)

            # immediately async prefetch next batch while model is doing
            # the forward pass on the GPU
            X, Y = data.get_batch("train")

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

        # clip the gradient
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time()
        dt = t1 - t0
        t0 = t1
        if it % config.log_interval == 0 and master_process:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            if it >= 5:  # let the training loop settle a bit
                _mfu = raw_model.estimate_mfu(config.n_batch * gas, dt)
                mfu = _mfu if mfu is None else 0.9 * mfu + 0.1 * _mfu
            print(f"iter {it}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {mfu*100:.2f}%")
