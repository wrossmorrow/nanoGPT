from __future__ import annotations

import os.path

from math import cos, pi
from time import time
from typing import Any, cast, Optional, Union
from warnings import warn

import torch
from torch import nn

from .config import CheckpointConfig, NanoGPTContext, TrainingConfig
from .data import DataLoader
from .model import NanoGPT


class NanoGPTTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

        # TODO: ?
        # self.best_val_loss: float = 1.0e9  # best validation loss
        self.mfu: Optional[float] = None

    def overrides(self, **kwargs) -> None:
        for attr, value in kwargs.items():
            if not hasattr(self.config, attr):
                raise AttributeError(f"{self.config.__class__.__name__} has no attribute {attr}")
            setattr(self.config, attr, value)

    def train(
        self,
        model: nn.Module,  # expect a NanoGPT, but not important actually
        data: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoints: CheckpointConfig,
        context: NanoGPTContext,
    ) -> Any:

        # local data (convenience and o/w)
        best_val_loss: float = 1.0e9  # best validation loss
        grad_accm_steps: int = self.config.gradient_accumulation_steps
        fwdbwd_per_iter = self.config.n_batch * grad_accm_steps
        raw_model = cast(NanoGPT, model.module if context.ddp_enabled else model)

        # compile the model
        if self.config.torch_compile:
            warn("compiling the model... (takes a ~minute)")
            model = torch.compile(model)  # NOTE: requires PyTorch 2.0

        # get a "GradScaler"
        scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == "float16"))

        # training loop
        X, Y = data.get_batch("train")  # fetch the very first batch

        ts = time()
        for it in range(self.config.max_iters):

            # determine and set the learning rate for this iteration
            lr = self.get_lr(it)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if it % self.config.eval_interval == 0 and context.main_process:
                dt = time() - ts
                tl, vl = data.estimate_loss(model, context, self.config.eval_iters)
                self.on_eval(it, dt, tl, vl, best_val_loss, self.mfu)
                if vl < best_val_loss or self.config.always_save_checkpoint:
                    best_val_loss = vl
                    if it > 0:
                        checkpoints.save(it, best_val_loss, raw_model, raw_model.config, optimizer, self.config)

            # forward backward update, with optional gradient accumulation to
            # simulate larger batch size and using the GradScaler if data type
            # is float16
            for ms in range(grad_accm_steps):
                if context.ddp_enabled:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = ms == grad_accm_steps - 1  # type: ignore

                # compute logits and loss, possibly using mixed precision
                with context.amp_context:
                    logits, loss = model(X, Y)

                # immediately async prefetch next batch while model is doing
                # the forward pass on the GPU
                X, Y = data.get_batch("train")

                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # clip the gradient
            if self.config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()

            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            te = time()
            dt = te - ts
            ts = te
            if it % self.config.log_interval == 0 and context.main_process:
                if it >= 5:  # let the training loop settle a bit
                    self.estimate_mfu(raw_model, fwdbwd_per_iter, dt)
                self.on_log(it, dt, loss, self.mfu)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it: int) -> float:

        # 0) using?
        if not self.config.decay_lr:
            return self.config.learning_rate

        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters

        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr

        # 3) in between, use cosine decay down to min learning rate
        num = it - self.config.warmup_iters
        den = self.config.lr_decay_iters - self.config.warmup_iters
        decay_ratio = num / den
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + cos(pi * decay_ratio))  # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    def estimate_mfu(self, model: NanoGPT, fwdbwd_per_iter: Union[float, int], dt: float) -> Optional[float]:
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS

        This is a little specific to NanoGPT.
        """

        if not self.config.estimate_mfu:
            return self.mfu

        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N, cfg = model.get_num_params(), model.config

        L, T, C = cfg.n_layer, cfg.n_block, cfg.n_embed
        flops_per_token = 6 * N + 12 * L * C * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter / dt  # per second
        flops_promised = 312.0e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS

        mfu = flops_achieved / flops_promised
        self.mfu = mfu if self.mfu is None else (0.9 * self.mfu + 0.1 * mfu)

        return self.mfu

    def on_eval(self, it: int, dt: float, tl: float, vl: float, bvl: float, mfu: Optional[float]) -> bool:
        print(f"step {it}: train loss {tl:.4f}, val loss {vl:.4f}, best val loss {bvl:.4f}")
        return True

    def on_log(self, it: int, dt: float, loss: torch.Tensor, mfu: Optional[float]) -> bool:
        return True

    @staticmethod
    def from_checkpoint(config: CheckpointConfig, device: str) -> NanoGPTTrainer:

        filename = os.path.join(config.checkpoint_dir, config.train_checkpoint)
        checkpoint = torch.load(filename, map_location=device)

        # iter_num = checkpoint["iter_num"]
        # best_val_loss = checkpoint["best_val_loss"]
        train_config = checkpoint["train_config"]
        return NanoGPTTrainer(train_config)


# def train(
#     model: nn.Module,  # expect a NanoGPT, but not important actually
#     data: DataLoader,
#     optimizer: torch.optim.Optimizer,
#     context: Any,
#     config: TrainingConfig,
#     ddp: bool = False,
#     main_process: bool = True,
# ) -> Any:

#     best_val_loss: float = 1.0e9  # best validation loss
#     grad_accm_steps: int = config.gradient_accumulation_steps

#     # compile the model
#     if config.compile:
#         warn("compiling the model... (takes a ~minute)")
#         model = torch.compile(model)  # requires PyTorch 2.0

#     # training loop
#     X, Y = data.get_batch("train")  # fetch the very first batch

#     # get "GradScaler" (WTF)
#     scaler = config.scaler()

#     t0 = time()
#     raw_model = model.module if ddp else model  # unwrap DDP container if needed
#     mfu: Optional[float] = None

#     if config.wandb_log and main_process:
#         import wandb

#         wandb.init(
#             project=config.wandb_project,
#             name=config.wandb_run_name,
#             # config=config, # TODO: what here?
#         )

#     for it in range(config.max_iters):

#         # determine and set the learning rate for this iteration
#         lr = config.get_lr(it)
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = lr

#         # evaluate the loss on train/val sets and write checkpoints
#         if it % config.eval_interval == 0 and main_process:
#             tl, vl = data.estimate_loss(model, config.eval_iters, context)
#             print(f"step {it}: train loss {tl:.4f}, val loss {vl:.4f}")

#             if config.wandb_log:
#                 wandb.log(
#                     {
#                         "iter": it,
#                         "train/loss": tl,
#                         "val/loss": vl,
#                         "lr": lr,
#                         "mfu": 0.0 if mfu is None else 100 * mfu,  # convert to percentage
#                     }
#                 )

#             if vl < best_val_loss or config.always_save_checkpoint:
#                 best_val_loss = vl
#                 if it > 0:
#                     checkpoint = {
#                         "model": raw_model.state_dict(),
#                         "optimizer": optimizer.state_dict(),
#                         "model_config": asdict(model.config),
#                         "iter_num": it,
#                         "best_val_loss": best_val_loss,
#                         "config": asdict(config),
#                     }
#                     filename = config.checkpoint_file()
#                     print(f'saving checkpoint to "{filename}"')
#                     torch.save(checkpoint, filename)

#         # forward backward update, with optional gradient accumulation to
#         # simulate larger batch size and using the GradScaler if data type
#         # is float16
#         for ms in range(grad_accm_steps):
#             if ddp:
#                 # in DDP training we only need to sync gradients at the last micro step.
#                 # the official way to do this is with model.no_sync() context manager, but
#                 # I really dislike that this bloats the code and forces us to repeat code
#                 # looking at the source of that context manager, it just toggles this variable
#                 model.require_backward_grad_sync = ms == grad_accm_steps - 1

#             with context:
#                 logits, loss = model(X, Y)

#             # immediately async prefetch next batch while model is doing
#             # the forward pass on the GPU
#             X, Y = data.get_batch("train")

#             # backward pass, with gradient scaling if training in fp16
#             scaler.scale(loss).backward()

#         # clip the gradient
#         if config.grad_clip != 0.0:
#             scaler.unscale_(optimizer)
#             nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

#         # step the optimizer and scaler if training in fp16
#         scaler.step(optimizer)
#         scaler.update()

#         # flush the gradients as soon as we can, no need for this memory anymore
#         optimizer.zero_grad(set_to_none=True)

#         # timing and logging
#         t1 = time()
#         dt = t1 - t0
#         t0 = t1
#         if it % config.log_interval == 0 and main_process:
#             lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
#             if it >= 5:  # let the training loop settle a bit
#                 _mfu = raw_model.estimate_mfu(config.n_batch * grad_accm_steps, dt)
#                 mfu = _mfu if mfu is None else 0.9 * mfu + 0.1 * _mfu
#             print(f"iter {it}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {mfu*100:.2f}%")
