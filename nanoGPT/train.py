from __future__ import annotations

import os.path

from math import cos, pi
from time import time
from datetime import datetime
from typing import Any, Optional, Union
from typing import cast  # noqa: F401
from warnings import warn

import torch
from torch import nn

from nanoGPT.config import CheckpointConfig, NanoGPTContext, TrainingConfig
from nanoGPT.data import DataLoader, EstimatedLosses
from nanoGPT.model import NanoGPT
from nanoGPT.optim import save_checkpoint as optim_to_checkpoint
from nanoGPT.identification import LinearSubspaceProjectionNaive  # noqa: F401
from nanoGPT.identification import LinearSubspaceProjectionConstr  # noqa: F401
from nanoGPT.identification import LinearSubspaceProjectionDPP  # noqa: F401


IDENTIFICATION_STUDY_OUT = "identification.csv"


def isonow() -> str:
    return datetime.now().isoformat()


class NanoGPTTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        prev_iters: int = 0,
        prev_best_val_loss: float = 1.0e9,
    ) -> None:
        self.config = config

        # TODO: ?
        self.prev_iters = prev_iters
        self.prev_best_val_loss = prev_best_val_loss
        self.mfu: Optional[float] = None

    def overrides(self, **kwargs) -> None:
        for attr, value in kwargs.items():
            if not hasattr(self.config, attr):
                raise AttributeError(f"{self.config.__class__.__name__} has no attribute {attr}")
            setattr(self.config, attr, value)

    def train(
        self,
        model: nn.Module,  # expect a NanoGPT; nn.Module should be enough; DDP -> Union[Tensor, Module]?
        data: DataLoader,
        optimizer: torch.optim.Optimizer,
        checkpoints: CheckpointConfig,
        context: NanoGPTContext,
    ) -> Any:

        # local data (convenience and o/w)
        best_val_loss: float = self.prev_best_val_loss
        grad_accm_steps: int = self.config.gradient_accumulation_steps
        fwdbwd_per_iter: float = self.config.n_batch * grad_accm_steps

        # raw_model = cast(NanoGPT, model.module if context.ddp_enabled else model)
        raw_model = model.module if context.ddp_enabled else model

        # compile the model if desired
        # NOTE: requires PyTorch 2.0
        # NOTE: compile returns a Callable, so is type incompatible with nn.Module
        if self.config.torch_compile:
            warn("compiling the model...  (takes a ~minute)")
            model = torch.compile(model)  # type: ignore [assignment]

        # get a "GradScaler"
        scaler = torch.cuda.amp.GradScaler(enabled=(self.config.dtype == "float16"))

        filename = checkpoints.checkpoint_filename("status")
        with open(filename, "w") as status:
            status.write(f"number of parameters: {model.get_num_params()}\n")
            status.write("curr time,iter num,last iter us,train loss,test loss,best loss,mfu\n")

        # IDENTIFICATION STUDY
        LSP = LinearSubspaceProjectionDPP(
            model.config.n_embed,
            model.config.n_heads,
            model.config.n_qkdim,
        )
        with open(IDENTIFICATION_STUDY_OUT, "w") as f:
            f.write("time,iter,lsp_duration,iter_loss,")
            for h in range(model.config.n_heads):
                f.write(f"reslv_head_{h+1},")
            for n in range(6):
                f.write(f"unf_1e({-6-n}),")
            for n in range(6):
                f.write(f"tid_1e({-6-n}),")
            f.write("\n")

        # training loop
        X, Y = data.get_batch("train")  # fetch the very first batch

        ts = time()
        for it in range(self.config.max_iters):

            # IDENTIFICATION STUDY
            pre_weights = [{k: W.data.clone() for k, W in qkvo.items()} for qkvo in model.weights()]

            # determine and set the learning rate for this iteration
            lr = self.get_lr(it)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate step: evaluate both train/test loss and write checkpoints
            if it % self.config.eval_interval == 0 and context.main_process:
                dt = 1.0e6 * (time() - ts)
                losses = data.estimate_loss(model, context, self.config.eval_iters)
                filename = checkpoints.checkpoint_filename("status")
                with open(filename, "a") as status:
                    status.write(f"{isonow()},{it},{dt:.6f},{losses.to_csv()},{best_val_loss:.4f},")
                    status.write("-\n" if self.mfu is None else f"{self.mfu:0.4f}\n")
                self.on_eval(it, dt, losses, best_val_loss, self.mfu)
                if losses.val.full.mean < best_val_loss or self.config.always_save_checkpoint:
                    best_val_loss = losses.val.full.mean
                    if it > 0:
                        # checkpoints.save(it, best_val_loss, raw_model, raw_model.config, optimizer, self.config)
                        raw_model.to_checkpoint(it, best_val_loss, checkpoints)  # type: ignore
                        self.to_checkpoint(it, best_val_loss, checkpoints)
                        optim_to_checkpoint(it, best_val_loss, checkpoints, optimizer)

            # forward backward update, with optional gradient accumulation to
            # simulate larger batch size and using the GradScaler if data type
            # is float16
            for ms in range(grad_accm_steps):
                if context.ddp_enabled:
                    # In DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code.
                    # Looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = ms == grad_accm_steps - 1  # type: ignore

                # compute logits and loss, possibly using mixed precision
                with context.amp_context:
                    logits, loss = model(X, Y)

                # immediately async prefetch next batch while model is doing
                # the forward pass on the GPU
                X, Y = data.get_batch("train")

                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # print(model.state_dict())

            # clip the gradient
            if self.config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()

            # IDENTIFICATION STUDY
            for i, qkvo in enumerate(model.weights()):
                # for n in qkvo:
                #     DW = qkvo[n] - pre_weights[i][n]
                #     # print(i, n, torch.linalg.norm(DW))

                WQ, WK = pre_weights[i]["Q"], pre_weights[i]["K"]
                dWQ, dWK = qkvo["Q"] - WQ, qkvo["K"] - WK
                if (torch.linalg.norm(dWQ) >= 1.0e-8) and (torch.linalg.norm(dWK) >= 1.0e-8):
                    lsp_start = time()
                    status, reslv, unid, idd = LSP.solve(
                        WQ.detach(),
                        WK.detach(),
                        dWQ.detach(),
                        dWK.detach(),
                    )
                    # losses = data.estimate_loss(model, context, self.config.eval_iters)
                    lsp_end = time()
                    lsp_duration = lsp_end - lsp_start
                    with open(IDENTIFICATION_STUDY_OUT, "a") as f:
                        f.write(f"{isonow()},{it},")
                        f.write(f"{lsp_duration},{loss},")
                        f.write(",".join([f"{v}" for v in reslv]) + ",")
                        f.write(",".join([f"{v}" for v in unid]) + ",")
                        f.write(",".join([f"{v}" for v in idd]))
                        f.write("\n")
                    print(
                        f"""{it} ({lsp_end - lsp_start}):
  re-solve solution norm: {', '.join([f'{i:.6}' for i in reslv])}
  fraction unidentified: {', '.join([f'{i:.3}' for i in unid])}
  total fraction identified: {', '.join([f'{i:.3}' for i in idd])}
"""
                    )

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

        # completed max_iters training steps; always consider this an evaluate step
        dt = time() - ts
        losses = data.estimate_loss(model, context, self.config.eval_iters)
        filename = checkpoints.checkpoint_filename("status")
        with open(filename, "a") as status:
            status.write(f"{isonow()},{it},{dt:.6f},{losses.to_csv()},{best_val_loss:.4f},")
            status.write("-\n" if self.mfu is None else f"{self.mfu:0.4f}\n")
        self.on_eval(it, dt, losses, best_val_loss, self.mfu)
        if losses.val.full.mean < best_val_loss or self.config.always_save_checkpoint:
            best_val_loss = losses.val.full.mean
            if it > 0:
                # checkpoints.save(it, best_val_loss, raw_model, raw_model.config, optimizer, self.config)
                raw_model.to_checkpoint(it, best_val_loss, checkpoints)  # type: ignore
                self.to_checkpoint(it, best_val_loss, checkpoints)
                optim_to_checkpoint(it, best_val_loss, checkpoints, optimizer)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it: int) -> float:

        # return learning rate if we aren't using a dynamic learning rate
        if not self.config.decay_lr:
            return self.config.learning_rate

        # linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters

        # if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr

        # in between, use cosine decay down to min learning rate
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

    def append_status(self, checkpoints, it, dt, losses, bvl) -> None:
        filename = checkpoints.checkpoint_filename("status")
        with open(filename, "a") as status:
            tl, vl = losses.train.full.mean, losses.val.full.mean
            status.write(f"{isonow()},{it},{dt:.6f},{tl:.4f},{vl:.4f},{bvl:.4f},")
            status.write("-\n" if self.mfu is None else f"{self.mfu:0.4f}\n")

    def on_eval(self, it: int, dt: float, losses: EstimatedLosses, bvl: float, mfu: Optional[float]) -> bool:
        print(f"step {it}: {losses}, (prev) best val loss {bvl:.4f}")
        return True

    def on_log(self, it: int, dt: float, loss: torch.Tensor, mfu: Optional[float]) -> bool:
        return True

    def to_checkpoint(self, iter_num: int, best_val_loss: float, config: CheckpointConfig) -> None:
        torch.save(
            {
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "train_config": self.config.dict(),
            },
            config.checkpoint_filename("train"),
        )

    @staticmethod
    def from_checkpoint(config: CheckpointConfig, device: str) -> NanoGPTTrainer:

        filename = os.path.join(config.checkpoint_dir, config.train_checkpoint)
        checkpoint = torch.load(filename, map_location=device)

        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        train_config = TrainingConfig(**checkpoint["train_config"])
        return NanoGPTTrainer(train_config, prev_iters=iter_num, prev_best_val_loss=best_val_loss)
