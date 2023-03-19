"""
FORMERLY train.py...

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python -m nanoGPT --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 nanoGPT/__main__.py

To run with DDP on 4 gpus across 2 nodes, example:

- Run on the first (master) node with example IP 123.456.123.456:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
        --master_addr=123.456.123.456 --master_port=1234 nanoGPT/__main__.py

- Run on the worker node:

    $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
        --master_addr=123.456.123.456 --master_port=1234 nanoGPT/__main__.py

(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

"""

# TODO: decide whether to include GPT2; it's "just" an initialization
# from a HuggingFace model anyway, so why recreate? If we want to fine-
# tune a GPT2 variant, we could come up with another mechanism to pre-
# process GPT2 weights separately and load like a "resume" command.
#
# See pretrained.py for removed code

import os

from contextlib import nullcontext

import torch
from torch.distributed import init_process_group, destroy_process_group

from . import config
from .cli import DefaultCLI
from .data import DataLoader
from .model import NanoGPT
from .optim import configure_optimizer, load_checkpoint
from .train import NanoGPTTrainer


# Read CLI args, parse them into configs, and validate for commands

command, configs, args = DefaultCLI.parse_config()
print(command, args)
for name, conf in configs.items():
    if conf is not None:
        print(name, conf)

chkpt_config = configs.get(config.CheckpointConfig.__name__)
datas_config = configs.get(config.DatasetConfig.__name__)
# devic_config = configs.get(config.DeviceConfig.__name__)
evalm_config = configs.get(config.EvaluateConfig.__name__)
gener_config = configs.get(config.GenerateConfig.__name__)
model_config = configs.get(config.NanoGPTConfig.__name__)
train_config = configs.get(config.TrainingConfig.__name__)

assert datas_config is not None, "Dataset configuration required"
# assert devic_config is not None, "Device configuration required"

if command == "train":  # init a new model from scratch and train it
    assert model_config is not None, "Model configuration required when training"
    assert train_config is not None, "Training configuration required when training"
    print("Training a new model from scratch")
    n_block = model_config.n_block
    n_batch = train_config.n_batch

else:
    assert chkpt_config is not None, "Checkpoint configuration required"
    if command == "eval":  # evaluate an existing model
        assert evalm_config is not None, "Evaluation configuration required when evaluating"
        print("Evaluating model from checkpoint")
    elif command == "resume":  # resume training from a checkpoint.
        # assert train_config is not None # training config is optional here
        print("Resuming training from checkpoint")
    elif command == "generate":  # generate text from a checkpoint.
        assert gener_config is not None, "Generation configuration required when generating"
        print("Generating from checkpoint")


# DDP backend
#
# TODO: environment? another config obj?
ddp_backend: str = "nccl"  # 'nccl', 'gloo', etc.

# various inits, derived attributes, I/O setup

main_process: bool = True
seed_offset: int = 0
n_block: int = 0
n_batch: int = 0
device: str = args.device if hasattr(args, "device") else "cpu"
dtype: str

ddp_config = config.DDPConfig.from_env()  # is this a ddp run?
if ddp_config.enabled:
    init_process_group(backend=ddp_backend)
    main_process = ddp_config.main_process
    seed_offset = ddp_config.seed_offset
    torch.cuda.set_device(ddp_config.device)
    device = ddp_config.device
else:
    # if not ddp, we are running on a single gpu, and one process
    if command in ["train", "resume"]:
        train_config.gradient_accumulation_steps *= 8  # simulate 8 gpus

if "cuda" in device:
    assert torch.cuda.is_available()

# generic torch setup for this process
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

# define the model, and maybe trainer and optimizer; also define
# the runtime block and batch sizes for the dataset loader/wrapper
if command == "eval":
    model = NanoGPT.from_checkpoint(chkpt_config, device)

    n_block = model.config.n_block
    n_batch = evalm_config.n_batch
    dtype = evalm_config.dtype

elif command == "train":
    model = NanoGPT(model_config)
    trainer = NanoGPTTrainer(train_config)
    optimizer = configure_optimizer(model, train_config, device)

    n_block = model.config.n_block
    n_batch = trainer.config.n_batch
    dtype = trainer.config.dtype

    if main_process:
        os.makedirs(train_config.out_dir, exist_ok=True)
        # TODO: checkpoint dirs, logging dirs?

elif command == "resume":
    model = NanoGPT.from_checkpoint(chkpt_config, device)
    trainer = NanoGPTTrainer.from_checkpoint(chkpt_config, device)
    trainer.overrides(**(train_config.dict() if train_config else {}))  # update if any passed
    # TODO: with all the defaults I actually don't think this is effective...
    # Specifically, we'll always override everything and that's not the idea
    optimizer = configure_optimizer(model, trainer.config, device)
    load_checkpoint(chkpt_config, device, optimizer)

    n_block = model.config.n_block
    n_batch = trainer.config.n_batch
    dtype = trainer.config.dtype

elif command == "generate":
    model = NanoGPT.from_checkpoint(chkpt_config, device)

    n_block = model.config.n_block
    n_batch = 1  # NOTE: batch size not actually relevant here
    dtype = gener_config.dtype

# construct dataset loader/wrapper using block/batch sizes case-determined above
data = DataLoader(datas_config, n_block, n_batch, device)

# put model on device
model.to(device)

# wrap model into DDP container (no-op if not enabled)
model = ddp_config.wrap(model)

# evaluate/training context (not used in generation?)
context = config.NanoGPTContext(
    ddp_enabled=ddp_config.enabled,
    main_process=main_process,
    amp_context=torch.amp.autocast(device_type="cuda", dtype=config.TORCH_TYPES[dtype])
    if "cuda" in device
    else nullcontext(),
)

# if command in ["train", "resume"]:
#     trainer.train(model, data, optimizer, context)
# elif command == "eval":
#     model.eval()
#     data.evaluate(model, eval_iters, context)
# elif command == "generate":
#     model.eval()
#     model.generate(idx, gener_config, context)

if ddp_config.enabled:
    destroy_process_group()
