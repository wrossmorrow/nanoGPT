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

import os
import pickle
from contextlib import nullcontext

import torch
from torch.distributed import init_process_group, destroy_process_group

from .config import NanoGPTConfig, TrainingConfig, DDPConfig
from .data import DataLoader
from .model import NanoGPT
from .train import evaluate, train

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
data_dir = "data"
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup

master_process: bool = True
seed_offset: int = 0

ddp_config = DDPConfig.from_env()  # is this a ddp run?
if ddp_config is None:
    # if not ddp, we are running on a single gpu, and one process
    gradient_accumulation_steps *= 8  # simulate 8 gpus
else:
    init_process_group(backend=backend)
    master_process = ddp_config.master_process
    seed_offset = ddp_config.seed_offset

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# TODO: device_type = config.device_type()

# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
# TODO: ptdtype = config.ptdtype()

context = nullcontext() if device_type == "cpu" else (torch.amp.autocast(device_type=device_type, dtype=ptdtype))
# TODO: context = config.context()

data = DataLoader(dataset, block_size, batch_size, device)
# TODO: data = DataLoader(dataset, config.n_block, config.n_batch, config.device)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=None, dropout=dropout
)  # start with model_args from command line

if init_from == "scratch":  # init a new model from scratch
    print("Initializing a new model from scratch")

    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")

    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304

    gptconf = NanoGPTConfig(**model_args)
    model = NanoGPT(gptconf)

elif init_from == "resume":  # resume training from a checkpoint.
    print(f"Resuming training from {out_dir}")

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]

    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    # create the model
    gptconf = NanoGPTConfig(**model_args)
    model = NanoGPT(gptconf)

    state_dict = checkpoint["model"]

    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

elif init_from.startswith("gpt2"):  # initialize from OpenAI GPT-2 weights
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")

    override_args = {"dropout": dropout}
    model = NanoGPT.from_pretrained(init_from, override_args)

    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size  # so that the checkpoint will have the right value

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
# TODO: config.scaler()

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp_config:
    model = ddp_config.wrap(model)

# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

if eval_only:
    evaluate(model, data, context, config.eval_iters)
else:
    train(
        model,
        data,
        optimizer,
        context,
        config,
        ddp=(ddp_config is not None),
        master_process=master_process,
    )

if ddp_config:
    destroy_process_group()
