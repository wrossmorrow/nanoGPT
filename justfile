
default:
    just --lists

prepare dataset:
    uv run data/{{dataset}}/prepare.py

train:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device=mps \
        --compile=False \
        --dtype=float32 \
        --bias=True
sample:
    uv run sample.py \
        --out_dir=out-shakespeare-char \
        --device=mps \
        --compile=False \
        --dtype=float32

train-all: train-sgd-wd-0 train-sgd-wd-01 train-adamw
train-sgd-wd-0:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device=mps \
        --compile=False \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=0.0 \
        --learning_rate=0.25
    mkdir -p results
    mv stats.jsonl results/sgd-wo-weight-decay.jsonl
train-sgd-wd-01:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device=mps \
        --compile=False \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=0.1 \
        --learning_rate=0.25
    mkdir -p results
    mv stats.jsonl results/sgd-w-weight-decay.jsonl
train-adamw:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device=mps \
        --compile=False \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=adamw \
        --weight_decay=0.1
    mkdir -p results
    mv stats.jsonl results/adamw.jsonl
