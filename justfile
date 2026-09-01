
device := "mps"
compile := "False"

default:
    just --lists

prepare dataset:
    uv run data/{{dataset}}/prepare.py

train:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True
sample:
    uv run sample.py \
        --out_dir=out-shakespeare-char \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32

train-all: train-sgd-wd-0 train-sgd-wd-01 train-adamw

train-sgd-wd-0:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=0.0 \
        --learning_rate=0.25
    mkdir -p results
    mv stats.jsonl results/sgd-wo-weight-decay.jsonl

train-sgd-wd-0-cpu:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device=cpu \
        --compile={{compile}} \
        --dtype=float64 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=0.0 \
        --learning_rate=0.25 \
        --max_iters=1000
    mkdir -p results
    mv stats.jsonl results/sgd-wo-weight-decay-cpu.jsonl

train-sgd-wd-01:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=4e-4 \
        --learning_rate=0.25
    mkdir -p results
    mv stats.jsonl results/sgd-w-weight-decay.jsonl

train-adamw:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=adamw \
        --weight_decay=0.1
    mkdir -p results
    mv stats.jsonl results/adamw.jsonl

train-adamw-rebalance:
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=adamw \
        --weight_decay=0.1 \
        --reset_interval=200
    mkdir -p results
    mv stats.jsonl results/adamw-state-clear.jsonl

train-sgd-wd-0-sweep:
    mkdir -p results/sweep
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=0.0 \
        --learning_rate=0.5
    mv stats.jsonl results/sgd-wo-weight-decay-sweep/0_5.jsonl
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=0.0 \
        --learning_rate=0.25
    mv stats.jsonl results/sgd-wo-weight-decay-sweep/0_25.jsonl
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=0.0 \
        --learning_rate=0.125
    mv stats.jsonl results/sgd-wo-weight-decay-sweep/0_125.jsonl
    uv run train.py \
        config/train_shakespeare_char.py \
        --device={{device}} \
        --compile={{compile}} \
        --dtype=float32 \
        --bias=True \
        --optimizer_name=sgd \
        --weight_decay=0.0 \
        --learning_rate=0.0625
    mv stats.jsonl results/sgd-wo-weight-decay-sweep/0_0625.jsonl
