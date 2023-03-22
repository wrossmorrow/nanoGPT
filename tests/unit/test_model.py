import pytest

from contextlib import nullcontext
from os import path

import torch

from nanoGPT.config import (
    CheckpointConfig, DatasetConfig, NanoGPTConfig, NanoGPTContext, TrainingConfig
)
from nanoGPT.data import DataLoader
from nanoGPT.model import NanoGPT
from nanoGPT.train import NanoGPTTrainer
from nanoGPT.optim import configure_optimizer


# @pytest.mark.parametrize(...)
def test_train_model(device) -> None:

    filename = path.join(path.dirname(__file__), "config", "test.yaml")
    model_config = NanoGPTConfig.from_yaml_file(filename)
    train_config = TrainingConfig.from_yaml_file(filename)
    chkpt_config = CheckpointConfig.from_yaml_file(filename)
    datas_config = DatasetConfig.from_yaml_file(filename)
    n_block, n_batch = model_config.n_block, train_config.n_batch
    
    data = DataLoader(datas_config, n_block, n_batch, device)
    model = NanoGPT(model_config)
    trainer = NanoGPTTrainer(train_config)
    optimizer = configure_optimizer(model, train_config, device)
    context = NanoGPTContext(False, True, nullcontext())

    trainer.train(model, data, optimizer, chkpt_config, context)
