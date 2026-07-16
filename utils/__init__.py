#!/usr/bin/env python

from functools import partial

import numpy as np
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


def preserve_default_initialization(_config):
    return lambda _module: None


initialization_registry = {"none": preserve_default_initialization}


def float_image_to_uint8(values):
    return (255 * np.clip(values, 0, 1)).astype(np.uint8)


def build_optimizer(config, modules):
    parameters = []
    for module in modules:
        if isinstance(module, nn.Parameter):
            parameters.append(module)
        elif isinstance(module, nn.Module):
            parameters.extend(module.parameters())
    if config.optimizer != "adam":
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return Adam(
        parameters,
        lr=config.lr,
        eps=1e-8,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.99),
    )


def exponential_decay_factor(decay_gamma, stop_epoch, decay_epoch, epoch):
    if epoch > stop_epoch:
        return 0.0
    return decay_gamma ** (epoch / decay_epoch)


def build_learning_rate_scheduler(config, optimizer, _iterations_per_epoch):
    if config.lr_scheduler != "exp":
        raise ValueError(f"Unsupported scheduler: {config.lr_scheduler}")
    return LambdaLR(
        optimizer,
        partial(
            exponential_decay_factor,
            config.decay_gamma,
            getattr(config, "stop_epoch", float("inf")),
            float(config.decay_epoch),
        ),
    )
