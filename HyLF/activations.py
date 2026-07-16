#!/usr/bin/env python

import torch
from torch import nn


class LeakyReLU(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        slope = cfg.a if not isinstance(cfg, str) and "a" in cfg else 0.01
        self.activation = nn.LeakyReLU(slope, inplace=kwargs.get("inplace", True))

    def forward(self, values):
        return self.activation(values)


class Sigmoid(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.inner_fac = cfg.inner_fac if "inner_fac" in cfg else 1.0
        self.outer_fac = cfg.outer_fac if "outer_fac" in cfg else cfg.get("fac", 1.0)
        self.shift = cfg.shift if "shift" in cfg else 0.0

    def forward(self, values):
        return torch.sigmoid(values * self.inner_fac + self.shift) * self.outer_fac


class Tanh(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.inner_fac = cfg.inner_fac if "inner_fac" in cfg else 1.0
        self.outer_fac = cfg.outer_fac if "outer_fac" in cfg else cfg.get("fac", 1.0)
        self.shift = cfg.shift if "shift" in cfg else 0.0

    def forward(self, values):
        return torch.tanh(values * self.inner_fac + self.shift) * self.outer_fac


class Identity(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.inner_fac = cfg.inner_fac if "inner_fac" in cfg else 1.0
        self.outer_fac = cfg.outer_fac if "outer_fac" in cfg else cfg.get("fac", 1.0)
        self.shift = cfg.shift if "shift" in cfg else 0.0

    def forward(self, values):
        return (values * self.inner_fac + self.shift) * self.outer_fac


class EaseValue(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.activation = get_activation(cfg.activation, **kwargs)
        self.start_value = getattr(cfg, "start_value", 0.0)
        self.wait_iters = getattr(cfg, "wait_iters", 0)
        self.window_iters = getattr(cfg, "window_iters", 0)
        self.current_iteration = 0

    def forward(self, values):
        output = self.activation(values)
        if self.current_iteration >= self.window_iters:
            return output
        if self.window_iters == 0:
            return torch.ones_like(output) * self.start_value
        weight = min(
            max(self.current_iteration / float(self.window_iters), 0.0),
            1.0,
        )
        return weight * output + (1 - weight) * self.start_value

    def set_iteration(self, iteration):
        self.current_iteration = iteration - self.wait_iters


activation_registry = {
    "leaky_relu": LeakyReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "identity": Identity,
    "ease_value": EaseValue,
}


def get_activation(cfg, **kwargs):
    if isinstance(cfg, str):
        return activation_registry[cfg]({}, **kwargs)
    return activation_registry[cfg.type](cfg, **kwargs)
