#!/usr/bin/env python

from typing import Dict

import torch
from torch import nn

from hylf.embedding import embedding_registry
from hylf.nets import network_registry
from hylf.param import RayParameterization


class HybridColorDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__()
        self.net = network_registry[cfg.net.type](
            in_channels,
            out_channels,
            cfg.net,
        )

    def forward(self, values: Dict[str, torch.Tensor]):
        return self.net(values)

    def set_iteration(self, iteration):
        if hasattr(self.net, "set_iteration"):
            self.net.set_iteration(iteration)


class HyLFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ray_parameterization = RayParameterization(cfg.param)
        self.embedding_model = embedding_registry[cfg.embedding.type](
            self.ray_parameterization.out_channels,
            cfg.embedding,
        )
        self.color_model = HybridColorDecoder(
            self.embedding_model.out_channels,
            getattr(cfg, "num_outputs", 3),
            cfg.color,
        )

    def embed(self, rays: torch.Tensor):
        rays = self.ray_parameterization(rays)
        return self.embedding_model(rays)

    def forward(self, rays):
        return self.color_model(self.embed(rays))

    def set_iteration(self, iteration):
        self.embedding_model.set_iteration(iteration)
        self.color_model.set_iteration(iteration)


model_registry = {"hylf": HyLFModel}
