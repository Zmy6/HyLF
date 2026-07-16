#!/usr/bin/env python

from typing import Dict

import torch
from torch import nn

from hylf.activations import get_activation
from hylf.intersect import intersector_registry
from hylf.nets import network_registry
from hylf.param import RayParameterization
from hylf.pe import encoding_registry


class HybridRayPredictor(nn.Module):
    def __init__(self, _in_channels, cfg):
        super().__init__()
        self.rays_name = getattr(cfg, "rays_name", "rays")

        self.params = nn.ModuleList()
        self.pes = nn.ModuleList()
        self.param_channels = []
        encoded_channels = 0
        for param_cfg in cfg.params.values():
            self.param_channels.append((param_cfg.start, param_cfg.end))
            parameterization = RayParameterization(param_cfg.param)
            encoding = encoding_registry[param_cfg.pe.type](
                parameterization.out_channels,
                param_cfg.pe,
            )
            self.params.append(parameterization)
            self.pes.append(encoding)
            encoded_channels += encoding.out_channels

        self.z_channels = cfg.z_channels
        self.outputs = cfg.outputs
        self.output_names = list(self.outputs.keys())
        self.output_shapes = [output.channels for output in self.outputs.values()]
        total_out_channels = self.z_channels * sum(self.output_shapes)

        if "depth" in cfg.net:
            cfg.net.depth -= 2
            cfg.net.linear_last = False
        self.net = network_registry[cfg.net.type](
            encoded_channels,
            total_out_channels,
            cfg.net,
        )
        self.activations = nn.ModuleList(
            [
                get_activation(
                    output.activation if "activation" in output else "identity"
                )
                for output in self.outputs.values()
            ]
        )

    def forward(self, values: Dict[str, torch.Tensor]):
        rays = values[self.rays_name]
        encoded = []
        for channels, parameterization, encoding in zip(
            self.param_channels, self.params, self.pes
        ):
            encoded.append(encoding(parameterization(rays[:, channels[0]:channels[1]])))

        predictions = self.net(torch.cat(encoded, dim=-1))
        predictions = predictions.reshape(rays.shape[0], self.z_channels, -1)
        predictions = torch.split(predictions, self.output_shapes, dim=-1)
        for name, activation, prediction in zip(
            self.output_names, self.activations, predictions
        ):
            values[name] = activation(prediction)
        return values

    def set_iteration(self, iteration):
        for module in [*self.activations, *self.pes]:
            if hasattr(module, "set_iteration"):
                module.set_iteration(iteration)


class PlanarRayIntersection(nn.Module):
    def __init__(self, _in_channels, cfg):
        super().__init__()
        self.rays_name = getattr(cfg, "rays_name", "rays")
        self.intersector = intersector_registry[cfg.intersect.type](
            cfg.z_channels,
            cfg.intersect,
        )

    def forward(self, values: Dict[str, torch.Tensor]):
        return self.intersector(values[self.rays_name], values)


ray_embedding_registry = {
    "hybrid_ray_predictor": HybridRayPredictor,
    "planar_ray_intersection": PlanarRayIntersection,
}
