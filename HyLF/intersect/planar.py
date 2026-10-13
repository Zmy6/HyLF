#!/usr/bin/env python

from typing import Dict

import torch
from torch import nn

from HyLF.activations import get_activation
from utils.intersect_utils import intersect_rays_with_axis_plane, sort_depths


class PlanarIntersector(nn.Module):
    def __init__(self, z_channels, cfg):
        super().__init__()
        self.z_channels = z_channels
        self.density_field = cfg.in_density_field
        self.near = getattr(cfg, "near", 0.0)
        self.far = getattr(cfg, "far", float("inf"))
        self.activation = get_activation(getattr(cfg, "activation", "identity"))

        initial = getattr(cfg, "initial", 0.0)
        end = getattr(cfg, "end", 1.0)
        samples = torch.linspace(initial, end, z_channels).reshape(-1, 1)
        self.register_buffer("samples", samples)
        if z_channels > 1:
            z_scale = getattr(cfg, "z_scale", torch.abs(samples[1] - samples[0]))
        else:
            z_scale = getattr(cfg, "z_scale", 1.0)
        self.register_buffer("z_scale", torch.as_tensor(z_scale).reshape(-1, 1))

    def forward(
        self,
        rays: torch.Tensor,
        values: Dict[str, torch.Tensor],
    ):
        z_offsets = values["z_vals"].reshape(rays.shape[0], self.z_channels, -1)
        sigma = values[self.density_field].reshape(
            rays.shape[0],
            self.z_channels,
            -1,
        )

        z_values = self.activation(z_offsets) * (1 - sigma)
        z_values = (
            z_values * self.z_scale.unsqueeze(0) + self.samples.unsqueeze(0)
        ).reshape(rays.shape[0], -1)
        distances = intersect_rays_with_axis_plane(rays[..., None, :], z_values, 2)
        weights = torch.ones_like(distances)
        distances = torch.where(
            (distances <= self.near) | (distances >= self.far),
            torch.zeros_like(distances),
            distances,
        )

        distances, _ = sort_depths(distances, 1, False)

        distances = distances.unsqueeze(-1)
        points = rays[..., None, :3] + rays[..., None, 3:6] * distances
        values["weights"] = weights.unsqueeze(-1)
        values["points"] = points
        values["distances"] = distances
        return values


planar_intersector_registry = {"parallel_planes": PlanarIntersector}
