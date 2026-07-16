#!/usr/bin/env python

import torch
from torch import nn

from utils.intersect_utils import intersect_rays_with_axis_plane


def identity_parameterization(_cfg):
    return lambda rays: rays


def origin_direction_parameterization(_cfg):
    def parameterize(rays):
        rays = rays.reshape(rays.shape[0], -1, 6)
        origins = rays[..., :3]
        directions = torch.nn.functional.normalize(rays[..., 3:6] - origins, dim=-1)
        return torch.cat([origins, directions], dim=-1).reshape(rays.shape[0], -1)

    return parameterize


class TwoPlaneParameterization(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.out_channels = getattr(cfg, "n_dims", 4)
        self.near = getattr(cfg, "near", -1.0)
        self.far = getattr(cfg, "far", 0.0)
        self.register_buffer(
            "origin",
            torch.tensor(getattr(cfg, "origin", [0.0, 0.0, 0.0])),
        )

    def forward(self, rays):
        origins = rays[..., :3] - self.origin.unsqueeze(0)
        directions = rays[..., 3:6]
        centered_rays = torch.cat([origins, directions], dim=-1)
        near_distance = intersect_rays_with_axis_plane(centered_rays, self.near, 2)
        far_distance = intersect_rays_with_axis_plane(centered_rays, self.far, 2)
        return torch.cat(
            [
                origins[..., :2] + directions[..., :2] * near_distance.unsqueeze(-1),
                origins[..., :2] + directions[..., :2] * far_distance.unsqueeze(-1),
            ],
            dim=-1,
        )


parameterization_registry = {
    "identity": identity_parameterization,
    "origin_direction": origin_direction_parameterization,
    "two_plane": TwoPlaneParameterization,
}


class RayParameterization(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.parameterize = parameterization_registry[cfg.fn](cfg)
        self.out_channels = getattr(cfg, "n_dims", getattr(cfg, "in_channels", 6))
        # Preserve reference RNG consumption so seeded MLP weights match.
        self.dummy_layer = nn.Linear(1, 1)

    def forward(self, rays):
        return self.parameterize(rays)
