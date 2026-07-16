#!/usr/bin/env python

import torch


def sort_depths(values, dim=-1, descending=False):
    indices = torch.argsort(values, dim=dim, descending=descending)
    return torch.gather(values, dim, indices), indices


def intersect_rays_with_axis_plane(rays, plane_position, axis):
    origins = rays[..., :3]
    directions = rays[..., 3:6]
    safe_directions = torch.where(
        torch.abs(directions) < 1e-5,
        torch.full_like(directions, 1e12),
        directions,
    )
    return (plane_position - origins[..., axis]) / safe_directions[..., axis]
