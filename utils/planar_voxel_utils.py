#!/usr/bin/env python

import torch
import torch.nn.functional as F


def voxel_count_to_grid_resolution(voxel_count, bounding_box):
    xyz_min, xyz_max = bounding_box
    voxel_size = ((xyz_max - xyz_min).prod() / voxel_count).pow(1 / 3)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


class _PositiveCumprod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values):
        product = values.cumprod(-1)
        ctx.save_for_backward(values, product)
        return product

    @staticmethod
    def backward(ctx, gradient):
        values, product = ctx.saved_tensors
        accumulated = (gradient * product).flip(-1).cumsum(-1).flip(-1)
        return accumulated / values


def compute_volume_rendering_weights(sigma, distance, fast_cumprod=False):
    alpha = 1.0 - torch.exp(-sigma * distance)
    factors = torch.cat(
        [torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10],
        dim=-1,
    )
    transmittance = (_PositiveCumprod.apply(factors) if fast_cumprod
                     else factors.cumprod(-1))
    return alpha * transmittance[:, :-1]


def apply_color_correction(rgb, color_scale, color_shift):
    color_scale = color_scale.reshape_as(rgb)
    color_shift = color_shift.reshape_as(rgb)
    return rgb * (color_scale + 1.0) + color_shift


class OccupancyGridMask(torch.nn.Module):
    def __init__(self, aabb, occupancy_volume):
        super().__init__()
        device = aabb.device
        self.register_buffer("mask_aabb", aabb.to(device))
        self.register_buffer(
            "occupancy_volume",
            occupancy_volume.to(device=device, dtype=torch.float32).reshape(
                1,
                1,
                *occupancy_volume.shape[-3:],
            ),
        )
        aabb_size = self.mask_aabb[1] - self.mask_aabb[0]
        self.register_buffer("inverse_grid_size", 2.0 / aabb_size)
        self.register_buffer(
            "grid_size",
            torch.tensor(
                [
                    occupancy_volume.shape[-1],
                    occupancy_volume.shape[-2],
                    occupancy_volume.shape[-3],
                ],
                dtype=torch.long,
                device=device,
            ),
        )

    def normalize_points(self, points):
        return (points - self.mask_aabb[0]) * self.inverse_grid_size - 1

    def sample_occupancy(self, points):
        points = self.normalize_points(points)
        return F.grid_sample(
            self.occupancy_volume,
            points.reshape(1, -1, 1, 1, 3),
            align_corners=True,
        ).reshape(-1)
