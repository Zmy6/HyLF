#!/usr/bin/env python

from typing import Dict

import torch
import torch.nn.functional as F

from .planar_voxel_factors import PlanarVoxelFactorization

from utils.planar_voxel_utils import (
    apply_color_correction,
    compute_volume_rendering_weights,
)


class PlanarVoxelGrid(PlanarVoxelFactorization):
    def compute_density_features(self, xyz_sampled):
        coordinate_plane = torch.stack(
            (
                xyz_sampled[:, self.plane_axes[0]],
                xyz_sampled[:, self.plane_axes[1]],
                xyz_sampled[:, self.plane_axes[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[:, self.line_axes[0]],
                xyz_sampled[:, self.line_axes[1]],
                xyz_sampled[:, self.line_axes[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)

        for idx_plane, (plane, line) in enumerate(
            zip(self.density_planes, self.density_lines)
        ):
            plane_coef_point = F.grid_sample(
                plane, coordinate_plane[[idx_plane]], align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            line_coef_point = F.grid_sample(
                line, coordinate_line[[idx_plane]], align_corners=True
            ).view(-1, xyz_sampled.shape[0])
            sigma_feature = sigma_feature + torch.sum(
                plane_coef_point * line_coef_point, dim=0
            )

        return sigma_feature

    def compute_appearance_features(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = torch.stack(
            (
                xyz_sampled[:, self.plane_axes[0]],
                xyz_sampled[:, self.plane_axes[1]],
                xyz_sampled[:, self.plane_axes[2]],
            )
        ).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (
                xyz_sampled[:, self.line_axes[0]],
                xyz_sampled[:, self.line_axes[1]],
                xyz_sampled[:, self.line_axes[2]],
            )
        )
        coordinate_line = torch.stack(
            (torch.zeros_like(coordinate_line), coordinate_line), dim=-1
        ).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane, (plane, line) in enumerate(
            zip(self.appearance_planes, self.appearance_lines)
        ):
            plane_coef_point.append(
                F.grid_sample(
                    plane, coordinate_plane[[idx_plane]], align_corners=True
                ).view(-1, xyz_sampled.shape[0])
            )
            line_coef_point.append(
                F.grid_sample(
                    line, coordinate_line[[idx_plane]], align_corners=True
                ).view(-1, xyz_sampled.shape[0])
            )

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(
            line_coef_point
        )
        return self.appearance_basis((plane_coef_point * line_coef_point).T)

    def forward(self, x: Dict[str, torch.Tensor]):
        batch_size = x["points"].shape[0]
        num_samples = x["points"].shape[-1] // 3
        xyz_sampled = x["points"].view(batch_size, num_samples, 3)
        distances = x["distances"].view(batch_size, -1)
        deltas = torch.cat(
            [
                distances[..., 1:] - distances[..., :-1],
                1e10 * torch.ones_like(distances[:, :1]),
            ],
            dim=1,
        )

        weights = x["weights"].view(batch_size, -1, 1)

        ray_valid = self.valid_sample_mask(xyz_sampled) & (distances > 0)
        xyz_sampled = self.normalize_points(xyz_sampled)
        sigma = xyz_sampled.new_zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)

        if ray_valid.any():
            sigma_feature = self.compute_density_features(xyz_sampled[ray_valid])

            sigma_feature = sigma_feature * weights[ray_valid].view(sigma_feature.shape[0])

            valid_sigma = self.activate_density(sigma_feature)

            assert valid_sigma is not None
            assert ray_valid is not None

            sigma[ray_valid] = valid_sigma

        weight = compute_volume_rendering_weights(
            sigma,
            deltas * self.distance_scale,
        )
        appearance_mask = weight > self.render_weight_threshold

        rgb = xyz_sampled.new_zeros(
            (xyz_sampled.shape[0], xyz_sampled.shape[1], 3), device=xyz_sampled.device
        )

        if appearance_mask.any():
            appearance_features = self.compute_appearance_features(xyz_sampled[appearance_mask])

            valid_rgbs = torch.sigmoid(appearance_features)
            rgb = valid_rgbs.new_zeros(
                (xyz_sampled.shape[0], xyz_sampled.shape[1], 3), device=xyz_sampled.device
            )
            assert valid_rgbs is not None
            assert appearance_mask is not None
            rgb[appearance_mask] = valid_rgbs

        if 'color_scale' in x:
            color_scale = x['color_scale'].view(rgb.shape[0], rgb.shape[1], 3)
            color_shift = x['color_shift'].view(rgb.shape[0], rgb.shape[1], 3)
            rgb = apply_color_correction(rgb, color_scale, color_shift)

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[:, :, None] * rgb, -2)

        random_white_background = self.training and torch.rand(()) < 0.5
        if (
            self.white_background or random_white_background
        ) and not self.black_background:
            rgb_map = rgb_map + (1.0 - acc_map[:, None])

        if not self.training:
            rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * distances, -1)
            depth_map = depth_map + (1. - acc_map) * x["rays"][..., -1]

        outputs = {
            "rgb": rgb_map,
            "depth": depth_map.reshape(-1, 1),
        }

        return outputs


planar_voxel_grid_registry = {
    "planar_voxel_grid": PlanarVoxelGrid,
}
