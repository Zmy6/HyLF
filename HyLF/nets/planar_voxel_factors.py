#!/usr/bin/env python

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from utils.planar_voxel_utils import (
    OccupancyGridMask,
    voxel_count_to_grid_resolution,
)


class PlanarVoxelFactorBase(torch.nn.Module):
    def __init__(self, _in_channels, _out_channels, cfg):
        super().__init__()

        self.white_background = bool(cfg.white_background)
        self.black_background = bool(cfg.black_background)

        self.register_buffer("aabb", torch.tensor(cfg.aabb, dtype=torch.float32))

        grid_size = voxel_count_to_grid_resolution(
            cfg.initial_voxel_count,
            self.aabb,
        )
        self.register_buffer("grid_size", torch.tensor(grid_size))

        self.occupancy_update_iters = cfg.occupancy_update_iters
        self.upsample_iters = cfg.upsample_iters

        self.voxel_count_schedule = (
            torch.round(
                torch.exp(
                    torch.linspace(
                        np.log(cfg.initial_voxel_count),
                        np.log(cfg.final_voxel_count),
                        len(self.upsample_iters) + 1,
                    )
                )
            ).long()
        ).tolist()[1:]

        self.density_component_counts = list(cfg.density_components)
        self.appearance_component_counts = list(cfg.appearance_components)
        self.appearance_feature_channels = cfg.appearance_feature_channels
        self.occupancy_mask = None

        self.occupancy_threshold = cfg.occupancy_threshold
        self.render_weight_threshold = cfg.render_weight_threshold
        self.distance_scale = cfg.distance_scale

        self.update_grid_geometry(self.grid_size)

        self.plane_axes = [[0, 1], [0, 2], [1, 2]]
        self.line_axes = [2, 1, 0]

        if self.appearance_feature_channels != 3:
            raise ValueError("The HyLF planar voxel grid requires three RGB features")
        self.initialize_factors()

    def update_grid_geometry(self, grid_size):
        print("aabb", self.aabb.view(-1))
        print("grid size", grid_size)
        aabb_size = self.aabb[1] - self.aabb[0]
        self.register_buffer("aabb_size", aabb_size)
        self.register_buffer("inverse_aabb_size", 2.0 / aabb_size)
        grid_size = torch.as_tensor(
            grid_size,
            dtype=torch.long,
            device=self.aabb.device,
        )
        self.register_buffer("grid_size", grid_size)
        self.register_buffer("units", aabb_size / (self.grid_size - 1))

    def initialize_factors(self):
        raise NotImplementedError

    def compute_density_features(self, xyz_sampled):
        raise NotImplementedError

    def compute_appearance_features(self, xyz_sampled):
        raise NotImplementedError

    def normalize_points(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.inverse_aabb_size - 1

    def valid_sample_mask(self, rays_pts):
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(
            dim=-1
        )
        return ~mask_outbbox

    def shrink(self, new_aabb):
        raise NotImplementedError

    @torch.no_grad()
    def compute_dense_opacity(self, grid_size):
        device = self.aabb.device
        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, grid_size[0], device=device),
                torch.linspace(0, 1, grid_size[1], device=device),
                torch.linspace(0, 1, grid_size[2], device=device),
                indexing="ij",
            ),
            -1,
        )
        dense_xyz = self.aabb[0] * (1 - samples) + self.aabb[1] * samples

        opacity = torch.zeros_like(dense_xyz[..., 0])
        for i in range(grid_size[0]):
            opacity[i] = self.compute_opacity(dense_xyz[i].view(-1, 3), 0.01).view(
                (grid_size[1], grid_size[2])
            )
        return opacity, dense_xyz

    @torch.no_grad()
    def update_occupancy_mask(self, grid_size=(200, 200, 200)):
        opacity, dense_xyz = self.compute_dense_opacity(grid_size)
        dense_xyz = dense_xyz.transpose(0, 2).contiguous()
        opacity = opacity.clamp(0, 1).transpose(0, 2).contiguous()[None, None]
        total_voxels = grid_size[0] * grid_size[1] * grid_size[2]

        kernel_size = 3
        opacity = F.max_pool3d(
            opacity,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=1,
        ).view(grid_size[::-1])
        occupancy = opacity >= self.occupancy_threshold

        valid_xyz = dense_xyz[occupancy]
        if valid_xyz.numel() == 0:
            self.occupancy_mask = None
            print(
                "occupancy update skipped: no voxel reached "
                f"threshold {self.occupancy_threshold}"
            )
            return None

        self.occupancy_mask = OccupancyGridMask(self.aabb, occupancy)
        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))
        occupied_percent = occupancy.sum().item() / total_voxels * 100
        print(f"occupancy bounding box: {new_aabb}; occupied: {occupied_percent:.2f}%")
        return new_aabb

    def activate_density(self, density_features):
        return F.relu(density_features)

    def compute_opacity(self, xyz_locs, length=0.01):
        if self.occupancy_mask is not None:
            occupancy = self.occupancy_mask.sample_occupancy(xyz_locs) > 0
        else:
            occupancy = torch.ones_like(xyz_locs[:, 0], dtype=bool)

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if occupancy.any():
            xyz_sampled = self.normalize_points(xyz_locs[occupancy])
            sigma_feature = self.compute_density_features(xyz_sampled)
            valid_sigma = self.activate_density(sigma_feature)
            sigma[occupancy] = valid_sigma

        return 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

    def set_iteration(self, iteration):
        self.current_iteration = iteration

        if not self.training:
            return

        if iteration in self.occupancy_update_iters:
            mask_resolution = tuple(int(value) for value in self.grid_size.tolist())
            if mask_resolution[0] > 200:
                mask_resolution = (200, 200, 200)
            new_aabb = self.update_occupancy_mask(mask_resolution)

            if new_aabb is not None and iteration == self.occupancy_update_iters[0]:
                self.shrink(new_aabb)

        if iteration in self.upsample_iters:
            schedule_index = list(self.upsample_iters).index(iteration)
            voxel_count = self.voxel_count_schedule[schedule_index]
            target_resolution = voxel_count_to_grid_resolution(voxel_count, self.aabb)
            self.upsample_grid(target_resolution)

class PlanarVoxelFactorization(PlanarVoxelFactorBase):
    def initialize_factors(self):
        device = self.aabb.device
        self.density_planes, self.density_lines = self.initialize_density_factors(
            self.density_component_counts,
            self.grid_size,
            1e-2,
            device,
        )
        self.appearance_planes, self.appearance_lines = self.initialize_appearance_factors(
            self.appearance_component_counts, self.grid_size, 0.1, device
        )
        self.appearance_basis = torch.nn.Linear(
            sum(self.appearance_component_counts),
            self.appearance_feature_channels,
            bias=False,
        ).to(device)
        self.opt_group = {
            "color": [
                self.density_lines,
                self.density_planes,
                self.appearance_lines,
                self.appearance_planes,
                self.appearance_basis,
            ],
        }

    def initialize_appearance_factors(self, n_component, grid_size, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.line_axes)):
            vec_id = self.line_axes[i]
            mat_id_0, mat_id_1 = self.plane_axes[i]

            plane_coef.append(
                torch.nn.Parameter(
                    scale
                    * torch.randn(
                        (1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0])
                    )
                )
            )
            line_coef.append(
                torch.nn.Parameter(
                    scale * torch.randn((1, n_component[i], grid_size[vec_id], 1))
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def initialize_density_factors(self, n_component, grid_size, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.line_axes)):
            vec_id = self.line_axes[i]
            mat_id_0, mat_id_1 = self.plane_axes[i]

            plane_coef.append(
                torch.nn.Parameter(
                    scale
                    * torch.rand(
                        (1, n_component[i], grid_size[mat_id_1], grid_size[mat_id_0])
                    ).clamp(1e-2, 1e8)
                )
            )
            line_coef.append(
                torch.nn.Parameter(
                    scale
                    * torch.rand(
                        (1, n_component[i], grid_size[vec_id], 1)
                    ).clamp(1e-2, 1e8)
                )
            )

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(
            line_coef
        ).to(device)

    def density_l1_loss(self):
        total = 0
        for idx in range(len(self.density_planes)):
            if self.density_planes[idx].shape[1] == 0:
                continue

            total = (
                total
                + torch.mean(torch.abs(self.density_planes[idx]))
                + torch.mean(torch.abs(self.density_lines[idx]))
            )
        return total

    def density_tv_loss(self, reg):
        total = 0
        for idx in range(len(self.density_planes)):
            if self.density_planes[idx].shape[1] == 0:
                continue

            total = total + reg(self.density_planes[idx]) * 1e-2
        return total

    def appearance_tv_loss(self, reg):
        total = 0
        for idx in range(len(self.appearance_planes)):
            if self.appearance_planes[idx].shape[1] == 0:
                continue

            total = total + reg(self.appearance_planes[idx]) * 1e-2
        return total

    @torch.no_grad()
    def upsample_factors(self, plane_coef, line_coef, target_resolution):

        for i in range(len(self.line_axes)):
            vec_id = self.line_axes[i]
            mat_id_0, mat_id_1 = self.plane_axes[i]
            if plane_coef[i].shape[1] > 0:
                plane_coef[i] = torch.nn.Parameter(
                    F.interpolate(
                        plane_coef[i].data,
                        size=(target_resolution[mat_id_1], target_resolution[mat_id_0]),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            else:
                plane_coef[i] = torch.nn.Parameter(
                    plane_coef[i].new_empty(
                        1,
                        0,
                        target_resolution[mat_id_1],
                        target_resolution[mat_id_0],
                    )
                )
            if line_coef[i].shape[1] > 0:
                line_coef[i] = torch.nn.Parameter(
                    F.interpolate(
                        line_coef[i].data,
                        size=(target_resolution[vec_id], 1),
                        mode="bilinear",
                        align_corners=True,
                    )
                )
            else:
                line_coef[i] = torch.nn.Parameter(
                    line_coef[i].new_empty(
                        1,
                        0,
                        target_resolution[vec_id],
                        1,
                    )
                )

        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_grid(self, target_resolution):
        self.appearance_planes, self.appearance_lines = self.upsample_factors(
            self.appearance_planes, self.appearance_lines, target_resolution
        )
        self.density_planes, self.density_lines = self.upsample_factors(
            self.density_planes, self.density_lines, target_resolution
        )

        self.update_grid_geometry(target_resolution)
        print(f"upsampling to {target_resolution}")

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (
            xyz_max - self.aabb[0]
        ) / self.units
        t_l, b_r = torch.round(t_l).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.grid_size]).amin(0)

        for i in range(len(self.line_axes)):
            mode0 = self.line_axes[i]
            self.density_lines[i] = torch.nn.Parameter(
                self.density_lines[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            self.appearance_lines[i] = torch.nn.Parameter(
                self.appearance_lines[i].data[..., t_l[mode0] : b_r[mode0], :]
            )
            mode0, mode1 = self.plane_axes[i]
            self.density_planes[i] = torch.nn.Parameter(
                self.density_planes[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )
            self.appearance_planes[i] = torch.nn.Parameter(
                self.appearance_planes[i].data[
                    ..., t_l[mode1] : b_r[mode1], t_l[mode0] : b_r[mode0]
                ]
            )

        if not torch.all(self.occupancy_mask.grid_size == self.grid_size):
            t_l_r, b_r_r = t_l / (self.grid_size - 1), (b_r - 1) / (self.grid_size - 1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1 - t_l_r) * self.aabb[0] + t_l_r * self.aabb[1]
            correct_aabb[1] = (1 - b_r_r) * self.aabb[0] + b_r_r * self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        new_size = b_r - t_l
        self.register_buffer('aabb', new_aabb)
        self.update_grid_geometry((new_size[0], new_size[1], new_size[2]))
