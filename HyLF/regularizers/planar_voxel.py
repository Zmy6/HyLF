#!/usr/bin/env python

import numpy as np
import torch
from torch import nn

from .base import BaseRegularizer


class TotalVariationLoss(nn.Module):
    def forward(self, values):
        batch_size = values.shape[0]
        height_count = values[:, :, 1:, :].numel()
        width_count = values[:, :, :, 1:].numel()
        height_tv = torch.square(values[:, :, 1:, :] - values[:, :, :-1, :]).sum()
        width_tv = torch.square(values[:, :, :, 1:] - values[:, :, :, :-1]).sum()
        return 2 * (height_tv / height_count + width_tv / width_count) / batch_size


class PlanarVoxelRegularizer(BaseRegularizer):
    def __init__(self, system, cfg):
        super().__init__(system, cfg)
        self.tv_loss = TotalVariationLoss()
        self.lr_factor = cfg.lr_decay_target_ratio ** (1 / cfg.n_iters)
        self.total_num_tv_iters = getattr(
            cfg,
            "total_num_tv_iters",
            int(
                np.round(
                    np.log(1e-4)
                    / np.log(cfg.lr_decay_target_ratio)
                    * cfg.n_iters
                )
            ),
        )
        self.l1_weight = cfg.initial_l1_weight
        self.configured_tv_density_weight = cfg.tv_density_weight
        self.configured_tv_appearance_weight = cfg.tv_appearance_weight
        self.tv_density_weight = cfg.tv_density_weight
        self.tv_appearance_weight = cfg.tv_appearance_weight

    def _compute_loss(self):
        voxel_grid = self.get_hylf_system().render_fn.model.color_model.net
        total_loss = 0.0

        if self.l1_weight > 0:
            total_loss += self.l1_weight * voxel_grid.density_l1_loss()
        if self.current_iteration > self.total_num_tv_iters:
            return total_loss

        tv_loss = 0.0
        if self.tv_density_weight > 0:
            self.tv_density_weight *= self.lr_factor
            tv_loss = (
                voxel_grid.density_tv_loss(self.tv_loss)
                * self.configured_tv_density_weight
            )
            total_loss += tv_loss
        if self.tv_appearance_weight > 0:
            self.tv_appearance_weight *= self.lr_factor
            tv_loss += (
                voxel_grid.appearance_tv_loss(self.tv_loss)
                * self.configured_tv_appearance_weight
            )
            total_loss += tv_loss
        return total_loss
