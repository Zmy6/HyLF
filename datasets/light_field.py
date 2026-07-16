#!/usr/bin/env python

from itertools import product

import torch

from .base import BaseDataset
from utils.ray_utils import build_light_field_rays


class LightFieldDataset(BaseDataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = getattr(cfg.dataset, "split", split)
        self.dataset_cfg = getattr(cfg.dataset, self.split, cfg.dataset)
        grid = self.dataset_cfg.angular_grid

        self.rows = grid.rows
        self.cols = grid.cols
        self.step = grid.step
        self.start_row = getattr(grid, "start_row", 0)
        self.end_row = getattr(grid, "end_row", self.rows)
        self.start_col = getattr(grid, "start_col", 0)
        self.end_col = getattr(grid, "end_col", self.cols)
        self.near_plane = getattr(grid, "near", -1.0)
        self.far_plane = getattr(grid, "far", 0.0)

        row_indices = range(self.start_row, self.end_row)
        col_indices = range(self.start_col, self.end_col)
        self.all_views = [
            (col_idx, row_idx)
            for row_idx, col_idx in product(row_indices, col_indices)
        ]
        self.training_views = [
            (col_idx, row_idx)
            for row_idx, col_idx in product(
                range(self.start_row, self.end_row, self.step),
                range(self.start_col, self.end_col, self.step),
            )
        ]

        val_pairs = list(getattr(self.dataset_cfg, "val_pairs", []))
        if val_pairs:
            self.evaluation_views = list(zip(val_pairs[::2], val_pairs[1::2]))
        elif getattr(self.dataset_cfg, "val_all", False):
            self.evaluation_views = self.all_views
        else:
            training_set = set(self.training_views)
            self.evaluation_views = [view for view in self.all_views if view not in training_set]

        self.num_test_images = len(self.evaluation_views)

        super().__init__(cfg, split)

    def prepare_training_data(self):
        self.active_views = self.training_views
        rays = [self.get_rays(*view) for view in self.active_views]
        colors = [self.load_rgb_view(*view) for view in self.active_views]
        self.set_training_data(torch.cat(rays, dim=0), torch.cat(colors, dim=0))

    def prepare_test_data(self):
        self.active_views = self.evaluation_views

    def normalize_angular_coordinates(self, view):
        s = view[0] / (self.cols - 1) * 2 - 1 if self.cols > 1 else 0
        t = -(view[1] / (self.rows - 1) * 2 - 1) if self.rows > 1 else 0
        return s, t

    def get_rays(self, s_idx, t_idx):
        s, t = self.normalize_angular_coordinates((s_idx, t_idx))
        return build_light_field_rays(
            self.image_size[0],
            self.image_size[1],
            s,
            t,
            self.aspect,
            near=self.near_plane,
            far=self.far_plane,
        )

    def load_rgb_view(self, s_idx, t_idx):
        raise NotImplementedError

    def __len__(self):
        if self.split == "train":
            return len(self.training_rays) if hasattr(self, "training_rays") else 1
        return self.num_test_images

    def __getitem__(self, index):
        if self.split == "train":
            batch = {"training_samples": self.training_samples[index]}
        else:
            s_idx, t_idx = self.active_views[index]
            batch = {
                "rays": self.get_rays(s_idx, t_idx),
                "view_index": index,
                "s_idx": s_idx,
                "t_idx": t_idx,
            }
            if self.split != "render":
                batch["rgb"] = self.load_rgb_view(s_idx, t_idx)

        width, height, batch = self.crop_batch(batch)
        batch["width"] = width
        batch["height"] = height
        return batch
