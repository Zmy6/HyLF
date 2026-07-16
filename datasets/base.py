#!/usr/bin/env python

import os

import torch
from iopath.common.file_io import NativePathHandler, PathManager
from torch.utils.data import Dataset
from torchvision import transforms as T


class BaseDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = getattr(cfg.dataset, "split", split)
        self.dataset_cfg = getattr(cfg.dataset, self.split, cfg.dataset)

        self.path_manager = PathManager()
        self.path_manager.register_handler(NativePathHandler())
        self.root_dir = os.path.expanduser(self.dataset_cfg.root_dir)

        if self.dataset_cfg.image_size is not None:
            self._native_image_size = tuple(self.dataset_cfg.image_size)
            self.image_size = self._native_image_size
            self.downsample = 1
        else:
            self.image_size = None
            self.downsample = getattr(self.dataset_cfg, "downsample", 1)

        self.validation_crop = getattr(self.dataset_cfg, "validation_crop", 1.0)
        self.render_crop = self.dataset_cfg.render_params.crop
        self.transform = T.ToTensor()
        self.prepare_data()

    def prepare_data(self):
        self.read_metadata()
        if self.split == "train":
            if self.cfg.params.render_only or self.cfg.params.test_only:
                self.training_samples = [0]
            else:
                self.prepare_training_data()
        elif self.split == "val":
            self.prepare_validation_data()
        elif self.split == "test":
            self.prepare_test_data()
        elif self.split == "render":
            self.prepare_render_data()

    def read_metadata(self):
        raise NotImplementedError

    def prepare_training_data(self):
        raise NotImplementedError

    def prepare_validation_data(self):
        self.prepare_test_data()

    def prepare_test_data(self):
        raise NotImplementedError

    def prepare_render_data(self):
        self.prepare_test_data()

    def set_training_data(self, rays, rgb):
        self.training_rays = rays
        self.training_rgb = rgb
        self.training_weights = torch.ones_like(rays[..., :1])
        self.training_samples = torch.cat(
            [self.training_rays, self.training_rgb, self.training_weights], dim=-1
        )

    def format_training_batch(self, batch):
        ray_channels = self.training_rays.shape[-1]
        batch["rays"] = batch["training_samples"][..., :ray_channels]
        batch["rgb"] = batch["training_samples"][..., ray_channels:ray_channels + 3]
        batch["weight"] = batch["training_samples"][..., -1:]
        del batch["training_samples"]
        return batch

    def crop_batch(self, batch):
        width, height = self.image_size
        crop = (
            self.validation_crop
            if self.split in {"val", "test"}
            else self.render_crop
        )
        if crop >= 1.0:
            return width, height, batch

        original_width, original_height = width, height
        half_width = int(width // 2 * crop)
        half_height = int(height // 2 * crop)
        for key, value in batch.items():
            if not torch.is_tensor(value) or value.ndim == 0:
                continue
            value = value.view(1, original_height, original_width, -1)
            value = value[
                :,
                original_height // 2 - half_height:original_height // 2 + half_height + 1,
                original_width // 2 - half_width:original_width // 2 + half_width + 1,
            ]
            height, width = value.shape[1:3]
            batch[key] = value.reshape(-1, value.shape[-1])
        return width, height, batch
