#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

from .light_field import LightFieldDataset


class SparseAngularLightFieldDataset(LightFieldDataset):
    def read_metadata(self):
        self.image_paths = sorted(self.path_manager.ls(self.root_dir))
        expected_images = self.rows * self.cols
        if len(self.image_paths) < expected_images:
            raise ValueError(
                f"Expected at least {expected_images} views in {self.root_dir}, "
                f"found {len(self.image_paths)}"
            )

        if self.image_size is None:
            image_path = os.path.join(self.root_dir, self.image_paths[0])
            with self.path_manager.open(image_path, "rb") as image_file:
                image = np.array(Image.open(image_file).convert("RGB"))
            self._native_image_size = (
                image.shape[1] // self.downsample,
                image.shape[0] // self.downsample,
            )
            self.image_size = self._native_image_size

        self.aspect = float(self.image_size[0]) / self.image_size[1]

    def load_rgb_view(self, s_idx, t_idx):
        image_path = self.image_paths[t_idx * self.cols + s_idx]
        with self.path_manager.open(
            os.path.join(self.root_dir, image_path),
            "rb",
        ) as image_file:
            image = Image.open(image_file).convert("RGB")

        image = image.resize(self._native_image_size, Image.LANCZOS)
        if self.image_size != self._native_image_size:
            image = image.resize(self.image_size, Image.BOX)
        return self.transform(image).view(3, -1).permute(1, 0)
