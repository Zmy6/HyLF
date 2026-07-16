#!/usr/bin/env python

import torch
from torch import nn

from .point import point_embedding_registry
from .ray import ray_embedding_registry


embedding_registry = {**ray_embedding_registry, **point_embedding_registry}


class HybridRayEmbedding(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.embeddings = nn.ModuleList()

        for embedding_cfg in cfg.embeddings.values():
            self.embeddings.append(
                embedding_registry[embedding_cfg.type](in_channels, embedding_cfg)
            )

        self.out_channels = in_channels

    def forward(self, rays: torch.Tensor):
        values = {"rays": rays}
        for embedding in self.embeddings:
            values = embedding(values)
        return {key: value.reshape(rays.shape[0], -1) for key, value in values.items()}

    def set_iteration(self, iteration):
        for embedding in self.embeddings:
            if hasattr(embedding, "set_iteration"):
                embedding.set_iteration(iteration)


embedding_registry["hybrid_ray"] = HybridRayEmbedding
