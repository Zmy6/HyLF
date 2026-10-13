#!/usr/bin/env python

from collections import defaultdict

import torch
from torch import nn


class HybridLightFieldRenderer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rays):
        return self.model(rays.reshape(-1, rays.shape[-1]))

renderer_registry = {"hylf": HybridLightFieldRenderer}


def render_chunked(rays, renderer, chunk_size):
    results = defaultdict(list)
    if rays.shape[0] <= chunk_size:
        return renderer(rays)
        
    for start in range(0, rays.shape[0], chunk_size):
        outputs = renderer(rays[start:start + chunk_size])
        for key, value in outputs.items():
            results[key].append(value)

    return {key: torch.cat(values, 0) for key, values in results.items()}
