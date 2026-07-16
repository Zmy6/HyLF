#!/usr/bin/env python

from typing import Dict

import torch
from torch import nn

class FieldExtractor(nn.Module):
    def __init__(self, _in_channels, cfg):
        super().__init__()
        self.fields = list(cfg.fields)

    def forward(self, values: Dict[str, torch.Tensor]):
        return {field: values[field] for field in self.fields}


class HybridPointOffset(nn.Module):
    def __init__(self, _in_channels, cfg):
        super().__init__()
        self.density_field = cfg.in_density_field

    def forward(self, values: Dict[str, torch.Tensor]):
        points = values["points"]
        sigma = values[self.density_field]
        offset = values["point_offset"] * (1 - sigma)
        values["point_offset"] = offset
        values["points"] = points + offset
        return values


point_embedding_registry = {
    "field_extractor": FieldExtractor,
    "hybrid_point_offset": HybridPointOffset,
}
