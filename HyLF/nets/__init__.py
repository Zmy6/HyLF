#!/usr/bin/env python

from .mlp import mlp_registry
from .planar_voxel_grid import planar_voxel_grid_registry


network_registry = {**mlp_registry, **planar_voxel_grid_registry}
