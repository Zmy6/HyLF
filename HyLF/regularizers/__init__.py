#!/usr/bin/env python

from .planar_voxel import PlanarVoxelRegularizer

regularizer_registry = {
    "planar_voxel": PlanarVoxelRegularizer,
}
