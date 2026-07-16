#!/usr/bin/env python

from .sparse_angular import SparseAngularLightFieldDataset

dataset_registry = {
    "sparse_angular": SparseAngularLightFieldDataset,
}
