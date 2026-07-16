#!/usr/bin/env python

import numpy as np
from torch import nn


class BaseRegularizer(nn.Module):
    def __init__(self, system, cfg):
        super().__init__()
        self.weight = getattr(cfg, "weight", 0.0)
        self.systems = [system]
        self.current_iteration = 0

    def get_hylf_system(self):
        return self.systems[0]

    def compute_loss(self):
        return self._compute_loss()

    def compute_weight(self):
        if isinstance(self.weight, float):
            return self.weight
        if self.weight.type == "exponential_decay":
            system = self.get_hylf_system()
            iterations = (
                self.weight.num_epochs
                * len(system.trainer.datamodule.train_dataset)
                // system.trainer.datamodule.current_batch_size
            )
            exponent = self.current_iteration / max(iterations, 1)
            return self.weight.start * np.power(self.weight.decay, exponent)
        return self.weight

    def set_iteration(self, iteration):
        self.current_iteration = iteration
