#!/usr/bin/env python

from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(self, _config):
        super().__init__()
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, prediction, target, **kwargs):
        return self.loss(prediction, target)


loss_registry = {"mse": ReconstructionLoss}
