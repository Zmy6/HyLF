#!/usr/bin/env python

import numpy as np
import torch
from torch import nn


class WindowedPE(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.current_iteration = 0
        self.wait_iters = cfg.wait_iters
        self.max_freq_iter = float(cfg.max_freq_iter)
        self.n_freqs = cfg.n_freqs
        self.freq_multiplier = getattr(cfg, "freq_multiplier", 2.0)
        self.register_buffer(
            "freq_bands",
            self.freq_multiplier ** torch.linspace(1.0, cfg.n_freqs, cfg.n_freqs),
        )
        self.exclude_identity = getattr(cfg, "exclude_identity", False)
        self.out_channels = in_channels * (
            2 * cfg.n_freqs + (0 if self.exclude_identity else 1)
        )
        # Preserve reference RNG consumption so seeded MLP weights match.
        self.dummy_layer = nn.Linear(1, 1)

        if self.n_freqs > 0:
            window_size = self.max_freq_iter / self.n_freqs
            self.window_iters = [
                (
                    window_size * index + self.wait_iters,
                    window_size * (index + 1) + self.wait_iters,
                )
                for index in range(self.n_freqs)
            ]

    def weight(self, index):
        if self.max_freq_iter == 0 or self.current_iteration > self.max_freq_iter:
            return 1.0
        if self.current_iteration < self.wait_iters:
            return 0.0
        start, end = self.window_iters[index]
        if end == start:
            return float(self.current_iteration >= start)
        alpha = (self.current_iteration - start) / float(end - start)
        return (1.0 - np.cos(np.pi * np.clip(alpha, 0.0, 1.0))) / 2

    def forward(self, values):
        encoded = [] if self.exclude_identity else [values]
        for index, frequency in enumerate(self.freq_bands):
            weight = self.weight(index)
            encoded.extend(
                [
                    weight * torch.sin(frequency * values),
                    weight * torch.cos(frequency * values),
                ]
            )
        return torch.cat(encoded, dim=-1)

    def set_iteration(self, iteration):
        self.current_iteration = iteration


encoding_registry = {
    "windowed": WindowedPE,
}
