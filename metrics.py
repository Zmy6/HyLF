#!/usr/bin/env python

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mean_squared_error(prediction, target):
    return torch.mean(torch.square(prediction - target))


def compute_psnr(prediction, target):
    return peak_signal_noise_ratio(
        np.asarray(target),
        np.asarray(prediction),
        data_range=1.0,
    )


def compute_ssim(prediction, target):
    try:
        return structural_similarity(
            np.asarray(target),
            np.asarray(prediction),
            win_size=11,
            channel_axis=-1,
            gaussian_weights=True,
            data_range=1.0,
        )
    except TypeError:
        return structural_similarity(
            np.asarray(target),
            np.asarray(prediction),
            win_size=11,
            multichannel=True,
            gaussian_weights=True,
            data_range=1.0,
        )


def compute_tensor_psnr(prediction, target):
    return -10 * torch.log10(mean_squared_error(prediction, target))


def average_step_outputs(outputs, cpu=False):
    grouped = {}
    for output in outputs:
        for key, value in output.items():
            grouped.setdefault(key, []).append(value)

    if cpu:
        return {key: np.stack(values).mean() for key, values in grouped.items()}
    return {key: torch.stack(values).mean() for key, values in grouped.items()}
