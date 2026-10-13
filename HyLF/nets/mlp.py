#!/usr/bin/env python

import torch
from torch import nn

from HyLF.activations import get_activation


class DenseMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cfg,
        default_pad_to=None,
        return_rgb_dict=False,
    ):
        super().__init__()
        self.opt_group = getattr(cfg, "group", "color")
        self.return_rgb_dict = return_rgb_dict
        self.in_channels = in_channels
        self.out_channels = out_channels if cfg.depth != 0 else self.in_channels
        self.pad_to = getattr(cfg, "pad_to", default_pad_to)
        self.skips = list(getattr(cfg, "skips", []))
        self.linear_last = getattr(cfg, "linear_last", True)
        self.bias = getattr(cfg, "bias", True)
        net_in_channels = self.pad_to or self.in_channels

        depth = cfg.depth
        hidden_channels = cfg.hidden_channels
        layer_activation = getattr(cfg, "layer_activation", "leaky_relu")
        self.layers = nn.ModuleList()
        for index in range(depth + 2):
            if index == 0:
                layer = nn.Linear(net_in_channels, hidden_channels, bias=self.bias)
            elif index == depth + 1:
                layer = nn.Linear(hidden_channels, self.out_channels, bias=self.bias)
            elif index in self.skips:
                layer = nn.Linear(
                    hidden_channels + net_in_channels,
                    hidden_channels,
                    bias=self.bias,
                )
            else:
                layer = nn.Linear(hidden_channels, hidden_channels, bias=self.bias)

            apply_activation = index < depth if self.linear_last else index < depth + 1
            if apply_activation:
                layer = nn.Sequential(layer, get_activation(layer_activation))
            self.layers.append(layer)

        self.output_activation = get_activation(getattr(cfg, "activation", "identity"))

    def forward(self, values):
        if isinstance(values, dict):
            values = values["points"]
        if self.pad_to is not None:
            padding = self.pad_to - values.shape[-1]
            if padding < 0:
                raise ValueError(
                    f"MLP input has {values.shape[-1]} channels, larger than pad_to={self.pad_to}"
                )
            values = torch.cat([values, values.new_ones(values.shape[0], padding)], dim=-1)

        residual = values
        for index, layer in enumerate(self.layers):
            if index in self.skips:
                values = torch.cat([residual, values], dim=-1)
            values = layer(values)

        output = self.output_activation(values)
        if self.return_rgb_dict:
            return {"rgb": output, "depth": output[..., 0]}
        return output

class FeatureMLP(DenseMLP):
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__(in_channels, out_channels, cfg)


class RGBDecoderMLP(DenseMLP):
    def __init__(self, in_channels, out_channels, cfg):
        super().__init__(
            in_channels,
            out_channels,
            cfg,
            default_pad_to=96,
            return_rgb_dict=True,
        )


mlp_registry = {
    "feature_mlp": FeatureMLP,
    "rgb_decoder_mlp": RGBDecoderMLP,
}
