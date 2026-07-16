#!/usr/bin/env python

import torch


def build_light_field_rays(width, height, s, t, aspect, near=-1.0, far=0.0):
    u = torch.linspace(-1, 1, width, dtype=torch.float32)
    v = torch.linspace(1, -1, height, dtype=torch.float32) / aspect
    v, u = torch.meshgrid(v, u, indexing="ij")
    s = torch.full_like(u, s)
    t = torch.full_like(v, t)
    return torch.stack(
        [
            s,
            t,
            torch.full_like(s, near),
            u,
            v,
            torch.full_like(s, far - near),
        ],
        dim=-1,
    ).reshape(-1, 6)
