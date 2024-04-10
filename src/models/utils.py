# types

import torch
from torch import nn


def exists(val):
    return val is not None


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    """https://github.com/lucidrains/egnn-pytorch"""
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1) if include_self else x
    return x


class CoorsNorm(nn.Module):
    """https://github.com/lucidrains/egnn-pytorch"""

    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


def init_weights(m: nn.Module, act: nn.Module, gain: float = 1.0):
    if type(m) in {nn.Linear}:
        if (type(act.func()) is nn.SELU) or (type(act) is nn.SELU):
            nn.init.kaiming_normal_(m.weight, nonlinearity="linear", mode="fan_in")
            nn.init.zeros_(m.bias)
        else:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(m.weight, gain=gain)
            nn.init.zeros_(m.bias)