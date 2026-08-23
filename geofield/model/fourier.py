"""Random Fourier feature embeddings.

phi(x) = [sin(2 pi B x), cos(2 pi B x)], B ~ N(0, sigma^2) fixed at init.
Output dim = 2 * n_freq.

Default sigma: the smallest resolvable wavelength should be ~ 1/32 of the
bounding box (bbox = 2.1 for the unit-sphere-normalized scenes). Gaussian B
has ~2 sigma tail, so sigma ~= 32 / (2 * bbox) puts the shortest wavelength
1/(2 sigma) at bbox/32. Exposed as a hyperparameter.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

BBOX = 2.1


def default_sigma(bbox: float = BBOX, finest_fraction: float = 1 / 32) -> float:
    return 1.0 / (2 * bbox * finest_fraction)


class FourierFeatures(nn.Module):
    def __init__(self, dim_in: int = 3, n_freq: int = 64,
                 sigma: float | None = None, seed: int = 0):
        super().__init__()
        sigma = default_sigma() if sigma is None else sigma
        gen = torch.Generator().manual_seed(seed)
        B = torch.randn(n_freq, dim_in, generator=gen) * sigma
        self.register_buffer("B", B)
        self.dim_out = 2 * n_freq

    def forward(self, x: Tensor) -> Tensor:
        proj = 2 * math.pi * x @ self.B.T
        return torch.cat([proj.sin(), proj.cos()], dim=-1)
