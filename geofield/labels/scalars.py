"""Per-record scalar labels derived from fields and labelers.

All fractions are computed over near-surface samples (|f| below a band) so
they measure surface area fractions, not volume fractions. `mass` uses Monte
Carlo volume of {f < 0} inside the sampling ball times material density.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from ..fields.primitives import Field
from . import manufacturability as mfg


def surface_mask(f: Tensor, band: float = 0.02) -> Tensor:
    return f.abs() < band


def support_fraction(support: Tensor, f: Tensor, band: float = 0.02) -> float:
    m = surface_mask(f, band)
    return float(support[m].mean().item()) if m.any() else 0.0


def inaccessible_fraction(access_binary: Tensor, f: Tensor, band: float = 0.02) -> float:
    m = surface_mask(f, band)
    return float((1.0 - access_binary[m]).mean().item()) if m.any() else 0.0


def min_thickness(thick: Tensor, f: Tensor, band: float = 0.02,
                  percentile: float = 0.05) -> float:
    """Robust minimum: the 5th percentile of surface-sample thickness.

    Pass RAY-CHORD thickness (labels.manufacturability.ray_thickness), not the
    sphere-growing field label: the inscribed-sphere radius legitimately
    collapses to zero at convex and reentrant edges, so its low percentiles
    measure edge geometry, not wall thickness."""
    m = surface_mask(f, band)
    if not m.any():
        return 0.0
    return float(torch.quantile(thick[m], percentile).item())


def volume(field: Field, n: int = 200_000, radius: float = 1.05,
           seed: int = 0, device="cpu") -> float:
    """Monte Carlo volume of the solid inside the sampling ball."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(n, 3, generator=gen)
    v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
    r = radius * torch.rand(n, 1, generator=gen).pow(1 / 3)
    x = (v * r).to(device)
    with torch.no_grad():
        frac = (field(x) < 0).float().mean().item()
    return frac * (4.0 / 3.0) * math.pi * radius ** 3


def mass(field: Field, density: float, **kw) -> float:
    return volume(field, **kw) * density
