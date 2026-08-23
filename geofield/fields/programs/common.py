"""Shared helpers for shape programs: random rotations, unit-sphere
normalization. (Moved from the retired random-bracket program.)"""
from __future__ import annotations

import torch
from torch import Tensor

from ..primitives import Field

NORM_RADIUS = 0.85  # shapes normalized to fit inside this radius


def random_rotation(gen: torch.Generator) -> Tensor:
    """Uniform random rotation via QR of a Gaussian matrix."""
    A = torch.randn(3, 3, generator=gen)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def normalize_to_unit_sphere(field: Field, tokens: list, seed: int = 0,
                             n_probe: int = 4096, probe_scale: float = 0.8
                             ) -> tuple[Field, list, float, Tensor]:
    """Center the surface at the origin and scale to radius NORM_RADIUS.

    Surface points are found by projecting random probes along -f * grad
    (exact for a distance field, near-exact through blends).

    Returns (field, tokens, s, center) where s is the applied scale factor
    (normalized units per input unit) and center the applied translation —
    callers needing real-world units keep 1/s as meters(or mm)-per-unit.
    """
    gen = torch.Generator().manual_seed(seed ^ 0x5EED)
    x = torch.randn(n_probe, 3, generator=gen) * probe_scale
    for _ in range(4):
        with torch.no_grad():
            f = field(x)
            g = field.grad(x)
        x = x - f.unsqueeze(-1) * g
    with torch.no_grad():
        f = field(x)
    tol = 5e-3 * probe_scale
    surf = x[f.abs() < tol]
    if surf.shape[0] < 32:  # degenerate; fall back to all probes
        surf = x
    center = (surf.amin(0) + surf.amax(0)) / 2
    radius = torch.linalg.vector_norm(surf - center, dim=-1).amax().item()
    s = NORM_RADIUS / max(radius, 1e-9)

    field = field.transform(torch.eye(3), -center).scale(s)
    tokens = [tok.transform(torch.eye(3), -center).scale(s) for tok in tokens]
    return field, tokens, s, center
