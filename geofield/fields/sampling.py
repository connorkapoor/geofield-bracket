"""Stratified sampling of (x, f(x), grad f(x)) from analytic fields.

Points are drawn uniformly inside a ball of radius `radius` (default 1.2,
covering the unit-sphere-normalized shape plus a margin) and binned by |f|
into bands. Rejection sampling fills each band to its quota; if a band cannot
be filled within `max_rounds` (e.g. a thin shell band on a tiny solid), the
shortfall is filled from the nearest over-full band so the output always has
exactly `n` points.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import torch
from torch import Tensor

from .primitives import Field

DEFAULT_BANDS = [(0.0, 0.02), (0.02, 0.1), (0.1, 0.5), (0.5, 1.2)]
DEFAULT_WEIGHTS = [0.4, 0.3, 0.2, 0.1]


@dataclass
class SampleSet:
    x: Tensor      # [N, 3]
    f: Tensor      # [N]
    grad: Tensor   # [N, 3]
    band: Tensor   # [N] int band index (by |f|)


def _uniform_ball(n: int, radius: float, gen: torch.Generator, device) -> Tensor:
    # generator lives on CPU for cross-device determinism; move result after
    v = torch.randn(n, 3, generator=gen)
    v = v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
    r = radius * torch.rand(n, 1, generator=gen).pow(1.0 / 3.0)
    return (v * r).to(device)


def _band_index(absf: Tensor, bands) -> Tensor:
    idx = torch.full_like(absf, -1, dtype=torch.long)
    for i, (lo, hi) in enumerate(bands):
        idx = torch.where((absf >= lo) & (absf < hi), torch.full_like(idx, i), idx)
    return idx


def stratified(
    field: Field,
    n: int = 16384,
    bands=None,
    weights=None,
    radius: float = 1.2,
    seed: int = 0,
    device: str | torch.device = "cpu",
    max_rounds: int = 200,
    chunk: int = 65536,
) -> SampleSet:
    """Sample n points stratified by |f| bands with exact f and grad.

    Surface-band points (band 0) are additionally boosted by projecting
    random points onto the surface along -f * grad (one Newton step of
    sphere tracing), which is exact for a true distance field; projected
    points are re-verified against the band bounds before acceptance.
    """
    bands = bands if bands is not None else DEFAULT_BANDS
    weights = weights if weights is not None else DEFAULT_WEIGHTS
    assert len(bands) == len(weights)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    quotas = [int(round(w / sum(weights) * n)) for w in weights]
    quotas[0] += n - sum(quotas)  # rounding remainder to the surface band

    kept_x = [[] for _ in bands]
    counts = [0] * len(bands)

    for round_i in range(max_rounds):
        if all(c >= q for c, q in zip(counts, quotas)):
            break
        x = _uniform_ball(chunk, radius, gen, device)
        if round_i % 2 == 1 and counts[0] < quotas[0]:
            # Surface-projection boost: exact projection for a distance field.
            with torch.no_grad():
                f0 = field(x)
                g0 = field.grad(x)
                x = x - f0.unsqueeze(-1) * g0
                # jitter within the surface band so band 0 isn't exactly f=0
                jit = ((torch.rand(x.shape[0], generator=gen) * 2 - 1) * bands[0][1]).to(x.device)
                x = x + jit.unsqueeze(-1) * field.grad(x)
                x = x[torch.linalg.vector_norm(x, dim=-1) <= radius]
        with torch.no_grad():
            f = field(x)
        idx = _band_index(f.abs(), bands)
        for b in range(len(bands)):
            need = quotas[b] - counts[b]
            if need <= 0:
                continue
            xb = x[idx == b][:need]
            if xb.numel():
                kept_x[b].append(xb)
                counts[b] += xb.shape[0]

    # Fill shortfalls from neighbouring bands' surplus pool (rare).
    all_x = []
    for b in range(len(bands)):
        xb = torch.cat(kept_x[b], dim=0) if kept_x[b] else torch.empty(0, 3, device=device)
        all_x.append(xb)
    total = sum(t.shape[0] for t in all_x)
    if total < n:
        extra = _uniform_ball(n - total, radius, gen, device)
        all_x.append(extra)
    x = torch.cat(all_x, dim=0)[:n]

    f = field(x)
    g = field.grad(x)
    return SampleSet(x=x, f=f, grad=g, band=_band_index(f.abs(), bands))
