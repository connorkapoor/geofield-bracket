"""Rotation study: equivariance error distribution + same/different-shape
discrimination vs rotation angle from the pooled latent.

For the equivariant model the error should sit at float noise; for the
baseline it measures how well SE(3) augmentation approximated the symmetry.
"""
from __future__ import annotations

import math

import torch

from ..fields.programs.common import random_rotation


def _axis_angle_rotation(angle: float, axis: torch.Tensor) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    return torch.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


@torch.no_grad()
def equivariance_error(model, record: dict, n_rot: int = 16,
                       device: str = "cpu", n_input: int = 1024,
                       n_query: int = 512, seed: int = 0) -> list[float]:
    """Relative sdf discrepancy under random rotations for one record."""
    gen = torch.Generator().manual_seed(seed)
    x = record["x"].unsqueeze(0).to(device)
    f = record["f"].unsqueeze(0).to(device)
    g = record["grad"].unsqueeze(0).to(device)
    ii = torch.randperm(x.shape[1], generator=gen)[:n_input]
    qi = torch.randperm(x.shape[1], generator=gen)[:n_query]
    lat0 = model.encode(x[:, ii], f[:, ii], g[:, ii])
    o0 = model.decode(lat0, x[:, qi], "sdf")
    scale = o0.abs().max().item() + 1e-9
    errs = []
    for _ in range(n_rot):
        R = random_rotation(gen).to(device)
        t = (torch.randn(3, generator=gen) * 0.1).to(device)
        lat = model.encode(x[:, ii] @ R.T + t, f[:, ii], g[:, ii] @ R.T)
        o = model.decode(lat, x[:, qi] @ R.T + t, "sdf")
        errs.append(float((o - o0).abs().max().item() / scale))
    return errs


@torch.no_grad()
def pooled_rotation_similarity(model, records: list[dict], angles=None,
                               device: str = "cpu", n_input: int = 1024,
                               seed: int = 0) -> dict:
    """Cosine similarity of pooled() between a shape and its rotated copy,
    and between DIFFERENT shapes, binned by rotation angle (0-180 deg).
    A rotation-robust latent keeps same-shape similarity >> cross-shape
    similarity at every angle. Returns {angle_deg: {same, diff}}."""
    angles = angles if angles is not None else [0, 30, 60, 90, 120, 150, 180]
    gen = torch.Generator().manual_seed(seed)
    pooled0 = []
    subsets = []
    for rec in records:
        x = rec["x"].unsqueeze(0).to(device)
        fv = rec["f"].unsqueeze(0).to(device)
        g = rec["grad"].unsqueeze(0).to(device)
        ii = torch.randperm(x.shape[1], generator=gen)[:n_input]
        subsets.append((x[:, ii], fv[:, ii], g[:, ii]))
        pooled0.append(model.encode(*subsets[-1]).pooled().squeeze(0))
    pooled0 = torch.stack(pooled0)
    pooled0n = torch.nn.functional.normalize(pooled0, dim=-1)

    out = {}
    for ang in angles:
        axis = torch.randn(3, generator=gen)
        R = _axis_angle_rotation(math.radians(ang), axis).to(device)
        same, diff = [], []
        for i, (x, fv, g) in enumerate(subsets):
            p = model.encode(x @ R.T, fv, g @ R.T).pooled().squeeze(0)
            pn = torch.nn.functional.normalize(p, dim=0)
            sims = pooled0n @ pn
            same.append(float(sims[i]))
            diff.append(float((sims.sum() - sims[i]) / max(len(records) - 1, 1)))
        out[ang] = {"same": sum(same) / len(same), "diff": sum(diff) / len(diff)}
    return out
