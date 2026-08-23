"""Latent 'closeness' diagnostics.

1. Spearman correlation between latent context distance ||c_m - c_m'|| and
   pose distance ||p_m - p_m'|| within a shape: fine latents should be
   locally organized (nearby elements describe nearby geometry).
2. sdf error binned by |f| (surface band vs far field), on train-like and
   OOD splits — where does reconstruction degrade OOD?
"""
from __future__ import annotations

import torch

from .probes import spearman


@torch.no_grad()
def context_pose_spearman(model, record: dict, device: str = "cpu",
                          n_input: int = 1024, n_pairs: int = 4096,
                          seed: int = 0) -> float:
    gen = torch.Generator().manual_seed(seed)
    x = record["x"].unsqueeze(0).to(device)
    f = record["f"].unsqueeze(0).to(device)
    g = record["grad"].unsqueeze(0).to(device)
    ii = torch.randperm(x.shape[1], generator=gen)[:n_input]
    lat = model.encode(x[:, ii], f[:, ii], g[:, ii])
    p = lat.fine.pos.squeeze(0).cpu()
    c = lat.fine.h.squeeze(0).cpu()
    M = p.shape[0]
    i = torch.randint(M, (n_pairs,), generator=gen)
    j = torch.randint(M, (n_pairs,), generator=gen)
    keep = i != j
    i, j = i[keep], j[keep]
    dp = torch.linalg.vector_norm(p[i] - p[j], dim=-1)
    dc = torch.linalg.vector_norm(c[i] - c[j], dim=-1)
    return spearman(dc, dp)


@torch.no_grad()
def sdf_error_by_band(model, record: dict, device: str = "cpu",
                      n_input: int = 2048,
                      bands=((0, .02), (.02, .1), (.1, .5), (.5, 1.2)),
                      seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    x = record["x"].unsqueeze(0).to(device)
    f = record["f"].unsqueeze(0).to(device)
    g = record["grad"].unsqueeze(0).to(device)
    ii = torch.randperm(x.shape[1], generator=gen)[:n_input]
    lat = model.encode(x[:, ii], f[:, ii], g[:, ii])
    pred = model.decode(lat, x, "sdf").squeeze().cpu()
    err = (pred - record["f"]).abs()
    out = {}
    for lo, hi in bands:
        m = (record["f"].abs() >= lo) & (record["f"].abs() < hi)
        out[f"band_{lo}_{hi}"] = float(err[m].mean()) if m.any() else float("nan")
    return out
