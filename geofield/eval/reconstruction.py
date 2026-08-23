"""Reconstruction metrics: Chamfer distance and occupancy IoU between a
decoded model field and the ground-truth analytic field, plus the masked
completion variant. Used both during training (every 5k steps on a few val
shapes) and in the Phase-3 sweep across splits.
"""
from __future__ import annotations

import torch

from ..export.marching_cubes import (field_to_grid, grid_to_mesh,
                                     sample_mesh_surface)


def chamfer(a: torch.Tensor, b: torch.Tensor) -> float:
    """Symmetric Chamfer-L2 (mean of squared nearest distances) between two
    point sets [Na,3], [Nb,3]."""
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    d = torch.cdist(a, b)
    return float(d.min(dim=1).values.pow(2).mean()
                 + d.min(dim=0).values.pow(2).mean())


@torch.no_grad()
def occupancy_iou(f_pred: torch.Tensor, f_true: torch.Tensor) -> float:
    """IoU of {f<0} from two field evaluations on the same points/grid."""
    p, t = f_pred < 0, f_true < 0
    union = (p | t).sum().item()
    return (p & t).sum().item() / union if union else float("nan")


@torch.no_grad()
def reconstruction_metrics(model, record: dict, device: str = "cpu",
                           res: int = 128, n_surf: int = 8192,
                           n_input: int = 2048, true_field=None,
                           seed: int = 0) -> dict:
    """Encode a dataset record, decode its sdf, compare to ground truth.

    Ground-truth surface points come from `true_field` (rebuilt analytic
    field) when given, else from the record's near-surface samples.
    Returns {chamfer, iou, sdf_mae_band0}.
    """
    x = record["x"].unsqueeze(0).to(device)
    f = record["f"].unsqueeze(0).to(device)
    g = record["grad"].unsqueeze(0).to(device)
    gen = torch.Generator().manual_seed(seed)
    ii = torch.randperm(x.shape[1], generator=gen)[:n_input]
    lat = model.encode(x[:, ii], f[:, ii], g[:, ii],
                       [record["tokens"]] if record["tokens"] else None)

    def pred_fn(pts):
        return model.decode(lat, pts.unsqueeze(0).to(device), "sdf").squeeze(0).squeeze(-1)

    grid = field_to_grid(pred_fn, res=res, device=device)
    verts, faces = grid_to_mesh(grid)
    pred_pts = torch.from_numpy(sample_mesh_surface(verts, faces, n_surf, seed))

    if true_field is not None:
        tg = field_to_grid(lambda p: true_field(p), res=res, device=device)
        tv, tf_ = grid_to_mesh(tg)
        true_pts = torch.from_numpy(sample_mesh_surface(tv, tf_, n_surf, seed))
        iou = occupancy_iou(grid, tg)
    else:
        band = record["f"].abs() < 0.02
        true_pts = record["x"][band][:n_surf]
        with torch.no_grad():
            fp = pred_fn(record["x"].to(device)).cpu()
        iou = occupancy_iou(fp, record["f"])

    band0 = record["f"].abs() < 0.02
    with torch.no_grad():
        f_pred_band = pred_fn(record["x"][band0].to(device)).cpu()
    mae0 = float((f_pred_band - record["f"][band0]).abs().mean())

    return {"chamfer": chamfer(pred_pts, true_pts), "iou": iou,
            "sdf_mae_band0": mae0}


@torch.no_grad()
def masked_completion_metrics(model, record: dict, device: str = "cpu",
                              radius: float = 0.15, seed: int = 0) -> dict:
    """Drop inputs inside a surface-centred ball; measure sdf error inside it."""
    x = record["x"].to(device)
    f = record["f"].to(device)
    g = record["grad"].to(device)
    gen = torch.Generator().manual_seed(seed)
    surf_ids = (f.abs() < 0.02).nonzero().squeeze(-1)
    ctr = x[surf_ids[int(torch.randint(len(surf_ids), (), generator=gen))]]
    keep = torch.linalg.vector_norm(x - ctr, dim=-1) > radius
    ii = torch.randperm(int(keep.sum()), generator=gen)[:2048]
    xk, fk, gk = x[keep][ii], f[keep][ii], g[keep][ii]
    lat = model.encode(xk.unsqueeze(0), fk.unsqueeze(0), gk.unsqueeze(0))
    inside = ~keep
    if not inside.any():
        return {"masked_sdf_mae": float("nan")}
    pred = model.decode(lat, x[inside].unsqueeze(0), "sdf").squeeze()
    return {"masked_sdf_mae": float((pred - f[inside]).abs().mean())}
