"""Surrogate accuracy: predicted vs FEA von Mises on held-out brackets.

Reports relative error on peak VM, field R^2, and the architecture's
go/no-go: error binned by distance from the load token (a flat curve means
the coarse/global latent route carries load information everywhere; a rising
curve means physics stays local and M_c must grow).
"""
from __future__ import annotations

import torch

from ..model.heads import field_specs


@torch.no_grad()
def surrogate_metrics(model, record: dict, device: str = "cpu",
                      n_input: int = 2048, n_bins: int = 6,
                      seed: int = 0) -> dict | None:
    """One labeled record -> {peak_rel_err, r2, err_by_dist: [(dist, mae)]}."""
    key = next((k for k in record["fields"] if k.startswith("von_mises|")), None)
    if key is None:
        return None
    cond_ids = record["field_conditions"].get(key, [])
    cond = [[record["tokens"][i] for i in cond_ids]]
    load = next((t for t in cond[0] if t.type == "load"), None)

    x = record["x"].unsqueeze(0).to(device)
    f = record["f"].unsqueeze(0).to(device)
    g = record["grad"].unsqueeze(0).to(device)
    gen = torch.Generator().manual_seed(seed)
    ii = torch.randperm(x.shape[1], generator=gen)[:n_input]
    lat = model.encode(x[:, ii], f[:, ii], g[:, ii], [record["tokens"]])
    pred = model.decode(lat, x, "von_mises", cond).squeeze(0).squeeze(-1).cpu()
    true = record["fields"][key].float()

    solid = record["f"] < 0.02  # in/near solid, where FEA values are meaningful
    p, t = pred[solid], true[solid]
    peak_rel = float(abs(p.max() - t.max()) / t.max().clamp_min(1e-9))
    ss_res = (p - t).pow(2).sum()
    ss_tot = (t - t.mean()).pow(2).sum().clamp_min(1e-12)
    r2 = float(1 - ss_res / ss_tot)

    err_by_dist = []
    if load is not None and load.position is not None:
        d = torch.linalg.vector_norm(
            record["x"][solid] - load.position, dim=-1)
        edges = torch.quantile(d, torch.linspace(0, 1, n_bins + 1))
        for b in range(n_bins):
            m = (d >= edges[b]) & (d < edges[b + 1] + (1e-6 if b == n_bins - 1 else 0))
            if m.any():
                mae = float((p[m] - t[m]).abs().mean())
                rel = mae / float(t.abs().mean().clamp_min(1e-9))
                err_by_dist.append({"dist": float(d[m].mean()), "rel_mae": rel})
    return {"peak_rel_err": peak_rel, "r2": r2, "err_by_dist": err_by_dist}


def aggregate_surrogate(results: list[dict]) -> dict:
    results = [r for r in results if r]
    if not results:
        return {}
    peaks = torch.tensor([r["peak_rel_err"] for r in results])
    r2s = torch.tensor([r["r2"] for r in results])
    # merge distance bins across records by rank
    n_bins = max(len(r["err_by_dist"]) for r in results)
    curve = []
    for b in range(n_bins):
        pts = [r["err_by_dist"][b] for r in results if len(r["err_by_dist"]) > b]
        if pts:
            curve.append({
                "dist": sum(p["dist"] for p in pts) / len(pts),
                "rel_mae": sum(p["rel_mae"] for p in pts) / len(pts)})
    return {"peak_rel_err_median": float(peaks.median()),
            "peak_rel_err_mean": float(peaks.mean()),
            "r2_median": float(r2s.median()),
            "err_vs_load_distance": curve}
