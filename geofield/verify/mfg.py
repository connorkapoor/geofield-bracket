"""Verification: analytic manufacturability checks on the decoded field."""
from __future__ import annotations

import torch

from ..labels import manufacturability as mfg_labels
from ..labels import scalars as scal
from ..tokens.schema import Token
from .fea import _CallableField, decoded_field_fn


@torch.no_grad()
def verify_mfg(model, lat, tokens: list[Token],
               s_max: float | None = None, a_max: float | None = None,
               t_min: float | None = None, device: str = "cpu",
               n_query: int = 8192, seed: int = 0, field_fn=None) -> dict:
    """Analytic support / access / thickness checks under the given tokens.
    Thresholds are per-process: s_max (additive support-area fraction),
    a_max (machining inaccessible fraction), t_min (min wall thickness).
    field_fn overrides the geometry (hybrid compositions; numeric grads)."""
    if field_fn is not None:
        field = _CallableField(field_fn, numeric_grad=True)
        device = "cpu"
    else:
        field = _CallableField(decoded_field_fn(model, lat, device),
                               model, lat, device)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n_query, 3, generator=gen).to(device) * 0.45
    f = field(x)
    surf = mfg_labels.project_to_surface(field, x)
    out: dict = {"pass": True}

    build = next((t for t in tokens if t.type == "build_dir"), None)
    if build is not None:
        d = build.params["direction"].to(device)
        plate = (surf[0] @ d).amin()
        sn = mfg_labels.support_need(field, x, d, surf=surf, plate=plate)
        frac = scal.support_fraction(sn, f.cpu())
        out["support_fraction"] = frac
        if s_max is not None:
            out["support_pass"] = frac <= s_max
            out["pass"] &= out["support_pass"]

    spindle = next((t for t in tokens if t.type == "spindle_dir"), None)
    if spindle is not None:
        d = spindle.params["direction"].to(device)
        _, acc = mfg_labels.tool_access(field, x, d, surf=surf)
        frac = scal.inaccessible_fraction(acc.cpu(), f.cpu())
        out["inaccessible_fraction"] = frac
        if a_max is not None:
            out["access_pass"] = frac <= a_max
            out["pass"] &= out["access_pass"]

    rt = mfg_labels.ray_thickness(field, x, surf=surf)
    mt = scal.min_thickness(rt.cpu(), f.cpu())
    out["min_thickness"] = mt
    if t_min is not None:
        out["thickness_pass"] = mt >= t_min
        out["pass"] &= out["thickness_pass"]

    # interface validity: material ring around every fixed_point hole,
    # void at the hole center (same acceptance as the bracket program)
    holes_ok = True
    for tok in tokens:
        if tok.type != "fixed_point" or tok.position is None:
            continue
        c = tok.position.to(device)
        if float(field(c.unsqueeze(0))) <= 0:
            holes_ok = False
            continue
        axis = tok.params["axis"].to(device)
        a = torch.tensor([1.0, 0, 0], device=device)
        if abs(float(axis @ a)) > 0.9:
            a = torch.tensor([0.0, 1, 0], device=device)
        u = torch.linalg.cross(axis, a)
        u = u / torch.linalg.vector_norm(u)
        v = torch.linalg.cross(axis, u)
        ang = torch.linspace(0, 6.2832, 8, device=device)
        ring = c + 0.75 * float(tok.params["diameter"]) * (
            torch.cos(ang).unsqueeze(-1) * u + torch.sin(ang).unsqueeze(-1) * v)
        if float((field(ring) < 0).float().mean()) < 0.6:
            holes_ok = False
    out["holes_pass"] = holes_ok
    out["pass"] &= holes_ok
    return out
