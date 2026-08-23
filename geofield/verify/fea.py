"""Verification: decoded latent -> FEA solve -> compare to predictions.

The generated bracket is never trusted: its decoded sdf becomes the immersed
domain for the SAME voxel solver used to label training data, under the
sample's condition tokens.
"""
from __future__ import annotations

import torch

from ..labels.physics import VoxelFEA
from ..tokens.schema import Token


@torch.no_grad()
def decoded_field_fn(model, lat, device: str):
    """Latent -> field callable [N,3] -> [N] for solver/labelers/export."""
    def fn(pts: torch.Tensor) -> torch.Tensor:
        return model.decode(lat, pts.unsqueeze(0).to(device), "sdf") \
            .squeeze(0).squeeze(-1)
    return fn


class _CallableField:
    """Adapts a plain callable to the Field protocol. Gradients: model
    autograd when a latent is attached, else numeric central differences
    (composed/hybrid fields run under no_grad internally)."""

    def __init__(self, fn, model=None, lat=None, device="cpu",
                 numeric_grad: bool = False, h: float = 2e-3):
        self.fn = fn
        self.model, self.lat, self.device = model, lat, device
        self.numeric_grad, self.h = numeric_grad, h

    def __call__(self, x):
        return self.fn(x)

    def grad(self, x):
        if self.numeric_grad:
            g = torch.zeros_like(x)
            for a in range(3):
                d = torch.zeros(3, device=x.device, dtype=x.dtype)
                d[a] = self.h
                g[:, a] = (self.fn(x + d) - self.fn(x - d)) / (2 * self.h)
            return g
        if self.model is not None:
            _, g = self.model.sdf_and_grad(self.lat, x.unsqueeze(0).to(self.device))
            return g.squeeze(0)
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(self.fn(xg).sum(), xg)
        return g

    def transform(self, R, t):  # pragma: no cover - verification never moves fields
        raise NotImplementedError


@torch.no_grad()
def verify_physics(model, lat, tokens: list[Token], sigma_max: float,
                   device: str = "cpu", fea_res: int = 96,
                   n_query: int = 4096, seed: int = 0,
                   field_fn=None) -> dict:
    """Solve the decoded field under `tokens`; compare peak VM to prediction
    and to the constraint sigma_max. field_fn overrides the geometry (hybrid
    CSG+model compositions) — gradients go numeric in that case."""
    if field_fn is not None:
        field = _CallableField(field_fn, numeric_grad=True)
    else:
        field = _CallableField(decoded_field_fn(model, lat, device),
                               model, lat, device)
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(n_query, 3, generator=gen) * 0.45
    fea = VoxelFEA(res=fea_res, device=device)
    out = fea.solve(field, tokens, q)

    solid = out.mask > 0.5
    pred_peak = None
    if model is not None and lat is not None:
        cond = [[t for t in tokens
                 if t.type in ("fixed_point", "load", "material")]]
        pred_vm = model.decode(lat, q.unsqueeze(0).to(device), "von_mises",
                               cond).squeeze().cpu()
        pred_peak = float(pred_vm[solid].max()) if solid.any() \
            else float(pred_vm.max())

    return {
        "verified_peak_vm": out.peak_vm,
        "predicted_peak_vm": pred_peak,
        "peak_rel_err": (abs(out.peak_vm - pred_peak) / max(out.peak_vm, 1e-9)
                         if pred_peak is not None else None),
        "sigma_max": sigma_max,
        "pass": out.peak_vm <= sigma_max,
        "max_disp": out.max_disp,
        "solver_residual": out.residual,
    }
