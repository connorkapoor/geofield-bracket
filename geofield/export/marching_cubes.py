"""Field -> mesh via marching cubes, plus STL export and surface sampling.

Used at EXPORT and EVAL time only (ground rule: no meshes inside the model).
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from skimage import measure


@torch.no_grad()
def field_to_grid(field_fn, res: int = 128, domain: float = 1.05,
                  device: str = "cpu", chunk: int = 262144) -> torch.Tensor:
    """Evaluate a field callable on a res^3 grid. field_fn: [N,3] -> [N]."""
    lin = torch.linspace(-domain, domain, res, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    out = []
    for i in range(0, pts.shape[0], chunk):
        out.append(field_fn(pts[i:i + chunk]))
    return torch.cat(out).reshape(res, res, res)


def grid_to_mesh(grid: torch.Tensor, domain: float = 1.05
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Zero-isosurface mesh. Returns (verts [V,3] world coords, faces [F,3])."""
    g = grid.detach().cpu().numpy()
    res = g.shape[0]
    if g.min() > 0 or g.max() < 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    verts, faces, _, _ = measure.marching_cubes(g, level=0.0)
    verts = verts / (res - 1) * 2 * domain - domain
    return verts.astype(np.float32), faces.astype(np.int64)


def _grad_of(field, pts: torch.Tensor, h: float = 1e-3) -> torch.Tensor:
    """Field gradient: exact if the object exposes .grad(), else central diff."""
    if hasattr(field, "grad"):
        try:
            return field.grad(pts)
        except Exception:  # noqa: BLE001 - fall through to numeric
            pass
    g = torch.zeros_like(pts)
    for a in range(3):
        d = torch.zeros(3, dtype=pts.dtype, device=pts.device)
        d[a] = h
        g[:, a] = (field(pts + d) - field(pts - d)) / (2 * h)
    return g


@torch.no_grad()
def refine_vertices(verts: np.ndarray, field, iters: int = 3,
                    chunk: int = 65536) -> np.ndarray:
    """Project marching-cubes vertices onto the TRUE zero level set.

    For a unit-gradient field the Newton step x <- x - f(x) * grad f(x) lands
    exactly on the surface in one iteration on planar regions (|grad f| = 1,
    so f IS the signed distance to travel); a few iterations clean up
    curvature and blend regions. This removes the staircase/faceting that
    grid resolution alone would leave, without raising the grid size.
    """
    if len(verts) == 0:
        return verts
    v = torch.from_numpy(np.ascontiguousarray(verts)).float()
    for _ in range(iters):
        out = []
        for i in range(0, v.shape[0], chunk):
            p = v[i:i + chunk]
            f = field(p)
            g = _grad_of(field, p)
            gn = torch.linalg.vector_norm(g, dim=-1, keepdim=True).clamp_min(1e-6)
            out.append(p - f.unsqueeze(-1) * g / gn)
        v = torch.cat(out)
    return v.numpy().astype(np.float32)


@torch.no_grad()
def analytic_normals(verts: np.ndarray, field, chunk: int = 65536) -> np.ndarray:
    """Per-vertex normals from the field gradient (exact for UGF primitives),
    which shades curved faces smoothly while keeping CSG edges crisp —
    mesh-topology normals cannot do both."""
    if len(verts) == 0:
        return verts
    v = torch.from_numpy(np.ascontiguousarray(verts)).float()
    out = []
    for i in range(0, v.shape[0], chunk):
        g = _grad_of(field, v[i:i + chunk])
        out.append(g / torch.linalg.vector_norm(g, dim=-1, keepdim=True).clamp_min(1e-9))
    return torch.cat(out).numpy().astype(np.float32)


def mesh_to_stl(verts: np.ndarray, faces: np.ndarray, path: str | Path) -> None:
    """Binary STL writer (no external deps)."""
    tri = verts[faces]                                   # [F,3,3]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            fh.write(struct.pack("<3f", *n[i]))
            for v in tri[i]:
                fh.write(struct.pack("<3f", *v))
            fh.write(struct.pack("<H", 0))


def sample_mesh_surface(verts: np.ndarray, faces: np.ndarray, n: int,
                        seed: int = 0) -> np.ndarray:
    """Uniform-by-area surface point samples for Chamfer distance."""
    if len(faces) == 0:
        return np.zeros((0, 3), np.float32)
    rng = np.random.default_rng(seed)
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)
    p = area / area.sum().clip(1e-12)
    fi = rng.choice(len(faces), size=n, p=p)
    r1, r2 = rng.random(n), rng.random(n)
    s1 = np.sqrt(r1)
    a, b, c = tri[fi, 0], tri[fi, 1], tri[fi, 2]
    return ((1 - s1)[:, None] * a + (s1 * (1 - r2))[:, None] * b
            + (s1 * r2)[:, None] * c).astype(np.float32)
