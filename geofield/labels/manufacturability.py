"""Analytic manufacturability labels computed directly from the field.

All labelers take (field, x[N,3], direction) and return per-point tensors.
Query points are first projected to the nearest surface point (exact for a
distance field: x_s = x - f * grad), so every query carries the label of the
surface it is closest to — this makes the labels well-defined as fields over
all of R^3, which is what the decoder is trained on.

Conventions:
  * `build_dir` b: the printing direction ("up"). Outward normal n = grad f.
  * overhang(x) = angle(n, b) in radians (0 = facing straight up,
    pi = facing straight down).
  * support_need: downward-facing by more than 45 deg past horizontal
    (dot(n, b) < -cos(45 deg)) AND not within `plate_tol` of the build plate
    (the lowest surface point along b).
  * tool_access along spindle s: ray-march from just outside the surface
    along -s; accessible iff the ray leaves the unit sphere without
    re-entering the solid. Soft value = min f along the ray (clearance).
  * thickness: sphere-growing along -n; fixed-point iteration
    r <- -f(x_s - n r) converges to the maximal inscribed sphere radius
    touching x_s; thickness = 2 r.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from ..fields.primitives import Field

SUPPORT_ANGLE_DEG = 45.0


def project_to_surface(field: Field, x: Tensor, iters: int = 3) -> tuple[Tensor, Tensor]:
    """Nearest surface point and outward unit normal for each query."""
    xs = x
    for _ in range(iters):
        with torch.no_grad():
            f = field(xs)
            g = field.grad(xs)
        xs = xs - f.unsqueeze(-1) * g
    with torch.no_grad():
        n = field.grad(xs)
    n = n / torch.linalg.vector_norm(n, dim=-1, keepdim=True).clamp_min(1e-12)
    return xs, n


def overhang(field: Field, x: Tensor, build_dir: Tensor, surf=None) -> Tensor:
    """Angle (radians) between the surface normal and the build direction.

    `surf`: optional precomputed (xs, n) from project_to_surface, shared
    across labelers to avoid re-projecting the same queries (all labelers
    accept it)."""
    b = torch.as_tensor(build_dir, dtype=x.dtype, device=x.device)
    b = b / torch.linalg.vector_norm(b).clamp_min(1e-12)
    _, n = surf if surf is not None else project_to_surface(field, x)
    return torch.acos((n @ b).clamp(-1.0, 1.0))


def plate_height(field: Field, build_dir: Tensor, n_probe: int = 2048,
                 seed: int = 0) -> Tensor:
    """Height (along build_dir) of the build plate: the lowest surface point
    of the WHOLE shape, found from a fixed-seed global probe set."""
    b = torch.as_tensor(build_dir, dtype=torch.float32)
    b = b / torch.linalg.vector_norm(b).clamp_min(1e-12)
    gen = torch.Generator().manual_seed(seed)
    probes = (torch.randn(n_probe, 3, generator=gen) * 0.6).to(b.device)
    xs, _ = project_to_surface(field, probes)
    with torch.no_grad():
        f = field(xs)
    surf = xs[f.abs() < 5e-3]
    if surf.shape[0] < 8:
        surf = xs
    return (surf @ b).amin()


def support_need(field: Field, x: Tensor, build_dir: Tensor,
                 plate_tol: float = 0.03, surf=None,
                 plate: Tensor | float | None = None) -> Tensor:
    """Binary (0/1 float): surface needs support material when printed along b.

    `plate`: optional precomputed build-plate height along b. When `surf`
    covers the whole shape (e.g. a record's global query set), the plate can
    be derived from it: (xs @ b).amin()."""
    b = torch.as_tensor(build_dir, dtype=x.dtype, device=x.device)
    b = b / torch.linalg.vector_norm(b).clamp_min(1e-12)
    xs, n = surf if surf is not None else project_to_surface(field, x)
    cos_nb = n @ b
    steep_down = cos_nb < -math.cos(math.radians(SUPPORT_ANGLE_DEG))
    height = xs @ b
    if plate is None:
        plate = plate_height(field, b).to(height)
    on_plate = height <= plate + plate_tol
    return (steep_down & ~on_plate).float()


def tool_access(field: Field, x: Tensor, spindle_dir: Tensor,
                n_steps: int = 64, sphere_radius: float = 1.05,
                surface_eps: float = 5e-3, surf=None) -> tuple[Tensor, Tensor]:
    """(soft, binary) 3-axis tool access.

    Convention: `spindle_dir` points from the part TOWARD the machine head
    (a vertical mill machining the top of a part has spindle_dir = +z).
    A surface point is accessible iff the escape ray from just outside the
    surface along +spindle_dir leaves the bounding sphere without re-entering
    the solid.

    soft   = min f along the escape ray (clearance; > 0 means clear),
    binary = 1.0 if the ray never re-enters the solid, else 0.0.
    """
    s = torch.as_tensor(spindle_dir, dtype=x.dtype, device=x.device)
    s = s / torch.linalg.vector_norm(s).clamp_min(1e-12)
    d = s
    xs, n = surf if surf is not None else project_to_surface(field, x)
    x0 = xs + n * surface_eps  # start just outside the solid

    # exit distance of the ray x0 + t d from the bounding sphere
    b_half = (x0 * d).sum(-1)
    c = (x0 * x0).sum(-1) - sphere_radius**2
    t_exit = -b_half + torch.sqrt((b_half * b_half - c).clamp_min(0.0))

    ts = torch.linspace(0.0, 1.0, n_steps, device=x.device, dtype=x.dtype)
    # skip the immediate neighbourhood of the start point (its own surface)
    t0 = 4 * surface_eps
    pts = x0.unsqueeze(1) + d.reshape(1, 1, 3) * (
        t0 + ts.reshape(1, -1, 1) * (t_exit.reshape(-1, 1, 1) - t0)).clamp_min(0.0)
    with torch.no_grad():
        fv = field(pts.reshape(-1, 3)).reshape(x.shape[0], n_steps)
    soft = fv.amin(dim=-1)
    binary = (soft > -1e-3).float()
    # a surface facing away from the tool (normal opposed to escape dir) is
    # unreachable regardless of clearance along the ray
    facing = (n @ d) > -0.999  # only fully opposed normals excluded
    binary = binary * facing.float()
    return soft, binary


def thickness(field: Field, x: Tensor, r_max: float = 0.6,
              n_march: int = 32, n_bisect: int = 8, tol: float = 2e-3,
              surf=None) -> Tensor:
    """Local wall thickness: 2x the maximal inscribed sphere radius touching
    the nearest surface point.

    A sphere of radius r centered at c(r) = x_s - n r is inscribed iff
    -f(c(r)) >= r (its center is at least r deep). This predicate is true for
    r up to the medial-axis radius and false beyond, so the largest r is
    found by a coarse march followed by bisection.
    """
    xs, n = surf if surf is not None else project_to_surface(field, x)
    N = xs.shape[0]
    rs = torch.linspace(0.0, r_max, n_march, device=x.device, dtype=x.dtype)
    centers = xs.unsqueeze(1) - n.unsqueeze(1) * rs.reshape(1, -1, 1)
    with torch.no_grad():
        fc = field(centers.reshape(-1, 3)).reshape(N, n_march)
    ok = (-fc) >= rs.reshape(1, -1) - tol
    # largest index where the predicate still holds, given it held at 0
    ok_cum = ok.cummin(dim=-1).values  # once false, stays false
    last_ok = ok_cum.float().sum(dim=-1).long().clamp(1, n_march) - 1
    lo = rs[last_ok]
    hi = (lo + rs[1]).clamp_max(r_max)
    for _ in range(n_bisect):
        mid = (lo + hi) / 2
        c = xs - n * mid.unsqueeze(-1)
        with torch.no_grad():
            fc = field(c)
        good = (-fc) >= mid - tol
        lo = torch.where(good, mid, lo)
        hi = torch.where(good, hi, mid)
    return 2.0 * lo


def ray_thickness(field: Field, x: Tensor, t_max: float = 1.2,
                  n_steps: int = 96, surf=None) -> Tensor:
    """Chord length of the solid along the inward normal from the nearest
    surface point: the distance until f becomes positive again.

    Unlike sphere-growing, this does not collapse to zero at convex or
    reentrant edges, which makes it the right estimator for the
    `min_thickness` SCALAR (walls measure their true thickness); the
    sphere-growing `thickness` remains the per-point field label.
    """
    xs, n = surf if surf is not None else project_to_surface(field, x)
    eps = 5e-3
    ts = torch.linspace(eps, t_max, n_steps, device=x.device, dtype=x.dtype)
    pts = xs.unsqueeze(1) - n.unsqueeze(1) * ts.reshape(1, -1, 1)
    with torch.no_grad():
        fv = field(pts.reshape(-1, 3)).reshape(x.shape[0], n_steps)
    exited = fv > 0
    first_exit = torch.where(exited.any(dim=-1),
                             exited.float().argmax(dim=-1),
                             torch.full(xs.shape[:1], n_steps - 1,
                                        device=x.device, dtype=torch.long))
    return ts[first_exit]
