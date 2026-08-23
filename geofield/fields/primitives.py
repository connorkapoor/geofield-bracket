"""Exact unit-gradient-field (UGF / signed-distance) primitives.

Every primitive is a `Field`: a function f: R^3 -> R with

  * f(x) < 0 inside the solid, f(x) > 0 outside,
  * |grad f(x)| = 1 everywhere away from the medial axis (exact Euclidean
    distance, not a bound),
  * an exact gradient, computed analytically where simple and via autograd
    of the exact distance formula where the piecewise algebra is error-prone
    (autograd of an exact formula is the exact gradient wherever f is
    differentiable; on the measure-zero non-differentiable set a valid
    subgradient is returned).

`transform(R, t)` returns the field of the *moved* solid: if g = f.transform(R, t)
then g(R @ x + t) == f(x). `scale(s)` returns the field of the solid scaled by
s about the origin, with the distance property preserved: g(x) = s * f(x / s).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

_EPS = 1e-12


@runtime_checkable
class Field(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...
    def grad(self, x: Tensor) -> Tensor: ...
    def transform(self, R: Tensor, t: Tensor) -> "Field": ...


class AnalyticField:
    """Base class: autograd-exact gradient, rigid transform, uniform scale."""

    def __call__(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def grad(self, x: Tensor) -> Tensor:
        """Exact gradient via autograd of the exact distance formula."""
        with torch.enable_grad():
            xg = x.detach().clone().requires_grad_(True)
            f = self(xg)
            (g,) = torch.autograd.grad(f.sum(), xg)
        return g

    def transform(self, R: Tensor, t: Tensor) -> "Field":
        return Transformed(self, R, t)

    def scale(self, s: float) -> "Field":
        return Scaled(self, s)


class Transformed(AnalyticField):
    """Rigidly moved field: evaluates the base field in body coordinates.

    g(y) = f(R^T (y - t));  grad g(y) = R grad f(R^T (y - t)).
    Composition of transforms collapses into a single (R, t).
    """

    def __init__(self, base: Field, R: Tensor, t: Tensor):
        R = torch.as_tensor(R, dtype=torch.float32)
        t = torch.as_tensor(t, dtype=torch.float32)
        if isinstance(base, Transformed):
            # (R2, t2) after (R1, t1): y = R2(R1 x + t1) + t2
            self.base = base.base
            self.R = R @ base.R
            self.t = R @ base.t + t
        else:
            self.base = base
            self.R = R
            self.t = t

    def __call__(self, x: Tensor) -> Tensor:
        local = (x - self.t.to(x)) @ self.R.to(x)  # == R^T (x - t) row-wise
        return self.base(local)

    def grad(self, x: Tensor) -> Tensor:
        local = (x - self.t.to(x)) @ self.R.to(x)
        return self.base.grad(local) @ self.R.to(x).T

    def transform(self, R: Tensor, t: Tensor) -> "Field":
        return Transformed(self, R, t)


class Scaled(AnalyticField):
    """Uniformly scaled solid; g(x) = s * f(x / s) keeps |grad g| = 1."""

    def __init__(self, base: Field, s: float):
        if isinstance(base, Scaled):
            self.base, self.s = base.base, float(s) * base.s
        else:
            self.base, self.s = base, float(s)

    def __call__(self, x: Tensor) -> Tensor:
        return self.s * self.base(x / self.s)

    def grad(self, x: Tensor) -> Tensor:
        return self.base.grad(x / self.s)


def _norm(v: Tensor, dim: int = -1) -> Tensor:
    return torch.linalg.vector_norm(v, dim=dim)


class Sphere(AnalyticField):
    """f(x) = |x| - r."""

    def __init__(self, radius: float):
        self.r = float(radius)

    def __call__(self, x: Tensor) -> Tensor:
        return _norm(x) - self.r

    def grad(self, x: Tensor) -> Tensor:
        return x / _norm(x).clamp_min(_EPS).unsqueeze(-1)


class Box(AnalyticField):
    """Exact Euclidean box distance.

    q = |x| - h (componentwise).
    outside: f = |max(q, 0)|            (distance to nearest face/edge/corner)
    inside:  f = max(q) <= 0            (negative distance to nearest face)
    Combined exact form: f = |max(q,0)| + min(max(q_x, q_y, q_z), 0).
    Gradient: outside, max(q,0)*sign(x)/|max(q,0)|; inside, sign(x_k) e_k for
    the axis k of the largest q (nearest face normal).
    """

    def __init__(self, half_extents):
        self.h = torch.as_tensor(half_extents, dtype=torch.float32)

    def __call__(self, x: Tensor) -> Tensor:
        q = x.abs() - self.h.to(x)
        outside = _norm(q.clamp_min(0.0))
        inside = q.amax(dim=-1).clamp_max(0.0)
        return outside + inside

    def grad(self, x: Tensor) -> Tensor:
        q = x.abs() - self.h.to(x)
        qpos = q.clamp_min(0.0)
        out_norm = _norm(qpos)
        is_out = (out_norm > 0).unsqueeze(-1)
        g_out = qpos / out_norm.clamp_min(_EPS).unsqueeze(-1)
        k = q.argmax(dim=-1, keepdim=True)
        g_in = torch.zeros_like(x).scatter_(-1, k, 1.0)
        g = torch.where(is_out, g_out, g_in)
        return g * torch.where(x >= 0, 1.0, -1.0)


class Capsule(AnalyticField):
    """Capsule from a to b with radius r: distance-to-segment minus r."""

    def __init__(self, a, b, radius: float):
        self.a = torch.as_tensor(a, dtype=torch.float32)
        self.b = torch.as_tensor(b, dtype=torch.float32)
        self.r = float(radius)

    def _closest(self, x: Tensor) -> Tensor:
        a, b = self.a.to(x), self.b.to(x)
        ab = b - a
        t = ((x - a) @ ab / (ab @ ab).clamp_min(_EPS)).clamp(0.0, 1.0)
        return a + t.unsqueeze(-1) * ab

    def __call__(self, x: Tensor) -> Tensor:
        return _norm(x - self._closest(x)) - self.r

    def grad(self, x: Tensor) -> Tensor:
        d = x - self._closest(x)
        return d / _norm(d).clamp_min(_EPS).unsqueeze(-1)


class Cylinder(AnalyticField):
    """Exact capped cylinder along +z, radius r, half-height h.

    2-D box distance in (radial, axial) coordinates:
    d = (|x_xy| - r, |x_z| - h); f = min(max(d), 0) + |max(d, 0)|.
    Gradient assembled from the radial and axial unit directions.
    """

    def __init__(self, radius: float, half_height: float):
        self.r = float(radius)
        self.h = float(half_height)

    def _d(self, x: Tensor):
        rad = _norm(x[..., :2])
        return rad - self.r, x[..., 2].abs() - self.h, rad

    def __call__(self, x: Tensor) -> Tensor:
        dr, dz, _ = self._d(x)
        d = torch.stack([dr, dz], dim=-1)
        return d.amax(dim=-1).clamp_max(0.0) + _norm(d.clamp_min(0.0))

    def grad(self, x: Tensor) -> Tensor:
        dr, dz, rad = self._d(x)
        e_r = torch.zeros_like(x)
        e_r[..., :2] = x[..., :2] / rad.clamp_min(_EPS).unsqueeze(-1)
        e_z = torch.zeros_like(x)
        e_z[..., 2] = torch.where(x[..., 2] >= 0, 1.0, -1.0)
        drp, dzp = dr.clamp_min(0.0), dz.clamp_min(0.0)
        out_norm = torch.sqrt(drp * drp + dzp * dzp)
        outside = out_norm > 0
        g_out = (drp.unsqueeze(-1) * e_r + dzp.unsqueeze(-1) * e_z) / out_norm.clamp_min(_EPS).unsqueeze(-1)
        g_in = torch.where((dr > dz).unsqueeze(-1), e_r, e_z)
        return torch.where(outside.unsqueeze(-1), g_out, g_in)


class Torus(AnalyticField):
    """Torus in the xy-plane: f = |(|x_xy| - R, x_z)| - r."""

    def __init__(self, major_radius: float, minor_radius: float):
        self.R = float(major_radius)
        self.r = float(minor_radius)

    def __call__(self, x: Tensor) -> Tensor:
        q1 = _norm(x[..., :2]) - self.R
        return torch.sqrt(q1 * q1 + x[..., 2] * x[..., 2]) - self.r


class Cone(AnalyticField):
    """Exact capped cone along z from radius r1 (at z=-h) to r2 (at z=+h).

    Inigo Quilez's exact capped-cone distance in (radial, axial) coordinates.
    Gradient via autograd of the exact formula (see module docstring).
    """

    def __init__(self, r1: float, r2: float, half_height: float):
        self.r1, self.r2, self.h = float(r1), float(r2), float(half_height)

    def __call__(self, x: Tensor) -> Tensor:
        r1, r2, h = self.r1, self.r2, self.h
        q = torch.stack([_norm(x[..., :2]), x[..., 2]], dim=-1)
        k1 = q.new_tensor([r2, h])
        k2 = q.new_tensor([r2 - r1, 2.0 * h])
        rq = torch.where(q[..., 1] < 0, q.new_tensor(r1), q.new_tensor(r2))
        ca = torch.stack([q[..., 0] - torch.minimum(q[..., 0], rq),
                          q[..., 1].abs() - h], dim=-1)
        t = (((k1 - q) * k2).sum(-1) / (k2 @ k2).clamp_min(_EPS)).clamp(0.0, 1.0)
        cb = q - k1 + k2 * t.unsqueeze(-1)
        inside = (cb[..., 0] < 0) & (ca[..., 1] < 0)
        s = torch.where(inside, -1.0, 1.0)
        return s * torch.sqrt(torch.minimum((ca * ca).sum(-1), (cb * cb).sum(-1)).clamp_min(_EPS))


class Halfspace(AnalyticField):
    """f(x) = <x, n> - d with unit normal n; solid where f < 0."""

    def __init__(self, normal, offset: float):
        n = torch.as_tensor(normal, dtype=torch.float32)
        self.n = n / _norm(n).clamp_min(_EPS)
        self.d = float(offset)

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.n.to(x) - self.d

    def grad(self, x: Tensor) -> Tensor:
        return self.n.to(x).expand_as(x).clone()
