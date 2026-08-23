"""CSG operators on fields.

Formulas (a, b are child field values at x):

  union(a, b)      = min(a, b)          exact distance outside, bound inside
  intersect(a, b)  = max(a, b)
  subtract(a, b)   = max(a, -b)
  smooth_min(a,b,r): quadratic polynomial smooth-min (Quilez),
                     h = max(r - |a-b|, 0)/r,  f = min(a,b) - h^2 * r / 4.
                     NOTE: this is the standard fallback blend. It is C^1 and
                     bounded by min(a,b), but it is NOT unit-gradient inside
                     the blend region (|grad f| <= 1 there). A UGF-preserving
                     blend from the domain expert should replace it; every
                     instance is flagged via `GradPreservingBlend.APPROXIMATE`.
  offset(a, d)     = a - d              (dilate by d > 0, erode by d < 0)
  shell(a, t)      = |a| - t/2          (hollow shell of total thickness t)

Gradients of min/max are the gradient of the selected child (exact away from
the tie set, valid subgradient on it). smooth_min's gradient is the exact
analytic gradient of the polynomial blend.
"""
from __future__ import annotations

import warnings

import torch
from torch import Tensor

from .primitives import AnalyticField, Field


class Union(AnalyticField):
    """union = min over children; gradient of the winning child."""

    def __init__(self, *children: Field):
        assert len(children) >= 2
        self.children = list(children)

    def __call__(self, x: Tensor) -> Tensor:
        return torch.stack([c(x) for c in self.children], dim=0).amin(dim=0)

    def grad(self, x: Tensor) -> Tensor:
        vals = torch.stack([c(x) for c in self.children], dim=0)
        grads = torch.stack([c.grad(x) for c in self.children], dim=0)
        idx = vals.argmin(dim=0)
        return grads.gather(0, idx[None, ..., None].expand(1, *x.shape)).squeeze(0)

    def transform(self, R, t):
        return Union(*[c.transform(R, t) for c in self.children])


class Intersect(AnalyticField):
    """intersect = max over children; gradient of the winning child."""

    def __init__(self, *children: Field):
        assert len(children) >= 2
        self.children = list(children)

    def __call__(self, x: Tensor) -> Tensor:
        return torch.stack([c(x) for c in self.children], dim=0).amax(dim=0)

    def grad(self, x: Tensor) -> Tensor:
        vals = torch.stack([c(x) for c in self.children], dim=0)
        grads = torch.stack([c.grad(x) for c in self.children], dim=0)
        idx = vals.argmax(dim=0)
        return grads.gather(0, idx[None, ..., None].expand(1, *x.shape)).squeeze(0)

    def transform(self, R, t):
        return Intersect(*[c.transform(R, t) for c in self.children])


class Negate(AnalyticField):
    """Complement of a solid: f -> -f."""

    def __init__(self, child: Field):
        self.child = child

    def __call__(self, x: Tensor) -> Tensor:
        return -self.child(x)

    def grad(self, x: Tensor) -> Tensor:
        return -self.child.grad(x)

    def transform(self, R, t):
        return Negate(self.child.transform(R, t))


def Subtract(a: Field, b: Field) -> Field:
    """subtract(a, b) = max(a, -b): remove b's solid from a's."""
    return Intersect(a, Negate(b))


class SmoothMin(AnalyticField):
    """Quadratic polynomial smooth union (see module docstring).

    f = min(a, b) - h^2 * r / 4 with h = max(r - |a - b|, 0) / r.
    grad = (1 - m) grad_a + m grad_b with m = the standard mix weight
    clamp(0.5 + 0.5 (b - a)/r, 0, 1); this is the exact gradient of the
    equivalent mix form of the polynomial smooth-min.
    """

    def __init__(self, a: Field, b: Field, r: float):
        self.a, self.b, self.r = a, b, float(r)

    def _mix(self, x: Tensor):
        fa, fb = self.a(x), self.b(x)
        m = (0.5 + 0.5 * (fb - fa) / self.r).clamp(0.0, 1.0)
        return fa, fb, m

    def __call__(self, x: Tensor) -> Tensor:
        fa, fb, m = self._mix(x)
        return fb + m * (fa - fb) - self.r * m * (1.0 - m)

    def grad(self, x: Tensor) -> Tensor:
        fa, fb, m = self._mix(x)
        ga, gb = self.a.grad(x), self.b.grad(x)
        # d f / d fa = m, d f / d fb = 1 - m (the r m (1-m) term's dependence
        # on fa, fb cancels against the mix term's — standard smooth-min result).
        return m.unsqueeze(-1) * ga + (1.0 - m).unsqueeze(-1) * gb

    def transform(self, R, t):
        return SmoothMin(self.a.transform(R, t), self.b.transform(R, t), self.r)


class GradPreservingBlend(SmoothMin):
    """Placeholder for the domain expert's UGF-preserving blend.

    Currently the quadratic smooth-min (APPROXIMATE = True). When the exact
    formula arrives, implement it here; all call sites already route through
    this class, so nothing else changes.
    """

    APPROXIMATE = True
    _warned = False

    def __init__(self, a: Field, b: Field, r: float):
        if not GradPreservingBlend._warned:
            warnings.warn(
                "GradPreservingBlend: using quadratic smooth-min fallback; "
                "|grad| < 1 inside blend regions. Replace when the exact "
                "UGF-preserving formula is available.",
                stacklevel=2,
            )
            GradPreservingBlend._warned = True
        super().__init__(a, b, r)

    def transform(self, R, t):
        return GradPreservingBlend(self.a.transform(R, t), self.b.transform(R, t), self.r)


class Offset(AnalyticField):
    """offset(a, d) = a - d: dilate by d > 0, erode by d < 0."""

    def __init__(self, child: Field, d: float):
        self.child, self.d = child, float(d)

    def __call__(self, x: Tensor) -> Tensor:
        return self.child(x) - self.d

    def grad(self, x: Tensor) -> Tensor:
        return self.child.grad(x)

    def transform(self, R, t):
        return Offset(self.child.transform(R, t), self.d)


class Shell(AnalyticField):
    """shell(a, t) = |a| - t/2: hollow shell of total thickness t around the surface."""

    def __init__(self, child: Field, t: float):
        self.child, self.t = child, float(t)

    def __call__(self, x: Tensor) -> Tensor:
        return self.child(x).abs() - self.t / 2.0

    def grad(self, x: Tensor) -> Tensor:
        f = self.child(x)
        return torch.where(f >= 0, 1.0, -1.0).unsqueeze(-1) * self.child.grad(x)

    def transform(self, R, t):
        return Shell(self.child.transform(R, t), self.t)
