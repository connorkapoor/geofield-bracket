"""Tests for field primitives, ops, and sampling: unit-gradient property,
sign correctness, transform equivariance, exact gradients."""
import math

import pytest
import torch

from geofield.fields.primitives import (
    Box, Capsule, Cone, Cylinder, Halfspace, Sphere, Torus)
from geofield.fields.ops import (
    GradPreservingBlend, Intersect, Offset, Shell, SmoothMin, Subtract, Union)
from geofield.fields.sampling import stratified, DEFAULT_BANDS
from geofield.fields.programs.common import random_rotation

torch.manual_seed(0)

PRIMITIVES = [
    Sphere(0.4),
    Box([0.3, 0.2, 0.4]),
    Capsule([-0.2, 0, 0], [0.3, 0.1, 0.2], 0.15),
    Cylinder(0.3, 0.25),
    Torus(0.35, 0.12),
    Cone(0.3, 0.1, 0.25),
    Halfspace([0, 0, 1], 0.1),
]


def _points(n=4096):
    g = torch.Generator().manual_seed(1)
    return torch.randn(n, 3, generator=g) * 0.5


@pytest.mark.parametrize("prim", PRIMITIVES, ids=lambda p: type(p).__name__)
def test_unit_gradient(prim):
    x = _points()
    f = prim(x)
    grad = prim.grad(x)
    gn = torch.linalg.vector_norm(grad, dim=-1)
    # away from the surface tie-set and medial axis: |grad| == 1.
    # exclude points very close to the medial axis by requiring the gradient
    # norm from autograd to be well-defined; test the 90th percentile bound
    ok = (gn - 1).abs() < 1e-4
    assert ok.float().mean() > 0.97, f"unit-gradient violated: {ok.float().mean()}"


@pytest.mark.parametrize("prim", PRIMITIVES, ids=lambda p: type(p).__name__)
def test_autograd_matches_analytic(prim):
    """The class grad() must equal autograd of __call__ wherever smooth."""
    x = _points(2048)
    g_cls = prim.grad(x)
    with torch.enable_grad():
        xg = x.clone().requires_grad_(True)
        (ag,) = torch.autograd.grad(prim(xg).sum(), xg)
    diff = torch.linalg.vector_norm(g_cls - ag, dim=-1)
    assert (diff < 1e-4).float().mean() > 0.99


def test_sphere_sign_and_value():
    s = Sphere(0.5)
    assert s(torch.zeros(1, 3)).item() == pytest.approx(-0.5)
    assert s(torch.tensor([[1.0, 0, 0]])).item() == pytest.approx(0.5)


def test_box_exact_distance():
    b = Box([0.5, 0.5, 0.5])
    # corner distance
    x = torch.tensor([[1.0, 1.0, 1.0]])
    assert b(x).item() == pytest.approx(math.sqrt(3) * 0.5, rel=1e-5)
    # inside: distance to nearest face
    x = torch.tensor([[0.2, 0.0, 0.0]])
    assert b(x).item() == pytest.approx(-0.3, rel=1e-5)


@pytest.mark.parametrize("prim", PRIMITIVES, ids=lambda p: type(p).__name__)
def test_transform_equivariance(prim):
    """g = f.transform(R, t) must satisfy g(R x + t) == f(x)."""
    g = torch.Generator().manual_seed(3)
    R = random_rotation(g)
    t = torch.randn(3, generator=g) * 0.3
    x = _points(1024)
    moved = prim.transform(R, t)
    assert torch.allclose(moved(x @ R.T + t), prim(x), atol=1e-5)
    # gradient rotates: grad g(Rx+t) == R grad f(x)
    gg = moved.grad(x @ R.T + t)
    gf = prim.grad(x) @ R.T
    assert (torch.linalg.vector_norm(gg - gf, dim=-1) < 1e-4).float().mean() > 0.99


def test_transform_composition_collapses():
    s = Box([0.3, 0.2, 0.1])
    g = torch.Generator().manual_seed(4)
    R1, R2 = random_rotation(g), random_rotation(g)
    t1 = torch.randn(3, generator=g)
    t2 = torch.randn(3, generator=g)
    once = s.transform(R1, t1).transform(R2, t2)
    assert once.base is s  # composition collapsed
    x = _points(512)
    ref = s(((x - t2) @ R2 - t1) @ R1)
    assert torch.allclose(once(x), ref, atol=1e-5)


def test_scale_preserves_unit_gradient():
    s = Box([0.3, 0.2, 0.1]).scale(2.5)
    x = _points()
    gn = torch.linalg.vector_norm(s.grad(x), dim=-1)
    assert ((gn - 1).abs() < 1e-4).float().mean() > 0.97


def test_csg_signs():
    a, b = Sphere(0.5), Sphere(0.3)
    x0 = torch.zeros(1, 3)
    assert Union(a, b)(x0).item() == pytest.approx(-0.5)
    assert Intersect(a, b)(x0).item() == pytest.approx(-0.3)
    assert Subtract(a, b)(x0).item() == pytest.approx(0.3)  # hole interior
    assert Offset(a, 0.1)(x0).item() == pytest.approx(-0.6)
    assert Shell(a, 0.2)(x0).item() == pytest.approx(0.4)


def test_union_grad_unit():
    u = Union(Sphere(0.4).transform(torch.eye(3), torch.tensor([0.3, 0, 0])),
              Box([0.2, 0.3, 0.2]))
    x = _points()
    gn = torch.linalg.vector_norm(u.grad(x), dim=-1)
    assert ((gn - 1).abs() < 1e-4).float().mean() > 0.95


def test_smooth_min_bounds_and_grad():
    a = Sphere(0.4)
    b = Sphere(0.4).transform(torch.eye(3), torch.tensor([0.5, 0, 0]))
    sm = SmoothMin(a, b, 0.1)
    x = _points()
    hard = torch.minimum(a(x), b(x))
    v = sm(x)
    assert (v <= hard + 1e-6).all()
    assert (v >= hard - 0.1 / 4 - 1e-6).all()
    # analytic grad matches autograd
    with torch.enable_grad():
        xg = x.clone().requires_grad_(True)
        (ag,) = torch.autograd.grad(sm(xg).sum(), xg)
    assert (torch.linalg.vector_norm(sm.grad(x) - ag, dim=-1) < 1e-4).float().mean() > 0.99


def test_blend_is_flagged_approximate():
    assert GradPreservingBlend.APPROXIMATE is True


def test_shell_grad_unit():
    sh = Shell(Sphere(0.5), 0.1)
    x = _points()
    gn = torch.linalg.vector_norm(sh.grad(x), dim=-1)
    assert ((gn - 1).abs() < 1e-4).float().mean() > 0.97


def test_stratified_sampling_bands_and_exactness():
    f = Union(Sphere(0.45), Box([0.35, 0.25, 0.3]))
    n = 8192
    ss = stratified(f, n=n, seed=7)
    assert ss.x.shape == (n, 3)
    # returned f and grad are exact
    assert torch.allclose(ss.f, f(ss.x), atol=1e-6)
    assert torch.allclose(ss.grad, f.grad(ss.x), atol=1e-6)
    # band quotas roughly met (surface band dominates)
    frac0 = (ss.band == 0).float().mean().item()
    assert frac0 > 0.3, f"surface band underfilled: {frac0}"
    for b, (lo, hi) in enumerate(DEFAULT_BANDS):
        sel = ss.band == b
        if sel.any():
            af = ss.f[sel].abs()
            assert (af >= lo - 1e-6).all() and (af < hi + 1e-6).all()
