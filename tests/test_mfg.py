"""Manufacturability label tests on hand-computable shapes (box, T-bracket)."""
import math

import pytest
import torch

from geofield.fields.primitives import Box
from geofield.fields.ops import Union
from geofield.labels import manufacturability as mfg
from geofield.labels import scalars

UP = torch.tensor([0.0, 0.0, 1.0])


def _face_points(center, half, axis, sign, n=64, margin=0.7, seed=0):
    """Points on a box face, slightly outside the surface."""
    g = torch.Generator().manual_seed(seed)
    axes = [i for i in range(3) if i != axis]
    pts = torch.zeros(n, 3)
    for a in axes:
        pts[:, a] = (torch.rand(n, generator=g) * 2 - 1) * half[a] * margin + center[a]
    pts[:, axis] = center[axis] + sign * (half[axis] + 0.005)
    return pts


def test_overhang_box():
    b = Box([0.4, 0.3, 0.2])
    top = _face_points([0, 0, 0], [0.4, 0.3, 0.2], 2, +1)
    bot = _face_points([0, 0, 0], [0.4, 0.3, 0.2], 2, -1)
    side = _face_points([0, 0, 0], [0.4, 0.3, 0.2], 0, +1)
    assert mfg.overhang(b, top, UP).max() < 0.1
    assert mfg.overhang(b, bot, UP).min() > math.pi - 0.1
    assert (mfg.overhang(b, side, UP) - math.pi / 2).abs().max() < 0.1


def test_support_box_bottom_on_plate():
    """A box's bottom face sits ON the build plate -> no support anywhere."""
    b = Box([0.4, 0.3, 0.2])
    for pts in [_face_points([0, 0, 0], [0.4, 0.3, 0.2], 2, +1),
                _face_points([0, 0, 0], [0.4, 0.3, 0.2], 2, -1),
                _face_points([0, 0, 0], [0.4, 0.3, 0.2], 0, +1)]:
        assert mfg.support_need(b, pts, UP).max() == 0.0


def _t_bracket():
    """T on its side: vertical post 0.1 thick from plate up to z=0.5, plus a
    horizontal slab at height 0.4..0.5 that overhangs the post on both sides.
    The slab's downward faces away from the post need support."""
    post = Box([0.05, 0.2, 0.25]).transform(torch.eye(3), torch.tensor([0.0, 0.0, 0.25]))
    slab = Box([0.4, 0.2, 0.05]).transform(torch.eye(3), torch.tensor([0.0, 0.0, 0.45]))
    return Union(post, slab)


def test_support_t_bracket_overhang():
    t = _t_bracket()
    # under the slab wings (away from the post): downward faces high above plate
    g = torch.Generator().manual_seed(0)
    n = 64
    wing = torch.zeros(n, 3)
    wing[:, 0] = 0.2 + torch.rand(n, generator=g) * 0.15   # x in [0.2, 0.35]
    wing[:, 1] = (torch.rand(n, generator=g) * 2 - 1) * 0.15
    wing[:, 2] = 0.4 - 0.005                               # just below the slab
    assert mfg.support_need(t, wing, UP).mean() > 0.9
    # top of the slab: no support
    top = wing.clone()
    top[:, 2] = 0.5 + 0.005
    assert mfg.support_need(t, top, UP).max() == 0.0
    # post side wall: vertical, no support
    side = torch.zeros(n, 3)
    side[:, 0] = 0.05 + 0.005
    side[:, 1] = (torch.rand(n, generator=g) * 2 - 1) * 0.15
    side[:, 2] = 0.1 + torch.rand(n, generator=g) * 0.2
    assert mfg.support_need(t, side, UP).max() == 0.0


def test_tool_access_box():
    """+z spindle: top face reachable, bottom face blocked by the box itself."""
    b = Box([0.3, 0.3, 0.2])
    top = _face_points([0, 0, 0], [0.3, 0.3, 0.2], 2, +1)
    bot = _face_points([0, 0, 0], [0.3, 0.3, 0.2], 2, -1)
    _, acc_top = mfg.tool_access(b, top, UP)
    _, acc_bot = mfg.tool_access(b, bot, UP)
    assert acc_top.mean() > 0.9
    assert acc_bot.mean() < 0.1


def test_tool_access_t_bracket_under_slab():
    """The wing undersides of the T are shadowed from a +z spindle."""
    t = _t_bracket()
    g = torch.Generator().manual_seed(1)
    n = 64
    wing = torch.zeros(n, 3)
    wing[:, 0] = 0.2 + torch.rand(n, generator=g) * 0.15
    wing[:, 1] = (torch.rand(n, generator=g) * 2 - 1) * 0.15
    wing[:, 2] = 0.4 - 0.005
    _, acc = mfg.tool_access(t, wing, UP)
    assert acc.mean() < 0.1
    soft, acc_top = mfg.tool_access(t, wing.clone().index_fill_(
        1, torch.tensor(2), 0.5 + 0.005), UP)
    assert acc_top.mean() > 0.9


def test_thickness_slab():
    """Thickness of a 0.2-thick slab is 0.2 on its faces."""
    slab = Box([0.5, 0.5, 0.1])
    pts = _face_points([0, 0, 0], [0.5, 0.5, 0.1], 2, +1, margin=0.5)
    th = mfg.thickness(slab, pts)
    assert (th - 0.2).abs().max() < 0.02


def test_thickness_varies_with_geometry():
    thick = Box([0.2, 0.2, 0.2])
    thin = Box([0.2, 0.2, 0.02])
    p_thick = _face_points([0, 0, 0], [0.2, 0.2, 0.2], 2, +1, margin=0.3)
    p_thin = _face_points([0, 0, 0], [0.2, 0.2, 0.02], 2, +1, margin=0.3)
    assert mfg.thickness(thick, p_thick).median() > 3 * mfg.thickness(thin, p_thin).median()


def test_scalar_fractions():
    t = _t_bracket()
    g = torch.Generator().manual_seed(2)
    x = torch.rand(8192, 3, generator=g) * 2 - 1
    f = t(x)
    sup = mfg.support_need(t, x, UP)
    frac = scalars.support_fraction(sup, f)
    assert 0.0 < frac < 0.5  # wings need support; most of the surface doesn't
    th = mfg.ray_thickness(t, x)
    mt = scalars.min_thickness(th, f)
    assert 0.05 < mt < 0.25
