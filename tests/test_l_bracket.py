"""Tests for the L-bracket program: determinism, hole validity, variants,
frame mapping, machinability sanity, and load-case sampling."""
import pytest
import torch

from geofield.fields.programs import l_bracket
from geofield.fields.programs.l_bracket import LBracketConfig
from geofield.fields.sampling import stratified
from geofield.labels import manufacturability as mfg
from geofield.labels.physics import OOD_LOAD_BAND, l_load_cases


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
def test_deterministic_and_valid(seed):
    f1, t1, m1 = l_bracket.sample(seed)
    f2, t2, m2 = l_bracket.sample(seed)
    x = torch.randn(512, 3) * 100.0  # mm frame
    assert torch.allclose(f1(x), f2(x), atol=1e-4)
    assert m1["Lw"] == m2["Lw"]
    fps = [t for t in t1 if t.type == "fixed_point" and t.params["fixed"]]
    assert 2 <= len(fps) <= 4
    assert any(t.type == "envelope" for t in t1)
    assert any(t.type == "spindle_dir" for t in t1)


@pytest.mark.parametrize("seed", [0, 3, 11, 25, 31])
def test_mount_holes_valid(seed):
    """Void at hole centers, solid ring around each mounting hole (mm frame)."""
    f, toks, meta = l_bracket.sample(seed)
    for t in toks:
        if t.type != "fixed_point":
            continue
        pos = t.position
        assert f(pos.unsqueeze(0)).item() > 0, "hole center must be void"
        r_ring = 0.75 * t.params["diameter"]
        axis = t.params["axis"]
        a = torch.tensor([1.0, 0, 0])
        if abs(float(axis @ a)) > 0.9:
            a = torch.tensor([0.0, 1, 0])
        u = torch.linalg.cross(axis, a)
        u = u / torch.linalg.vector_norm(u)
        v = torch.linalg.cross(axis, u)
        ang = torch.linspace(0, 6.2832, 8)
        ring = pos + r_ring * (torch.cos(ang).unsqueeze(-1) * u
                               + torch.sin(ang).unsqueeze(-1) * v)
        assert (f(ring) < 0).float().mean() > 0.6


def test_solid_inside_envelope():
    for seed in range(6):
        f, toks, meta = l_bracket.sample(seed)
        env = next(t for t in toks if t.type == "envelope")
        g = torch.Generator().manual_seed(seed)
        # random points OUTSIDE the (tight, pre-slack) AABB must be void
        lo = torch.tensor(meta["aabb"]["lo"])
        hi = torch.tensor(meta["aabb"]["hi"])
        span = hi - lo
        pts = lo - span * (0.05 + torch.rand(64, 3, generator=g) * 0.3)
        assert (f(pts) > 0).all()


def test_variants_appear():
    reinf, lw = set(), set()
    for seed in range(30):
        _, _, m = l_bracket.sample(seed)
        reinf.add(m["reinforcement"])
        lw.add(m["lightweight"])
    assert {"none", "gusset"} <= reinf
    assert "none" in lw and len(lw) >= 2


def test_ood_geometry_carveout():
    cfg = LBracketConfig()
    for seed in range(8):
        _, _, m = l_bracket.sample(seed, cfg)
        in_corner = (m["Lw"] > cfg.ood_leg and m["Lf"] > cfg.ood_leg
                     and min(m["tw"], m["tf"]) < cfg.ood_thick)
        assert not in_corner, "train draw fell in the OOD corner"
    _, _, m = l_bracket.sample(0, cfg, force_ood_geometry=True)
    assert m["Lw"] > cfg.ood_leg and m["Lf"] > cfg.ood_leg
    assert min(m["tw"], m["tf"]) < cfg.ood_thick


def test_finalize_frame_roundtrip_and_scale_token():
    f_mm, toks, meta = l_bracket.sample(4)
    f_n, toks_n, frame = l_bracket.finalize(f_mm, toks, meta, seed=4)
    g = torch.Generator().manual_seed(0)
    x_n = torch.randn(256, 3, generator=g) * 0.5
    x_mm = l_bracket.norm_to_mm(x_n, frame)
    back = l_bracket.mm_to_norm(x_mm, frame)
    assert torch.allclose(back, x_n, atol=1e-4)
    # field values correspond: f_n(x_n) == s * f_mm(x_mm)
    assert torch.allclose(f_n(x_n), frame["s"] * f_mm(x_mm), atol=1e-3)
    sc = next(t for t in toks_n if t.type == "scale")
    assert sc.params["unit_mm"] == pytest.approx(1.0 / frame["s"], rel=1e-5)
    # normalized: surface inside unit sphere
    ss = stratified(f_n, n=2048, seed=0)
    surf = ss.x[ss.f.abs() < 0.01]
    assert torch.linalg.vector_norm(surf, dim=-1).max() < 1.001


def test_machinability_fixture_direction():
    """Top surfaces reachable from the fixture spindle (+y away from wall);
    the wall-contact face is not."""
    f, toks, meta = l_bracket.sample(1)  # canonical mm frame
    d = torch.tensor([0.0, 1.0, 0.0])
    g = torch.Generator().manual_seed(0)
    n = 64
    # points on the free leg outer face (y = Lf side is reachable from +y? the
    # free leg TOP surface z=0 faces +z; test instead the wall-contact face
    # y=0 (unreachable) vs the free-leg end face y=Lf (reachable)
    wall_face = torch.stack([
        (torch.rand(n, generator=g) * 0.8 - 0.4) * meta["W"],
        torch.full((n,), -0.5),
        -(0.2 + torch.rand(n, generator=g) * 0.6) * meta["Lw"]], dim=-1)
    end_face = torch.stack([
        (torch.rand(n, generator=g) * 0.8 - 0.4) * meta["W"],
        torch.full((n,), meta["Lf"] + 0.5),
        -(torch.rand(n, generator=g) * 0.9) * meta["tf"]], dim=-1)
    _, acc_wall = mfg.tool_access(f, wall_face, d, sphere_radius=1.2 * max(
        meta["Lw"], meta["Lf"]), surface_eps=0.5)
    _, acc_end = mfg.tool_access(f, end_face, d, sphere_radius=1.2 * max(
        meta["Lw"], meta["Lf"]), surface_eps=0.5)
    assert acc_wall.mean() < 0.2
    assert acc_end.mean() > 0.8


def test_load_cases_structure_and_ood_band():
    _, _, meta = l_bracket.sample(5)
    cases = l_load_cases(meta, seed=5, n_cases=10)
    assert len(cases) == 10
    for c in cases:
        d = torch.linalg.vector_norm(c["dir"])
        assert abs(float(d) - 1.0) < 1e-5
        assert 0.1 <= c["target_yield_fraction"] <= 1.5
        if c["kind"] == "patch":
            frac = float(c["pos_mm"][1]) / meta["Lf"]
            assert not (OOD_LOAD_BAND[0] <= frac <= OOD_LOAD_BAND[1]), \
                "train case landed in the held-out band"
            assert c["pos_mm"][2] == 0.0  # top surface
    ood = l_load_cases(meta, seed=5, n_cases=10, ood_load=True)
    fracs = [float(c["pos_mm"][1]) / meta["Lf"] for c in ood
             if c["kind"] == "patch"]
    assert all(OOD_LOAD_BAND[0] <= fr <= OOD_LOAD_BAND[1] for fr in fracs)
    # deterministic
    again = l_load_cases(meta, seed=5, n_cases=10)
    assert torch.allclose(again[0]["dir"], cases[0]["dir"])
