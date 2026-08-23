"""FEA adapter tests: element stiffness sanity, the cantilever benchmark
(analytic Timoshenko deflection within 10%), anisotropic tight-domain grids,
and calibrated L-bracket load-case solving."""
import math

import pytest
import torch

from geofield.fields.primitives import Box
from geofield.labels.physics import (
    MATERIALS, VoxelFEA, element_stiffness, l_load_cases, material_token,
    solve_l_case)
from geofield.tokens.schema import Token


def test_element_stiffness_properties():
    KE, B0 = element_stiffness(h=0.1, nu=0.3)
    assert torch.allclose(KE, KE.T, atol=1e-8)
    evals = torch.linalg.eigvalsh(KE)
    # exactly 6 rigid-body modes (zero eigenvalues), rest positive
    assert (evals[:6].abs() < 1e-6 * evals[-1]).all()
    assert (evals[6:] > 0).all()
    # rigid translation produces zero force
    u = torch.zeros(24, dtype=KE.dtype)
    u[0::3] = 1.0
    assert torch.linalg.vector_norm(KE @ u) < 1e-8 * torch.linalg.vector_norm(KE)


def test_materials_table():
    for name, m in MATERIALS.items():
        assert 0 < m["nu"] < 0.5
        assert m["E"] > 1e8
        tok = material_token(name)
        assert tok.params["name"] == name


def test_anisotropic_bounds_grid():
    """Tight AABB mode: grid covers the bounds with cubic cells of side h."""
    fea = VoxelFEA(bounds=(torch.tensor([-0.05, 0.0, -0.2]),
                           torch.tensor([0.05, 0.15, 0.0])), h=0.005)
    nx, ny, nz = fea.dims
    assert nx >= 20 and ny >= 30 and nz >= 40
    assert fea.h == pytest.approx(0.005)
    c = fea._element_centers()
    assert c.shape == (nx, ny, nz, 3)
    # centers span the (padded) bounds
    assert c[..., 1].min() > -0.01 and c[..., 1].max() < 0.16


@pytest.mark.slow
def test_cantilever_deflection_within_10pct():
    """End-loaded cantilever beam vs Timoshenko analytic solution.

    Beam: x in [-0.6, 0.6], width b=0.3 (y), height t=0.2 (z). All nodes with
    x <= -0.5 are clamped exactly (extra_fixed plane, unambiguous geometry);
    tip load -z centered at x=+0.5. Span L = 1.0.
    delta = F L^3 / (3 E I) + F L / (kappa G A), kappa = 5/6.
    """
    b_half, t_half = 0.15, 0.1
    beam = Box([0.6, b_half, t_half])
    E, nu = 10e9, 0.3
    F = 1000.0
    x_clamp, x_load = -0.5, 0.5

    load = Token("load", [x_load, 0.0, 0.0, 0, 0, 0, 1],
                 {"direction": [0.0, 0, -1.0], "magnitude": F, "kind": "force",
                  "grip": 0.06})
    mat = Token("material", None, {"E": E, "nu": nu, "yield": 1e9,
                                   "density": 2700.0, "name": "test"})

    solver = VoxelFEA(res=128, tol=1e-7, max_iter=8000)
    q = torch.tensor([[x_load, 0.0, 0.0]])
    out = solver.solve(beam, [load, mat], q,
                       extra_fixed=lambda p: p[:, 0] <= x_clamp)

    span = x_load - x_clamp
    I = (2 * b_half) * (2 * t_half) ** 3 / 12
    A = (2 * b_half) * (2 * t_half)
    G = E / (2 * (1 + nu))
    delta = F * span ** 3 / (3 * E * I) + F * span / (5 / 6 * G * A)

    tip = -out.displacement[0, 2].item()
    rel_err = abs(tip - delta) / delta
    assert rel_err < 0.10, f"tip {tip:.3e} vs analytic {delta:.3e} ({rel_err:.1%})"
    assert out.residual < 1e-4
    assert out.peak_vm > 0


@pytest.mark.slow
def test_l_case_calibration_and_solve():
    """Calibrated solve on a real L-bracket: peak stress must land exactly at
    the target yield fraction (linearity), fields finite, both materials."""
    from geofield.fields.programs import l_bracket
    field_mm, toks, meta = l_bracket.sample(2)
    wall_fp = [t for t in toks if t.type == "fixed_point" and t.params["fixed"]]
    cases = l_load_cases(meta, seed=2, n_cases=2)
    g = torch.Generator().manual_seed(0)
    # queries across the part in mm
    lo = torch.tensor(meta["aabb"]["lo"])
    hi = torch.tensor(meta["aabb"]["hi"])
    q_mm = lo + torch.rand(2048, 3, generator=g) * (hi - lo)
    for m in ("Al6061", "Steel1018"):
        out, F = solve_l_case(field_mm, wall_fp, cases[0], m, q_mm,
                              aabb_mm=(lo.tolist(), hi.tolist()),
                              t_min_mm=min(meta["tw"], meta["tf"]))
        target = cases[0]["target_yield_fraction"] * MATERIALS[m]["yield"]
        assert out.peak_vm == pytest.approx(target, rel=0.02)
        assert F > 0
        assert torch.isfinite(out.von_mises).all()
        assert torch.isfinite(out.displacement).all()
        assert 0 < out.mask.mean() < 1
        assert out.residual < 1e-3
