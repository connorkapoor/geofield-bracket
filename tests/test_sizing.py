"""Sizing checks: calibration against the validated FEA solver, monotonicity,
and load-direction-dependent strategy."""
import pytest

from geofield.model.sizing import (KT_FILLET, T_MAX_MM, Sizing,
                                   bending_stress_pa, required_thickness_mm,
                                   size_bracket)

YIELD_AL = 276e6
YIELD_STEEL = 370e6


def test_calibrated_against_fea():
    """The measured case: 500 N at 126 mm arm, W=60, tf=8.7 -> immersed-FEA
    solver reported 113 MPa (beam theory alone gives 83)."""
    s = bending_stress_pa(500.0, 126.0, 60.0, 8.7)
    assert s / 1e6 == pytest.approx(113.0, rel=0.12)


def test_thickness_inverts_stress():
    t = required_thickness_mm(500.0, 126.0, 60.0, YIELD_AL / 1.6)
    back = bending_stress_pa(500.0, 126.0, 60.0, t)
    assert back == pytest.approx(YIELD_AL / 1.6, rel=1e-6)


def test_heavier_load_needs_thicker_plate():
    a = size_bracket(200, (0, 0, -1), 120, 60, 200, YIELD_AL)
    b = size_bracket(1200, (0, 0, -1), 120, 60, 200, YIELD_AL)
    assert b.tf_min_mm > a.tf_min_mm * 1.8   # ~sqrt(6) scaling


def test_longer_arm_needs_thicker_plate():
    a = size_bracket(500, (0, 0, -1), 60, 60, 200, YIELD_AL)
    b = size_bracket(500, (0, 0, -1), 240, 60, 200, YIELD_AL)
    assert b.tf_min_mm > a.tf_min_mm * 1.6


def test_stronger_material_needs_less():
    al = size_bracket(800, (0, 0, -1), 150, 60, 200, YIELD_AL)
    st = size_bracket(800, (0, 0, -1), 150, 60, 200, YIELD_STEEL)
    assert st.tf_min_mm < al.tf_min_mm


def test_gusset_relief_reduces_thickness():
    plain = size_bracket(800, (0, 0, -1), 150, 60, 200, YIELD_AL,
                         reinforcement_reach_mm=0.0)
    braced = size_bracket(800, (0, 0, -1), 150, 60, 200, YIELD_AL,
                          reinforcement_reach_mm=90.0)
    assert braced.tf_min_mm < plain.tf_min_mm


def test_lateral_load_picks_ribs_and_offsets_web():
    s = size_bracket(600, (0.7, 0.0, -0.7), 140, 60, 200, YIELD_AL)
    assert s.style == "ribs"
    assert s.asym_x > 0.1
    assert any("lateral" in n for n in s.notes)


def test_down_load_picks_gusset_and_is_symmetric():
    s = size_bracket(600, (0, 0, -1), 140, 60, 200, YIELD_AL)
    assert s.style == "gusset"
    assert s.asym_x == 0.0


def test_pullout_adds_bolts_and_notes():
    s = size_bracket(600, (0, 1.0, -0.2), 140, 60, 200, YIELD_AL)
    assert s.n_holes_min >= 4
    assert any("pull-out" in n for n in s.notes)


def test_over_capacity_flag_and_clamp():
    s = size_bracket(50000, (0, 0, -1), 300, 25, 200, YIELD_AL)
    assert s.over_capacity
    assert s.tf_min_mm <= T_MAX_MM
    assert any("exceeds" in n for n in s.notes)


def test_utilisation_reported_at_target():
    s = size_bracket(700, (0, 0, -1), 150, 60, 200, YIELD_AL, safety_factor=2.0)
    assert 0.4 < s.utilisation < 0.6      # ~1/SF once sized to the floor
