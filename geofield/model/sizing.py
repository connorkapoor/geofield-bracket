"""Closed-form sizing checks for the L-bracket family.

The learned designer proposes proportions and style; this module enforces that
the result can actually carry the load. It is deliberately plain beam theory
plus a calibrated stress-concentration factor, because:

  * it is instant (no solve), so the UI can show utilisation while dragging,
  * it is auditable — an engineer can check every line by hand,
  * it gives the FEA verifier something to disagree with.

Calibration: for a plain (ungusseted) bracket with W=60 mm, tf=8.7 mm loaded
with 500 N at a 126 mm arm, beam theory gives 83 MPa at the free-leg root and
the validated immersed-FEA solver gives 113 MPa — the difference is the fillet
stress concentration. KT = 1.36 reproduces the solver, so it is used as the
default. See tests/test_sizing.py.

Frame convention (canonical, mm): wall plate at y=0 spanning z, free leg along
+y at the top, load applied on the free leg at `arm_mm` from the wall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

KT_FILLET = 1.36          # calibrated against the immersed-FEA solver
T_MIN_MM, T_MAX_MM = 3.0, 25.0
GUSSET_ARM_RELIEF = 0.85  # a gusset of reach a shortens the effective arm by
                          # 0.85*a (the load path turns into the diagonal)


def bending_stress_pa(force_n: float, arm_mm: float, width_mm: float,
                      thick_mm: float, kt: float = KT_FILLET) -> float:
    """Peak fibre stress of a rectangular section in bending, with the fillet
    concentration factor. Z = w t^2 / 6."""
    z_m3 = (width_mm * 1e-3) * (thick_mm * 1e-3) ** 2 / 6.0
    if z_m3 <= 0:
        return float("inf")
    return kt * (force_n * arm_mm * 1e-3) / z_m3


def required_thickness_mm(force_n: float, arm_mm: float, width_mm: float,
                          sigma_allow_pa: float, kt: float = KT_FILLET) -> float:
    """Invert the above for thickness."""
    if force_n <= 0 or arm_mm <= 0 or sigma_allow_pa <= 0:
        return T_MIN_MM
    need_z = kt * (force_n * arm_mm * 1e-3) / sigma_allow_pa      # m^3
    t_m = math.sqrt(6.0 * need_z / (width_mm * 1e-3))
    return max(T_MIN_MM, t_m * 1e3)


@dataclass
class Sizing:
    tf_min_mm: float          # free-leg (shelf) plate
    tw_min_mm: float          # wall plate
    style: str                # 'none' | 'gusset' | 'ribs'
    asym_x: float             # gusset offset across the width, -1..1
    n_holes_min: int
    utilisation: float        # of yield, at the sized geometry
    over_capacity: bool       # request needs more than T_MAX_MM
    notes: list = dc_field(default_factory=list)


def size_bracket(force_n: float, direction: tuple[float, float, float],
                 arm_mm: float, width_mm: float, wall_len_mm: float,
                 yield_pa: float, hole_dia_mm: float = 6.6,
                 safety_factor: float = 1.6,
                 reinforcement_reach_mm: float = 0.0,
                 style_hint: str | None = None) -> Sizing:
    """Size the two plates for a 3-D load and pick a reinforcement strategy.

    Three load components are checked independently and the governing one
    wins per plate:
      * dz (down/up): strong-axis bending of both plates -> usually governs
      * dx (lateral): weak-axis bending of the shelf (depth = width) and
        twist of the wall plate -> wants ribs, not a centred gusset
      * dy (pull-out / push-in): prying on the bolt column -> wall plate and
        bolt count
    """
    dx, dy, dz = (abs(direction[0]), direction[1], abs(direction[2]))
    n = math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
    dx, dy, dz = dx / n, dy / n, dz / n
    sig = yield_pa / max(safety_factor, 1.0)
    notes = []

    # a gusset/rib shortens the effective bending arm — this is exactly why
    # they exist, so reward it rather than treating style as cosmetic
    arm_eff = max(0.25 * arm_mm, arm_mm - GUSSET_ARM_RELIEF * reinforcement_reach_mm)

    f_z, f_x, f_y = force_n * dz, force_n * dx, force_n * max(dy, 0.0)

    # --- shelf plate -------------------------------------------------------
    tf_z = required_thickness_mm(f_z, arm_eff, width_mm, sig)
    # lateral: the shelf bends about its weak axis, depth = width (stiff), so
    # the thickness requirement is mild but non-zero
    tf_x = required_thickness_mm(f_x, arm_eff, max(width_mm * 0.5, 10.0), sig) \
        if f_x > 1.0 else T_MIN_MM
    tf = max(tf_z, tf_x)

    # --- wall plate --------------------------------------------------------
    tw_z = required_thickness_mm(f_z, arm_eff, width_mm, sig)
    # prying at the bolts under pull-out, lever ~1.5 x hole diameter
    tw_y = required_thickness_mm(f_y, 1.5 * hole_dia_mm, width_mm, sig) \
        if f_y > 1.0 else T_MIN_MM
    tw = max(tw_z, tw_y)

    # --- reinforcement strategy from the load direction --------------------
    if style_hint in ("none", "gusset", "ribs"):
        style = style_hint
    elif dx > 0.35:
        style = "ribs"      # off-axis: a single centred web does little
    elif dz > 0.5:
        style = "gusset"
    else:
        style = "none"
    if dx > 0.35:
        notes.append("lateral load component: twin ribs instead of a centred "
                     "gusset (a single web adds little torsional stiffness)")
    # asymmetric placement: push the web toward the side the load leans into
    asym_x = 0.0
    if abs(direction[0]) > 0.15:
        asym_x = max(-0.6, min(0.6, direction[0] * 1.2))
        notes.append(f"web offset {asym_x:+.2f} across the width to follow the "
                     "off-centre load path")
    if dy > 0.35:
        notes.append("pull-out component: wall plate sized for bolt prying")

    n_holes = 2 if wall_len_mm < 90 else (3 if wall_len_mm < 200 else 4)
    if f_y > 0.25 * force_n:
        n_holes = min(4, n_holes + 1)

    over = tf > T_MAX_MM or tw > T_MAX_MM
    tf = min(tf, T_MAX_MM)
    tw = min(tw, T_MAX_MM)
    if over:
        notes.append("load exceeds what this envelope can carry at "
                     f"{T_MAX_MM:.0f} mm plates — grow the box or drop the force")
    util = bending_stress_pa(f_z or force_n, arm_eff, width_mm, tf) / yield_pa
    return Sizing(tf_min_mm=tf, tw_min_mm=tw, style=style, asym_x=asym_x,
                  n_holes_min=n_holes, utilisation=util, over_capacity=over,
                  notes=notes)
