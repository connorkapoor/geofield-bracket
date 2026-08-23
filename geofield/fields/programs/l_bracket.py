"""Shelf-style L-bracket family -> (Field, tokens, meta).

Canonical frame (units: mm):
  * wall plane at y = 0, wall normal INTO the wall = -y
  * wall leg: x in [-W/2, W/2], y in [0, tw], z in [-Lw, 0]
  * free leg: x in [-W/2, W/2], y in [0, Lf], z in [-tf, 0] (top surface z = 0)
  * fixture: part clamped on its wall face -> 3-axis tool approaches from +y
    (opposite the fixed surface); spindle_dir token = +y

Variants:
  * corner reinforcement: none / triangular gusset plate / 1-2 cylindrical ribs
  * lightweighting: none / circular hole pattern / through-slots
  * 2-4 wall mounting holes (real fastener clearance sizes, M4-M10)
  * free leg: 70% plain (patch loading), 30% one bolt through-hole

`sample(seed, cfg)` returns CANONICAL-frame (field, tokens, meta) so load
cases can be placed with engineering meaning; `finalize(...)` then applies a
random SE(3), normalizes into the unit sphere, and appends the `scale` token
(mm per normalized unit). Deterministic in (seed, cfg).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict, replace

import torch
from torch import Tensor

from ..primitives import Box, Capsule, Cylinder, Field, Halfspace
from ..ops import GradPreservingBlend, Intersect, Subtract, Union
from ...tokens.schema import Token, rotmat_to_quat
from .common import normalize_to_unit_sphere, random_rotation

# fastener nominal -> clearance hole diameter (mm)
CLEARANCE = {"M4": 4.5, "M5": 5.5, "M6": 6.6, "M8": 9.0, "M10": 11.0}


@dataclass
class LBracketConfig:
    leg_len: tuple = (75.0, 300.0)       # mm, each leg independently
    width: tuple = (25.0, 75.0)
    thickness: tuple = (5.0, 12.0)       # plate thickness (per leg +-30%)
    fillet: tuple = (3.0, 12.0)
    p_reinforcement: tuple = (0.4, 0.4, 0.2)   # none / gusset / ribs
    p_lightweight: tuple = (0.5, 0.3, 0.2)     # none / holes / slots
    p_free_hole: float = 0.3
    n_mount_holes: tuple = (2, 4)
    fasteners: tuple = ("M4", "M5", "M6", "M8", "M10")
    # OOD carve-out: train samples with (both legs > ood_leg AND thickness <
    # ood_thick) are rejected; the ood_geometry split REQUIRES that corner.
    ood_leg: float = 260.0
    ood_thick: float = 6.5


def _u(gen, lo, hi):
    return lo + (hi - lo) * torch.rand((), generator=gen).item()


def _choice(gen, probs) -> int:
    r = torch.rand((), generator=gen).item()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1


def _in_ood_corner(meta: dict, cfg: LBracketConfig) -> bool:
    return (meta["Lw"] > cfg.ood_leg and meta["Lf"] > cfg.ood_leg
            and min(meta["tw"], meta["tf"]) < cfg.ood_thick)


def sample(seed: int, cfg: LBracketConfig | None = None,
           force_ood_geometry: bool = False
           ) -> tuple[Field, list[Token], dict]:
    """Canonical-frame L-bracket. Train draws reject the OOD geometry corner;
    force_ood_geometry narrows the parameter ranges INTO that corner (both
    legs long + thin plates), so ood draws succeed on the first attempt."""
    cfg = cfg or LBracketConfig()
    if force_ood_geometry:
        cfg = replace(cfg,
                      leg_len=(cfg.ood_leg + 5.0, cfg.leg_len[1]),
                      thickness=(cfg.thickness[0],
                                 min(cfg.ood_thick * 0.95, cfg.thickness[1])))
    for attempt in range(64):
        gen = torch.Generator().manual_seed(seed * 64 + attempt)
        out = _build(gen, cfg)
        if out is None:
            continue
        field, tokens, meta = out
        if force_ood_geometry != _in_ood_corner(meta, cfg):
            continue
        meta["seed"] = seed
        return field, tokens, meta
    raise RuntimeError(f"l_bracket.sample: no valid draw for seed {seed}")


def _build(gen, cfg: LBracketConfig):
    Lw = _u(gen, *cfg.leg_len)
    Lf = _u(gen, *cfg.leg_len)
    W = _u(gen, *cfg.width)
    t_base = _u(gen, *cfg.thickness)
    tw = t_base * _u(gen, 0.7, 1.3)
    tf = t_base * _u(gen, 0.7, 1.3)
    tw = max(min(tw, cfg.thickness[1]), cfg.thickness[0])
    tf = max(min(tf, cfg.thickness[1]), cfg.thickness[0])
    fillet = _u(gen, *cfg.fillet)

    e3 = torch.eye(3)
    wall_leg = Box([W / 2, tw / 2, Lw / 2]).transform(
        e3, torch.tensor([0.0, tw / 2, -Lw / 2]))
    free_leg = Box([W / 2, Lf / 2, tf / 2]).transform(
        e3, torch.tensor([0.0, Lf / 2, -tf / 2]))
    solid: Field = GradPreservingBlend(wall_leg, free_leg, fillet)

    meta = {"Lw": Lw, "Lf": Lf, "W": W, "tw": tw, "tf": tf, "fillet": fillet}

    # ---- corner reinforcement ----------------------------------------------
    reinf = _choice(gen, cfg.p_reinforcement)
    meta["reinforcement"] = ["none", "gusset", "ribs"][reinf]
    gusset_reach = (0.0, 0.0)  # (a along y, b along -z) for keep-outs
    if reinf == 1:
        a = _u(gen, 0.25, 0.65) * (Lf - tw) * 0.9
        b = _u(gen, 0.25, 0.65) * (Lw - tf) * 0.9
        a, b = max(a, 8.0), max(b, 8.0)
        full = torch.rand((), generator=gen).item() < 0.5
        gw = W if full else max(_u(gen, 0.15, 0.35) * W, 4.0)
        gbox = Box([gw / 2, (a + 2) / 2, (b + 2) / 2]).transform(
            e3, torch.tensor([0.0, tw + (a - 2) / 2, -tf - (b - 2) / 2]))
        nrm = torch.tensor([0.0, b, -a])
        nrm = nrm / torch.linalg.vector_norm(nrm)
        d = float(nrm @ torch.tensor([0.0, tw + a, -tf]))
        gusset = Intersect(gbox, Halfspace(nrm, d))
        solid = GradPreservingBlend(solid, gusset, min(fillet, 6.0))
        gusset_reach = (a, b)
        meta["gusset"] = {"a": a, "b": b, "gw": gw, "full_width": full}
    elif reinf == 2:
        n_ribs = 1 if torch.rand((), generator=gen).item() < 0.6 else 2
        rr = _u(gen, 0.35, 0.6) * min(tw, tf)
        a = _u(gen, 0.3, 0.7) * (Lf - tw) * 0.85
        b = _u(gen, 0.3, 0.7) * (Lw - tf) * 0.85
        xs = [0.0] if n_ribs == 1 else [-W * 0.28, W * 0.28]
        for xi in xs:
            A = torch.tensor([xi, tw + a, -tf + rr * 0.3])
            B = torch.tensor([xi, tw - rr * 0.3, -tf - b])
            solid = GradPreservingBlend(solid, Capsule(A, B, rr), min(fillet, 5.0))
        gusset_reach = (a, b)
        meta["ribs"] = {"n": n_ribs, "r": rr, "a": a, "b": b}

    # ---- wall mounting holes -------------------------------------------------
    fastener = cfg.fasteners[int(torch.randint(len(cfg.fasteners), (),
                                               generator=gen).item())]
    dia = CLEARANCE[fastener]
    n_m = int(torch.randint(cfg.n_mount_holes[0], cfg.n_mount_holes[1] + 1,
                            (), generator=gen).item())
    edge = max(1.5 * dia, 8.0)
    z_top = -(tf + gusset_reach[1] + edge)   # below corner/reinforcement
    z_bot = -(Lw - edge)
    if z_bot > z_top - (n_m - 1) * 2.2 * dia:
        n_m = max(2, int((z_top - z_bot) / (2.2 * dia)) + 1)
        n_m = min(n_m, 4)
    if z_bot >= z_top:
        return None  # wall leg too short for this fastener; redraw
    tokens: list[Token] = []
    holes: list[Field] = []
    zs = torch.linspace(z_top, z_bot, n_m)
    R_y = torch.tensor([[1.0, 0, 0], [0.0, 0, 1], [0.0, -1, 0]])  # z-axis -> y
    for z_i in zs.tolist():
        c = torch.tensor([0.0, tw / 2, z_i])
        holes.append(Cylinder(dia / 2, tw / 2 + 2).transform(R_y, c))
        tokens.append(Token("fixed_point", torch.cat([c, rotmat_to_quat(R_y)]),
                            {"axis": torch.tensor([0.0, 1.0, 0.0]),
                             "diameter": dia, "fixed": True}))
    meta["mount"] = {"fastener": fastener, "n": n_m, "dia": dia,
                     "z": zs.tolist()}

    # ---- optional free-leg bolt hole ------------------------------------------
    meta["free_hole"] = None
    yh_min = max(0.55 * Lf, tw + gusset_reach[0] + 2 * dia)  # clear of gusset
    if torch.rand((), generator=gen).item() < cfg.p_free_hole and yh_min < 0.9 * Lf:
        yh = _u(gen, yh_min / Lf, 0.9) * Lf
        c = torch.tensor([0.0, yh, -tf / 2])
        holes.append(Cylinder(dia / 2, tf / 2 + 2).transform(e3, c))
        tokens.append(Token("fixed_point", torch.cat([c, torch.tensor([0.0, 0, 0, 1])]),
                            {"axis": torch.tensor([0.0, 0.0, 1.0]),
                             "diameter": dia, "fixed": False}))
        meta["free_hole"] = {"y": yh, "dia": dia}

    # ---- lightweighting ---------------------------------------------------------
    lw = _choice(gen, cfg.p_lightweight)
    meta["lightweight"] = ["none", "holes", "slots"][lw]
    cuts: list[Field] = []
    if lw:
        margin = max(0.22 * W, 1.6 * dia)

        def keepout_free(y):  # distance to free-leg keep-outs
            ok = tw + gusset_reach[0] + margin < y < Lf - margin
            if meta["free_hole"]:
                ok &= abs(y - meta["free_hole"]["y"]) > margin
            return ok

        def keepout_wall(z):
            ok = z_bot + margin / 2 < z < z_top  # mounting-hole column margin
            ok &= all(abs(z - zi) > margin for zi in zs.tolist())
            ok &= z < -(tf + gusset_reach[1] + margin)
            return ok

        if lw == 1:  # circular holes
            r_c = min(0.26 * W, 0.9 * margin)
            n_c = int(torch.randint(2, 6, (), generator=gen).item())
            for _ in range(n_c):
                if torch.rand((), generator=gen).item() < 0.5:
                    y = _u(gen, 0, 1) * Lf
                    if keepout_free(y):
                        cuts.append(Cylinder(r_c, tf / 2 + 2).transform(
                            e3, torch.tensor([0.0, y, -tf / 2])))
                else:
                    z = -_u(gen, 0, 1) * Lw
                    if keepout_wall(z):
                        cuts.append(Cylinder(r_c, tw / 2 + 2).transform(
                            R_y, torch.tensor([0.0, tw / 2, z])))
        else:  # slot: stadium through-cut along the free leg
            r_s = min(0.18 * W, 0.8 * margin)
            y1 = tw + gusset_reach[0] + margin + r_s
            y2 = Lf - margin - r_s
            if meta["free_hole"]:
                y2 = min(y2, meta["free_hole"]["y"] - margin - r_s)
            if y2 > y1 + 3 * r_s:
                cuts.append(_slot_y(y1, y2, r_s, tf, e3))
        if not cuts:
            meta["lightweight"] = "none"  # keep-outs rejected every candidate

    # subtract everything LAST so nothing re-fills holes
    for h in holes + cuts:
        solid = Subtract(solid, h)

    # ---- envelope + spindle tokens -----------------------------------------
    y_max = max(Lf, tw + gusset_reach[0])
    z_min = -max(Lw, tf + gusset_reach[1])
    lo = torch.tensor([-W / 2, 0.0, z_min])
    hi = torch.tensor([W / 2, y_max, 0.0])
    slack = 1.0 + _u(gen, 0.0, 0.15)
    center = (lo + hi) / 2
    half = (hi - lo) / 2 * slack
    tokens.append(Token("envelope",
                        torch.cat([center, torch.tensor([0.0, 0, 0, 1])]),
                        {"half_extents": half}))
    tokens.append(Token("spindle_dir", None,
                        {"direction": torch.tensor([0.0, 1.0, 0.0])}))
    meta["aabb"] = {"lo": lo.tolist(), "hi": hi.tolist()}
    return solid, tokens, meta


def build_from_params(p: dict) -> tuple[Field, list[Token], dict]:
    """Deterministic construction from EXPLICIT parameters (the learned
    designer's output), mirroring _build's geometry paths. Canonical mm frame.

    p keys: Lw, Lf, W, tw, tf, fillet, reinforcement ('none'|'gusset'|'ribs'),
    gusset_a, gusset_b, gusset_full (bool), ribs_n, ribs_r, lightweight
    ('none'|'holes'|'slots'), n_holes, hole_dia.
    """
    Lw, Lf, W = float(p["Lw"]), float(p["Lf"]), float(p["W"])
    tw, tf = float(p["tw"]), float(p["tf"])
    fillet = float(p.get("fillet", 6.0))
    e3 = torch.eye(3)
    wall_leg = Box([W / 2, tw / 2, Lw / 2]).transform(
        e3, torch.tensor([0.0, tw / 2, -Lw / 2]))
    free_leg = Box([W / 2, Lf / 2, tf / 2]).transform(
        e3, torch.tensor([0.0, Lf / 2, -tf / 2]))
    solid: Field = GradPreservingBlend(wall_leg, free_leg, fillet)
    meta = {"Lw": Lw, "Lf": Lf, "W": W, "tw": tw, "tf": tf, "fillet": fillet,
            "reinforcement": p.get("reinforcement", "none"),
            "lightweight": p.get("lightweight", "none")}
    reach = (0.0, 0.0)
    if p.get("reinforcement") == "gusset":
        a = max(8.0, min(float(p.get("gusset_a", 0.4)) * (Lf - tw), Lf - tw - 5))
        b = max(8.0, min(float(p.get("gusset_b", 0.4)) * (Lw - tf), Lw - tf - 5))
        gw = W if p.get("gusset_full", True) else max(0.25 * W, 4.0)
        gbox = Box([gw / 2, (a + 2) / 2, (b + 2) / 2]).transform(
            e3, torch.tensor([0.0, tw + (a - 2) / 2, -tf - (b - 2) / 2]))
        nrm = torch.tensor([0.0, b, -a])
        nrm = nrm / torch.linalg.vector_norm(nrm)
        d = float(nrm @ torch.tensor([0.0, tw + a, -tf]))
        solid = GradPreservingBlend(solid, Intersect(gbox, Halfspace(nrm, d)),
                                    min(fillet, 6.0))
        reach = (a, b)
        meta["gusset"] = {"a": a, "b": b, "gw": gw}
    elif p.get("reinforcement") == "ribs":
        n_ribs = max(1, min(2, int(p.get("ribs_n", 1))))
        rr = max(2.0, float(p.get("ribs_r", 0.45)) * min(tw, tf))
        a = 0.5 * (Lf - tw) * 0.85
        b = 0.5 * (Lw - tf) * 0.85
        xs = [0.0] if n_ribs == 1 else [-W * 0.28, W * 0.28]
        for xi in xs:
            A = torch.tensor([xi, tw + a, -tf + rr * 0.3])
            B = torch.tensor([xi, tw - rr * 0.3, -tf - b])
            solid = GradPreservingBlend(solid, Capsule(A, B, rr), min(fillet, 5.0))
        reach = (a, b)
        meta["ribs"] = {"n": n_ribs, "r": rr, "a": a, "b": b}

    dia = float(p.get("hole_dia", 6.6))
    n_m = max(2, min(4, int(p.get("n_holes", 3))))
    edge = max(1.5 * dia, 8.0)
    z_top = -(tf + reach[1] + edge)
    z_bot = -(Lw - edge)
    tokens: list[Token] = []
    holes: list[Field] = []
    if z_bot < z_top:
        zs = torch.linspace(z_top, z_bot, n_m)
        R_y = torch.tensor([[1.0, 0, 0], [0.0, 0, 1], [0.0, -1, 0]])
        for z_i in zs.tolist():
            c = torch.tensor([0.0, tw / 2, z_i])
            holes.append(Cylinder(dia / 2, tw / 2 + 2).transform(R_y, c))
            tokens.append(Token("fixed_point", torch.cat([c, rotmat_to_quat(R_y)]),
                                {"axis": torch.tensor([0.0, 1.0, 0.0]),
                                 "diameter": dia, "fixed": True}))
        meta["mount"] = {"n": n_m, "dia": dia, "z": zs.tolist()}

    if p.get("lightweight") == "holes":
        margin = max(0.22 * W, 1.6 * dia)
        r_c = min(0.26 * W, 0.9 * margin)
        y_lo = tw + reach[0] + margin
        y_hi = Lf - margin
        if y_hi > y_lo + 2 * r_c:
            for y in torch.linspace(y_lo + r_c, y_hi - r_c,
                                    3).tolist():
                holes.append(Cylinder(r_c, tf / 2 + 2).transform(
                    e3, torch.tensor([0.0, y, -tf / 2])))
    for h in holes:
        solid = Subtract(solid, h)

    y_max = max(Lf, tw + reach[0])
    z_min = -max(Lw, tf + reach[1])
    lo = torch.tensor([-W / 2, 0.0, z_min])
    hi = torch.tensor([W / 2, y_max, 0.0])
    tokens.append(Token("envelope",
                        torch.cat([(lo + hi) / 2, torch.tensor([0.0, 0, 0, 1])]),
                        {"half_extents": (hi - lo) / 2}))
    tokens.append(Token("spindle_dir", None,
                        {"direction": torch.tensor([0.0, 1.0, 0.0])}))
    meta["aabb"] = {"lo": lo.tolist(), "hi": hi.tolist()}
    return solid, tokens, meta


def _slot_y(y1: float, y2: float, r: float, tf: float, e3: Tensor) -> Field:
    """Stadium through-slot in the free leg along y at x=0."""
    c1 = Cylinder(r, tf / 2 + 2).transform(e3, torch.tensor([0.0, y1, -tf / 2]))
    c2 = Cylinder(r, tf / 2 + 2).transform(e3, torch.tensor([0.0, y2, -tf / 2]))
    mid = Box([r, (y2 - y1) / 2, tf / 2 + 2]).transform(
        e3, torch.tensor([0.0, (y1 + y2) / 2, -tf / 2]))
    return Union(c1, c2, mid)


# ---------------------------------------------------------------------------
# Finalize: SE(3) + normalization + scale token
# ---------------------------------------------------------------------------

def finalize(field: Field, tokens: list[Token], meta: dict, seed: int,
             rotate: bool = True) -> tuple[Field, list[Token], dict]:
    """Random SE(3) then unit-sphere normalization; appends the `scale` token.

    Returns (field_norm, tokens_norm, frame) with frame = {R, center, s}:
      x_norm = s * (R @ x_mm - center);  x_mm = R.T @ (x_norm / s + center).
    """
    gen = torch.Generator().manual_seed(seed ^ 0xF17A)
    R = random_rotation(gen) if rotate else torch.eye(3)
    field = field.transform(R, torch.zeros(3))
    tokens = [t.transform(R, torch.zeros(3)) for t in tokens]
    extent = max(meta["Lw"], meta["Lf"], meta["W"])
    field, tokens, s, center = normalize_to_unit_sphere(
        field, tokens, seed=seed, probe_scale=0.6 * extent)
    tokens = list(tokens) + [Token("scale", None, {"unit_mm": 1.0 / s})]
    frame = {"R": R, "center": center, "s": s}
    return field, tokens, frame


def norm_to_mm(x_norm: Tensor, frame: dict) -> Tensor:
    return (x_norm / frame["s"] + frame["center"]) @ frame["R"]


def mm_to_norm(x_mm: Tensor, frame: dict) -> Tensor:
    return (x_mm @ frame["R"].T - frame["center"]) * frame["s"]


def config_dict(cfg: LBracketConfig | None = None) -> dict:
    return asdict(cfg or LBracketConfig())
