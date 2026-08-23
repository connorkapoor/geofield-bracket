"""GeoField demo backend (FastAPI).

Wraps the trained checkpoints behind five endpoints:

  POST /encode    {seed, split?}                 -> {latent_id}   (dataset shapes)
  POST /generate  {tokens[], n?, guidance?}      -> {candidates: [{latent_id,
                     peak_vm_pred, mass_kg, volume}]}
  POST /query     {latent_id, field_id, tokens?, resolution?}
                                                 -> {dims, values_b64(float32),
                                                     lo, hi}   (dense grid)
  POST /verify    {latent_id, tokens[], thresholds{sigma_max?, a_max?, t_min?}}
                                                 -> verify report (real FEA!)
  POST /export    {latent_id, resolution?}       -> binary STL

Tokens travel as the JSON schema of tokens.schema (type/pose/params).
One GPU worker; model calls serialized by a lock. Latents cached in memory.

Run:  python -m geofield.demo.backend.app --stage_b <ckpt> --flow <ckpt> \\
        --latent_stats <pt> [--data data/l1] [--host 0.0.0.0 --port 8600]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import threading
import uuid
from pathlib import Path

import torch

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install fastapi uvicorn pydantic") from e

from ...data.dataset import ShardDataset
from ...fields.primitives import Box, Cylinder
from ...fields.ops import Intersect, Subtract, Union
from ...export.marching_cubes import (analytic_normals, field_to_grid,
                                      grid_to_mesh, mesh_to_stl,
                                      refine_vertices)
from ...labels.physics import MATERIALS, material_token
from ...model.flow import LatentNormalizer, sample as flow_sample
from ...model.guidance import ConstraintGuidance
from ...model.latent import LatentSet
from ...tokens.schema import Token
from ...train.loop import build_model
from ...verify.fea import decoded_field_fn, verify_physics
from ...verify.mfg import verify_mfg
from ...labels import scalars as scal


class Engine:
    """Holds models + latent cache; serializes GPU access."""

    def __init__(self, stage_b: str, flow: str | None, latent_stats: str | None,
                 data_dir: str | None, device: str | None = None,
                 designer: str | None = None, linear_heads: bool = False):
        """linear_heads=True for checkpoints trained BEFORE the log-space head
        change (their raw outputs must not be run through expm1, which would
        turn ~300 into ~1e130). Drop the flag once serving a log-space ckpt."""
        if linear_heads:
            from ...model.heads import field_specs
            for spec in field_specs().values():
                spec.log_scale = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # stage_b is OPTIONAL: without it the demo still designs, builds exact
        # geometry, verifies with FEA and exports — it just cannot predict
        # fields or rank candidates (that is the geometric model's job). This
        # keeps the public repo runnable with only the 324 KB designer.
        self.model, self.cfg = None, {}
        if stage_b and Path(stage_b).exists():
            sd = torch.load(stage_b, map_location=self.device, weights_only=True)
            self.model = build_model(sd["cfg"]).to(self.device)
            self.model.load_state_dict(sd["model"])
            self.model.eval()
            self.cfg = sd["cfg"]

        self.flow = None
        self.norm = None
        if flow and Path(flow).exists() and self.model is not None:
            fsd = torch.load(flow, map_location=self.device, weights_only=True)
            from ...model.flow import LatentFlow
            stats = fsd["stats"]
            self.flow = LatentFlow(
                latent_dim=stats["latent_dim"], m_fine=stats["m_fine"],
                m_coarse=fsd["cfg"].get("m_coarse", 32) if "m_coarse" in fsd.get("cfg", {})
                else self.cfg.get("m_coarse", 32),
                dim=fsd["cfg"].get("flow_dim", 384),
                n_layers=fsd["cfg"].get("flow_layers", 8),
                k=fsd["cfg"].get("flow_k", 32)).to(self.device)
            self.flow.load_state_dict(fsd["model"])
            self.flow.eval()
            self.norm = LatentNormalizer.load(
                torch.load(latent_stats, map_location="cpu", weights_only=True))
        self.data = ShardDataset(data_dir, "val") if data_dir else None
        self.designer = None
        if designer and Path(designer).exists():
            from ...model.designer import ParamDesigner
            self.designer = ParamDesigner()
            dsd = torch.load(designer, map_location="cpu", weights_only=True)
            self.designer.load_state_dict(dsd["model"])
            self.designer.eval()
        self.latents: dict[str, dict] = {}
        self.lock = threading.Lock()

    # -- learned designer path --------------------------------------------------

    def ensure_latent(self, lid: str):
        """Lazily encode a designed candidate's exact field into the geometric
        model's latent — only needed when someone asks to paint or verify, so
        live-preview design stays cheap."""
        entry = self.get(lid)
        if entry.get("lat") is not None or self.model is None:
            return entry
        from ...fields.sampling import stratified
        ss = stratified(entry["field_obj"], n=1536, seed=0, device=self.device)
        with self.lock, torch.no_grad():
            entry["lat"] = self.model.encode(
                ss.x.unsqueeze(0), ss.f.unsqueeze(0), ss.grad.unsqueeze(0),
                [entry["tokens"]])
        return entry

    def design_candidates(self, tokens: list[Token], n: int, seed: int,
                          rank: bool = True) -> list[dict]:
        """Designer -> params -> exact CSG build -> (optional) surrogate rank.

        rank=False skips the encode + stress decode entirely: that is the
        live-preview path while the user drags sliders/handles (~10x faster);
        latents are filled in lazily by ensure_latent when needed.
        """
        from ...fields.programs import l_bracket
        from ...fields.sampling import stratified
        from ...model.designer import featurize
        env = next(t for t in tokens if t.type == "envelope")
        load = next(t for t in tokens if t.type == "load")
        mat = next(t for t in tokens if t.type == "material")
        scale_tok = next(t for t in tokens if t.type == "scale")
        u = float(scale_tok.params["unit_mm"])
        half = env.params["half_extents"]
        env_mm = half * 2 * u
        lp = load.position
        pf = torch.tensor([0.5,
                           (float(lp[1]) + float(half[1])) / (2 * float(half[1])),
                           (float(lp[2]) + float(half[2])) / (2 * float(half[2]))])
        # engineering geometry of the request, in mm
        arm_mm = (float(lp[1]) + float(half[1])) * u        # wall -> load
        room_down = (float(lp[2]) + float(half[2])) * u     # below the shelf
        room_up = (float(half[2]) - float(lp[2])) * u       # above the shelf
        flip = room_up > room_down   # put the wall leg where the room is
        avail_h = max(room_up, room_down)
        feats = featurize(env_mm, pf, load.params["direction"],
                          float(load.params["magnitude"]),
                          float(mat.params["E"]), float(mat.params["yield"]),
                          arm_mm=arm_mm)
        designs = self.designer.design(feats, n=n, seed=seed)
        out = []
        # canonical frame: wall back face at y=0, shelf TOP at z=0, wall leg
        # descending. Place canonical z=0 at the LOAD's height so the shelf
        # meets the load; flip 180deg about y when the room is above it.
        R = (torch.tensor([[-1.0, 0, 0], [0.0, 1, 0], [0.0, 0, -1]])
             if flip else torch.eye(3))
        offset = torch.tensor([0.0, -float(half[1]), float(lp[2])])
        for i, p in enumerate(designs):
            # HARD requirement constraints override the designer's extents:
            # the shelf must reach the load, the wall leg must fit the room
            # and still hold its bolt column.
            p["W"] = min(p["W"], float(env_mm[0]) * 0.98)
            p["Lf"] = float(min(max(p["Lf"], arm_mm + 10.0),
                                float(env_mm[1]) * 0.98))
            min_wall = min(4.0 * p.get("hole_dia", 6.6) + 20.0, avail_h * 0.9)
            p["Lw"] = float(min(max(p["Lw"], min_wall), avail_h * 0.98))
            if p["Lw"] < 25.0 or p["Lf"] < 25.0:
                continue   # request leaves no room for a real bracket here
            try:
                f_mm, toks_c, meta = l_bracket.build_from_params(p)
            except Exception:  # noqa: BLE001
                continue
            f_norm = f_mm.scale(1.0 / u).transform(R, offset)
            toks_n = [t.scale(1.0 / u).transform(R, offset) for t in toks_c]
            gen0 = torch.Generator().manual_seed(0)
            probes = torch.randn(2048, 3, generator=gen0) * 0.45
            lat, peak = None, None
            do_rank = rank and self.model is not None
            if do_rank:
                # surrogate: encode the NEW geometry, predict stress to rank
                ss = stratified(f_norm, n=1536, seed=seed + i, device=self.device)
                cond_phys = [[t for t in tokens if t.type in ("load", "material")]
                             + [t for t in toks_n if t.type == "fixed_point"]]
                with self.lock, torch.no_grad():
                    lat = self.model.encode(ss.x.unsqueeze(0), ss.f.unsqueeze(0),
                                            ss.grad.unsqueeze(0),
                                            [toks_n + [load, mat]])
                    vm = self.model.decode(lat, probes.to(self.device).unsqueeze(0),
                                           "von_mises", cond_phys).squeeze().cpu()
            fn = (lambda ff: (lambda pts: ff(pts)))(f_norm)
            sdfv = f_norm(probes)
            inside = sdfv < 0
            if do_rank:
                peak = float(vm[inside].max()) if inside.any() else float(vm.max())
            vol = float(inside.float().mean())
            # authoritative token set for THIS geometry: the request's load /
            # material / envelope plus the holes actually drilled by the
            # builder. The request's auto-placed fixed_points are dropped —
            # verifying against holes that were never cut fails spuriously.
            tokens_geo = [t for t in tokens if t.type != "fixed_point"] + toks_n
            lid = self.store(lat, tokens_geo, field_fn=fn, field_obj=f_norm)
            out.append({"latent_id": lid, "volume": vol, "peak_vm_pred": peak,
                        "designed": True, "ranked": bool(do_rank),
                        "mass_g": vol * (4 / 3 * 3.14159 * 0.45 ** 3)
                            * (u * 1e-3) ** 3
                            * float(mat.params["density"]) * 1000.0,
                        "arm_mm": round(arm_mm, 1),
                        "params": {k: (round(v, 2) if isinstance(v, float) else v)
                                   for k, v in p.items()}})
        if out and out[0]["peak_vm_pred"] is not None:
            out.sort(key=lambda c: (c["peak_vm_pred"], c["volume"]))
        return out

    # -- helpers ---------------------------------------------------------------

    def store(self, lat: LatentSet, tokens: list[Token],
              field_fn=None, field_obj=None) -> str:
        lid = uuid.uuid4().hex[:12]
        self.latents[lid] = {"lat": lat, "tokens": tokens,
                             "field_fn": field_fn, "field_obj": field_obj}
        return lid

    def field_of(self, entry: dict):
        """The sdf callable for an entry: hybrid CSG composition if present,
        else the decoded latent field."""
        return entry["field_fn"] or decoded_field_fn(
            self.model, entry["lat"], self.device)

    # -- hybrid CSG scaffold ---------------------------------------------------

    def build_scaffold(self, tokens: list[Token], variant_seed: int = 0):
        """A parametric CANDIDATE VARIANT of the L-skeleton (normalized frame):
        sampled plate thicknesses, corner reinforcement (plain / triangular
        gusset / ribs), optional shelf lightweighting — plus the guaranteed
        wall/shelf plates, bolt holes and envelope clamp. Load magnitude
        biases the thickness draw (heavier -> beefier). Returns
        (base_solid, holes, env_box, variant_descriptor)."""
        from ...fields.primitives import Capsule, Halfspace
        gen = torch.Generator().manual_seed(variant_seed)

        def u(lo, hi):
            return lo + (hi - lo) * torch.rand((), generator=gen).item()

        env = next(t for t in tokens if t.type == "envelope")
        half = env.params["half_extents"]
        center = env.position
        load = next((t for t in tokens if t.type == "load"), None)
        load_n = float(load.params["magnitude"]) if load is not None else 300.0
        beef = min(1.6, max(0.7, (load_n / 400.0) ** 0.5))  # sizing bias
        t_wall = float(min(0.30 * half[1], max(0.04, u(0.10, 0.22) * half[1] * beef)))
        t_shelf = float(min(0.30 * half[1], max(0.04, u(0.10, 0.22) * half[1] * beef)))
        e3 = torch.eye(3)
        wall = Box([half[0] * 0.98, t_wall / 2, half[2] * 0.98]).transform(
            e3, center + torch.tensor([0.0, float(-half[1]) + t_wall / 2, 0.0]))
        z_top = float(load.position[2]) if load is not None else float(half[2]) * 0.8
        z_top = min(z_top, float(half[2]) - 0.01)
        shelf = Box([half[0] * 0.98, half[1] * 0.97, t_shelf / 2]).transform(
            e3, center + torch.tensor([0.0, 0.0, z_top - t_shelf / 2]))
        base = Union(wall, shelf)

        wall_y = float(-half[1]) + t_wall
        under_z = z_top - t_shelf
        reach_y = float(half[1]) * 0.9
        span_z = under_z - float(-half[2]) * 0.9
        kind = ["plain", "gusset", "ribs"][int(torch.randint(3, (), generator=gen))]
        desc = {"style": kind, "t_wall": round(t_wall, 3),
                "t_shelf": round(t_shelf, 3)}
        if kind == "gusset" and span_z > 0.08 and reach_y - wall_y > 0.08:
            a = u(0.35, 0.8) * (reach_y - wall_y)
            b = u(0.35, 0.8) * span_z
            gw = half[0] * (2 * 0.9 if u(0, 1) < 0.5 else u(0.25, 0.5))
            gy0, gz0 = wall_y, under_z
            gbox = Box([gw / 2, (a + 0.01) / 2, (b + 0.01) / 2]).transform(
                e3, center + torch.tensor([0.0, gy0 + a / 2, gz0 - b / 2]))
            nrm = torch.tensor([0.0, b, a])
            nrm = nrm / torch.linalg.vector_norm(nrm)
            d = float(nrm @ (center + torch.tensor([0.0, gy0 + a, gz0])))
            base = Union(base, Intersect(gbox, Halfspace(nrm, d)))
            desc["gusset"] = {"a": round(a, 3), "b": round(b, 3)}
        elif kind == "ribs" and span_z > 0.08:
            n_r = 1 if u(0, 1) < 0.6 else 2
            rr = u(0.3, 0.55) * min(t_wall, t_shelf)
            a = u(0.3, 0.7) * (reach_y - wall_y)
            b = u(0.3, 0.7) * span_z
            xs = [0.0] if n_r == 1 else [-float(half[0]) * 0.5, float(half[0]) * 0.5]
            for xi in xs:
                A = center + torch.tensor([xi, wall_y + a, under_z + rr * 0.2])
                B = center + torch.tensor([xi, wall_y - rr * 0.2, under_z - b])
                base = Union(base, Capsule(A, B, rr))
            desc["ribs"] = {"n": n_r, "r": round(rr, 3)}
        # optional shelf lightweighting holes
        if u(0, 1) < 0.4 and float(half[0]) > 0.08:
            r_c = min(0.05, float(half[0]) * 0.35)
            ys = torch.linspace(wall_y + 0.15, reach_y - 0.1, 3)
            R_id = torch.eye(3)
            for y_i in ys.tolist():
                cpos = center + torch.tensor([0.0, y_i, z_top - t_shelf / 2])
                base = Subtract(base, Cylinder(r_c, t_shelf).transform(R_id, cpos))
            desc["lightweight"] = True

        R_y = torch.tensor([[1.0, 0, 0], [0.0, 0, 1], [0.0, -1, 0]])
        holes = []
        for tok in tokens:
            if tok.type == "fixed_point" and tok.params.get("fixed"):
                holes.append(Cylinder(float(tok.params["diameter"]) / 2,
                                      t_wall * 2).transform(R_y, tok.position))
        env_box = Box(half * 1.0).transform(e3, center)
        return base, holes, env_box, desc

    def compose_hybrid(self, lat: LatentSet, tokens: list[Token],
                       variant_seed: int = 0):
        """final = Subtract(Intersect(Union(variant, generated), envelope), holes)."""
        base, holes, env_box, desc = self.build_scaffold(tokens, variant_seed)
        model, device, lock = self.model, self.device, self.lock

        def gen_sdf(pts: torch.Tensor) -> torch.Tensor:
            with lock, torch.no_grad():
                outs = []
                for i in range(0, pts.shape[0], 131072):
                    v = model.decode(lat, pts[i:i + 131072].unsqueeze(0).to(device),
                                     "sdf")
                    outs.append(v.squeeze(0).squeeze(-1).float())
                return torch.cat(outs).to(pts.device)

        def composed(pts: torch.Tensor) -> torch.Tensor:
            v = torch.minimum(base(pts), gen_sdf(pts))     # union w/ scaffold
            v = torch.maximum(v, env_box(pts))             # clamp to envelope
            for h in holes:                                # re-drill the bolts
                v = torch.maximum(v, -h(pts))
            return v

        return composed, desc

    def get(self, lid: str) -> dict:
        if lid not in self.latents:
            raise HTTPException(404, f"unknown latent_id {lid}")
        return self.latents[lid]

    def encode_record(self, seed_index: int) -> str:
        assert self.data is not None, "no dataset attached"
        rec = self.data[seed_index % len(self.data)]
        x = rec["x"].unsqueeze(0).to(self.device)
        f = rec["f"].unsqueeze(0).to(self.device)
        g = rec["grad"].unsqueeze(0).to(self.device)
        gen = torch.Generator().manual_seed(0)
        ii = torch.randperm(x.shape[1], generator=gen)[:self.cfg.get("n_input", 1536)]
        with self.lock, torch.no_grad():
            lat = self.model.encode(x[:, ii], f[:, ii], g[:, ii], [rec["tokens"]])
        return self.store(lat, rec["tokens"])

    def generate(self, tokens: list[Token], n: int, guidance_scale: float,
                 thresholds: dict, seed: int) -> list[str]:
        assert self.flow is not None, "flow model not loaded"
        guidance_fn = None
        if any(v is not None for v in thresholds.values()):
            guidance_fn = ConstraintGuidance(
                self.model, self.norm, tokens, thresholds,
                m_fine=self.flow.m_fine, latent_dim=self.flow.latent_dim,
                device=self.device, gamma=thresholds.get("gamma", 0.5) or 0.5)
        with self.lock, torch.no_grad():
            z = flow_sample(self.flow, [tokens] * n, n=n, steps=48,
                            guidance_scale=guidance_scale, device=self.device,
                            seed=seed, guidance_fn=guidance_fn)
            z = self.norm.denorm(z)
        ids = []
        for i in range(n):
            lat = LatentSet.unflatten(z[i:i + 1], self.flow.m_fine,
                                      self.flow.latent_dim)
            ids.append(self.store(lat, tokens))
        return ids

    def quick_scalars(self, lid: str) -> dict:
        """Cheap predicted scalars for candidate cards."""
        entry = self.get(lid)
        lat, tokens = entry["lat"], entry["tokens"]
        gen = torch.Generator().manual_seed(0)
        probes = (torch.randn(2048, 3, generator=gen) * 0.45).to(self.device)
        cond_phys = [[t for t in tokens
                      if t.type in ("fixed_point", "load", "material")]]
        with self.lock, torch.no_grad():
            sdf = self.model.decode(lat, probes.unsqueeze(0), "sdf").squeeze()
            out = {"volume": float((sdf < 0).float().mean())}
            if cond_phys[0]:
                vm = self.model.decode(lat, probes.unsqueeze(0), "von_mises",
                                       cond_phys).squeeze()
                inside = sdf < 0
                out["peak_vm_pred"] = float(vm[inside].max()) if inside.any() \
                    else float(vm.max())
        mat = next((t for t in tokens if t.type == "material"), None)
        scale_tok = next((t for t in tokens if t.type == "scale"), None)
        if mat is not None and scale_tok is not None:
            unit_m = float(scale_tok.params["unit_mm"]) * 1e-3
            # volume fraction of the probe ball -> m^3
            vol_m3 = out["volume"] * (4 / 3 * 3.14159 * 0.45 ** 3) * unit_m ** 3
            out["mass_kg"] = vol_m3 * float(mat.params["density"])
        return out


ENGINE: Engine | None = None
app = FastAPI(title="GeoField demo")


class EncodeReq(BaseModel):
    seed_index: int = 0


class GenerateReq(BaseModel):
    tokens: list[dict]
    n: int = 4
    guidance_scale: float = 2.0
    seed: int = 0
    thresholds: dict = {}
    hybrid: bool = True   # CSG scaffold (guaranteed plates+holes) + model detail
    mode: str = "designer"  # 'designer' (learned params) | 'hybrid' | 'flow'
    rank: bool = True     # False = fast live preview (skip surrogate ranking)


class QueryReq(BaseModel):
    latent_id: str
    field_id: str = "sdf"
    tokens: list[dict] | None = None
    resolution: int = 64


class VerifyReq(BaseModel):
    latent_id: str
    tokens: list[dict] | None = None
    thresholds: dict = {}


class ExportReq(BaseModel):
    latent_id: str
    resolution: int = 192


@app.get("/health")
def health():
    return {"ok": True,
            "designer": ENGINE.designer is not None,
            "geometric_model": ENGINE.model is not None,
            "flow_loaded": ENGINE.flow is not None,
            "can_predict_fields": ENGINE.model is not None,
            "latents_cached": len(ENGINE.latents)}


class DesignReq(BaseModel):
    """User-facing design request in REAL units (mm / N)."""
    envelope_mm: list[float]          # [width_x, depth_y, height_z] of the box
    load_pos_frac: list[float]        # [0..1]^3 position within the envelope
    load_dir: list[float]             # direction of the force
    load_N: float = 500.0
    material: str = "Al6061"
    n_holes: int = 3


@app.post("/make_tokens")
def make_tokens(req: DesignReq):
    """Build condition tokens from a designer's box + load case, mirroring the
    training convention: wall = the y=0 face of the envelope, tool from +y,
    mounting holes auto-placed on the wall face, geometry normalized to the
    unit sphere with a scale token carrying mm-per-unit."""
    import math
    w, d, h = req.envelope_mm
    # normalization: envelope corner-to-corner diagonal fits radius 0.85
    diag = math.sqrt(w * w + d * d + h * h)
    unit_mm = diag / (2 * 0.85)
    half = torch.tensor([w, d, h]) / 2.0 / unit_mm
    center = torch.zeros(3)   # normalized frame is centered on the envelope
    toks: list[Token] = []
    # auto mounting holes: column on the wall face (y = -half.y), spread in z
    dia_mm = max(4.5, min(11.0, 0.08 * min(w, h)))
    zs = torch.linspace(-0.6 * half[2], 0.6 * half[2], max(req.n_holes, 2))
    for z_i in zs.tolist():
        pos = torch.tensor([0.0, -float(half[1]) * 0.9, z_i])
        toks.append(Token("fixed_point",
                          torch.cat([pos, torch.tensor([0.0, 0, 0, 1])]),
                          {"axis": torch.tensor([0.0, 1.0, 0.0]),
                           "diameter": dia_mm / unit_mm, "fixed": True}))
    lp = (torch.tensor(req.load_pos_frac) - 0.5) * 2 * half
    toks.append(Token("load", torch.cat([lp, torch.tensor([0.0, 0, 0, 1])]),
                      {"direction": torch.tensor(req.load_dir, dtype=torch.float32),
                       "magnitude": req.load_N, "kind": "force",
                       "grip": max(6.0, 0.12 * min(w, d)) / unit_mm}))
    toks.append(material_token(req.material))
    toks.append(Token("envelope", torch.cat([center, torch.tensor([0.0, 0, 0, 1])]),
                      {"half_extents": half}))
    toks.append(Token("spindle_dir", None, {"direction": torch.tensor([0.0, 1.0, 0.0])}))
    toks.append(Token("scale", None, {"unit_mm": unit_mm}))
    return {"tokens": [t.to_dict() for t in toks], "unit_mm": unit_mm,
            "half_extents_norm": half.tolist()}


@app.post("/encode")
def encode(req: EncodeReq):
    if ENGINE.model is None or ENGINE.data is None:
        raise HTTPException(503, "needs --stage_b and --data")
    lid = ENGINE.encode_record(req.seed_index)
    toks = [t.to_dict() for t in ENGINE.latents[lid]["tokens"]]
    return {"latent_id": lid, "tokens": toks}


class MeshReq(BaseModel):
    latent_id: str
    resolution: int = 96


@app.post("/mesh")
def mesh(req: MeshReq):
    """Marching-cubes mesh as JSON, refined with the field's unit-gradient
    property: vertices are Newton-projected onto the exact zero level set and
    normals come from the analytic gradient — so a modest grid gives clean
    faces and crisp CSG edges instead of staircase artifacts."""
    entry = ENGINE.get(req.latent_id)
    res = min(max(req.resolution, 48), 224)
    fobj = entry.get("field_obj")
    if entry.get("field_fn"):  # composed/exact field, locks internally
        grid = field_to_grid(entry["field_fn"], res=res, device="cpu")
    else:
        with ENGINE.lock, torch.no_grad():
            grid = field_to_grid(
                decoded_field_fn(ENGINE.model, entry["lat"], ENGINE.device),
                res=res, device=ENGINE.device)
    verts, faces = grid_to_mesh(grid)
    if len(faces) == 0:
        raise HTTPException(422, "empty solid")
    normals = None
    if fobj is not None:            # exact field available -> refine + normals
        verts = refine_vertices(verts, fobj, iters=3)
        normals = analytic_normals(verts, fobj)
    out = {"verts": verts.round(5).tolist(), "faces": faces.tolist()}
    if normals is not None:
        out["normals"] = normals.round(4).tolist()
    return out


class PointsReq(BaseModel):
    latent_id: str
    field_id: str
    tokens: list[dict] | None = None
    points: list[list[float]]


@app.post("/query_points")
def query_points(req: PointsReq):
    """Field values at arbitrary points (used to paint mesh vertices)."""
    if ENGINE.model is None:
        raise HTTPException(503, "no geometric model loaded — field prediction "
                                 "needs --stage_b; geometry/verify/export work")
    entry = ENGINE.ensure_latent(req.latent_id)
    cond = [[Token.from_dict(d) for d in req.tokens]] if req.tokens else \
        ([entry["tokens"]] if entry["tokens"] else None)
    pts = torch.tensor(req.points, dtype=torch.float32, device=ENGINE.device)
    with ENGINE.lock, torch.no_grad():
        vals = []
        for i in range(0, pts.shape[0], 65536):
            v = ENGINE.model.decode(entry["lat"], pts[i:i + 65536].unsqueeze(0),
                                    req.field_id, cond)
            v = v.squeeze(0)
            if v.shape[-1] == 3:  # vector field -> magnitude
                v = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
            vals.append(v.squeeze(-1).float().cpu())
    v = torch.cat(vals)
    # range diagnostics so the UI can say "this head is flat" instead of
    # silently rendering one uniform colour
    q = torch.quantile(v, torch.tensor([0.02, 0.5, 0.98]))
    spread = float((q[2] - q[0]) / (q[1].abs() + 1e-9))
    return {"values": v.tolist(),
            "stats": {"min": float(v.min()), "max": float(v.max()),
                      "p2": float(q[0]), "p50": float(q[1]), "p98": float(q[2]),
                      "rel_spread": spread, "flat": spread < 0.02}}


@app.post("/generate")
def generate(req: GenerateReq):
    tokens = [Token.from_dict(d) for d in req.tokens]
    if req.mode == "designer" and ENGINE.designer is not None:
        kept = ENGINE.design_candidates(tokens, req.n, req.seed, rank=req.rank)
        return {"candidates": kept, "resample_rounds": 1, "note": "designer"}
    if req.hybrid and req.mode != "flow":
        # Parametric variant scaffolds (different thickness/gusset/rib/
        # lightweighting draws) guarantee n DIFFERENT valid brackets; the
        # generator's latent unions extra material on top; the MODEL ranks
        # candidates by predicted peak stress and material volume.
        ids = ENGINE.generate(tokens, req.n, req.guidance_scale,
                              req.thresholds, req.seed)
        kept = []
        gen0 = torch.Generator().manual_seed(0)
        probes = torch.randn(2048, 3, generator=gen0) * 0.45
        cond_phys = [[t for t in tokens
                      if t.type in ("fixed_point", "load", "material")]]
        for i, lid in enumerate(ids):
            entry = ENGINE.get(lid)
            ffn, desc = ENGINE.compose_hybrid(entry["lat"], tokens,
                                              variant_seed=req.seed * 31 + i)
            entry["field_fn"] = ffn
            sdfv = ffn(probes)
            vol = float((sdfv < 0).float().mean())
            with ENGINE.lock, torch.no_grad():
                vm = ENGINE.model.decode(entry["lat"],
                                         probes.unsqueeze(0).to(ENGINE.device),
                                         "von_mises", cond_phys).squeeze().cpu()
            inside = sdfv < 0
            peak = float(vm[inside].max()) if inside.any() else float(vm.max())
            kept.append({"latent_id": lid, "volume": vol, "hybrid": True,
                         "variant": desc, "peak_vm_pred": peak})
        # rank: prefer low predicted stress, break ties with low material use
        kept.sort(key=lambda c: (c["peak_vm_pred"], c["volume"]))
        return {"candidates": kept, "resample_rounds": 1, "note": "hybrid"}
    # pure-generative path (rejection sampling; pilot generator drops empties)
    kept = []
    attempts = 0
    seed = req.seed
    while len(kept) < req.n and attempts < 6:
        ids = ENGINE.generate(tokens, req.n, req.guidance_scale,
                              req.thresholds, seed)
        for lid in ids:
            sc = ENGINE.quick_scalars(lid)
            if sc.get("volume", 0.0) >= 0.002:
                kept.append({"latent_id": lid, **sc})
            else:
                ENGINE.latents.pop(lid, None)
            if len(kept) >= req.n:
                break
        attempts += 1
        seed += 1000
    return {"candidates": kept, "resample_rounds": attempts,
            "note": "" if kept else "generator produced no valid solids"}


@app.post("/query")
def query(req: QueryReq):
    entry = ENGINE.get(req.latent_id)
    lat = entry["lat"]
    cond = [[Token.from_dict(d) for d in req.tokens]] if req.tokens else None
    res = min(max(req.resolution, 16), 160)
    lin = torch.linspace(-1.05, 1.05, res, device=ENGINE.device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    with ENGINE.lock, torch.no_grad():
        vals = []
        for i in range(0, pts.shape[0], 131072):
            v = ENGINE.model.decode(lat, pts[i:i + 131072].unsqueeze(0),
                                    req.field_id, cond)
            vals.append(v.squeeze(0).squeeze(-1).float().cpu())
        v = torch.cat(vals)
    return {"dims": [res, res, res], "lo": -1.05, "hi": 1.05,
            "values_b64": base64.b64encode(v.numpy().tobytes()).decode()}


@app.post("/verify")
def verify(req: VerifyReq):
    entry = ENGINE.get(req.latent_id)
    # a designed/composed candidate carries its own self-consistent tokens
    # (real hole positions); only the neural-latent path trusts req.tokens
    tokens = (entry["tokens"] if entry.get("field_obj") is not None
              else ([Token.from_dict(d) for d in req.tokens]
                    if req.tokens else entry["tokens"]))
    th = {"sigma_max": req.thresholds.get("sigma_max"),
          "a_max": req.thresholds.get("a_max"),
          "t_min": req.thresholds.get("t_min")}
    ffn = entry.get("field_fn")
    rep: dict = {"thresholds": th, "hybrid": bool(ffn)}
    if ffn is not None:  # composed field locks internally; no outer lock
        rep["mfg"] = verify_mfg(ENGINE.model, entry["lat"], tokens,
                                a_max=th["a_max"], t_min=th["t_min"],
                                field_fn=ffn)
        if th["sigma_max"] and any(t.type == "load" for t in tokens):
            rep["physics"] = verify_physics(ENGINE.model, entry["lat"], tokens,
                                            th["sigma_max"], device="cpu",
                                            field_fn=ffn)
    else:
        if ENGINE.model is None:
            raise HTTPException(503, "verifying a neural-decoded latent needs "
                                     "--stage_b (designed candidates verify fine)")
        with ENGINE.lock:
            rep["mfg"] = verify_mfg(ENGINE.model, entry["lat"], tokens,
                                    a_max=th["a_max"], t_min=th["t_min"],
                                    device=ENGINE.device)
            if th["sigma_max"] and any(t.type == "load" for t in tokens):
                rep["physics"] = verify_physics(ENGINE.model, entry["lat"],
                                                tokens, th["sigma_max"],
                                                device=ENGINE.device)
    rep["pass"] = rep["mfg"]["pass"] and rep.get("physics", {}).get("pass", True)
    return json.loads(json.dumps(rep, default=float))


@app.post("/export")
def export(req: ExportReq):
    entry = ENGINE.get(req.latent_id)
    res = min(max(req.resolution, 64), 256)
    if entry.get("field_fn"):
        grid = field_to_grid(entry["field_fn"], res=res, device="cpu")
    else:
        with ENGINE.lock, torch.no_grad():
            grid = field_to_grid(
                decoded_field_fn(ENGINE.model, entry["lat"], ENGINE.device),
                res=res, device=ENGINE.device)
    verts, faces = grid_to_mesh(grid)
    if len(faces) == 0:
        raise HTTPException(422, "empty solid; nothing to export")
    if entry.get("field_obj") is not None:
        verts = refine_vertices(verts, entry["field_obj"], iters=3)
    buf = io.BytesIO()
    # mesh_to_stl writes to a path; reuse its triangle packing inline
    tri = verts[faces]
    import numpy as np
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)
    buf.write(b"\0" * 80)
    buf.write(struct.pack("<I", len(faces)))
    for i in range(len(faces)):
        buf.write(struct.pack("<3f", *n[i]))
        for v in tri[i]:
            buf.write(struct.pack("<3f", *v))
        buf.write(struct.pack("<H", 0))
    return Response(content=buf.getvalue(), media_type="model/stl",
                    headers={"Content-Disposition":
                             f"attachment; filename=bracket_{req.latent_id}.stl"})


@app.get("/")
def index():
    fe = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if fe.exists():
        return HTMLResponse(fe.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>GeoField backend up</h1><p>frontend not built yet;"
                        " see /docs for the API.</p>")


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_b", default=None,
                    help="geometric-model checkpoint; omit to run the "
                         "design + exact-geometry + FEA-verify demo only")
    ap.add_argument("--flow", default=None)
    ap.add_argument("--latent_stats", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--designer", default=None)
    ap.add_argument("--linear_heads", action="store_true",
                    help="serving a pre-log-space checkpoint")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    ENGINE = Engine(args.stage_b, args.flow, args.latent_stats, args.data,
                    args.device, designer=args.designer,
                    linear_heads=args.linear_heads)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
