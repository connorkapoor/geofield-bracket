"""Per-sample verification reports and aggregate constraint satisfaction.

`verify_sample` -> JSON-able dict {predicted, verified, pass/fail, mass}.
`satisfaction_run` -> rate over n samples per condition set (Phase-4
acceptance: >= 70% across 10 held-out condition sets).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..labels import scalars as scal
from ..tokens.schema import Token, tokens_to_json
from .fea import _CallableField, decoded_field_fn, verify_physics
from .mfg import verify_mfg


@torch.no_grad()
def verify_sample(model, lat, tokens: list[Token], thresholds: dict,
                  device: str = "cpu", seed: int = 0) -> dict:
    """thresholds: {sigma_max, s_max?, a_max?, t_min?} (None = unconstrained)."""
    rep: dict = {"tokens": json.loads(tokens_to_json(tokens)),
                 "thresholds": thresholds}
    has_load = any(t.type == "load" for t in tokens)
    if has_load and thresholds.get("sigma_max") is not None:
        rep["physics"] = verify_physics(model, lat, tokens,
                                        thresholds["sigma_max"],
                                        device=device, seed=seed)
    rep["mfg"] = verify_mfg(model, lat, tokens,
                            s_max=thresholds.get("s_max"),
                            a_max=thresholds.get("a_max"),
                            t_min=thresholds.get("t_min"),
                            device=device, seed=seed)
    field = _CallableField(decoded_field_fn(model, lat, device))
    density = next((float(t.params["density"]) for t in tokens
                    if t.type == "material"), 1000.0)
    rep["mass"] = scal.mass(field, density, seed=seed, device=device)
    rep["pass"] = rep["mfg"]["pass"] and rep.get("physics", {}).get("pass", True)
    return rep


def satisfaction_run(reports: list[dict], out_path: str | Path | None = None) -> dict:
    agg = {
        "n": len(reports),
        "satisfaction_rate": sum(r["pass"] for r in reports) / max(len(reports), 1),
        "physics_pass_rate": sum(r.get("physics", {}).get("pass", True)
                                 for r in reports) / max(len(reports), 1),
        "mfg_pass_rate": sum(r["mfg"]["pass"] for r in reports) / max(len(reports), 1),
        "mass_mean": sum(r["mass"] for r in reports) / max(len(reports), 1),
    }
    if out_path:
        Path(out_path).write_text(json.dumps(
            {"aggregate": agg, "samples": reports}, indent=2))
    return agg
