"""Learned parameter-designer: requirements -> bracket program parameters.

The generative-geometry route failed for a structural reason (the analysis
latent space is non-convex between designs — verified by interpolation
tests), so generation moves to the space that IS well-behaved: the bracket
program's own ~15 parameters. This model learns

    (envelope, load position/direction/magnitude, material)
        -> (leg lengths, thicknesses, reinforcement type + dims,
            lightweighting, hole count/size)

from dataset pairs where the bracket's calibrated stress utilization was in
the sensible design band (0.30-0.95 x yield) — i.e. pairs where this bracket
is a genuinely adequate, efficient answer to that load. Candidate variety
comes from Monte-Carlo dropout sampling; the surrogate ranks; FEA verifies.

All inputs/outputs are canonical-frame invariant scalars, so no equivariance
machinery is needed here — the frame is re-attached at build time.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

# continuous parameter slots (all normalized to [0,1] by the ranges below)
CONT = ["Lw", "Lf", "W", "tw", "tf", "fillet", "gusset_a", "gusset_b",
        "ribs_r", "hole_dia"]
RANGES = {"Lw": (75.0, 300.0), "Lf": (75.0, 300.0), "W": (25.0, 75.0),
          "tw": (5.0, 12.0), "tf": (5.0, 12.0), "fillet": (3.0, 12.0),
          "gusset_a": (0.0, 1.0), "gusset_b": (0.0, 1.0),
          "ribs_r": (0.3, 0.6), "hole_dia": (4.5, 11.0)}
REINF = ["none", "gusset", "ribs"]
LIGHT = ["none", "holes", "slots"]
N_FEAT = 15


def featurize(env_wdh_mm: Tensor, load_pos_frac: Tensor, load_dir: Tensor,
              load_n: float, e_pa: float, yield_pa: float,
              arm_mm: float | None = None) -> Tensor:
    """Invariant requirement features -> [N_FEAT].

    Includes explicit ENGINEERING features, not just raw geometry: the moment
    arm (distance from the wall to the load), the bending moment F*arm, and
    the required section modulus M/sigma_yield. Those are what actually set
    plate thickness in hand calculations, so handing them to the network
    directly is far more sample-efficient than making it rediscover the
    product of two of its own inputs.
    """
    arm = float(arm_mm) if arm_mm is not None else \
        float(load_pos_frac[1]) * float(env_wdh_mm[1])
    arm = max(arm, 1.0)
    moment = max(load_n, 1.0) * arm                       # N*mm
    sec_mod = moment / max(yield_pa * 1e-6, 1.0)          # mm^3 (sigma in MPa)
    return torch.tensor([
        env_wdh_mm[0] / 300.0, env_wdh_mm[1] / 300.0, env_wdh_mm[2] / 300.0,
        float(load_pos_frac[1]), float(load_pos_frac[2]),
        float(load_dir[0]), float(load_dir[1]), float(load_dir[2]),
        torch.log10(torch.tensor(max(load_n, 1.0))).item() / 4.0,
        torch.log10(torch.tensor(e_pa)).item() / 12.0,
        torch.log10(torch.tensor(yield_pa)).item() / 10.0,
        arm / 300.0,
        torch.log10(torch.tensor(moment)).item() / 6.0,
        torch.log10(torch.tensor(max(sec_mod, 1e-3))).item() / 4.0,
        1.0,
    ], dtype=torch.float32)


class ParamDesigner(nn.Module):
    def __init__(self, hidden: int = 192, p_drop: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEAT, hidden), nn.SiLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(p_drop))
        self.cont = nn.Linear(hidden, len(CONT))       # sigmoid -> [0,1]
        self.reinf = nn.Linear(hidden, len(REINF))
        self.light = nn.Linear(hidden, len(LIGHT))
        self.n_holes = nn.Linear(hidden, 3)            # 2/3/4

    def forward(self, x: Tensor):
        h = self.net(x)
        return (torch.sigmoid(self.cont(h)), self.reinf(h),
                self.light(h), self.n_holes(h))

    @torch.no_grad()
    def design(self, feats: Tensor, n: int = 4, seed: int = 0) -> list[dict]:
        """n candidate parameter sets via MC-dropout sampling (dropout stays
        ON) — dispersion reflects genuine design ambiguity in the data."""
        torch.manual_seed(seed)
        self.train()  # keep dropout active for sampling
        outs = []
        for i in range(n):
            c, r, l, k = self.forward(feats.unsqueeze(0))
            p = {name: RANGES[name][0] + float(c[0, j]) *
                 (RANGES[name][1] - RANGES[name][0])
                 for j, name in enumerate(CONT)}
            # first sample = argmax (the designer's best guess); later samples
            # draw from the softened class distributions for variety
            if i == 0:
                p["reinforcement"] = REINF[int(r.argmax())]
                p["lightweight"] = LIGHT[int(l.argmax())]
                p["n_holes"] = 2 + int(k.argmax())
            else:
                p["reinforcement"] = REINF[int(torch.multinomial(
                    torch.softmax(r[0] / 0.8, -1), 1))]
                p["lightweight"] = LIGHT[int(torch.multinomial(
                    torch.softmax(l[0] / 0.8, -1), 1))]
                p["n_holes"] = 2 + int(torch.multinomial(
                    torch.softmax(k[0] / 0.8, -1), 1))
            p["ribs_n"] = 1 + (i % 2)
            p["gusset_full"] = True
            outs.append(p)
        self.eval()
        return outs


def designer_loss(pred, target_cont: Tensor, target_reinf: Tensor,
                  target_light: Tensor, target_holes: Tensor,
                  mask: Tensor | None = None) -> Tensor:
    """mask [B, len(CONT)]: 0 where a continuous target is meaningless for
    that record (e.g. gusset dimensions on an unbraced bracket)."""
    c, r, l, k = pred
    if mask is None:
        mse = nn.functional.mse_loss(c, target_cont)
    else:
        se = (c - target_cont) ** 2 * mask
        mse = se.sum() / mask.sum().clamp_min(1.0)
    ce = (nn.functional.cross_entropy(r, target_reinf)
          + nn.functional.cross_entropy(l, target_light)
          + nn.functional.cross_entropy(k, target_holes))
    return mse * 4.0 + ce
