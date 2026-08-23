"""Linear probes on pooled() for physics/manufacturability scalars.

The shared-latent claim: a GEOMETRY-trained latent should already contain
(linearly decodable) physics and manufacturability information, and Stage-B
co-training should sharpen it. Probes are ridge regressions at
N in {50, 100, 500, 2000} training shapes, evaluated by Spearman rank
correlation on held-out shapes, run on (a) the Stage-B model, (b) the
Stage-A geometry-only checkpoint, (c) the baseline.

`cross_direction_probe` trains the support_fraction probe under one
build-direction distribution and evaluates under another (the latent, not the
probe, must carry the geometry that makes support predictable).
"""
from __future__ import annotations

import torch

PROBE_TARGETS = ["worst_peak_vm", "inaccessible_fraction",
                 "min_thickness", "mass"]
PROBE_NS = [20, 50, 100, 300]  # sized for the 500-geometry pilot's val pool


def _scalar_target(name: str, scalars: dict) -> float:
    """Derive probe targets from the per-record scalar dict."""
    if name == "worst_peak_vm":
        peaks = [v for k, v in scalars.items() if k.startswith("peak_vm|")]
        return max(peaks) if peaks else float("nan")
    if name == "mass":
        masses = [v for k, v in scalars.items() if k.startswith("mass|")]
        return masses[0] if masses else float("nan")
    return float(scalars.get(name, float("nan")))


@torch.no_grad()
def collect_pooled(model, records: list[dict], device: str = "cpu",
                   n_input: int = 1024, seed: int = 0) -> tuple[torch.Tensor, dict]:
    """Pooled latents [N,d] + scalar targets {name: [N]} (nan if absent)."""
    gen = torch.Generator().manual_seed(seed)
    zs, targets = [], {k: [] for k in PROBE_TARGETS}
    for rec in records:
        x = rec["x"].unsqueeze(0).to(device)
        f = rec["f"].unsqueeze(0).to(device)
        g = rec["grad"].unsqueeze(0).to(device)
        ii = torch.randperm(x.shape[1], generator=gen)[:n_input]
        zs.append(model.encode(x[:, ii], f[:, ii], g[:, ii]).pooled().squeeze(0).cpu())
        for k in PROBE_TARGETS:
            targets[k].append(_scalar_target(k, rec["scalars"]))
    return torch.stack(zs), {k: torch.tensor(v) for k, v in targets.items()}


def ridge_fit_predict(z_tr: torch.Tensor, y_tr: torch.Tensor,
                      z_te: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    zc = torch.cat([z_tr, torch.ones(len(z_tr), 1)], dim=1)
    A = zc.T @ zc + lam * torch.eye(zc.shape[1])
    w = torch.linalg.solve(A, zc.T @ y_tr.unsqueeze(-1))
    return (torch.cat([z_te, torch.ones(len(z_te), 1)], dim=1) @ w).squeeze(-1)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std().clamp_min(1e-9)
    rb = (rb - rb.mean()) / rb.std().clamp_min(1e-9)
    return float((ra * rb).mean())


def probe_panel(z: torch.Tensor, targets: dict, ns=None, test_frac: float = 0.3,
                seed: int = 0) -> dict:
    """{target: {N: spearman}} with a fixed held-out test split."""
    ns = ns or PROBE_NS
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, y in targets.items():
        ok = ~torch.isnan(y)
        zk, yk = z[ok], y[ok]
        # log-scale wide-ranged targets for a better-conditioned probe
        if "vm" in name or name == "mass":
            yk = yk.clamp_min(1e-9).log()
        perm = torch.randperm(len(zk), generator=gen)
        n_te = max(int(len(zk) * test_frac), 10)
        te, tr = perm[:n_te], perm[n_te:]
        out[name] = {}
        for n in ns:
            if n > len(tr):
                continue
            pred = ridge_fit_predict(zk[tr[:n]], yk[tr[:n]], zk[te])
            out[name][n] = spearman(pred, yk[te])
    return out
