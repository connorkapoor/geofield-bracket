"""Sampling-time constraint guidance for the flow generator.

At each ODE step the current endpoint estimate z1_hat is decoded through the
FROZEN field heads on a fixed probe set and differentiable constraint losses
are computed; their gradient w.r.t. the latent is subtracted from the
velocity. Schedule: gamma = 0 for t < t_on (let the flow settle the topology
first), then linear ramp to the user-set strength.

Constraints (all optional, per condition set):
  peak(von_mises) <= sigma_max                       hinge on softmax-peak
  mean(support_need) <= s_max        (additive)      hinge on mean prob
  mean(1 - tool_access) <= a_max     (machined)      hinge
  sdf < 0 on a ring around each fixed_point hole     material around holes
  sdf > 0 at hole centers                            holes stay open
  sdf >= 0 outside the envelope box                  stay inside envelope
"""
from __future__ import annotations

import torch
from torch import Tensor

from .latent import LatentSet
from ..tokens.schema import Token


def _hole_probes(tokens: list[Token], n_ring: int = 8) -> tuple[Tensor, Tensor]:
    """(ring points [K,3] that must be SOLID, centers [H,3] that must be VOID)."""
    rings, centers = [], []
    for tok in tokens:
        if tok.type != "fixed_point" or tok.position is None:
            continue
        c = tok.position
        centers.append(c)
        axis = tok.params["axis"]
        a = torch.tensor([1.0, 0, 0])
        if abs(float(axis @ a)) > 0.9:
            a = torch.tensor([0.0, 1, 0])
        u = torch.linalg.cross(axis, a)
        u = u / torch.linalg.vector_norm(u).clamp_min(1e-9)
        v = torch.linalg.cross(axis, u)
        ang = torch.linspace(0, 6.2832, n_ring)
        rings.append(c + 0.75 * float(tok.params["diameter"]) *
                     (torch.cos(ang).unsqueeze(-1) * u
                      + torch.sin(ang).unsqueeze(-1) * v))
    ring = torch.cat(rings) if rings else torch.zeros(0, 3)
    ctr = torch.stack(centers) if centers else torch.zeros(0, 3)
    return ring, ctr


class ConstraintGuidance:
    """Callable guidance_fn(z1_hat_normalized, t) -> velocity correction."""

    def __init__(self, model, normalizer, tokens: list[Token], thresholds: dict,
                 m_fine: int, latent_dim: int, device: str = "cpu",
                 gamma: float = 1.0, t_on: float = 0.3, n_probe: int = 2048,
                 seed: int = 0):
        self.model, self.norm = model, normalizer
        self.tokens = tokens
        self.th = thresholds
        self.m_fine, self.latent_dim = m_fine, latent_dim
        self.device, self.gamma, self.t_on = device, gamma, t_on
        gen = torch.Generator().manual_seed(seed)
        self.probes = (torch.randn(n_probe, 3, generator=gen) * 0.45).to(device)
        self.ring, self.centers = _hole_probes(tokens)
        self.ring = self.ring.to(device)
        self.centers = self.centers.to(device)
        self.cond_phys = [[t for t in tokens
                           if t.type in ("fixed_point", "load", "material")]]
        self.cond_build = [[t for t in tokens if t.type == "build_dir"]]
        self.cond_spindle = [[t for t in tokens if t.type == "spindle_dir"]]
        self.envelope = next((t for t in tokens if t.type == "envelope"), None)

    def losses(self, lat: LatentSet) -> Tensor:
        model, th = self.model, self.th
        x = self.probes.unsqueeze(0)
        total = x.new_zeros(())
        sdf = model.decode(lat, x, "sdf").squeeze(0).squeeze(-1)

        if th.get("sigma_max") is not None and self.cond_phys[0]:
            vm = model.decode(lat, x, "von_mises", self.cond_phys).squeeze()
            inside = torch.sigmoid(-sdf / 0.02).detach()
            peak = ((vm * inside).logsumexp(dim=0))  # soft peak over probes
            total = total + torch.relu(peak - th["sigma_max"]) / max(th["sigma_max"], 1e-9)
        if th.get("s_max") is not None and self.cond_build[0]:
            sn = torch.sigmoid(model.decode(lat, x, "support_need",
                                            self.cond_build).squeeze())
            band = (sdf.abs() < 0.05).float().detach()
            frac = (sn * band).sum() / band.sum().clamp_min(1.0)
            total = total + torch.relu(frac - th["s_max"])
        if th.get("a_max") is not None and self.cond_spindle[0]:
            acc = torch.sigmoid(model.decode(lat, x, "tool_access",
                                             self.cond_spindle).squeeze())
            band = (sdf.abs() < 0.05).float().detach()
            frac = ((1 - acc) * band).sum() / band.sum().clamp_min(1.0)
            total = total + torch.relu(frac - th["a_max"])

        if self.ring.numel():
            ring_sdf = model.decode(lat, self.ring.unsqueeze(0), "sdf").squeeze()
            total = total + torch.relu(ring_sdf + 0.01).mean()      # want solid
        if self.centers.numel():
            ctr_sdf = model.decode(lat, self.centers.unsqueeze(0), "sdf").squeeze(-1)
            total = total + torch.relu(0.01 - ctr_sdf).mean()       # want void
        if self.envelope is not None:
            he = self.envelope.params["half_extents"].to(self.device)
            p = self.envelope.position.to(self.device)
            outside = ((self.probes - p).abs() > he).any(dim=-1)
            if outside.any():
                total = total + torch.relu(-sdf[outside]).mean()    # no material
        return total

    def __call__(self, z1_hat: Tensor, t: float) -> Tensor:
        if t < self.t_on or self.gamma == 0:
            return torch.zeros_like(z1_hat)
        ramp = min((t - self.t_on) / max(1 - self.t_on, 1e-6), 1.0)
        with torch.enable_grad():
            z = z1_hat.detach().requires_grad_(True)
            lat = LatentSet.unflatten(self.norm.denorm(z), self.m_fine,
                                      self.latent_dim)
            loss = sum(self.losses(LatentSet(
                fine=_index(lat.fine, b), coarse=_index(lat.coarse, b)))
                for b in range(z.shape[0]))
            (grad,) = torch.autograd.grad(loss, z)
        return -self.gamma * ramp * grad


def _index(state, b: int):
    from .encoder import TokenState
    return TokenState(pos=state.pos[b:b + 1], h=state.h[b:b + 1],
                      v=state.v[b:b + 1], fval=state.fval[b:b + 1],
                      global_mask=state.global_mask[b:b + 1])
