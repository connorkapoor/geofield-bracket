"""Conditional flow matching over the latent set.

Target: z = LatentSet.flatten() in R^{(M_f+M_c) x (3 + d + 3C)} — poses,
contexts, vector channels. Poses/vector channels are geometric (rotate with
the condition tokens); contexts are invariant. The velocity field is built
from the SAME equivariant attention as the encoder, so generation is
SE(3)-equivariant w.r.t. condition tokens: rotate the fixed points/loads and
the generated bracket rotates with them.

Objective: linear-interpolant CFM, E || v_theta(z_t, t, cond) - (z1 - z0) ||^2
with z0 ~ N(0, I) over normalized contexts and poses.

Permutation handling: latent elements are a set; each (noise, data) pair is
aligned with optimal-transport matching on poses (Hungarian, per sample,
fine and coarse blocks separately) to reduce multimodality.

Classifier-free guidance: condition tokens dropped with p=0.15 at train time;
`sample(..., guidance_scale)` mixes conditional/unconditional velocities.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .encoder import AttentionBlock, N_VEC, TokenEmbedder, TokenState
from .latent import LatentSet
from ..tokens.schema import Token


class LatentNormalizer:
    """Per-dimension normalization of flattened latents (fit on train set)."""

    def __init__(self, mean: Tensor, std: Tensor):
        self.mean, self.std = mean, std

    @staticmethod
    def fit(z: Tensor) -> "LatentNormalizer":
        return LatentNormalizer(z.mean(dim=(0, 1)), z.std(dim=(0, 1)).clamp_min(1e-4))

    def norm(self, z: Tensor) -> Tensor:
        return (z - self.mean.to(z)) / self.std.to(z)

    def denorm(self, z: Tensor) -> Tensor:
        return z * self.std.to(z) + self.mean.to(z)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    @staticmethod
    def load(sd) -> "LatentNormalizer":
        return LatentNormalizer(sd["mean"], sd["std"])


def ot_pair(z0: Tensor, z1: Tensor, m_fine: int,
            gen: torch.Generator | None = None) -> Tensor:
    """Reorder noise z0 toward data z1 by RANK PAIRING on poses: sort both
    sets along a random direction and pair by rank, fine and coarse blocks
    independently. An O(n log n), fully-batched approximation of optimal
    transport — the exact Hungarian was ~1.75 s/batch (90% of the step) with
    pathological worst cases that stalled training; for CFM the pairing only
    reduces target variance, so approximate transport is sufficient."""
    B, M, _ = z0.shape
    d = torch.randn(3, generator=gen).to(z0.device)
    d = d / torch.linalg.vector_norm(d).clamp_min(1e-9)
    out = z0.clone()
    for lo, hi in ((0, m_fine), (m_fine, M)):
        k0 = (z0[:, lo:hi, :3] @ d).argsort(dim=1)      # [B, n] noise ranks
        k1 = (z1[:, lo:hi, :3] @ d).argsort(dim=1)      # [B, n] data ranks
        # place the rank-r noise element at the position of the rank-r data
        # element: out[b, lo + k1[b, r]] = z0[b, lo + k0[b, r]]
        src = torch.gather(z0[:, lo:hi], 1,
                           k0.unsqueeze(-1).expand(-1, -1, z0.shape[-1]))
        out[:, lo:hi] = out[:, lo:hi].scatter(
            1, k1.unsqueeze(-1).expand(-1, -1, z0.shape[-1]), src)
    return out


class TimeEmbed(nn.Module):
    def __init__(self, dim: int, n_freq: int = 32):
        super().__init__()
        self.freqs = nn.Parameter(torch.exp(torch.linspace(
            math.log(1.0), math.log(1000.0), n_freq)), requires_grad=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))

    def forward(self, t: Tensor) -> Tensor:
        ang = t.unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))


class LatentFlow(nn.Module):
    """Equivariant velocity field over the (normalized) flattened latent set."""

    def __init__(self, latent_dim: int = 256, m_fine: int = 512,
                 m_coarse: int = 32, dim: int = 384, n_layers: int = 8,
                 k: int = 32):
        super().__init__()
        self.latent_dim, self.m_fine, self.m_coarse = latent_dim, m_fine, m_coarse
        self.dim = dim
        self.in_h = nn.Linear(latent_dim, dim)
        self.level_emb = nn.Embedding(2, dim)  # fine vs coarse element
        self.time_embed = TimeEmbed(dim)
        self.embedder = TokenEmbedder(dim)     # for condition tokens
        self.blocks = nn.ModuleList([AttentionBlock(dim, k=k) for _ in range(n_layers)])
        self.cond_blocks = nn.ModuleList(
            [AttentionBlock(dim, k=16) for _ in range(n_layers)])
        self.out_h = nn.Linear(dim, latent_dim)
        self.out_pos = nn.Linear(dim, N_VEC)   # scalar weights over v channels
        self.out_v = nn.Linear(dim, N_VEC * N_VEC)
        nn.init.zeros_(self.out_h.weight)
        nn.init.zeros_(self.out_h.bias)
        nn.init.zeros_(self.out_pos.weight)
        nn.init.zeros_(self.out_pos.bias)
        nn.init.zeros_(self.out_v.weight)
        nn.init.zeros_(self.out_v.bias)

    def _state(self, z: Tensor, t: Tensor) -> TokenState:
        B, M, _ = z.shape
        pos = z[..., :3]
        ctx = z[..., 3:3 + self.latent_dim]
        v = z[..., 3 + self.latent_dim:].reshape(B, M, N_VEC, 3)
        lvl = torch.cat([torch.zeros(self.m_fine, dtype=torch.long, device=z.device),
                         torch.ones(self.m_coarse, dtype=torch.long, device=z.device)])
        h = self.in_h(ctx) + self.level_emb(lvl) + self.time_embed(t).unsqueeze(1)
        return TokenState(pos=pos, h=h, v=v, fval=torch.zeros(B, M, device=z.device),
                          global_mask=torch.zeros(B, M, dtype=torch.bool, device=z.device))

    def forward(self, z_t: Tensor, t: Tensor,
                cond: list[list[Token]] | None = None) -> Tensor:
        """Velocity in flattened latent coordinates. z_t [B,M,D], t [B]."""
        s = self._state(z_t, t)
        cond_state = None
        if cond is not None and any(cond):
            cond_state = self.embedder.embed_conditions(
                cond, z_t.device, center=s.pos.mean(dim=1))
        for blk, cblk in zip(self.blocks, self.cond_blocks):
            s = blk(s)
            if cond_state is not None:
                s = cblk(s, cond_state)
        # equivariant readout: pose/vector velocities from vector channels
        dpos = torch.einsum("bmc,bmcd->bmd", self.out_pos(s.h), s.v)
        dctx = self.out_h(s.h)
        Wv = self.out_v(s.h).reshape(*s.h.shape[:2], N_VEC, N_VEC)
        dv = torch.einsum("bmce,bmed->bmcd", Wv, s.v)
        return torch.cat([dpos, dctx, dv.reshape(*z_t.shape[:2], N_VEC * 3)], dim=-1)


def cfm_loss(flow: LatentFlow, z1: Tensor, cond: list[list[Token]] | None,
             gen: torch.Generator, p_drop: float = 0.15) -> Tensor:
    """Linear-interpolant conditional flow-matching loss for one batch of
    normalized latents z1 [B,M,D]."""
    B = z1.shape[0]
    z0 = torch.randn(z1.shape, generator=gen, device="cpu").to(z1.device)
    z0 = ot_pair(z0, z1, flow.m_fine, gen=gen)
    t = torch.rand(B, generator=gen).to(z1.device)
    z_t = (1 - t).reshape(B, 1, 1) * z0 + t.reshape(B, 1, 1) * z1
    if cond is not None:
        keep = torch.rand(B, generator=gen) > p_drop
        cond = [c if keep[i] else [] for i, c in enumerate(cond)]
    v = flow(z_t, t, cond)
    return (v - (z1 - z0)).pow(2).mean()


@torch.no_grad()
def sample(flow: LatentFlow, cond: list[list[Token]] | None, n: int,
           steps: int = 64, guidance_scale: float = 2.0,
           device: str = "cpu", seed: int = 0,
           guidance_fn=None) -> Tensor:
    """Euler ODE integration of the learned velocity, with classifier-free
    guidance and optional constraint guidance (guidance_fn(z_hat1, t) -> grad
    added to the velocity; see model/guidance.py)."""
    gen = torch.Generator().manual_seed(seed)
    M = flow.m_fine + flow.m_coarse
    D = 3 + flow.latent_dim + N_VEC * 3
    z = torch.randn(n, M, D, generator=gen).to(device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        v_c = flow(z, t, cond)
        if guidance_scale != 1.0 and cond is not None:
            v_u = flow(z, t, None)
            v = v_u + guidance_scale * (v_c - v_u)
        else:
            v = v_c
        if guidance_fn is not None:
            z1_hat = z + (1 - i * dt) * v          # current endpoint estimate
            v = v + guidance_fn(z1_hat, i * dt)
        z = z + dt * v
    return z
