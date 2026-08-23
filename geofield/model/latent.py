"""Hierarchical posed latent set.

Fine set (M_f=512): poses initialized by farthest-point sampling of the input
tokens; refined each layer by an equivariant vector update (PoseRefine).
Coarse set (M_c=32): FPS over the fine poses; cross-attends from fine each
layer; fine attends back to coarse (the global route).

`LatentSet.pooled()` -> invariant R^d (mean of scalar contexts).
`LatentSet.flatten()`/`unflatten()` -> [M_f + M_c, 3 + d + 3*C] tensor for the
flow model (pose + context + vector channels).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .encoder import (AttentionBlock, N_VEC, PoseRefine, TokenState,
                      farthest_point_indices)


@dataclass
class LatentSet:
    fine: TokenState        # [B, M_f, ...]
    coarse: TokenState      # [B, M_c, ...]

    def pooled(self) -> Tensor:
        return torch.cat([self.fine.h, self.coarse.h], dim=1).mean(dim=1)

    # -- flat serialization for the flow model -------------------------------

    def flatten(self) -> Tensor:
        def flat(s: TokenState) -> Tensor:
            B, M, C, _ = s.v.shape
            return torch.cat([s.pos, s.h, s.v.reshape(B, M, C * 3)], dim=-1)
        return torch.cat([flat(self.fine), flat(self.coarse)], dim=1)

    @staticmethod
    def unflatten(z: Tensor, m_fine: int, dim: int, n_vec: int = N_VEC) -> "LatentSet":
        B, M, _ = z.shape

        def unflat(t: Tensor) -> TokenState:
            pos = t[..., :3]
            h = t[..., 3:3 + dim]
            v = t[..., 3 + dim:].reshape(B, t.shape[1], n_vec, 3)
            return TokenState(pos=pos, h=h, v=v,
                              fval=torch.zeros_like(pos[..., 0]),
                              global_mask=torch.zeros_like(pos[..., 0], dtype=torch.bool))
        return LatentSet(fine=unflat(z[:, :m_fine]), coarse=unflat(z[:, m_fine:]))


class LatentPyramid(nn.Module):
    """Per-layer latent refinement used inside the encoder stack."""

    def __init__(self, dim: int, k_fine_in: int = 32, k_fine_self: int = 16):
        super().__init__()
        self.fine_from_input = AttentionBlock(dim, k=k_fine_in)
        self.fine_self = AttentionBlock(dim, k=k_fine_self)
        self.coarse_from_fine = AttentionBlock(dim, k=64)   # dense-ish over 512
        self.fine_from_coarse = AttentionBlock(dim, k=32)   # all coarse
        self.fine_pose = PoseRefine(dim)
        self.coarse_pose = PoseRefine(dim)

    def forward(self, lat: LatentSet, inputs: TokenState) -> LatentSet:
        fine = self.fine_from_input(lat.fine, inputs)
        fine = self.fine_self(fine)
        coarse = self.coarse_from_fine(lat.coarse, fine)
        fine = self.fine_from_coarse(fine, coarse)
        fine = self.fine_pose(fine)
        coarse = self.coarse_pose(coarse)
        return LatentSet(fine=fine, coarse=coarse)


def init_latents(inputs: TokenState, m_fine: int, m_coarse: int) -> LatentSet:
    """FPS-initialized latent states inheriting features from their source."""
    def take(s: TokenState, idx: Tensor) -> TokenState:
        b = torch.arange(idx.shape[0], device=idx.device).unsqueeze(-1)
        return TokenState(pos=s.pos[b, idx], h=s.h[b, idx], v=s.v[b, idx],
                          fval=s.fval[b, idx],
                          global_mask=torch.zeros_like(s.fval[b, idx], dtype=torch.bool))
    fine_idx = farthest_point_indices(inputs.pos, m_fine)
    fine = take(inputs, fine_idx)
    coarse_idx = farthest_point_indices(fine.pos, m_coarse)
    coarse = take(fine, coarse_idx)
    return LatentSet(fine=fine, coarse=coarse)
