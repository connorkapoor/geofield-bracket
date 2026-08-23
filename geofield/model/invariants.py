"""Pairwise SE(3) invariants from (position, gradient, field value) tokens.

For a pair (i, j) with positions x, unit-ish gradients g, values f:

  [ ||dx||, <g_i, g_j>, <g_i, dxh>, <g_j, dxh>, f_i, f_j, det(g_i, g_j, dxh) ]

with dx = x_j - x_i and dxh = dx / ||dx||. All seven are invariant under
rotations + translations applied jointly to both tokens; the determinant flips
sign under reflection, so the representation is SE(3)- (not E(3)-) invariant,
preserving chirality.
"""
from __future__ import annotations

import torch
from torch import Tensor

N_INVARIANTS = 7


def pairwise_invariants(x_i: Tensor, g_i: Tensor, f_i: Tensor,
                        x_j: Tensor, g_j: Tensor, f_j: Tensor,
                        eps: float = 1e-6) -> Tensor:
    """Broadcasting invariants. Shapes: x/g [..., 3], f [...]. Returns [..., 7].

    Uses the smooth norm sqrt(|dx|^2 + eps^2): coincident points (dx = 0) are
    routine here (queries drawn from the same samples as latent poses), and
    the exact 2-norm has a NaN second derivative at zero, which poisons the
    eikonal/grad losses that differentiate THROUGH this function twice."""
    dx = x_j - x_i
    dist = torch.sqrt(dx.pow(2).sum(-1) + eps * eps)
    dxh = dx / dist.unsqueeze(-1)
    gi_gj = (g_i * g_j).sum(-1)
    gi_dx = (g_i * dxh).sum(-1)
    gj_dx = (g_j * dxh).sum(-1)
    det = (torch.linalg.cross(g_i, g_j, dim=-1) * dxh).sum(-1)
    return torch.stack([dist, gi_gj, gi_dx, gj_dx, f_i, f_j, det], dim=-1)


def knn_indices(q: Tensor, k_pos: Tensor, k: int) -> Tensor:
    """kNN of each query position among key positions.

    q [B, N, 3], k_pos [B, M, 3] -> [B, N, k] indices into M.
    Dense cdist implementation (no torch_cluster dependency); at the scales
    used here (N,M <= 2048) this is a single fused GPU op.
    """
    d = torch.cdist(q, k_pos)
    k = min(k, k_pos.shape[1])
    return d.topk(k, dim=-1, largest=False).indices
