"""Decoder: (latent set, query x, field_id, condition tokens) -> field values.

Query tokens start with NO absolute-position features (that would break
equivariance): h_q is a learned constant, v_q = 0, pos = x. Everything the
query learns about where it is comes from equivariant cross-attention to the
posed latents (invariants of relative geometry + vector-channel updates).

Backbone (field-agnostic, run once per query set):
  cross(fine, kNN=32) -> cross(coarse, all) -> cross(fine, kNN=32)
Per field: h += field-id embedding; one shared condition cross-attention
block over the tokens relevant to the field (registry `cond_types`); then the
field head (scalar heads read h_q; the displacement head mixes v_q channels
with scalar weights — equivariant).

`sdf` gradients come from autograd of the decoder w.r.t. x.

The full model (encoder + latents + decoder + heads) is `GeoFieldModel`.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoder import (AttentionBlock, N_VEC, TokenEmbedder, TokenState)
from .heads import build_heads, field_specs
from .latent import LatentPyramid, LatentSet, init_latents
from ..tokens.schema import Token


class QueryDecoder(nn.Module):
    def __init__(self, dim: int, k_fine: int = 32):
        super().__init__()
        self.query_seed = nn.Parameter(torch.randn(dim) * 0.02)
        self.cross_fine1 = AttentionBlock(dim, k=k_fine)
        self.cross_coarse = AttentionBlock(dim, k=64)
        self.cross_fine2 = AttentionBlock(dim, k=k_fine)
        self.field_emb = nn.ParameterDict({
            fid: nn.Parameter(torch.randn(dim) * 0.02) for fid in field_specs()})
        self.cond_cross = AttentionBlock(dim, k=16)
        self.heads = build_heads(dim)

    def backbone(self, lat: LatentSet, x: Tensor) -> TokenState:
        B, Q, _ = x.shape
        q = TokenState(
            pos=x,
            h=self.query_seed.expand(B, Q, -1).contiguous(),
            v=x.new_zeros(B, Q, N_VEC, 3),
            fval=x.new_zeros(B, Q),
            global_mask=torch.zeros(B, Q, dtype=torch.bool, device=x.device))
        q = self.cross_fine1(q, lat.fine)
        q = self.cross_coarse(q, lat.coarse)
        q = self.cross_fine2(q, lat.fine)
        return q

    def head(self, q: TokenState, field_id: str,
             cond_state: TokenState | None, raw: bool = True) -> Tensor:
        h = q.h + self.field_emb[field_id]
        qf = TokenState(q.pos, h, q.v, q.fval, q.global_mask, q.pad_mask)
        spec = field_specs()[field_id]
        if cond_state is not None and spec.cond_types:
            qf = self.cond_cross(qf, cond_state)
        out = self.heads[field_id](qf.h, qf.v)
        if spec.log_scale is not None and raw:
            # head lives in signed-log space; invert for physical units
            out = torch.sign(out) * spec.log_scale * torch.expm1(
                out.abs().clamp(max=25.0))
        return out


class GeoFieldModel(nn.Module):
    """Encoder stack + hierarchical latents + decoder + registered heads."""

    def __init__(self, dim: int = 256, n_layers: int = 6, m_fine: int = 512,
                 m_coarse: int = 32, k_input: int = 64, n_input_tokens: int = 2048):
        super().__init__()
        self.dim, self.m_fine, self.m_coarse = dim, m_fine, m_coarse
        self.n_input_tokens = n_input_tokens
        self.embedder = TokenEmbedder(dim)
        self.input_blocks = nn.ModuleList(
            [AttentionBlock(dim, k=k_input) for _ in range(n_layers)])
        self.pyramids = nn.ModuleList(
            [LatentPyramid(dim) for _ in range(n_layers)])
        self.decoder = QueryDecoder(dim)

    # -- encoding -------------------------------------------------------------

    def encode(self, x: Tensor, f: Tensor, g: Tensor,
               tokens: list[list[Token]] | None = None) -> LatentSet:
        """x [B,N,3], f [B,N], g [B,N,3], optional per-record condition tokens
        (joined into the input set so the latent can absorb interface info)."""
        state = self.embedder.embed_inputs(x, f, g)
        cond = (self.embedder.embed_conditions(tokens, x.device, center=x.mean(dim=1))
                if tokens is not None else None)
        if cond is not None:
            state = TokenState(
                pos=torch.cat([state.pos, cond.pos], dim=1),
                h=torch.cat([state.h, cond.h], dim=1),
                v=torch.cat([state.v, cond.v], dim=1),
                fval=torch.cat([state.fval, cond.fval], dim=1),
                global_mask=torch.cat([state.global_mask, cond.global_mask], dim=1),
                pad_mask=torch.cat(
                    [torch.zeros_like(state.global_mask), cond.pad_mask], dim=1))
        lat = init_latents(state, self.m_fine, self.m_coarse)
        for block, pyramid in zip(self.input_blocks, self.pyramids):
            state = block(state)
            lat = pyramid(lat, state)
        return lat

    # -- decoding -------------------------------------------------------------

    def decode(self, lat: LatentSet, x: Tensor, field_id: str,
               cond: list[list[Token]] | None = None, raw: bool = True) -> Tensor:
        """raw=True returns physical units (log-space heads are inverted);
        raw=False returns the head's native (log-space) output for training."""
        q = self.decoder.backbone(lat, x)
        cond_state = (self.embedder.embed_conditions(
            cond, x.device, center=lat.fine.pos.mean(dim=1)) if cond else None)
        return self.decoder.head(q, field_id, cond_state, raw=raw)

    def decode_many(self, lat: LatentSet, x: Tensor,
                    requests: list[tuple[str, str, list[list[Token]] | None]],
                    raw: bool = True) -> dict[str, Tensor]:
        """Decode several (key, field_id, cond) requests on the same queries
        with one backbone pass."""
        q = self.decoder.backbone(lat, x)
        center = lat.fine.pos.mean(dim=1)
        out = {}
        for key, field_id, cond in requests:
            cond_state = (self.embedder.embed_conditions(cond, x.device, center=center)
                          if cond else None)
            out[key] = self.decoder.head(q, field_id, cond_state, raw=raw)
        return out

    def sdf_and_grad(self, lat: LatentSet, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            f = self.decode(lat, xg, "sdf").squeeze(-1)
            (g,) = torch.autograd.grad(f.sum(), xg, create_graph=self.training)
        return f, g
