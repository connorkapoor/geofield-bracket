"""Non-equivariant baseline control.

Same interfaces and comparable parameter count as GeoFieldModel, but:
  * absolute Fourier positional encoding added to every token,
  * standard multi-head attention, no Gaussian windows, no pairwise invariants,
  * flat latent set of 544 pose-less vectors (learned queries),
  * absolute-coordinate decoder.
Trained with random SE(3) augmentation on every batch (in the loop).
Directions/axes of condition tokens enter as raw 3-vectors.

`encode` returns a LatentSet with all poses at zero so downstream code
(flow model, probes over pooled()) can treat both models uniformly.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .encoder import N_VEC, TokenState
from .fourier import FourierFeatures
from .heads import build_heads, field_specs
from .latent import LatentSet
from ..tokens.schema import Token


class BaselineModel(nn.Module):
    def __init__(self, dim: int = 256, n_layers: int = 6, m_latent: int = 544,
                 n_heads: int = 8):
        super().__init__()
        self.dim, self.m_latent = dim, m_latent
        self.pos_enc = FourierFeatures(3, n_freq=64)
        self.f_enc = FourierFeatures(1, n_freq=16)
        self.in_proj = nn.Linear(self.pos_enc.dim_out + self.f_enc.dim_out + 3, dim)
        self.cond_proj = nn.Linear(self.pos_enc.dim_out + 16 + 3 + 1, dim)
        self.type_emb = nn.Embedding(16, 16)
        self.type_index = {t: i for i, t in enumerate(
            ["fixed_point", "load", "material", "build_dir", "spindle_dir", "envelope"])}

        enc_layer = nn.TransformerEncoderLayer(
            dim, n_heads, dim_feedforward=2 * dim, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.latent_queries = nn.Parameter(torch.randn(m_latent, dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

        self.q_proj = nn.Linear(self.pos_enc.dim_out, dim)
        dec_layer = nn.TransformerDecoderLayer(
            dim, n_heads, dim_feedforward=2 * dim, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.decoder = nn.TransformerDecoder(dec_layer, 3)
        self.field_emb = nn.ParameterDict({
            fid: nn.Parameter(torch.randn(dim) * 0.02) for fid in field_specs()})
        self.cond_mix = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.heads = build_heads(dim)
        self.vec_out = nn.Linear(dim, 3)

    # -- embedding -------------------------------------------------------------

    def _embed_conditions(self, tokens: list[list[Token]], device) -> tuple[Tensor, Tensor] | None:
        known = [[t for t in toks if t.type in self.type_index] for toks in tokens]
        max_t = max((len(k) for k in known), default=0)
        if max_t == 0:
            return None
        B = len(tokens)
        feats = torch.zeros(B, max_t, self.pos_enc.dim_out + 16 + 3 + 1, device=device)
        pad = torch.ones(B, max_t, dtype=torch.bool, device=device)
        for b, toks in enumerate(known):
            for i, tok in enumerate(toks):
                pos = tok.pose[:3].to(device) if tok.pose is not None \
                    else torch.zeros(3, device=device)
                d = torch.zeros(3, device=device)
                for key in ("axis", "direction"):
                    if key in tok.params:
                        d = torch.as_tensor(tok.params[key], device=device)
                        break
                mag = 0.0
                for key in ("magnitude", "diameter", "E"):
                    if key in tok.params:
                        mag = float(torch.log10(torch.tensor(
                            abs(float(tok.params[key])) + 1.0)))
                        break
                feats[b, i] = torch.cat([
                    self.pos_enc(pos.unsqueeze(0)).squeeze(0),
                    self.type_emb.weight[self.type_index[tok.type]],
                    d, torch.tensor([mag], device=device)])
                pad[b, i] = False
        return self.cond_proj(feats), pad

    def encode(self, x: Tensor, f: Tensor, g: Tensor,
               tokens: list[list[Token]] | None = None) -> LatentSet:
        B = x.shape[0]
        feats = torch.cat([self.pos_enc(x), self.f_enc(f.unsqueeze(-1)), g], dim=-1)
        h = self.in_proj(feats)
        pad = torch.zeros(B, h.shape[1], dtype=torch.bool, device=x.device)
        if tokens is not None:
            cond = self._embed_conditions(tokens, x.device)
            if cond is not None:
                ch, cpad = cond
                h = torch.cat([h, ch], dim=1)
                pad = torch.cat([pad, cpad], dim=1)
        h = self.encoder(h, src_key_padding_mask=pad)
        lq = self.latent_queries.expand(B, -1, -1)
        z, _ = self.pool_attn(lq, h, h, key_padding_mask=pad)

        zeros = torch.zeros(B, self.m_latent, 3, device=x.device)
        state = TokenState(pos=zeros, h=z,
                           v=torch.zeros(B, self.m_latent, N_VEC, 3, device=x.device),
                           fval=torch.zeros(B, self.m_latent, device=x.device),
                           global_mask=torch.ones(B, self.m_latent, dtype=torch.bool,
                                                  device=x.device))
        # flat latent: put everything in 'fine', empty-ish coarse (1 token)
        coarse = TokenState(pos=zeros[:, :1], h=z[:, :1], v=state.v[:, :1],
                            fval=state.fval[:, :1], global_mask=state.global_mask[:, :1])
        return LatentSet(fine=state, coarse=coarse)

    def decode(self, lat: LatentSet, x: Tensor, field_id: str,
               cond: list[list[Token]] | None = None, raw: bool = True) -> Tensor:
        q = self.q_proj(self.pos_enc(x)) + self.field_emb[field_id]
        h = self.decoder(q, lat.fine.h)
        if cond:
            c = self._embed_conditions(cond, x.device)
            if c is not None:
                ch, cpad = c
                mixed, _ = self.cond_mix(h, ch, ch, key_padding_mask=cpad)
                h = h + mixed
        v = torch.zeros(*h.shape[:2], N_VEC, 3, device=x.device)
        spec = field_specs()[field_id]
        if spec.equivariant:
            # baseline emits vectors directly from scalars (not equivariant)
            out = self.vec_out(h)
        else:
            out = self.heads[field_id](h, v)
        if spec.log_scale is not None and raw:
            out = torch.sign(out) * spec.log_scale * torch.expm1(
                out.abs().clamp(max=25.0))
        return out

    def decode_many(self, lat: LatentSet, x: Tensor, requests,
                    raw: bool = True) -> dict[str, Tensor]:
        return {key: self.decode(lat, x, fid, cond, raw=raw)
                for key, fid, cond in requests}

    def sdf_and_grad(self, lat: LatentSet, x: Tensor) -> tuple[Tensor, Tensor]:
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            f = self.decode(lat, xg, "sdf").squeeze(-1)
            (g,) = torch.autograd.grad(f.sum(), xg, create_graph=self.training)
        return f, g
