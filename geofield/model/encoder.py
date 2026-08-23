"""SE(3)-equivariant attention encoder.

Representation per token:
  pos  [B, N, 3]   position (moves with the frame)
  h    [B, N, d]   scalar features (SE(3)-INVARIANT by construction)
  v    [B, N, C, 3] vector channels (rotate with the frame; C small)
  fval [B, N]      scalar anchor (input tokens: f(x); others: 0)

Anchor direction for the pairwise invariants is vector channel 0 (initialized
to grad f for input tokens, to the direction/axis param for condition tokens).

Attention:
  * windowed heads: kNN neighborhoods; logits = MLP(pairwise invariants)
    + q.k on scalars; per-head Gaussian window exp(-||dx||^2 / 2 sigma_h^2)
    with learned sigma_h. Pairs where the KEY is a global token bypass the
    window (global tokens have no meaningful position).
  * 2 global heads: dense attention with plain q.k logits on the invariant
    scalars (no pair MLP — dense pairwise invariants would be O(N^2 * 7)
    memory; scalar-only logits keep the head equivariant and cheap).
  * scalar values from h; vector update through the windowed path only:
      v_i += sum_j a_ij (w1_ij dx_ij + w2_ij g_j + w3_ij v_j)
    with per-pair per-channel scalar weights from the pair MLP — equivariant
    by construction.

Equivariance argument: h and all attention logits are functions of invariants
only; v updates are linear combinations of vectors that rotate with the frame
with invariant coefficients; poses update by adding vector-channel
combinations. Hence rotating inputs (x, g, token poses/directions) rotates
(pos, v) and leaves h unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .fourier import FourierFeatures
from .invariants import N_INVARIANTS, knn_indices, pairwise_invariants
from ..tokens.schema import Token

N_VEC = 8  # vector channels


@dataclass
class TokenState:
    pos: Tensor              # [B, N, 3]
    h: Tensor                # [B, N, d]
    v: Tensor                # [B, N, C, 3]
    fval: Tensor             # [B, N]
    global_mask: Tensor      # [B, N] bool: True for global (pose-less) tokens
    pad_mask: Tensor | None = None  # [B, N] bool: True for padding

    def anchor(self) -> Tensor:
        g = self.v[..., 0, :]
        # smooth norm: zero anchors (fresh query tokens) must stay
        # double-backward-safe for the sdf-gradient losses
        return g / torch.sqrt(g.pow(2).sum(-1, keepdim=True) + 1e-12)


def _gather_state(s: TokenState, idx: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Gather key tokens by [B, Nq, k] indices -> pos/h/v/fval/global."""
    B, Nq, k = idx.shape
    d = s.h.shape[-1]
    C = s.v.shape[-2]
    ie = idx.reshape(B, Nq * k)
    pos = s.pos.gather(1, ie.unsqueeze(-1).expand(-1, -1, 3)).reshape(B, Nq, k, 3)
    h = s.h.gather(1, ie.unsqueeze(-1).expand(-1, -1, d)).reshape(B, Nq, k, d)
    v = s.v.reshape(B, -1, C * 3).gather(1, ie.unsqueeze(-1).expand(-1, -1, C * 3))
    v = v.reshape(B, Nq, k, C, 3)
    fv = s.fval.gather(1, ie).reshape(B, Nq, k)
    gm = s.global_mask.gather(1, ie).reshape(B, Nq, k)
    return pos, h, v, fv, gm


class VecLayerNorm(nn.Module):
    """Equivariant normalization of vector channels: scale each channel by
    1/RMS(channel norms) and a learned per-channel gain."""

    def __init__(self, n_vec: int = N_VEC):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_vec))

    def forward(self, v: Tensor) -> Tensor:
        # squared norms directly (no sqrt-of-zero on the double-backward path)
        rms = (v.pow(2).sum(-1).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        return v * (self.gain / rms).unsqueeze(-1)


class EquivariantAttention(nn.Module):
    """Cross-attention from a query TokenState over a key TokenState."""

    def __init__(self, dim: int, n_heads: int = 8, n_global_heads: int = 2,
                 k: int = 64, n_vec: int = N_VEC, pair_hidden: int = 64):
        super().__init__()
        assert dim % n_heads == 0
        self.dim, self.h_all, self.h_glob = dim, n_heads, n_global_heads
        self.h_win = n_heads - n_global_heads
        self.dh = dim // n_heads
        self.k = k
        self.n_vec = n_vec

        self.q = nn.Linear(dim, dim, bias=False)
        self.kproj = nn.Linear(dim, dim, bias=False)
        self.vproj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

        # pair MLP: invariants -> windowed-head logits + vector weights
        self.pair = nn.Sequential(
            nn.Linear(N_INVARIANTS, pair_hidden), nn.SiLU(),
            nn.Linear(pair_hidden, self.h_win + 3 * n_vec))
        # learned per-head window widths (softplus-parameterized)
        self.win_raw = nn.Parameter(torch.full((self.h_win,), 0.0))
        self.vec_gate = nn.Linear(dim, n_vec)

    def window_sigma(self) -> Tensor:
        return nn.functional.softplus(self.win_raw) + 0.05

    def forward(self, q_state: TokenState, k_state: TokenState) -> tuple[Tensor, Tensor]:
        """Returns (dh [B,Nq,dim], dv [B,Nq,C,3]) residual updates."""
        B, Nq, dim = q_state.h.shape
        Q = self.q(q_state.h).reshape(B, Nq, self.h_all, self.dh)
        Kf = self.kproj(k_state.h)
        Vf = self.vproj(k_state.h)

        # ---- windowed kNN path ------------------------------------------------
        idx = knn_indices(q_state.pos, k_state.pos, self.k)      # [B,Nq,k]
        kpos, kh, kv, kfv, kglob = _gather_state(k_state, idx)
        k_eff = idx.shape[-1]
        Kk = Kf.gather(1, idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, dim)) \
            .reshape(B, Nq, k_eff, self.h_all, self.dh)
        Vk = Vf.gather(1, idx.reshape(B, -1).unsqueeze(-1).expand(-1, -1, dim)) \
            .reshape(B, Nq, k_eff, self.h_all, self.dh)

        qpos = q_state.pos.unsqueeze(2)
        qanc = q_state.anchor().unsqueeze(2)
        kanc = kv[..., 0, :]
        kanc = kanc / torch.linalg.vector_norm(kanc, dim=-1, keepdim=True).clamp_min(1e-6)
        inv = pairwise_invariants(
            qpos.expand_as(kpos), qanc.expand_as(kanc),
            q_state.fval.unsqueeze(-1).expand(B, Nq, k_eff),
            kpos, kanc, kfv)                                     # [B,Nq,k,7]
        pair_out = self.pair(inv)
        pair_logits = pair_out[..., :self.h_win]                 # [B,Nq,k,Hw]
        vec_w = pair_out[..., self.h_win:]                       # [B,Nq,k,3C]

        qk = torch.einsum("bnhd,bnkhd->bnkh", Q, Kk) / self.dh ** 0.5
        logits_win = qk[..., :self.h_win] + pair_logits
        dx = kpos - qpos                                          # [B,Nq,k,3]
        dist2 = dx.pow(2).sum(-1, keepdim=True)                   # [B,Nq,k,1]
        sig = self.window_sigma().reshape(1, 1, 1, -1)
        win = -dist2 / (2 * sig * sig)
        win = torch.where(kglob.unsqueeze(-1), torch.zeros_like(win), win)
        logits_win = logits_win + win
        # finite masking (-1e9) + post-softmax zeroing: all-padded rows give a
        # clean zero contribution with NO -inf softmax (whose backward is NaN)
        if k_state.pad_mask is not None:
            kpad = k_state.pad_mask.gather(1, idx.reshape(B, -1)).reshape(B, Nq, k_eff)
            logits_win = logits_win.masked_fill(kpad.unsqueeze(-1), -1e9)
        a_win = logits_win.softmax(dim=2)                        # [B,Nq,k,Hw]
        if k_state.pad_mask is not None:
            a_win = a_win.masked_fill(kpad.unsqueeze(-1), 0.0)

        out_win = torch.einsum("bnkh,bnkhd->bnhd", a_win, Vk[..., :self.h_win, :])

        # vector update via mean windowed attention
        a_vec = a_win.mean(dim=-1)                                # [B,Nq,k]
        w1, w2, w3 = vec_w.chunk(3, dim=-1)                       # each [B,Nq,k,C]
        contrib = (w1.unsqueeze(-1) * dx.unsqueeze(-2)
                   + w2.unsqueeze(-1) * kanc.unsqueeze(-2)
                   + w3.unsqueeze(-1) * kv)                       # [B,Nq,k,C,3]
        dv = torch.einsum("bnk,bnkcd->bncd", a_vec, contrib)
        dv = dv * self.vec_gate(q_state.h).unsqueeze(-1)

        # ---- global path (scalar-only logits) ---------------------------------
        qg = Q[..., self.h_win:, :]                               # [B,Nq,Hg,dh]
        Kg = Kf.reshape(B, -1, self.h_all, self.dh)[..., self.h_win:, :]
        Vg = Vf.reshape(B, -1, self.h_all, self.dh)[..., self.h_win:, :]
        logits_g = torch.einsum("bnhd,bmhd->bnmh", qg, Kg) / self.dh ** 0.5
        if k_state.pad_mask is not None:
            gpad = k_state.pad_mask.unsqueeze(1).unsqueeze(-1)
            logits_g = logits_g.masked_fill(gpad, -1e9)
        a_g = logits_g.softmax(dim=2)
        if k_state.pad_mask is not None:
            a_g = a_g.masked_fill(gpad, 0.0)
        out_g = torch.einsum("bnmh,bmhd->bnhd", a_g, Vg)

        out = torch.cat([out_win, out_g], dim=-2).reshape(B, Nq, dim)
        return self.out(out), dv


class AttentionBlock(nn.Module):
    """Pre-norm attention + FFN with equivariant vector residual."""

    def __init__(self, dim: int, n_heads: int = 8, n_global_heads: int = 2,
                 k: int = 64, ffn_mult: int = 2):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.attn = EquivariantAttention(dim, n_heads, n_global_heads, k)
        self.vnorm = VecLayerNorm()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_mult * dim), nn.SiLU(),
            nn.Linear(ffn_mult * dim, dim))

    def forward(self, q_state: TokenState, k_state: TokenState | None = None
                ) -> TokenState:
        ks = k_state or q_state
        ks_n = TokenState(ks.pos, self.norm_k(ks.h), ks.v, ks.fval,
                          ks.global_mask, ks.pad_mask)
        qs_n = TokenState(q_state.pos, self.norm_q(q_state.h), q_state.v,
                          q_state.fval, q_state.global_mask, q_state.pad_mask)
        dh, dv = self.attn(qs_n, ks_n)
        h = q_state.h + dh
        v = self.vnorm(q_state.v + dv)
        h = h + self.ffn(h)
        return TokenState(q_state.pos, h, v, q_state.fval,
                          q_state.global_mask, q_state.pad_mask)


class PoseRefine(nn.Module):
    """Equivariant pose update: p += sum_c w_c(h) * v_c, small-initialized."""

    def __init__(self, dim: int, n_vec: int = N_VEC, scale: float = 0.1):
        super().__init__()
        self.w = nn.Linear(dim, n_vec)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)
        self.scale = scale

    def forward(self, s: TokenState) -> TokenState:
        dp = torch.einsum("bnc,bncd->bnd", self.w(s.h), s.v) * self.scale
        return TokenState(s.pos + dp, s.h, s.v, s.fval, s.global_mask, s.pad_mask)


# ---------------------------------------------------------------------------
# Input + condition token embedding
# ---------------------------------------------------------------------------

TOKEN_TYPES = ["__input__", "fixed_point", "load", "material",
               "build_dir", "spindle_dir", "envelope", "scale"]


class TokenEmbedder(nn.Module):
    """Embeds input samples (x, f, g) and condition tokens into TokenState.

    Input tokens: h = MLP([Fourier(f), type_emb]); v0 = g.
    Condition tokens: h = MLP([type_emb, scalar params]); v0 = direction/axis
    param (zero if none); pos = pose position (origin for global tokens).
    Scalar params are embedded per type with fixed slots; unknown token types
    are skipped (with the schema-level warning already emitted).
    """

    N_PARAM_SLOTS = 6

    def __init__(self, dim: int):
        super().__init__()
        self.type_index = {t: i for i, t in enumerate(TOKEN_TYPES)}
        self.type_emb = nn.Embedding(len(TOKEN_TYPES), dim)
        self.f_fourier = FourierFeatures(dim_in=1, n_freq=32)
        self.input_mlp = nn.Sequential(
            nn.Linear(64 + dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.cond_mlp = nn.Sequential(
            nn.Linear(dim + self.N_PARAM_SLOTS, dim), nn.SiLU(), nn.Linear(dim, dim))

    def embed_inputs(self, x: Tensor, f: Tensor, g: Tensor) -> TokenState:
        B, N, _ = x.shape
        te = self.type_emb.weight[self.type_index["__input__"]]
        h = self.input_mlp(torch.cat(
            [self.f_fourier(f.unsqueeze(-1)), te.expand(B, N, -1)], dim=-1))
        v = x.new_zeros(B, N, N_VEC, 3)
        v[..., 0, :] = g
        return TokenState(pos=x, h=h, v=v, fval=f,
                          global_mask=torch.zeros(B, N, dtype=torch.bool, device=x.device))

    def _scalar_slots(self, tok: Token) -> list[float]:
        vals = []
        for k in sorted(tok.params):
            p = tok.params[k]
            if isinstance(p, bool):
                vals.append(1.0 if p else 0.0)
            elif isinstance(p, (int, float)):
                # squash wide-ranged physical scalars
                vals.append(float(torch.log10(torch.tensor(abs(p) + 1.0))))
            elif isinstance(p, Tensor) and p.numel() == 3 and k == "half_extents":
                # body-frame sizes are invariant scalars (rotation is in pose)
                vals.extend(float(v) for v in p)
        vals = vals[: self.N_PARAM_SLOTS]
        return vals + [0.0] * (self.N_PARAM_SLOTS - len(vals))

    def embed_conditions(self, tokens: list[list[Token]], device,
                         center: Tensor | None = None) -> TokenState | None:
        """Batch of per-record token lists -> padded TokenState.

        `center` [B,3]: pseudo-position for GLOBAL tokens. It must be an
        equivariant reference point (e.g. the input/query centroid) — a fixed
        world origin would leak translations into the invariants. Defaults to
        zeros only when no better reference exists."""
        max_t = max((len([t for t in toks if t.type in self.type_index])
                     for toks in tokens), default=0)
        if max_t == 0:
            return None
        B = len(tokens)
        dim = self.type_emb.weight.shape[1]
        if center is None:
            center = torch.zeros(B, 3, device=device)
        pos = torch.zeros(B, max_t, 3, device=device)
        v = torch.zeros(B, max_t, N_VEC, 3, device=device)
        fval = torch.zeros(B, max_t, device=device)
        glob = torch.zeros(B, max_t, dtype=torch.bool, device=device)
        pad = torch.ones(B, max_t, dtype=torch.bool, device=device)
        type_ids = torch.zeros(B, max_t, dtype=torch.long, device=device)
        slots = torch.zeros(B, max_t, self.N_PARAM_SLOTS, device=device)
        for b, toks in enumerate(tokens):
            i = 0
            for tok in toks:
                if tok.type not in self.type_index:
                    continue
                type_ids[b, i] = self.type_index[tok.type]
                slots[b, i] = torch.tensor(self._scalar_slots(tok), device=device)
                if tok.pose is not None:
                    pos[b, i] = tok.pose[:3].to(device)
                else:
                    pos[b, i] = center[b]
                    glob[b, i] = True
                for key in ("axis", "direction"):
                    if key in tok.params:
                        v[b, i, 0] = torch.as_tensor(tok.params[key], device=device)
                        break
                pad[b, i] = False
                i += 1
        h = self.cond_mlp(torch.cat([self.type_emb(type_ids), slots], dim=-1))
        return TokenState(pos=pos, h=h, v=v, fval=fval, global_mask=glob, pad_mask=pad)


# ---------------------------------------------------------------------------
# Farthest-point sampling
# ---------------------------------------------------------------------------

def farthest_point_indices(x: Tensor, m: int, seed_idx: int = 0) -> Tensor:
    """Deterministic FPS. x [B,N,3] -> [B,m] indices."""
    B, N, _ = x.shape
    idx = torch.zeros(B, m, dtype=torch.long, device=x.device)
    dist = torch.full((B, N), float("inf"), device=x.device)
    cur = torch.full((B,), seed_idx, dtype=torch.long, device=x.device)
    batch = torch.arange(B, device=x.device)
    for i in range(m):
        idx[:, i] = cur
        d = torch.linalg.vector_norm(x - x[batch, cur].unsqueeze(1), dim=-1)
        dist = torch.minimum(dist, d)
        cur = dist.argmax(dim=-1)
    return idx
