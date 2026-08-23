"""Field registry and per-field decoder heads.

Extension contract: `register_field(field_id, out_dim, loss_fn, head_cls,
equivariant, cond_types)` — adding a new physical field touches ONLY this
registry (plus a labeler and a verifier); the backbone never changes.

Heads receive the decoded query state (invariant scalars h_q [B,Q,d] and
equivariant vector channels v_q [B,Q,C,3]) plus embedded condition features.
Scalar heads read h_q; equivariant heads (displacement) emit
sum_c w_c(h_q) * v_qc — a rotation-equivariant vector by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import torch
from torch import Tensor, nn

from .encoder import N_VEC


class ScalarHead(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        last = nn.Linear(dim, out_dim)
        # small output init: keeps untrained field values AND their spatial
        # gradients near zero, so the eikonal term starts at ~1 instead of
        # exploding through the Fourier frequencies
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim),
                                 nn.SiLU(), last)

    def forward(self, h: Tensor, v: Tensor) -> Tensor:
        return self.net(h)


class VectorHead(nn.Module):
    """Equivariant: scalar-weighted sum of the query's vector channels."""

    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        assert out_dim == 3
        last = nn.Linear(dim, N_VEC + 1)
        nn.init.normal_(last.weight, std=1e-3)
        nn.init.zeros_(last.bias)
        self.w = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(),
                               last)

    def forward(self, h: Tensor, v: Tensor) -> Tensor:
        w = self.w(h)                       # [B,Q,C+1]; last = magnitude gain
        vec = torch.einsum("bqc,bqcd->bqd", w[..., :N_VEC], v)
        return vec * w[..., N_VEC:].exp().clamp(max=1e4)


@dataclass
class FieldSpec:
    field_id: str
    out_dim: int
    loss: str                      # 'l1' | 'bce' | 'l1_log'
    head_cls: type
    equivariant: bool
    cond_types: tuple = ()         # token types cross-attended at the head
    weight: float = 1.0
    # log-space output: the head predicts signed_log(value / log_scale) and
    # decode() inverts it for consumers. Raw-pascal outputs through a
    # LayerNorm-bounded MLP cannot span 1e4..4e8 — observed failure: the
    # stress head settling at a constant despite a healthy-looking log loss.
    log_scale: float | None = None


_FIELDS: dict[str, FieldSpec] = {}


def register_field(field_id: str, out_dim: int, loss: str, head_cls: type,
                   equivariant: bool, cond_types: tuple = (),
                   weight: float = 1.0, log_scale: float | None = None) -> None:
    _FIELDS[field_id] = FieldSpec(field_id, out_dim, loss, head_cls,
                                  equivariant, cond_types, weight, log_scale)


def field_specs() -> dict[str, FieldSpec]:
    return dict(_FIELDS)


# Phase-1/2 fields. Dataset label keys are '<field_id>|<cond_id>'.
register_field("sdf", 1, "l1", ScalarHead, equivariant=False, weight=1.0)
register_field("von_mises", 1, "l1_log", ScalarHead, equivariant=False,
               cond_types=("load", "fixed_point", "material"), weight=1.0,
               log_scale=1e5)     # head lives in log1p(Pa/1e5) space
register_field("displacement", 3, "l1_log", VectorHead, equivariant=True,
               cond_types=("load", "fixed_point", "material"), weight=1.0,
               log_scale=1e-4)    # signed_log(m/1e-4) space, per component
register_field("overhang", 1, "l1", ScalarHead, equivariant=False,
               cond_types=("build_dir",), weight=1.0)
register_field("support_need", 1, "bce", ScalarHead, equivariant=False,
               cond_types=("build_dir",), weight=1.0)
register_field("tool_access", 1, "bce", ScalarHead, equivariant=False,
               cond_types=("spindle_dir",), weight=1.0)
register_field("tool_access_soft", 1, "l1", ScalarHead, equivariant=False,
               cond_types=("spindle_dir",), weight=0.5)
register_field("thickness", 1, "l1", ScalarHead, equivariant=False, weight=1.0)


def build_heads(dim: int) -> nn.ModuleDict:
    return nn.ModuleDict({fid: spec.head_cls(dim, spec.out_dim)
                          for fid, spec in _FIELDS.items()})
