"""Training losses.

L_sdf   : L1 on f, weight 5 inside band 0 (|f| < 0.02), 1 elsewhere.
L_grad  : L1 on grad f at all queries.
L_eik   : (||grad f_hat|| - 1)^2 outside band 0.
L_mask  : sdf/grad supervision inside a dropped input-ball, weight 2
          (the drop itself happens in the loop; this just reweights).
L_field : per labeled field, masked by label presence:
          'l1' plain; 'l1_log' L1 on signed log1p scaling (wide-range physics
          magnitudes); 'bce' with logits.
Total   : L_sdf + L_grad + 0.1 L_eik + L_mask + sum_k lambda_k L_field_k.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

BAND0 = 0.02


def sdf_loss(pred: Tensor, target: Tensor, band_weight: float = 5.0) -> Tensor:
    w = torch.where(target.abs() < BAND0, band_weight, 1.0)
    return (w * (pred - target).abs()).mean()


def grad_loss(pred_g: Tensor, target_g: Tensor) -> Tensor:
    return (pred_g - target_g).abs().mean()


def eikonal_loss(pred_g: Tensor, target_f: Tensor) -> Tensor:
    outside = target_f.abs() >= BAND0
    gn = torch.linalg.vector_norm(pred_g, dim=-1)
    res = (gn - 1.0).pow(2)
    return res[outside].mean() if outside.any() else res.mean() * 0


def signed_log(x: Tensor, scale: float = 1.0) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs() / scale)


FIELD_SCALES = {  # log-space knee per field: values BELOW the knee live on the
    # linear part of log1p and get vanishing gradient. Stress spans 1e4..4e8 Pa
    # with most points near 1e5-1e6, so the knee must sit at the LOW end —
    # 1e7 (the first choice) let "predict ~0 everywhere" score loss ~0.35 and
    # the head collapsed to a constant. 1e5 spreads the range over ~0..8.3.
    "von_mises": 1e5,      # Pa
    "displacement": 1e-4,  # m (calibrated cases deflect ~0.1..10 mm) — healthy
}


def field_loss(field_id: str, loss_kind: str, pred: Tensor, target: Tensor,
               valid: Tensor | None = None) -> Tensor:
    """pred [B,Q,k], target [B,Q,(k)], valid [B] presence mask."""
    if target.dim() == pred.dim() - 1:
        target = target.unsqueeze(-1)
    if loss_kind == "bce":
        per = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    elif loss_kind == "l1_log":
        # pred arrives in the head's NATIVE log space (decode raw=False);
        # only the target needs transforming — gradients stay log-scaled
        s = FIELD_SCALES.get(field_id, 1.0)
        per = (pred - signed_log(target, s)).abs()
    else:
        per = (pred - target).abs()
    per = per.mean(dim=tuple(range(1, per.dim())))  # [B]
    if valid is not None:
        per = per * valid.to(per)
        denom = valid.to(per).sum().clamp_min(1.0)
        return per.sum() / denom
    return per.mean()
