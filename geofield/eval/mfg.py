"""Manufacturability head evaluation: IoU of predicted support_need and
tool_access against analytic labels on HELD-OUT directions (directions not in
the record's stored label set — recomputed analytically on the rebuilt field).
"""
from __future__ import annotations

import torch

from ..data import splits as split_defs
from ..data.gallery import rebuild_field
from ..labels import manufacturability as mfg_labels
from ..tokens.schema import Token


def binary_iou(pred_logit: torch.Tensor, target: torch.Tensor) -> float:
    p = pred_logit > 0
    t = target > 0.5
    union = (p | t).sum().item()
    return (p & t).sum().item() / union if union else float("nan")


@torch.no_grad()
def mfg_metrics(model, record: dict, split: str, device: str = "cpu",
                n_input: int = 2048, n_dirs: int = 2, seed: int = 1234) -> dict:
    """IoUs on fresh random directions (never seen in training labels)."""
    field = rebuild_field(record["program"], record["seed"], split)
    x = record["x"].unsqueeze(0).to(device)
    f = record["f"].unsqueeze(0).to(device)
    g = record["grad"].unsqueeze(0).to(device)
    gen = torch.Generator().manual_seed(seed)
    ii = torch.randperm(x.shape[1], generator=gen)[:n_input]
    lat = model.encode(x[:, ii], f[:, ii], g[:, ii], [record["tokens"]])

    surf = mfg_labels.project_to_surface(field, record["x"])
    out = {"access_iou": []}
    for _ in range(n_dirs):
        d = torch.randn(3, generator=gen)
        d = d / torch.linalg.vector_norm(d)
        s_tok = Token("spindle_dir", None, {"direction": d.clone()})
        _, true_acc = mfg_labels.tool_access(field, record["x"], d, surf=surf)
        pred_acc = model.decode(lat, x, "tool_access", [[s_tok]]).squeeze().cpu()
        band = record["f"].abs() < 0.05  # near-surface, where labels matter
        out["access_iou"].append(binary_iou(pred_acc[band], true_acc[band]))
    return {k: sum(v) / len(v) for k, v in out.items()}
