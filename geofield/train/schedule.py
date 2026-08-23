"""LR schedule: linear warmup then cosine decay to lr_min."""
from __future__ import annotations

import math


def lr_at(step: int, total_steps: int, lr: float, warmup: int = 2000,
          lr_min_ratio: float = 0.05) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    t = (step - warmup) / max(total_steps - warmup, 1)
    return lr * (lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))
