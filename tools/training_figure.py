"""Render docs/training_curves.png from the Stage A / Stage B metrics logs.

Usage: python tools/training_figure.py /tmp/gf/stage_a.jsonl /tmp/gf/stage_b.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#7d8697"


def load(p):
    rows = []
    for ln in Path(p).read_text(errors="ignore").splitlines():
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return rows


a = load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/gf/stage_a.jsonl")
b = load(sys.argv[2] if len(sys.argv) > 2 else "/tmp/gf/stage_b.jsonl")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
fig.patch.set_facecolor("#fcfcfb")

# Stage A: geometry
ax = axes[0]
for key, col, lab in [("sdf", AQUA, "distance-field error"),
                      ("eik", YELLOW, "unit-gradient violation"),
                      ("grad", BLUE, "surface-normal error")]:
    xs = [r["step"] for r in a if key in r]
    ys = [r[key] for r in a if key in r]
    ax.plot(xs, ys, color=col, lw=1.3, label=lab)
ax.set_yscale("log")
ax.set_xlabel("step", color=MUTED)
ax.set_title("Stage A — learning geometry\n(one latent must reconstruct the "
             "whole distance field)", fontsize=11, color=INK)
ax.legend(fontsize=8.5, frameon=False)
ax.grid(alpha=0.18)

# Stage B: physics added on the same latent
ax = axes[1]
vm = [(r["step"], sum(v for k, v in r.items() if k.startswith("von_mises"))
       / max(1, sum(1 for k in r if k.startswith("von_mises"))))
      for r in b if any(k.startswith("von_mises") for k in r)]
dp = [(r["step"], sum(v for k, v in r.items() if k.startswith("displacement"))
       / max(1, sum(1 for k in r if k.startswith("displacement"))))
      for r in b if any(k.startswith("displacement") for k in r)]
if vm:
    ax.plot([s for s, _ in vm], [v for _, v in vm], color=ORANGE, lw=1.3,
            label="von Mises stress error")
if dp:
    ax.plot([s for s, _ in dp], [v for _, v in dp], color=BLUE, lw=1.3,
            label="displacement error")
xs = [r["step"] for r in b if "eik" in r]
ax.plot(xs, [r["eik"] for r in b if "eik" in r], color=YELLOW, lw=1.1,
        ls="--", label="geometry held (unit-gradient)")
ax.set_yscale("log")
ax.set_xlabel("step", color=MUTED)
ax.set_title("Stage B — physics on the SAME latent\n(geometry must not "
             "regress while stress/deflection are learned)", fontsize=11,
             color=INK)
ax.legend(fontsize=8.5, frameon=False)
ax.grid(alpha=0.18)

for ax in axes:
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("Training the geometric model: geometry first, then physics "
             "on top of the same representation", fontsize=12.5, color=INK)
fig.tight_layout()
out = ROOT / "docs" / "training_curves.png"
fig.savefig(out, dpi=130, facecolor="#fcfcfb")
print("saved", out)
