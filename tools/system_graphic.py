"""Render the GeoField system diagram (PNG + SVG for slides).

Dataset -> Design Agent -> CSG Builder -> Geometric Model -> Verifier, with the
training links and the candidate-ranking loop. Boxes auto-size to their content
so nothing overflows. Colors: dataviz reference palette, light mode.
Usage: python tools/system_graphic.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GREY = "#0b0b0b", "#52514e", "#7d8697", "#9aa0ab"
SURFACE, PANEL = "#fcfcfb", "#f4f4f1"

TITLE_GAP, BADGE_GAP, LINE_STEP, PAD_BOT = 3.1, 2.9, 2.45, 1.7

fig, ax = plt.subplots(figsize=(16.0, 9.0))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.set_xlim(0, 100)
ax.set_ylim(0, 69)
ax.axis("off")


def box_height(lines, badge):
    return (TITLE_GAP + (BADGE_GAP if badge else 0)
            + len(lines) * LINE_STEP + PAD_BOT)


def draw_box(x, top, w, title, lines, color, badge=None, fc=PANEL,
             height=None, title_size=11.5, lw=2.0):
    h = height or box_height(lines, badge)
    y = top - h
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.5,rounding_size=1.0",
                                linewidth=lw, edgecolor=color, facecolor=fc,
                                zorder=2))
    ty = top - 1.6
    ax.text(x + w / 2, ty, title, ha="center", va="top", fontsize=title_size,
            fontweight="bold", color=INK, zorder=3)
    ty -= TITLE_GAP - 0.3
    if badge:
        ax.text(x + w / 2, ty, badge, ha="center", va="top", fontsize=8.0,
                color=color, fontweight="bold", zorder=3)
        ty -= BADGE_GAP - 0.2
    for ln in lines:
        ax.text(x + 1.7, ty, ln, ha="left", va="top", fontsize=8.7,
                color=INK2, zorder=3)
        ty -= LINE_STEP
    return y, h


def arrow(x1, y1, x2, y2, color=INK2, lw=1.9, ls="-", rad=0.0, scale=15):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=scale, linewidth=lw,
                                 color=color, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))


def label(x, y, txt, color=MUTED, size=8.3, style="italic"):
    ax.text(x, y, txt, ha="center", va="center", fontsize=size, color=color,
            style=style, zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="none"))


# ---------------------------------------------------------------- header
ax.text(1.0, 67.8, "GeoField", fontsize=21, fontweight="bold", color=INK,
        va="top")
ax.text(16.0, 66.8, "verified generative engineering for shelf brackets",
        fontsize=12, color=INK2, va="top")
ax.text(1.0, 63.4, "box + load in  →  designed, stress-checked, "
        "solver-certified bracket out  ·  entirely on local hardware",
        fontsize=9.8, color=MUTED, va="top")

# ---------------------------------------------------------------- top row
TOP = 51.0
rows = [
    dict(x=1.0, w=17.6, title="YOUR REQUEST", color=GREY, fc="#ffffff",
         badge=None, lines=[
             "• envelope  W × reach × H",
             "• load  point, direction, N",
             "• material  Al 6061 / Steel",
             "• stress limit (MPa)"]),
    dict(x=19.8, w=19.0, title="DESIGN AGENT", color=ORANGE,
         badge="learned  ·  MLP + MC-dropout", lines=[
             "requirements → 15 design params",
             "• leg lengths, plate thicknesses",
             "• gusset / ribs / plain + sizes",
             "• bolt count & diameter, cutouts",
             "learns sizing judgment from pairs",
             "at 0.3–0.95 × yield utilization"]),
    dict(x=41.1, w=16.0, title="CSG BUILDER", color=AQUA,
         badge="rules  ·  exact, deterministic", lines=[
             "params → exact geometry",
             "• unit-gradient distance fields",
             "• plates, fillets, gussets, bolts",
             "• clamped to your envelope",
             "invalid parts are impossible"]),
    dict(x=59.4, w=19.2, title="GEOMETRIC MODEL", color=BLUE,
         badge="learned  ·  16.3 M params", lines=[
             "encodes each new design into a",
             "416-token shared latent, reads:",
             "• von Mises stress, displacement",
             "• machinability, wall thickness",
             "milliseconds/candidate → ranking",
             "SE(3)-equivariant to 4.5e-7"]),
    dict(x=81.0, w=17.5, title="VERIFIER", color=YELLOW,
         badge="solver  ·  no ML", lines=[
             "the judge, not a prediction",
             "• immersed-voxel FEA, multigrid",
             "• analytic machinability + holes",
             "• PASS / FAIL vs your limit",
             "±10% vs Timoshenko benchmark"]),
]
H_ROW = max(box_height(r["lines"], r["badge"]) for r in rows)
for r in rows:
    draw_box(r["x"], TOP, r["w"], r["title"], r["lines"], r["color"],
             badge=r["badge"], fc=r.get("fc", PANEL), height=H_ROW)
BOT = TOP - H_ROW
mid = BOT + H_ROW / 2
for x1, x2 in [(18.6, 19.8), (38.8, 41.1), (57.1, 59.4), (78.6, 81.0)]:
    arrow(x1, mid, x2, mid)

# candidate-ranking loop, above the row
arrow(69.0, TOP + 0.4, 29.5, TOP + 0.4, color=MUTED, ls=(0, (3, 2)), lw=1.6,
      rad=0.24)
label(49.0, TOP + 6.2, "4 candidates ranked, best surfaced  ·  ~2 s "
      "(vs ~4 min of raw solver time)", size=8.8)

# ---------------------------------------------------------------- dataset
DS_TOP = BOT - 8.5
ds_lines = [
    "500 parametric L-brackets  ×  10 load cases  ×  2 materials  =  10 000 FEA solves",
    "calibrated so peak stress spans 0.1–1.5 × yield  ·  labels: stress, deflection,",
    "machinability, thickness, mass  ·  held-out geometry corner + load band",
]
ds_y, ds_h = draw_box(19.8, DS_TOP, 38.5, "DATASET",
                      ds_lines, AQUA, fc="#eef7f3",
                      badge="the factory that taught both models  ·  ~50 min "
                            "on the DGX Spark  ·  seed-reproducible")

arrow(28.0, DS_TOP + 0.3, 28.0, BOT - 0.3, color=ORANGE, ls=(0, (4, 2)))
label(34.6, DS_TOP + 4.4, "trains the agent", color=ORANGE)
arrow(52.0, DS_TOP + 0.3, 66.5, BOT - 0.3, color=BLUE, ls=(0, (4, 2)), rad=-0.14)
label(62.5, DS_TOP + 5.2, "trains the surrogate", color=BLUE)

# ---------------------------------------------------------------- output
out_y, out_h = draw_box(81.0, BOT - 3.0, 17.5, "OUTPUT", [
    "• 3D view + field overlays",
    "• STL for CAD / CAM / print",
    "• pass/fail report + mass",
], GREY, fc="#ffffff", title_size=10.8)
arrow(89.7, BOT - 0.3, 89.7, BOT - 2.7, color=INK2)

# verifier also produced the training labels
arrow(81.0, out_y + out_h * 0.45, 58.6, ds_y + ds_h * 0.5, color=YELLOW,
      ls=(0, (4, 2)), lw=1.7, rad=0.16)
label(69.0, ds_y + 2.2, "same solver made every training label", color=YELLOW)

# ---------------------------------------------------------------- footer
ax.text(1.0, 2.0, "Learning where rules are weak (design judgment)  ·  rules "
        "where they are exact (geometry)  ·  simulation where trust matters "
        "(certification)", fontsize=9.6, color=INK2, style="italic", va="center")

for ext in ("png", "svg"):
    out = ROOT / f"system_diagram.{ext}"
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print("saved", out)
