"""Render progress.png: one image summarizing generation + training state.

Reads status.json (written by tools/status.py) and training metrics.jsonl
files (local + Spark via ssh) and draws progress bars + loss curves.
Usage: python tools/snapshot.py [--out progress.png]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def load_metrics(name: str) -> list[dict]:
    p = ROOT / "runs" / name / "metrics.jsonl"
    rows = []
    if p.exists():
        for ln in p.read_text(errors="ignore").splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def load_remote_metrics() -> dict[str, list[dict]]:
    runs = {}
    for host in ("spark", "h100"):
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", host,
                 "for f in ~/geofield/runs/*/metrics.jsonl; do [ -f $f ] && "
                 "echo \"===$(basename $(dirname $f))\" && cat $f; done"],
                capture_output=True, text=True, timeout=25).stdout
        except Exception:
            continue
        for chunk in out.split("===")[1:]:
            name, _, body = chunk.partition("\n")
            rows = []
            for ln in body.splitlines():
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
            if rows:
                runs[f"{host}:{name.strip()}"] = rows
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "progress.png"))
    args = ap.parse_args()

    status = {}
    sp = ROOT / "status.json"
    if sp.exists():
        status = json.loads(sp.read_text())

    runs = {f"local:{p.parent.name}": load_metrics(p.parent.name)
            for p in (ROOT / "runs").glob("*/metrics.jsonl")} \
        if (ROOT / "runs").exists() else {}
    runs.update(load_remote_metrics())
    runs = {k: v for k, v in runs.items() if v}

    n_ax = 1 + min(len(runs), 3)
    fig, axes = plt.subplots(1, n_ax, figsize=(5.5 * n_ax, 4))
    axes = [axes] if n_ax == 1 else list(axes)

    # panel 1: generation progress bars
    ax = axes[0]
    bars, labels = [], []
    sg = status.get("spark_generation", {})
    if sg.get("total"):
        bars.append(100 * sg["done"] / sg["total"])
        eta = f" ETA {sg['eta_h']}h" if sg.get("eta_h") else ""
        labels.append(f"Spark data l1\n{sg['done']}/{sg['total']}{eta}")
    lp = status.get("local_generation", {}).get("progress", {})
    if lp.get("total"):
        bars.append(100 * lp["done"] / lp["total"])
        labels.append(f"local pilot\n{lp['done']}/{lp['total']}")
    if bars:
        ax.barh(range(len(bars)), bars, color="#3d7bfd")
        ax.set_yticks(range(len(bars)), labels, fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("%")
        ax.bar_label(ax.containers[0], fmt="%.0f%%")
    ax.set_title(f"dataset generation  ({datetime.now():%H:%M})", fontsize=10)

    # panels 2+: loss curves per training run
    for ax, (name, rows) in zip(axes[1:], sorted(runs.items())):
        steps = [r.get("step", 0) for r in rows]
        for key, color in [("total", "#3d7bfd"), ("loss", "#3d7bfd"),
                           ("sdf", "#41d6a6"), ("eik", "#f2b134")]:
            ys = [r.get(key) for r in rows]
            if any(y is not None for y in ys):
                pts = [(s, y) for s, y in zip(steps, ys) if y is not None]
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        label=key, color=color, lw=1.2)
        ax.set_yscale("log")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"snapshot -> {args.out}")


if __name__ == "__main__":
    main()
