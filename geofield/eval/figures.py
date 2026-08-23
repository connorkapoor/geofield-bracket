"""`make figures`: run the full Phase-3 evaluation from checkpoints and write
results.json + figures/*.png.

Usage:
  python -m geofield.eval.figures --data data/v0-pilot \\
      --model runs/stage_b/ckpt_XXXX.pt [--geom runs/stage_a/ckpt_XXXX.pt]
      [--baseline runs/baseline/ckpt_XXXX.pt] [--n 64] [--out results]
Every metric that a missing checkpoint would need is skipped, so this can run
after Stage A alone and be re-run as later stages land.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..data.dataset import ShardDataset
from ..data.gallery import rebuild_field
from ..train.loop import build_model
from . import closeness, mfg, probes, reconstruction, rotation, surrogate


def load_model(path: str, device: str):
    sd = torch.load(path, map_location=device, weights_only=True)
    model = build_model(sd["cfg"]).to(device)
    model.load_state_dict(sd["model"])
    model.eval()
    return model


def take(ds, n, stride=None):
    stride = stride or max(1, len(ds) // n)
    return [ds[i] for i in range(0, min(len(ds), n * stride), stride)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v0-pilot")
    ap.add_argument("--model", required=True, help="Stage-B (or -A) checkpoint")
    ap.add_argument("--geom", default=None, help="Stage-A geometry-only ckpt")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    dev = args.device

    models = {"model": load_model(args.model, dev)}
    if args.baseline:
        models["baseline"] = load_model(args.baseline, dev)
    if args.geom:
        models["geom_only"] = load_model(args.geom, dev)

    results: dict = {}
    val_b = ShardDataset(args.data, "val")
    recs_val = take(val_b, args.n)
    splits_ood = {}
    for s in ("ood_geometry", "ood_load"):
        try:
            splits_ood[s] = take(ShardDataset(args.data, s), max(args.n // 2, 8))
        except FileNotFoundError:
            pass

    # 1. rotation ------------------------------------------------------------
    results["rotation"] = {}
    for name, m in models.items():
        errs = sum((rotation.equivariance_error(m, r, n_rot=8, device=dev)
                    for r in recs_val[:8]), [])
        sim = rotation.pooled_rotation_similarity(m, recs_val[:12], device=dev)
        results["rotation"][name] = {
            "equiv_err_median": float(torch.tensor(errs).median()),
            "equiv_err_p95": float(torch.tensor(errs).quantile(0.95)),
            "pooled_similarity": sim}
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, r in results["rotation"].items():
        angs = sorted(r["pooled_similarity"])
        ax.plot(angs, [r["pooled_similarity"][a]["same"] for a in angs],
                marker="o", label=f"{name} same")
        ax.plot(angs, [r["pooled_similarity"][a]["diff"] for a in angs],
                marker="x", ls="--", label=f"{name} diff")
    ax.set_xlabel("rotation angle (deg)")
    ax.set_ylabel("pooled cosine similarity")
    ax.legend(fontsize=8)
    fig.savefig(out / "figures" / "rotation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2/3. closeness + reconstruction -----------------------------------------
    for name, m in models.items():
        res = {"context_pose_spearman": float(torch.tensor(
            [closeness.context_pose_spearman(m, r, device=dev)
             for r in recs_val[:8]]).mean())}
        for split_name, recs in [("val", recs_val)] + list(splits_ood.items()):
            mets = []
            for r in recs[: max(args.n // 2, 8)]:
                tf = rebuild_field(r["program"], r["seed"],
                                   split_name if split_name != "val" else "val")
                mets.append(reconstruction.reconstruction_metrics(
                    m, r, device=dev, true_field=tf))
            res[split_name] = {
                k: float(torch.tensor([x[k] for x in mets]).nanmedian())
                for k in mets[0]}
        res["masked"] = float(torch.tensor(
            [reconstruction.masked_completion_metrics(m, r, device=dev)["masked_sdf_mae"]
             for r in recs_val[:8]]).nanmedian())
        results.setdefault("reconstruction", {})[name] = res

    # 4. surrogate -------------------------------------------------------------
    ood_b = splits_ood.get("ood_load", recs_val)
    for name, m in models.items():
        agg = surrogate.aggregate_surrogate(
            [surrogate.surrogate_metrics(m, r, device=dev) for r in ood_b])
        results.setdefault("surrogate", {})[name] = agg
    if results["surrogate"].get("model", {}).get("err_vs_load_distance"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, agg in results["surrogate"].items():
            curve = agg.get("err_vs_load_distance", [])
            if curve:
                ax.plot([c["dist"] for c in curve], [c["rel_mae"] for c in curve],
                        marker="o", label=name)
        ax.set_xlabel("distance from load token")
        ax.set_ylabel("relative VM MAE")
        ax.legend()
        fig.savefig(out / "figures" / "err_vs_load_distance.png", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)

    # 5. manufacturability (held-out directions) --------------------------------
    for name, m in models.items():
        ms = [mfg.mfg_metrics(m, r, "val", device=dev) for r in recs_val[:8]]
        results.setdefault("mfg", {})[name] = {
            k: float(torch.tensor([x[k] for x in ms]).nanmean()) for k in ms[0]}

    # 6. probes ------------------------------------------------------------------
    probe_recs = take(val_b, min(len(val_b), 3000), stride=1)
    fig, axes = plt.subplots(1, len(probes.PROBE_TARGETS), figsize=(16, 3.5))
    for name, m in models.items():
        z, targets = probes.collect_pooled(m, probe_recs, device=dev)
        panel = probes.probe_panel(z, targets)
        results.setdefault("probes", {})[name] = panel
        for ax, tgt in zip(axes, probes.PROBE_TARGETS):
            if tgt in panel:
                ns = sorted(panel[tgt])
                ax.plot(ns, [panel[tgt][n] for n in ns], marker="o", label=name)
                ax.set_title(tgt, fontsize=9)
                ax.set_xscale("log")
                ax.set_xlabel("N probe shapes")
    axes[0].set_ylabel("spearman")
    axes[0].legend(fontsize=8)
    fig.savefig(out / "figures" / "probes.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"results -> {out / 'results.json'}")


if __name__ == "__main__":
    main()
