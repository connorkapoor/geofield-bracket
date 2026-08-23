"""Train the parameter-designer on (requirement -> adequate design) pairs.

Pairs are mined from the labeled dataset: for each bracket and each
(load case, material), the calibrated yield fraction tells us whether THIS
bracket is a sensible design for THAT load (utilization 0.30-0.95: neither
failing nor 10x overbuilt). Only in-band pairs supervise the designer.
Canonical-frame load positions are regenerated deterministically from the
record seed, so no rotation contamination.

Usage: python -m geofield.train.designer_loop --data data/l1 [--data2 data/l2]
       --out runs/designer [--steps 4000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..data.dataset import ShardDataset
from ..labels.physics import L_MATERIALS, MATERIALS, l_load_cases
from ..model.designer import (CONT, LIGHT, RANGES, REINF, ParamDesigner,
                              designer_loss, featurize)


def mine_pairs(data_dir: str, splits=("train", "val")) -> tuple:
    feats, cont, reinf, light, holes, masks = [], [], [], [], [], []
    n_rec = 0
    for split in splits:
        try:
            ds = ShardDataset(data_dir, split)
        except FileNotFoundError:
            continue
        for i in range(len(ds)):
            rec = ds[i]
            meta = rec.get("meta") or {}
            if not meta.get("Lw"):
                continue
            n_rec += 1
            cases = l_load_cases(meta, rec["seed"], n_cases=10,
                                 ood_load=(split == "ood_load"))
            lo = torch.tensor(meta["aabb"]["lo"])
            hi = torch.tensor(meta["aabb"]["hi"])
            env = torch.tensor([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]])
            # target parameter vector (same for all this record's pairs)
            g = meta.get("gusset") or {}
            r = meta.get("ribs") or {}
            tgt = {"Lw": meta["Lw"], "Lf": meta["Lf"], "W": meta["W"],
                   "tw": meta["tw"], "tf": meta["tf"],
                   "fillet": meta.get("fillet", 6.0),
                   "gusset_a": g.get("a", 0.0) / max(meta["Lf"] - meta["tw"], 1),
                   "gusset_b": g.get("b", 0.0) / max(meta["Lw"] - meta["tf"], 1),
                   "ribs_r": r.get("r", 0.45 * min(meta["tw"], meta["tf"]))
                             / min(meta["tw"], meta["tf"]),
                   "hole_dia": (meta.get("mount") or {}).get("dia", 6.6)}
            tvec = torch.tensor([
                (min(max(tgt[k], RANGES[k][0]), RANGES[k][1]) - RANGES[k][0])
                / (RANGES[k][1] - RANGES[k][0]) for k in CONT])
            # MASK the brace-geometry slots on records that have no brace:
            # regressing them toward 0 on 2/3 of the data collapsed the
            # prediction, which produced decorative 20%-reach "gussets"
            braced = meta.get("reinforcement", "none") != "none"
            mvec = torch.ones(len(CONT))
            if not braced:
                for k in ("gusset_a", "gusset_b", "ribs_r"):
                    mvec[CONT.index(k)] = 0.0
            t_reinf = REINF.index(meta.get("reinforcement", "none"))
            t_light = LIGHT.index(meta.get("lightweight", "none"))
            t_holes = min(max((meta.get("mount") or {}).get("n", 3), 2), 4) - 2
            for ci, case in enumerate(cases):
                for m in L_MATERIALS:
                    yf = rec["scalars"].get(f"yield_fraction|lc{ci}|{m}")
                    F = rec["scalars"].get(f"magnitude_N|lc{ci}|{m}")
                    if yf is None or F is None or not (0.30 <= yf <= 0.95):
                        continue
                    pos = case["pos_mm"]
                    pf = torch.tensor([
                        0.5,
                        float(pos[1]) / max(float(env[1]), 1.0),
                        float(pos[2] - lo[2]) / max(float(env[2]), 1.0)])
                    # canonical frame: wall inner face at y=0, so pos[1] IS
                    # the moment arm in mm
                    feats.append(featurize(env, pf, case["dir"], F,
                                           MATERIALS[m]["E"],
                                           MATERIALS[m]["yield"],
                                           arm_mm=float(pos[1])))
                    cont.append(tvec)
                    masks.append(mvec)
                    reinf.append(t_reinf)
                    light.append(t_light)
                    holes.append(t_holes)
    print(f"mined {len(feats)} pairs from {n_rec} records")
    return (torch.stack(feats), torch.stack(cont),
            torch.tensor(reinf), torch.tensor(light), torch.tensor(holes),
            torch.stack(masks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/l1")
    ap.add_argument("--data2", default=None)
    ap.add_argument("--out", default="runs/designer")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    packs = [mine_pairs(args.data)]
    if args.data2 and Path(args.data2).exists():
        packs.append(mine_pairs(args.data2))
    X = torch.cat([p[0] for p in packs])
    C = torch.cat([p[1] for p in packs])
    R = torch.cat([p[2] for p in packs])
    L = torch.cat([p[3] for p in packs])
    K = torch.cat([p[4] for p in packs])
    M = torch.cat([p[5] for p in packs])

    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X), generator=gen)
    n_val = max(64, len(X) // 10)
    va, tr = perm[:n_val], perm[n_val:]

    model = ParamDesigner()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = 1e9
    for step in range(1, args.steps + 1):
        idx = tr[torch.randint(len(tr), (args.batch,), generator=gen)]
        loss = designer_loss(model(X[idx]), C[idx], R[idx], L[idx], K[idx],
                             mask=M[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            model.eval()
            with torch.no_grad():
                vl = designer_loss(model(X[va]), C[va], R[va], L[va], K[va],
                                   mask=M[va])
                pred = model(X[va])
                acc_r = (pred[1].argmax(1) == R[va]).float().mean()
                mae_mm = ((pred[0][:, 0] - C[va][:, 0]).abs().mean()
                          * (RANGES["Lw"][1] - RANGES["Lw"][0]))
            model.train()
            print(f"step {step} train {loss.item():.4f} val {vl.item():.4f} "
                  f"reinf_acc {acc_r.item():.2f} Lw_mae {mae_mm.item():.1f}mm",
                  flush=True)
            if vl.item() < best:
                best = vl.item()
                torch.save({"model": model.state_dict(), "val": best},
                           out / "designer.pt")
    print(f"best val {best:.4f} -> {out / 'designer.pt'}")


if __name__ == "__main__":
    main()
