"""Co-training loop for Stage A (geometry) and Stage B (all heads).

Usage: python -m geofield.train.loop --config geofield/train/configs/stage_a.yaml
Every run is reproducible from (seed, config). Checkpoints + JSONL metrics in
cfg.out_dir. Resumes from the latest checkpoint automatically.

Batch pipeline per step:
  1. draw a mixed batch (geometry-only + labeled records),
  2. optional random SE(3) augmentation (baseline: always; equivariant model:
     off by default, it is equivariant by construction),
  3. subsample n_input tokens for the encoder and n_query points for the
     decoder from the record's 16384 samples,
  4. input masking: with prob p_mask, drop encoder inputs inside a ball
     (r ~ U[0.1, 0.2]) at a random surface point; queries inside the ball get
     L_mask weight 2 on sdf/grad,
  5. encode (with the record's tokens), decode sdf(+grad via autograd) and
     every labeled field key present, apply losses.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..data.dataset import MixedBatchSampler, ShardDataset, collate
from ..fields.programs.common import random_rotation
from ..model.baseline import BaselineModel
from ..model.decoder import GeoFieldModel
from ..model.heads import field_specs
from ..tokens.schema import Token
from . import losses as L
from .schedule import lr_at


def build_model(cfg: dict):
    if cfg.get("model", "geofield") == "baseline":
        # the sdf-gradient losses double-differentiate through attention;
        # flash/mem-efficient SDPA kernels have no second derivative
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        return BaselineModel(dim=cfg.get("dim", 256),
                             n_layers=cfg.get("n_layers", 6),
                             m_latent=cfg.get("m_fine", 512) + cfg.get("m_coarse", 32))
    return GeoFieldModel(dim=cfg.get("dim", 256), n_layers=cfg.get("n_layers", 6),
                         m_fine=cfg.get("m_fine", 512),
                         m_coarse=cfg.get("m_coarse", 32),
                         k_input=cfg.get("k_input", 64))


def se3_augment(batch: dict, gen: torch.Generator) -> dict:
    R = random_rotation(gen).to(batch["x"].device)
    t = (torch.randn(3, generator=gen) * 0.05).to(batch["x"].device)
    batch = dict(batch)
    batch["x"] = batch["x"] @ R.T + t
    batch["grad"] = batch["grad"] @ R.T
    batch["tokens"] = [[tok.transform(R.cpu(), t.cpu()) for tok in rec]
                       for rec in batch["tokens"]]
    # displacement labels are vectors: rotate them
    for k in list(batch["fields"]):
        if k.startswith("displacement"):
            batch["fields"][k] = batch["fields"][k] @ R.T.cpu()
    return batch


def subsample(batch: dict, n_input: int, n_query: int, gen: torch.Generator):
    B, N, _ = batch["x"].shape
    ii = torch.randperm(N, generator=gen)[:n_input]
    qi = torch.randperm(N, generator=gen)[:n_query]
    return ii, qi


def training_step(model, batch, cfg, gen, device):
    x, f, g = batch["x"].to(device), batch["f"].to(device), batch["grad"].to(device)
    ii, qi = subsample(batch, cfg["n_input"], cfg["n_query"], gen)
    xi, fi, gi = x[:, ii], f[:, ii], g[:, ii]
    xq, fq_t, gq_t = x[:, qi], f[:, qi], g[:, qi]

    # ---- input masking (L_mask) ---------------------------------------------
    mask_w = torch.ones_like(fq_t)
    if torch.rand((), generator=gen).item() < cfg.get("p_mask", 0.5):
        r = 0.1 + 0.1 * torch.rand((), generator=gen).item()
        surf = (fi.abs() < 0.02)
        # random surface point per batch element (fallback: first input)
        ctr_idx = torch.where(
            surf.any(dim=1),
            torch.multinomial(surf.float() + 1e-6, 1).squeeze(-1),
            torch.zeros(x.shape[0], dtype=torch.long, device=device))
        ctr = xi[torch.arange(x.shape[0]), ctr_idx]              # [B,3]
        keep = torch.linalg.vector_norm(
            xi - ctr.unsqueeze(1), dim=-1) > r                    # [B,Ni]
        # drop by replacing with a kept token (keeps shapes rectangular)
        rep = keep.float().argmax(dim=1)
        idx = torch.where(keep, torch.arange(xi.shape[1], device=device).unsqueeze(0),
                          rep.unsqueeze(1))
        b_ix = torch.arange(x.shape[0], device=device).unsqueeze(1)
        xi, fi, gi = xi[b_ix, idx], fi[b_ix, idx], gi[b_ix, idx]
        inside_q = torch.linalg.vector_norm(xq - ctr.unsqueeze(1), dim=-1) < r
        mask_w = torch.where(inside_q, 2.0, 1.0)

    tokens = batch["tokens"]
    lat = model.encode(xi, fi, gi, tokens)

    fq, gq = model.sdf_and_grad(lat, xq)
    w = mask_w
    l_sdf = (w * torch.where(fq_t.abs() < L.BAND0, 5.0, 1.0)
             * (fq - fq_t).abs()).mean()
    l_grad = (w.unsqueeze(-1) * (gq - gq_t).abs()).mean()
    l_eik = L.eikonal_loss(gq, fq_t)
    total = l_sdf + l_grad + 0.1 * l_eik
    logs = {"sdf": l_sdf.item(), "grad": l_grad.item(), "eik": l_eik.item()}

    # ---- labeled fields -------------------------------------------------------
    if cfg.get("use_field_heads", False) and batch["fields"]:
        specs = field_specs()
        # decode a random subset of the labeled field keys each step: with 44
        # label sets per record (10 cases x 2 materials x 2 fields + mfg),
        # decoding all of them per step is ~5x wasted head work — the subset
        # rotates every step, so coverage in expectation is unchanged
        keys = list(batch["fields"].keys())
        k_max = cfg.get("fields_per_step", 8)
        if len(keys) > k_max:
            sel = torch.randperm(len(keys), generator=gen)[:k_max]
            keys = [keys[i] for i in sel.tolist()]
        requests = []
        for key in keys:
            fid = key.split("|")[0]
            if fid not in specs or fid == "fea_mask":
                continue
            cond = []
            for b_i, rec_conds in enumerate(batch["field_conditions"]):
                tok_ids = rec_conds.get(key, [])
                cond.append([tokens[b_i][t] for t in tok_ids
                             if t < len(tokens[b_i])])
            requests.append((key, fid, cond))
        preds = model.decode_many(lat, xq, requests, raw=False)
        for key, fid, _ in requests:
            spec = specs[fid]
            target = batch["fields"][key][:, qi].to(device)
            valid = batch["field_masks"][key].to(device)
            if fid in ("von_mises", "displacement") and f"fea_mask|{key.split('|')[1]}" in batch["fields"]:
                pass  # outside-solid FEA values already resampled to surface
            lf = L.field_loss(fid, spec.loss, preds[key], target, valid)
            total = total + spec.weight * lf
            logs[key] = lf.item()

    logs["total"] = total.item()
    return total, logs


def main():
    # our collated batches carry dozens of tensors; the default fd-passing
    # sharing strategy exhausts file descriptors with worker processes
    torch.multiprocessing.set_sharing_strategy("file_system")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 0))
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))

    ds = ShardDataset(cfg["data"], "train",
                      cache_shards=cfg.get("cache_shards", 64))
    # each optimizer step consumes grad_accum batches, so the sampler must
    # provide steps * accum batches (else training silently halts halfway)
    accum_n = cfg.get("grad_accum", 1)
    sampler = MixedBatchSampler(ds, cfg.get("n_geom", 8),
                                cfg.get("n_labeled", 24),
                                n_batches=cfg["steps"] * accum_n,
                                seed=cfg.get("seed", 0))
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                    num_workers=cfg.get("workers", 0))

    model = build_model(cfg).to(device)
    if cfg.get("init_from"):
        init = Path(cfg["init_from"])
        if init.is_dir():  # a run dir: take the latest checkpoint
            init = sorted(init.glob("ckpt_*.pt"))[-1]
        sd = torch.load(init, map_location=device, weights_only=True)
        model.load_state_dict(sd["model"], strict=False)
        print(f"initialized from {init}")
    # Stage B trunk protection: fresh heads need full lr, but the pretrained
    # shared trunk must not be shoved around by their large early gradients
    # (two collapses traced to exactly this). Param groups carry lr_mult.
    trunk_mult = cfg.get("trunk_lr_mult", 0.25 if cfg.get("use_field_heads") else 1.0)
    trunk_prefixes = ("embedder", "input_blocks", "pyramids",
                      "decoder.query_seed", "decoder.cross_")
    trunk, heads_p = [], []
    for n, p in model.named_parameters():
        (trunk if n.startswith(trunk_prefixes) else heads_p).append(p)
    opt = torch.optim.AdamW(
        [{"params": trunk, "lr_mult": trunk_mult},
         {"params": heads_p, "lr_mult": 1.0}],
        lr=cfg["lr"], weight_decay=cfg.get("wd", 0.01))
    start_step = 0
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    if ckpts:
        sd = torch.load(ckpts[-1], map_location=device, weights_only=True)
        model.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start_step = sd["step"]
        print(f"resumed from {ckpts[-1]} at step {start_step}")
    # compile AFTER all state loading. Whole-model compile would not engage
    # (encode/decode are custom methods), so compile each attention block in
    # place — parameters stay on the original module, checkpoints stay clean.
    if cfg.get("compile", False):
        from ..model.encoder import AttentionBlock
        n_c = 0
        for m in model.modules():
            if isinstance(m, AttentionBlock):
                m.forward = torch.compile(m.forward, dynamic=False)
                n_c += 1
        print(f"compiled {n_c} attention blocks")
    run_model = model

    log_path = out_dir / "metrics.jsonl"
    accum = cfg.get("grad_accum", 1)
    use_amp = device.startswith("cuda") and cfg.get("bf16", True)
    t0 = time.time()
    step = start_step
    recent: list = []  # spike guard window of (total, eik, sdf)
    n_skipped = 0
    skip_streak = 0
    lr_scale = 1.0     # halved by rollback-on-collapse
    high_eik_rows = 0  # consecutive LOGGED rows with degenerate eik

    def rollback():
        """Collapse self-healing: reload the newest checkpoint, halve lr."""
        nonlocal lr_scale, high_eik_rows
        cks = sorted(out_dir.glob("ckpt_*.pt"))
        if not cks:
            print("[rollback] no checkpoint to restore; continuing", flush=True)
            return
        sd_r = torch.load(cks[-1], map_location=device, weights_only=True)
        model.load_state_dict(sd_r["model"])
        opt.load_state_dict(sd_r["opt"])
        lr_scale *= 0.5
        high_eik_rows = 0
        recent.clear()
        print(f"[rollback] eik degenerate on 3 logged rows -> restored "
              f"{cks[-1].name}, lr_scale now {lr_scale}", flush=True)
    opt.zero_grad(set_to_none=True)
    for i, batch in enumerate(dl):
        if step >= cfg["steps"]:
            break
        if cfg.get("se3_augment", False):
            batch = se3_augment(batch, gen)
        lr = lr_at(step, cfg["steps"], cfg["lr"], cfg.get("warmup", 2000)) * lr_scale
        for pg in opt.param_groups:
            pg["lr"] = lr * pg.get("lr_mult", 1.0)
        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            loss, logs = training_step(run_model, batch, cfg, gen, device)
        # spike guard: one bad batch at full lr can throw the model into a
        # degenerate basin (observed twice: eik pinned at 1.0). The guard is
        # COMPONENT-AWARE: in Stage B the physics terms dominate `total`, so a
        # geometry-only collapse hides inside a normal-looking total — watch
        # sdf and eik against their own medians too.
        def _med(xs):
            return sorted(xs)[len(xs) // 2] if len(xs) >= 16 else None

        med = _med([r[0] for r in recent])
        med_eik = _med([r[1] for r in recent])
        med_sdf = _med([r[2] for r in recent])
        eik_v, sdf_v = logs.get("eik", 0.0), logs.get("sdf", 0.0)
        # margins have absolute FLOORS so a lucky low median can't wedge the
        # guard shut, and eik carries the collapse detection (its degenerate
        # value 1.0 is unambiguous); sdf/total guards are coarse backstops
        bad = (not torch.isfinite(loss)
               or (med is not None and loss.item() > max(6 * med, med + 3.0))
               or (med_eik is not None and eik_v > max(0.45, 6 * med_eik))
               or (med_sdf is not None and sdf_v > max(0.30, 6 * med_sdf)))
        # deadlock breaker: a long skip streak means the reference window is
        # stale (the loss distribution legitimately moved) — accept one batch
        # to refresh it rather than spinning forever (observed: 1700+ skips)
        if bad and skip_streak >= 25:
            print(f"[guard] streak-breaker: accepting batch after "
                  f"{skip_streak} consecutive skips to refresh reference",
                  flush=True)
            bad = False
            recent.clear()
        if bad:
            opt.zero_grad(set_to_none=True)
            n_skipped += 1
            skip_streak += 1
            if n_skipped % 5 == 1:
                print(f"[guard] skipped spike step (total {loss.item():.3f}/"
                      f"med {med}, eik {eik_v:.3f}/med {med_eik}, "
                      f"sdf {sdf_v:.3f}/med {med_sdf}), skipped {n_skipped}",
                      flush=True)
            continue
        skip_streak = 0
        recent.append((loss.item(), eik_v, sdf_v))
        if len(recent) > 64:
            recent.pop(0)
        (loss / accum).backward()
        if (i + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.get("log_every", 20) == 0:
                logs.update(step=step, lr=lr, sec=round(time.time() - t0, 1))
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(logs) + "\n")
                # rollback-on-collapse: streak-breakers can re-baseline the
                # guard to a collapsed state, so the LOGGED eik is the final
                # arbiter — 3 consecutive degenerate rows trigger a restore
                if logs.get("eik", 0.0) > 0.6 and step > 300:
                    high_eik_rows += 1
                    if high_eik_rows >= 3:
                        rollback()
                else:
                    high_eik_rows = 0
                print(f"step {step} " + " ".join(
                    f"{k}={v:.4f}" for k, v in logs.items()
                    if isinstance(v, float)), flush=True)
            if step % cfg.get("ckpt_every", 5000) == 0 or step == cfg["steps"]:
                if logs.get("eik", 0.0) > 0.5:
                    print(f"[ckpt] SKIPPED saving at step {step}: eik "
                          f"{logs['eik']:.3f} looks collapsed", flush=True)
                else:
                    torch.save({"model": model.state_dict(),
                                "opt": opt.state_dict(),
                                "step": step, "cfg": cfg},
                               out_dir / f"ckpt_{step:07d}.pt")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step, "cfg": cfg}, out_dir / f"ckpt_{step:07d}.pt")
    print(f"done at step {step}")


if __name__ == "__main__":
    main()
