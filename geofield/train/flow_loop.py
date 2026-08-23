"""Stage C: train the conditional flow-matching generator over encoded latents.

Usage: python -m geofield.train.flow_loop --config geofield/train/configs/stage_c.yaml
Config keys: latents (dir from encode_latents), out_dir, steps, lr, batch,
flow_dim, flow_layers, p_drop, seed, device, ckpt_every.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..model.flow import LatentFlow, LatentNormalizer, cfm_loss
from ..tokens.schema import Token
from .schedule import lr_at


class LatentShards:
    def __init__(self, lat_dir: str, split: str = "train"):
        self.paths = sorted(Path(lat_dir).glob(f"{split}_latents_*.npz"))
        assert self.paths, f"no latent shards in {lat_dir}"
        zs, toks = [], []
        for p in self.paths:
            with np.load(p) as z:
                zs.append(torch.from_numpy(z["z"]))
                toks.extend(json.loads(str(z["tokens"])))
        self.z = torch.cat(zs)
        self.tokens = [[Token.from_dict(d) for d in rec] for rec in toks]
        assert len(self.tokens) == self.z.shape[0]

    def batch(self, n: int, gen: torch.Generator):
        idx = torch.randint(self.z.shape[0], (n,), generator=gen)
        return self.z[idx], [self.tokens[i] for i in idx.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 0))
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))

    stats = torch.load(Path(cfg["latents"]) / "latent_stats.pt",
                       map_location="cpu", weights_only=True)
    norm = LatentNormalizer.load(stats)
    data = LatentShards(cfg["latents"], "train")
    print(f"{data.z.shape[0]} latents, dim {tuple(data.z.shape[1:])}")

    flow = LatentFlow(latent_dim=stats["latent_dim"], m_fine=stats["m_fine"],
                      m_coarse=data.z.shape[1] - stats["m_fine"],
                      dim=cfg.get("flow_dim", 384),
                      n_layers=cfg.get("flow_layers", 8),
                      k=cfg.get("flow_k", 32)).to(device)
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"flow params: {n_params / 1e6:.1f}M")
    opt = torch.optim.AdamW(flow.parameters(), lr=cfg["lr"], weight_decay=0.01)

    start = 0
    ckpts = sorted(out_dir.glob("ckpt_*.pt"))
    if ckpts:
        sd = torch.load(ckpts[-1], map_location=device, weights_only=True)
        flow.load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        start = sd["step"]
        print(f"resumed at {start}")

    use_amp = device.startswith("cuda") and cfg.get("bf16", True)
    t0 = time.time()
    for step in range(start, cfg["steps"]):
        z1, toks = data.batch(cfg.get("batch", 64), gen)
        z1 = norm.norm(z1).to(device)
        lr = lr_at(step, cfg["steps"], cfg["lr"], cfg.get("warmup", 2000))
        for pg in opt.param_groups:
            pg["lr"] = lr
        with torch.autocast("cuda", torch.bfloat16, enabled=use_amp):
            loss = cfm_loss(flow, z1, toks, gen, p_drop=cfg.get("p_drop", 0.15))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()
        if (step + 1) % cfg.get("log_every", 50) == 0:
            with open(out_dir / "metrics.jsonl", "a") as fh:
                fh.write(json.dumps({"step": step + 1, "loss": loss.item(),
                                     "lr": lr, "sec": round(time.time() - t0, 1)}) + "\n")
            print(f"step {step + 1} loss {loss.item():.4f}", flush=True)
        if (step + 1) % cfg.get("ckpt_every", 10000) == 0 or step + 1 == cfg["steps"]:
            torch.save({"model": flow.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "cfg": cfg, "stats": stats},
                       out_dir / f"ckpt_{step + 1:07d}.pt")
    print("flow training done")


if __name__ == "__main__":
    main()
