"""Live observability dashboard -> status.html (auto-refresh) + status.json.

Per 45s tick it gathers, across this PC / the Spark / the H100:
  * full training histories (metrics.jsonl) -> SVG loss charts w/ crosshair
  * pipeline stage states (parsed from pipeline logs + checkpoints)
  * per-run progress, ETA, s/step, learning rate
  * GPU util/mem/temp tiles, H100 cost meter
  * event feed (pipeline log lines), dataset summary (manifest.json)

Chart colors are the dataviz reference palette's dark-mode categorical slots
(validated adjacent-pair order for line charts on a dark surface).
Run forever: python tools/status.py [--interval 45]; one shot: --once.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HOSTS = ("spark", "h100")
STEPS_TOTAL = {"stage_a_l1": 6000, "stage_b_l1": 8000,
               "baseline_l1": 6000, "stage_c": 10000}
H100_RATE = 4.29
H100_START = datetime(2026, 8, 22, 21, 5)

# dataviz reference palette, dark-mode steps (validated order)
SERIES = [("total", "#3987e5"), ("sdf", "#d95926"),
          ("grad", "#199e70"), ("eik", "#c98500")]
INK1, INK2, MUTED = "#ffffff", "#c3c2b7", "#7d8697"
SURFACE, CARD = "#0e0f13", "#1a1a19"
GOOD, RUN, PEND, BAD = "#008300", "#3987e5", "#4a4f5c", "#e66767"


def sh(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout
    except Exception:
        return ""


def ssh(host: str, cmd: str, timeout: int = 20) -> str:
    return sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6",
               host, cmd], timeout)


# ---------------------------------------------------------------------------
# gathering
# ---------------------------------------------------------------------------

def gather_runs() -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}

    def parse(text: str):
        rows = []
        for ln in text.splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if "total" not in r and "loss" in r:
                r["total"] = r["loss"]  # flow runs log a single 'loss'
            rows.append(r)
        return rows

    if (ROOT / "runs").exists():
        for mp in (ROOT / "runs").glob("*/metrics.jsonl"):
            rows = parse(mp.read_text(errors="ignore"))
            if rows:
                runs[f"local:{mp.parent.name}"] = rows
    for host in HOSTS:
        out = ssh(host, "for f in ~/geofield/runs/*/metrics.jsonl; do "
                        "[ -f $f ] && echo \"===$(basename $(dirname $f))\" && cat $f; done")
        for chunk in out.split("===")[1:]:
            name, _, body = chunk.partition("\n")
            rows = parse(body)
            if rows:
                runs[f"{host}:{name.strip()}"] = rows
    return runs


def gather_pipeline_events() -> list[str]:
    events = []
    for host, logs in (("spark", "logs/baseline.log"),
                       ("h100", "logs/pipeline.log logs/stage_c.log")):
        out = ssh(host, f"cd ~/geofield && grep -hE '\\[pipeline|\\[stage-c|Error|Traceback' {logs} 2>/dev/null | tail -6")
        for ln in out.splitlines():
            if ln.strip():
                events.append(f"[{host}] {ln.strip()[:140]}")
    return events[-12:]


def gather_stages(runs: dict, events: list[str]) -> list[dict]:
    """Pipeline stage chips with states derived from runs + logs."""
    ev = " ".join(events)

    def run_state(key: str, total: int):
        r = runs.get(key)
        if not r:
            return "pending", ""
        step = r[-1].get("step", 0)
        if step >= total:
            return "done", f"{step}/{total}"
        return "running", f"{step}/{total}"

    stages = [{"name": "dataset l1 (580 rec)", "state": "done", "info": "10k solves"}]
    for label, key, tot in [("Stage A geometry", "h100:stage_a_l1", 6000),
                            ("Stage B all heads", "h100:stage_b_l1", 8000),
                            ("baseline (control)", "spark:baseline_l1", 6000)]:
        st, info = run_state(key, tot)
        stages.append({"name": label, "state": st, "info": info})
    lat = "done" if "latents encoded" in ev else "pending"
    stages.append({"name": "encode latents", "state": lat, "info": ""})
    st, info = run_state("h100:stage_c", STEPS_TOTAL["stage_c"])
    if st == "pending" and "flow trained" in ev:
        st = "done"
    stages.append({"name": "Stage C generator", "state": st, "info": info})
    stages.append({"name": "eval figures",
                   "state": "done" if "ALL DONE" in ev else "pending", "info": ""})
    if "Traceback" in ev or "Error" in ev:
        for s in stages:
            if s["state"] == "running":
                s["info"] += " (!) check events"
    return stages


def gather_gpus() -> dict:
    q = ("--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
         "--format=csv,noheader")
    out = {"4060 (this PC)": sh(["nvidia-smi"] + q.split()).strip() or "-"}
    out["GB10 (spark)"] = ssh("spark", f"nvidia-smi {q}").strip() or "-"
    out["H100 (lambda)"] = ssh("h100", f"nvidia-smi {q}").strip() or "-"
    return out


def gather_dataset() -> dict:
    out = ssh("h100", "cat ~/geofield/data/l1/manifest.json 2>/dev/null")
    try:
        m = json.loads(out)
        return {k: v["records"] for k, v in m["splits"].items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# svg chart (line chart, log y, crosshair tooltip layer)
# ---------------------------------------------------------------------------

def svg_chart(rows: list[dict], cid: str, w: int = 560, h: int = 200) -> str:
    pad_l, pad_r, pad_t, pad_b = 44, 88, 10, 22
    pts = {name: [(r["step"], r[name]) for r in rows
                  if isinstance(r.get(name), (int, float)) and r[name] > 0]
           for name, _ in SERIES}
    allv = [v for s in pts.values() for _, v in s]
    if not allv:
        return "<div class='small'>no data yet</div>"
    x_max = max(s for s_list in pts.values() for s, _ in s_list)
    x_min = min(s for s_list in pts.values() for s, _ in s_list)
    x_max = max(x_max, x_min + 1)
    lo, hi = min(allv) * 0.85, max(allv) * 1.15
    llo, lhi = math.log10(lo), math.log10(hi)

    def X(s):
        return pad_l + (s - x_min) / (x_max - x_min) * (w - pad_l - pad_r)

    def Y(v):
        return pad_t + (lhi - math.log10(v)) / (lhi - llo) * (h - pad_t - pad_b)

    parts = [f"<svg class='chart' data-cid='{cid}' width='{w}' height='{h}' "
             f"viewBox='0 0 {w} {h}'>"]
    # recessive grid: log decades + halves within range
    tick = 10 ** math.floor(llo)
    ticks = []
    while tick <= hi * 1.01:
        for m in (1, 2, 5):
            tv = tick * m
            if lo <= tv <= hi:
                ticks.append(tv)
        tick *= 10
    for tv in ticks:
        y = Y(tv)
        parts.append(f"<line x1='{pad_l}' y1='{y:.1f}' x2='{w - pad_r}' y2='{y:.1f}' "
                     f"stroke='#252a35' stroke-width='1'/>")
        parts.append(f"<text x='{pad_l - 6}' y='{y + 3:.1f}' text-anchor='end' "
                     f"font-size='10' fill='{MUTED}'>{tv:g}</text>")
    for name, color in SERIES:
        s_list = pts.get(name)
        if not s_list:
            continue
        d = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in s_list)
        parts.append(f"<polyline points='{d}' fill='none' stroke='{color}' "
                     f"stroke-width='2' stroke-linejoin='round'/>")
        ls, lv = s_list[-1]
        parts.append(
            f"<text x='{X(ls) + 5:.1f}' y='{Y(lv) + 3:.1f}' font-size='10' "
            f"fill='{INK2}'>"
            f"<tspan fill='{color}'>&#9679;</tspan> {name} {lv:.3g}</text>")
    parts.append(f"<text x='{w - pad_r}' y='{h - 6}' text-anchor='end' "
                 f"font-size='10' fill='{MUTED}'>step</text>")
    # crosshair layer (populated by shared JS)
    parts.append(f"<line class='xh' x1='0' x2='0' y1='{pad_t}' y2='{h - pad_b}' "
                 f"stroke='{MUTED}' stroke-width='1' visibility='hidden'/>")
    parts.append("</svg>")
    data = {name: pts.get(name, []) for name, _ in SERIES}
    parts.append(f"<script type='application/json' id='data-{cid}'>"
                 f"{json.dumps({'x0': x_min, 'x1': x_max, 'lo': lo, 'hi': hi, 'pts': data, 'pl': pad_l, 'pr': pad_r, 'pt': pad_t, 'pb': pad_b, 'w': w, 'h': h})}</script>")
    return "".join(parts)


CROSSHAIR_JS = """
<div id='tip' style='position:fixed;display:none;background:#232833;color:#c3c2b7;
 border:1px solid #3a4150;border-radius:6px;padding:6px 9px;font-size:11px;
 pointer-events:none;z-index:9;font-family:ui-monospace,monospace'></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('svg.chart').forEach(svg => {
  const cfg = JSON.parse(document.getElementById('data-' + svg.dataset.cid).textContent);
  const xh = svg.querySelector('.xh');
  svg.addEventListener('mousemove', e => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) * cfg.w / r.width;
    const frac = Math.min(1, Math.max(0, (px - cfg.pl) / (cfg.w - cfg.pl - cfg.pr)));
    const step = cfg.x0 + frac * (cfg.x1 - cfg.x0);
    let rows = [];
    for (const [name, pts] of Object.entries(cfg.pts)) {
      if (!pts.length) continue;
      let best = pts[0];
      for (const p of pts) if (Math.abs(p[0]-step) < Math.abs(best[0]-step)) best = p;
      rows.push(name + ': ' + best[1].toPrecision(3) + '  @' + best[0]);
    }
    xh.setAttribute('x1', px); xh.setAttribute('x2', px);
    xh.setAttribute('visibility', 'visible');
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 14) + 'px';
    tip.style.top = (e.clientY + 10) + 'px';
    tip.innerHTML = 'step ~' + Math.round(step) + '<br>' + rows.join('<br>');
  });
  svg.addEventListener('mouseleave', () => {
    xh.setAttribute('visibility', 'hidden'); tip.style.display = 'none';
  });
});
</script>"""


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

def run_card(key: str, rows: list[dict]) -> str:
    name = key.split(":", 1)[1]
    total = STEPS_TOTAL.get(name, 0)
    last = rows[-1]
    step = last.get("step", 0)
    pct = min(100, round(100 * step / total, 1)) if total else 0
    sps = "-"
    eta = ""
    if len(rows) > 4:
        a, b = rows[-5], rows[-1]
        ds, dt = b.get("step", 0) - a.get("step", 0), b.get("sec", 0) - a.get("sec", 0)
        if ds > 0 and dt > 0:
            sps = f"{dt / ds:.2f}"
            if total:
                fin = datetime.now() + timedelta(seconds=(total - step) * dt / ds)
                eta = f"ETA {fin:%H:%M}"
    lr = last.get("lr", 0)
    loss_bits = "  ".join(f"{k}={last[k]:.4f}" for k, _ in SERIES if k in last)
    extra = "  ".join(f"{k}={v:.3f}" for k, v in last.items()
                      if "|" in k and isinstance(v, float))[:180]
    cid = key.replace(":", "-")
    return f"""<div class='card'><h2>{html.escape(key)}
      <span class='small'>step {step:,}/{total:,} &middot; {sps} s/step &middot; lr {lr:.1e} &middot; {eta}</span></h2>
    <div class='bar'><div class='fill' style='width:{pct}%'></div><span>{pct}%</span></div>
    <div class='mono small'>{loss_bits}</div>
    {f"<div class='mono small' style='color:#8fb8ff'>{html.escape(extra)}</div>" if extra else ""}
    {svg_chart(rows, cid)}</div>"""


def render(status: dict) -> str:
    p = []
    cost = status["cost"]
    p.append(f"""<div class='tiles'>
      <div class='tile'><div class='k'>updated</div><div class='v'>{status['ts'][11:]}</div></div>
      <div class='tile'><div class='k'>H100 elapsed</div><div class='v'>{cost['hours']:.1f} h</div></div>
      <div class='tile'><div class='k'>H100 spend</div><div class='v'>${cost['usd']:.2f}</div></div>
      <div class='tile'><div class='k'>dataset</div><div class='v'>{sum(status['dataset'].values()) or '-'} rec</div></div>
    </div>""")

    chips = []
    for s in status["stages"]:
        col = {"done": GOOD, "running": RUN, "pending": PEND}.get(s["state"], BAD)
        icon = {"done": "OK", "running": "&gt;&gt;", "pending": "..."}.get(s["state"], "!")
        chips.append(f"<span class='chip' style='border-color:{col}'>"
                     f"<b style='color:{col}'>{icon}</b> {html.escape(s['name'])}"
                     f"{('<span class=small> ' + html.escape(s['info']) + '</span>') if s['info'] else ''}</span>")
    p.append(f"<div class='card'><h2>pipeline</h2><div>{' '.join(chips)}</div></div>")

    for key in sorted(status["runs"]):
        p.append(run_card(key, status["runs"][key]))

    tiles = []
    for label, raw in status["gpus"].items():
        tiles.append(f"<div class='tile'><div class='k'>{html.escape(label)}</div>"
                     f"<div class='v mono' style='font-size:13px'>{html.escape(raw)}</div></div>")
    p.append(f"<div class='card'><h2>GPUs (util, mem used/total, degC)</h2>"
             f"<div class='tiles'>{''.join(tiles)}</div></div>")

    if status["dataset"]:
        d = "  ".join(f"{k.split('/')[0]}={v}" for k, v in status["dataset"].items())
        p.append(f"<div class='card small mono'>splits: {html.escape(d)}</div>")

    if status["events"]:
        ev = "<br>".join(html.escape(e) for e in status["events"])
        p.append(f"<div class='card'><h2>events</h2><div class='small mono'>{ev}</div></div>")

    return f"""<!doctype html><html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='20'><title>GeoField observability</title><style>
body{{background:{SURFACE};color:{INK2};font-family:system-ui;margin:1.2rem;max-width:1200px}}
h1{{font-size:19px;color:{INK1}}} h2{{font-size:14px;color:#8fb8ff;margin:2px 0 8px}}
.card{{background:{CARD};border-radius:10px;padding:12px 16px;margin-bottom:12px}}
.tiles{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}}
.tile{{background:{CARD};border-radius:8px;padding:8px 14px;min-width:110px}}
.tile .k{{font-size:11px;color:{MUTED}}} .tile .v{{font-size:17px;color:{INK1}}}
.bar{{position:relative;background:#252a35;border-radius:6px;height:16px;margin:6px 0}}
.fill{{background:linear-gradient(90deg,#3987e5,#199e70);height:100%;border-radius:6px}}
.bar span{{position:absolute;left:50%;top:0;font-size:11px;color:{INK1}}}
.small{{color:{MUTED};font-size:11px}} .mono{{font-family:ui-monospace,monospace}}
.chip{{display:inline-block;border:1px solid;border-radius:14px;padding:2px 10px;
 margin:2px;font-size:12px;color:{INK2}}}
svg.chart{{margin-top:6px}}</style></head><body>
<h1>GeoField observability <span class='small'>auto-refresh 20s &middot; charts hover for values</span></h1>
{''.join(p)}
{CROSSHAIR_JS}
</body></html>"""


def tick() -> dict:
    runs = gather_runs()
    events = gather_pipeline_events()
    hours = max(0.0, (datetime.now() - H100_START).total_seconds() / 3600)
    status = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs,
        "events": events,
        "stages": gather_stages(runs, events),
        "gpus": gather_gpus(),
        "dataset": gather_dataset(),
        "cost": {"hours": hours, "usd": hours * H100_RATE},
    }
    (ROOT / "status.json").write_text(json.dumps(
        {k: v for k, v in status.items() if k != "runs"}
        | {"runs_last": {k: v[-1] for k, v in runs.items()}}, indent=2))
    (ROOT / "status.html").write_text(render(status), encoding="utf-8")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--local_gen_log", default=None)  # kept for cron compat
    args = ap.parse_args()
    while True:
        try:
            s = tick()
            print(f"[{s['ts']}] runs: {list(s['runs'])} "
                  f"| ${s['cost']['usd']:.2f}", flush=True)
        except Exception as e:  # noqa: BLE001 - dashboard must never die
            print(f"tick failed: {e}", flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
