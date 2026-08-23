# Trained weights

Three networks were trained for this project. Only the smallest one is small
enough to live in git; the two field models are published as **release assets**
on the [v0.1.0 release](https://github.com/connorkapoor/geofield-bracket/releases/tag/v0.1.0).

| Weights | Size | Where it lives | Needed for |
|---|---|---|---|
| `models/designer.pt` | 324 KB | **in this repo** | the design agent — the demo's generation path |
| `geofield_stage_a_geometry_l1.pt` | 179 MB | release asset | encoding a bracket into a latent, decoding the field, meshing from the model |
| `geofield_stage_b_surrogate_l1_8k.pt` | 186 MB | release asset | the predicted-stress overlay and candidate ranking |
| `latent_stats.pt` | 4 KB | release asset | latent normalisation used by Stage B |

```bash
bash scripts/fetch_models.sh          # pulls all three into models/
```

## What each one does

**`designer.pt` — the design agent (`geofield/model/designer.py`).**
15 invariant requirement features (envelope, load position/direction/magnitude,
material moduli, and the explicit engineering features arm / moment / required
section modulus) → the L-bracket program's ~15 parameters, plus categorical
heads for reinforcement style, lightweighting and bolt count. Trained on the
pairs in the labelled dataset where the calibrated stress utilisation landed in
the sensible design band (0.30–0.95 × yield), i.e. where *that* bracket is a
genuinely adequate and non-wasteful answer to *that* load. Candidate variety
comes from Monte-Carlo dropout. This is a 190k-parameter MLP, which is why it
fits in git.

**Stage A — geometry field model.** The SE(3)-equivariant encoder/decoder over
unit-gradient fields. Measured equivariance error 4.5e-7. Used by the demo for
`/encode`, `/mesh` from the model, and the latent-space experiments. The demo
**runs without it** (the designer plus the exact CSG builder cover the whole
generation path); the endpoints that need it return HTTP 503 when it is absent.

**Stage B — stress surrogate, step 8000.** Predicts the von Mises field from a
latent, and is what ranks candidates before FEA. ⚠️ **This checkpoint predates
the log-space field heads**, so its stress head is uncalibrated — its *spatial*
pattern is right and its ranking is useful, but the magnitudes are not. Launch
the demo with `--linear_heads` when using it, which is what tells the backend
not to apply the log-space inverse transform:

```bash
python -m geofield.demo.backend.app \
  --designer models/designer.pt \
  --stage_b models/geofield_stage_b_surrogate_l1_8k.pt --linear_heads \
  --data data/l1 --port 8600
```

A calibrated log-space retrain is in progress; it will ship as v0.1.1 and will
not need the flag.

## What the demo does *not* need weights for

Numbers the demo shows as engineering results do not come from a network:

- plate thicknesses, brace style and brace reach come from the closed-form
  sizing pass in `geofield/model/sizing.py` (beam theory with an FEA-calibrated
  stress-concentration factor K_t = 1.36), which **overrides** the agent's
  proposal;
- the geometry is built exactly by the CSG program in
  `geofield/fields/programs/l_bracket.py`;
- **Verify** runs the real immersed-voxel FEA solver in
  `geofield/labels/physics.py` on the true-millimetre field, at the actual
  requested force. That is the number to trust.

## Training data

The labelled shards (`data/l1`, ~500 geometries × 10 load cases × 2 materials)
are ~40 GB and are not published. `make data` regenerates them; the recipe and
seeds are deterministic, so a regenerated dataset matches the one used here.
