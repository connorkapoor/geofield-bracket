#!/usr/bin/env bash
# Fetch the field-model checkpoints (too large for git) from the GitHub release.
# The design agent (models/designer.pt) is already in the repo.
set -euo pipefail

REPO="${GEOFIELD_REPO:-connorkapoor/geofield-bracket}"
TAG="${GEOFIELD_MODELS_TAG:-v0.1.0}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/models"
ASSETS=(geofield_stage_a_geometry_l1.pt geofield_stage_b_surrogate_l1_8k.pt latent_stats.pt)

mkdir -p "$DEST"
for a in "${ASSETS[@]}"; do
  if [ -f "$DEST/$a" ]; then echo "have $a"; continue; fi
  echo "fetching $a ..."
  if command -v gh >/dev/null 2>&1; then
    gh release download "$TAG" --repo "$REPO" --pattern "$a" --dir "$DEST"
  else
    curl -fL --progress-bar -o "$DEST/$a" \
      "https://github.com/$REPO/releases/download/$TAG/$a"
  fi
done

echo
echo "models in $DEST:"
ls -lh "$DEST"
echo
echo "run the demo with the surrogate:"
echo "  python -m geofield.demo.backend.app --designer models/designer.pt \\"
echo "    --stage_b models/geofield_stage_b_surrogate_l1_8k.pt --linear_heads \\"
echo "    --data data/l1 --port 8600"
