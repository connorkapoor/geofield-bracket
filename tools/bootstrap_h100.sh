#!/bin/bash
# One-shot H100 box bootstrap: env + repo + data land via stdin-tar and scp
# from the Spark; then Stage A resumes from the latest Spark checkpoint.
# Usage (run ON the rental box): bash bootstrap_h100.sh
set -e
cd ~
mkdir -p geofield
cd geofield
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip -q
  .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126 -q
  .venv/bin/pip install numpy scipy pyyaml scikit-image matplotlib pillow pytest pyamg -q
fi
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p data/l1 runs logs
echo "READY - now push repo tar + data/l1 shards + runs/stage_a_l1 checkpoints, then:"
echo "  nohup ./run_pipeline.sh > logs/pipeline.log 2>&1 &"
