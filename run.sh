#!/bin/bash
# One run under frame.py on a rented machine, inside the rental's hour: the packages, training for the time budget
# and the evaluation, which train.py ends with. The summary, CORE's results and the training log go to results/, and
# the last weights, in 16 bits, to models/run. Each stage gets a stamp, so that its time can be read afterwards.
# Usage: run.sh [TRAIN_OPTION...]
set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%H:%M:%S $*"; }

stamp packages
# The image's Python refuses other packages by default; a rented machine is thrown away afterwards.
pip install -q --break-system-packages tokenizers pyyaml
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader
python -c 'import torch; print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())'
# A host whose GPU held another process's memory once ran training out of memory.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 1024 ]; then
    echo "the GPU already holds $used MiB"
    exit 1
fi
mkdir -p results models

# Attention within documents goes through PyTorch's varlen attention on a GPU, which tests without one cannot reach.
stamp check
python attention_check.py documents

stamp train
python -u train.py --out_dir=out/run "$@" > results/train.log 2>&1
cp out/run/summary.json out/run/core.json out/run/core.csv results/

stamp keep
python - <<'PY'
import os

import torch

checkpoint = torch.load("out/run/ckpt.pt", map_location="cpu")
weights = {key.removeprefix("_orig_mod."): value.to(torch.bfloat16) for key, value in checkpoint["model"].items()}
os.makedirs("models/run", exist_ok=True)
# A COUNT model needs to know where its table is.
config = {"count_table": checkpoint["config"].get("count_table", "")}
torch.save({"model_args": checkpoint["model_args"], "model": weights, "config": config}, "models/run/ckpt.pt")
PY
stamp finished
