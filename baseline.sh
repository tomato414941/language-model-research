#!/bin/bash
# Trains the baseline once for each seed on a rented machine and scores each, stamping each stage so
# that the time it takes can be read afterwards. A model keeps only its weights, in 16 bits.
# Usage: baseline.sh SEED...
set -euo pipefail
cd "$(dirname "$0")"
stamp() { date -u +"%H:%M:%S $*"; }

stamp packages
# The image's Python refuses other packages by default; a rented machine is thrown away afterwards.
pip install -q --break-system-packages tokenizers pyarrow
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader
python -c 'import torch; print("torch", torch.__version__)'
mkdir -p results models

for seed in "$@"; do
    stamp "train $seed"
    python -u train.py configs/baseline.py --seed="$seed" --out_dir="out/seed$seed" > "results/train-$seed.log" 2>&1
    stamp "score $seed"
    python scoring.py "out/seed$seed" --tokenizer data/climbmix/tokenizer.json --validation data/climbmix/val.bin \
        --questions data/questions.jsonl --output "results/seed$seed.json" > "results/score-$seed.log" 2>&1
    stamp "keep $seed"
    python - "$seed" <<'PY'
import os
import sys

import torch

seed = sys.argv[1]
checkpoint = torch.load(f"out/seed{seed}/ckpt.pt", map_location="cpu")
weights = {key.removeprefix("_orig_mod."): value.to(torch.bfloat16) for key, value in checkpoint["model"].items()}
os.makedirs(f"models/seed{seed}", exist_ok=True)
torch.save({"model_args": checkpoint["model_args"], "model": weights}, f"models/seed{seed}/ckpt.pt")
os.remove(f"out/seed{seed}/ckpt.pt")
PY
done
stamp finished
