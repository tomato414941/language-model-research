# Steps of the scripts that run on a rented machine, to be sourced. Each stage gets a stamp, so that the
# time it takes can be read afterwards.
export HF_HUB_DISABLE_PROGRESS_BARS=1
stamp() { date -u +"%H:%M:%S $*"; }

# Usage: prepare [PACKAGE...]
prepare() {
    stamp packages
    # The image's Python refuses other packages by default; a rented machine is thrown away afterwards.
    pip install -q --break-system-packages tokenizers pyarrow "$@"
    nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader
    python -c 'import torch; print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())'
    mkdir -p results models
}

# Trains a model and scores it, and keeps only its weights, in 16 bits. SCORING_OPTIONS go to the
# scoring, so that a small model can try this without a GPU.
# Usage: train_score_keep NAME CONFIG [TRAIN_OPTION...]
train_score_keep() {
    local name=$1 config=$2
    shift 2
    stamp "train $name"
    python -u train.py "$config" --out_dir="out/$name" "$@" > "results/train-$name.log" 2>&1
    stamp "score $name"
    python scoring.py "out/$name" --tokenizer data/climbmix/tokenizer.json --validation data/climbmix/val.bin \
        --questions data/questions.jsonl --output "results/$name.json" ${SCORING_OPTIONS:-} > "results/score-$name.log" 2>&1
    stamp "keep $name"
    python - "$name" <<'PY'
import os
import sys

import torch

name = sys.argv[1]
checkpoint = torch.load(f"out/{name}/ckpt.pt", map_location="cpu")
weights = {key.removeprefix("_orig_mod."): value.to(torch.bfloat16) for key, value in checkpoint["model"].items()}
os.makedirs(f"models/{name}", exist_ok=True)
torch.save({"model_args": checkpoint["model_args"], "model": weights}, f"models/{name}/ckpt.pt")
os.remove(f"out/{name}/ckpt.pt")
PY
}
