# Steps of the scripts that run on a rented machine, to be sourced. Each stage gets a stamp, so that the
# time it takes can be read afterwards.
export HF_HUB_DISABLE_PROGRESS_BARS=1
stamp() { date -u +"%H:%M:%S $*"; }

prepare() {
    stamp packages
    # The image's Python refuses other packages by default; a rented machine is thrown away afterwards.
    pip install -q --break-system-packages tokenizers pyarrow
    nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader
    python -c 'import torch; print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())'
    mkdir -p results models
    stamp harness packages
    # constraints.txt holds the versions that a try without a GPU ran with. The data directory holds the
    # harness's questions, so nothing is fetched from the Hugging Face Hub.
    pip install -q --break-system-packages -c data/constraints.txt "lm-eval[hf]==0.4.13"
    export HF_DATASETS_CACHE="$PWD/data/datasets" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
}

# Scores a saved model on all the questions of the English benchmarks with the LM Evaluation Harness.
# HARNESS_OPTIONS go to the harness, so that a small model can try this without a GPU.
# Usage: score_in_full NAME DIRECTORY
score_in_full() {
    stamp "harness $1"
    # A smaller batch than the harness's default, whose logits alone would nearly fill 24 GB.
    python harness.py "results/$1-full.json" --model "$2" --tokenizer data/climbmix/tokenizer.json \
        --batch-tokens 65536 ${HARNESS_OPTIONS:-} > "results/$1-full.log" 2>&1
}

# Scores a model that train.py saved in DIRECTORY, keeps only its weights, in 16 bits, as models/NAME, and
# scores those in full. SCORING_OPTIONS go to the first scoring, so that a small model can try this without a GPU.
# Usage: score_keep NAME DIRECTORY
score_keep() {
    local name=$1 directory=$2
    stamp "score $name"
    python scoring.py "$directory" --tokenizer data/climbmix/tokenizer.json --validation data/climbmix/val.bin \
        --questions data/questions.jsonl --output "results/$name.json" ${SCORING_OPTIONS:-} > "results/score-$name.log" 2>&1
    stamp "keep $name"
    python - "$directory" "models/$name" <<'PY'
import os
import sys

import torch

source, destination = sys.argv[1:]
checkpoint = torch.load(f"{source}/ckpt.pt", map_location="cpu")
weights = {key.removeprefix("_orig_mod."): value.to(torch.bfloat16) for key, value in checkpoint["model"].items()}
os.makedirs(destination, exist_ok=True)
# A COUNT model needs to know where its table is.
config = {"count_table": checkpoint["config"].get("count_table", "")}
torch.save({"model_args": checkpoint["model_args"], "model": weights, "config": config}, f"{destination}/ckpt.pt")
os.remove(f"{source}/ckpt.pt")
PY
    score_in_full "$name" "models/$name"
}

# Trains a model, then scores and keeps its last weights as NAME and, when the configuration asks train.py to
# mix them with the average of the last few, the mixed weights as NAME-mixed.
# Usage: train_score_keep NAME CONFIG [TRAIN_OPTION...]
train_score_keep() {
    local name=$1 config=$2
    shift 2
    stamp "train $name"
    python -u train.py "$config" --out_dir="out/$name" "$@" > "results/train-$name.log" 2>&1
    score_keep "$name" "out/$name"
    if [ -f "out/$name/mixed/ckpt.pt" ]; then
        score_keep "$name-mixed" "out/$name/mixed"
    fi
}
