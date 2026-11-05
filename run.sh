#!/bin/bash
# One run under frame.py on a rented machine, inside the rental's hour: the packages, training for the time budget and
# frame.py's evaluation of the last weights. olmes runs in an environment of its own, with the torch it pins, which is
# built and given its tasks' data while training runs. The summary, olmes's results and the logs go to results/, and
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

# olmes's environment and its tasks' data, beside training on the CPU. The requests come built from the processing
# machine (frame.py requests), and the tasks' data in its Hugging Face cache, data/hf; loading every task once checks
# that olmes runs here before training ends.
export HF_HOME=$PWD/data/hf
# The image's Python has no ensurepip, which `python -m venv` needs, so uv makes the environment.
(
    pip install -q --break-system-packages uv
    uv venv -q /root/olmes-venv --python "$(which python)"
    uv pip install -q --python /root/olmes-venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    uv pip install -q --python /root/olmes-venv/bin/python --no-deps -r olmes.txt
    PATH=/root/olmes-venv/bin:$PATH python frame.py data out/olmes-data data/olmes-requests
) > results/olmes-setup.log 2>&1 &
setup=$!

stamp train
python -u train.py --out_dir=out/run "$@" > results/train.log 2>&1

stamp wait for olmes
wait $setup
stamp evaluate
PATH=/root/olmes-venv/bin:$PATH python frame.py evaluate out/run data/olmes-requests > results/evaluate.log 2>&1
cp out/run/summary.json results/
cp out/run/olmes/metrics-all.jsonl results/olmes-metrics.jsonl

stamp keep
python - <<'PY'
import os

import torch

checkpoint = torch.load("out/run/ckpt.pt", map_location="cpu")
weights = {key.removeprefix("_orig_mod."): value.to(torch.bfloat16) for key, value in checkpoint["model"].items()}
os.makedirs("models/run", exist_ok=True)
# A COUNT model needs to know where its table is, and its evaluation which data it read.
config = {key: checkpoint["config"].get(key, "") for key in ("count_table", "dataset")}
torch.save({"model_args": checkpoint["model_args"], "model": weights, "config": config}, "models/run/ckpt.pt")
PY
stamp finished
