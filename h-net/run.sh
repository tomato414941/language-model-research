#!/bin/bash
# One H-Net run under frame.py on a rented machine, inside the rental's hour, as run.sh runs one of model.py: the
# packages, a check of the GPU, FineWeb's tokens written as bytes, training for frame.py's time and the evaluation of
# the last weights by h-net/evaluate_hnet.py. It starts from the repository's root once rent.py has sent the code, and
# waits for what rent.py sends after it: FineWeb's dataset, whose train.bin takes its name only once it has all come,
# then olmes's requests and its tasks' data, marked by data/olmes.sent. olmes runs in an environment of its own, with
# the torch it pins and mamba-ssm beside it, which is built and given its tasks' data while training runs. The
# summary, olmes's results and the logs go to results/, and the last weights, in 16 bits, to models/hnet. Each stage
# gets a stamp, so that its time can be read afterwards.
# Usage: bash h-net/run.sh [TRAIN_OPTION...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
stamp() { date -u +"%H:%M:%S $*"; }
wait_for() {
    for _ in $(seq 360); do
        [ -e "$1" ] && return 0
        sleep 5
    done
    echo "$1 never came"
    return 1
}
mkdir -p results models

stamp packages
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader
# A host whose GPU held another process's memory once ran training out of memory.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
if [ "$used" -gt 1024 ]; then
    echo "the GPU already holds $used MiB"
    exit 1
fi
# The image's Python refuses other packages by default; a rented machine is thrown away afterwards. Mamba's own
# requirements would replace the image's PyTorch, so it comes without them, and without its CUDA build, which needs a
# compiler the image lacks; its Mamba-2 needs only Triton.
pip install -q --break-system-packages tokenizers einops pyarrow
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -q --break-system-packages --no-deps --no-build-isolation "mamba-ssm==2.3.2.post1"
python -c 'import torch; print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())'

# olmes's environment, as run.sh builds it, then every task's data loaded once, which checks that olmes runs here
# before training ends, and the checks of the GPU in its PyTorch, which the evaluation runs in. mamba-ssm's Python
# files, all that Mamba-2's Triton kernels need, come from the install above, since its setup.py there would need
# the wheel package and try to fetch a build for that PyTorch. pip keeps no cache: the disk holds FineWeb as bytes too.
export HF_HOME=$PWD/data/hf
mamba=$(python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("mamba_ssm").origin))')
mkdir -p /root/mamba-ssm
cp -r "$mamba" /root/mamba-ssm/
olmes_python() { PATH=/root/olmes-venv/bin:$PATH PYTHONPATH=/root/mamba-ssm python "$@"; }
(
    python -m venv /root/olmes-venv
    /root/olmes-venv/bin/pip install -q --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    /root/olmes-venv/bin/pip install -q --no-cache-dir --no-deps -r olmes.txt
    wait_for data/olmes.sent
    olmes_python frame.py data out/olmes-data data/olmes-requests
    olmes_python h-net/check_gpu.py mamba2
) > results/olmes-setup.log 2>&1 &
setup=$!

stamp check
python h-net/check_gpu.py mamba2 > results/check.log 2>&1 &
check=$!
wait_for data/fineweb/train.bin
stamp bytes
python h-net/bytes_from_tokens.py data/fineweb data/hnet-bytes
# Training reads the bytes, and the evaluation only FineWeb's val.bin and tokenizer.
rm data/fineweb/train.bin
stamp "wait for check"
if ! wait $check; then
    cat results/check.log
    exit 1
fi
cat results/check.log

stamp train
python -u train.py h-net/configs/hnet_mamba2.py --out_dir=out/hnet "$@" > results/train.log 2>&1

stamp keep
python - <<'PY'
import os

import torch

checkpoint = torch.load("out/hnet/ckpt.pt", map_location="cpu")
weights = {key.removeprefix("_orig_mod."): value.to(torch.bfloat16) for key, value in checkpoint["model"].items()}
os.makedirs("models/hnet", exist_ok=True)
config = {key: checkpoint["config"].get(key, "") for key in ("dataset",)}
torch.save({"model_args": checkpoint["model_args"], "model": weights, "config": config}, "models/hnet/ckpt.pt")
PY
cp out/hnet/summary.json results/ 2>/dev/null || true

stamp "wait for olmes"
wait $setup
stamp evaluate
olmes_python h-net/evaluate_hnet.py out/hnet data/olmes-requests > results/evaluate.log 2>&1
cp out/hnet/summary.json results/
cp out/hnet/olmes/metrics-all.jsonl results/olmes-metrics.jsonl
stamp finished
