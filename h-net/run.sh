# Trains one H-Net on a rented H100 for the baseline's cost, in as little of the machine's time as it can, from the
# repository's root after rent.py send has put the code there: install what H-Net needs and check the GPU while
# rent.py send-data sends research's BPE dataset, then write the dataset as bytes, time the baseline and the H-Net,
# train it, and keep its weights in 16 bits as models/NAME. Scoring is left to score.sh on a cheaper GPU.
# Usage: bash h-net/run.sh DATASET NAME
set -uo pipefail
export PYTHONUNBUFFERED=1
stamp() { date -u +"%H:%M:%S $*"; }
dataset=$1
name=$2
kind=${name#hnet-}
mkdir -p results models

stamp packages
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
[ "${used:-0}" -lt 2000 ] || { echo "the GPU already holds ${used} MiB"; exit 1; }
# Mamba's own requirements would replace the image's PyTorch, so it comes without them, and without its CUDA build,
# which needs a compiler the image lacks; its Mamba-2 needs only Triton.
pip install -q --break-system-packages tokenizers einops "flash-linear-attention==0.6.0" || exit 1
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -q --break-system-packages --no-deps --no-build-isolation "mamba-ssm==2.3.2.post1" || exit 1

stamp check
python h-net/check_gpu.py "${kind//-/_}" || exit 1

stamp "wait for data"
for _ in $(seq 180); do
    [ -f "data/$dataset/train.bin" ] && break
    sleep 5
done
[ -f "data/$dataset/train.bin" ] || { echo "data/$dataset/train.bin never came"; exit 1; }

stamp bytes
python h-net/bytes_from_tokens.py "data/$dataset" data/hnet-bytes || exit 1

stamp calibrate
python h-net/calibrate.py "$dataset" "$name" > results/hnet-calibrate.log 2>&1 || exit 1
read -r config updates < <(python - "$name" <<'PY'
import json, sys
sys.path.insert(0, "h-net")
from calibrate import MODELS
name = sys.argv[1]
print(MODELS[name], json.load(open("results/hnet-speed.json"))["updates"].get(name, ""))
PY
)
[ -n "$updates" ] || { tail -n 20 results/hnet-calibrate.log; exit 1; }

stamp "train $name $updates"
python -u train.py "$config" --out_dir="out/$name" --max_iters="$updates" --lr_decay_iters="$updates" \
    --eval_interval="$updates" ${TRAIN_OPTIONS:-} > "results/train-$name.log" 2>&1 || { tail -n 20 "results/train-$name.log"; exit 1; }
python - "$name" "$config" <<'PY'
import os, sys
import torch
name, config = sys.argv[1:]
checkpoint = torch.load(f"out/{name}/ckpt.pt", map_location="cpu")
weights = {key.removeprefix("_orig_mod."): value.to(torch.bfloat16) for key, value in checkpoint["model"].items()}
os.makedirs(f"models/{name}", exist_ok=True)
torch.save({"model_args": checkpoint["model_args"], "model": weights, "config": config}, f"models/{name}/ckpt.pt")
os.remove(f"out/{name}/ckpt.pt")
PY
stamp done
