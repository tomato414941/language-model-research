# H-Net on a rented GPU, from the repository's root, after rent.py send has put research's data there (the BPE
# datasets of the two phases, the harness's questions, datasets and constraints.txt): write the datasets as bytes,
# check the kernels, time each H-Net against the baseline, then train and score each, keeping its weights in 16 bits
# as models/NAME. Usage: bash h-net/run.sh STABLE_DATASET DECAY_DATASET [NAME...]
set -uo pipefail
source ./rented.sh
stable=$1
decay=$2
shift 2
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(hnet-mamba2 hnet-mamba3 hnet-gated-deltanet)

prepare
stamp "outer layers"
# Mamba's own requirements would replace the image's PyTorch, so it comes without them; its Mamba-2 and Mamba-3 need
# only Triton.
pip install -q --break-system-packages "flash-linear-attention==0.6.0" || exit 1
pip install -q --break-system-packages --no-deps --no-build-isolation "mamba-ssm==2.3.2.post1" || exit 1
python -c 'import torch, triton; print("torch", torch.__version__, "triton", triton.__version__)'

stamp check
python h-net/check_gpu.py || exit 1

stamp bytes
python h-net/bytes_from_tokens.py "data/$stable" data/hnet-bytes-stable || exit 1
python h-net/bytes_from_tokens.py "data/$decay" data/hnet-bytes-decay || exit 1

stamp calibrate
python h-net/calibrate.py "$stable" "${names[@]}" > results/hnet-calibrate.log 2>&1 || exit 1
tail -n 1 results/hnet-calibrate.log

for name in "${names[@]}"; do
    read -r config updates < <(python - "$name" <<'PY'
import json, sys
sys.path.insert(0, "h-net")
from calibrate import MODELS
name = sys.argv[1]
print(MODELS[name], json.load(open("results/hnet-speed.json"))["updates"].get(name, ""))
PY
)
    if [ -z "$updates" ]; then
        stamp "skip $name"
        continue
    fi
    stamp "train $name $updates"
    if ! python -u train.py "$config" --out_dir="out/$name" --max_iters="$updates" --lr_decay_iters="$updates" \
        --eval_interval="$updates" ${TRAIN_OPTIONS:-} > "results/train-$name.log" 2>&1; then
        stamp "failed $name"
        tail -n 5 "results/train-$name.log"
        continue
    fi
    stamp "score $name"
    python h-net/score_hnet.py "results/$name.json" "out/$name" --tokens "data/$stable" ${SCORE_OPTIONS:-} \
        > "results/score-$name.log" 2>&1
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
done
stamp done
