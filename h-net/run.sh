# The H-Net comparison on a rented GPU, run from the repository's root with hnet-sources.json beside it (rent.py
# sources sends it): install, check the kernels, prepare the data, time each model against the baseline, then train
# and score each, keeping its weights in 16 bits as models/NAME.
# Usage: bash h-net/run.sh [NAME...]; PHASE_BYTES and TRAIN_OPTIONS make a short try of it.
set -uo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
stamp() { date -u +"%H:%M:%S $*"; }
mkdir -p results models
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(hnet-bpe hnet-mamba2 hnet-mamba3 hnet-gated-deltanet)

stamp packages
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader
# The image's Python refuses other packages by default; a rented machine is thrown away afterwards. Mamba's own
# requirements would replace the image's PyTorch, so it comes without them; its Mamba-2 and Mamba-3 need only Triton.
pip install -q --break-system-packages tokenizers pyarrow einops "lm-eval[hf]==0.4.13" "flash-linear-attention==0.6.0" \
    || exit 1
pip install -q --break-system-packages --no-deps --no-build-isolation "mamba-ssm==2.3.2.post1" || exit 1
python -c 'import torch, triton; print("torch", torch.__version__, "triton", triton.__version__)'

stamp check
python h-net/check_gpu.py || exit 1

stamp data
python h-net/prepare_bytes.py --sources hnet-sources.json ${PHASE_BYTES:+--phase-bytes "$PHASE_BYTES"} \
    > results/hnet-prepare.log 2>&1 || exit 1
rm -f hnet-sources.json
tail -n 4 results/hnet-prepare.log

stamp calibrate
python h-net/calibrate.py "${names[@]}" > results/hnet-calibrate.log 2>&1 || exit 1
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
    if [ "$name" = hnet-bpe ]; then
        python h-net/score_bytes.py "results/$name.json" --bpe "out/$name" ${SCORE_OPTIONS:-} > "results/score-$name.log" 2>&1
    else
        python h-net/score_bytes.py "results/$name.json" --hnet "out/$name" ${SCORE_OPTIONS:-} \
            > "results/score-$name.log" 2>&1
    fi
    python - "$name" "$config" <<'PY'
import os, sys
import torch
name, config = sys.argv[1:]
checkpoint = torch.load(f"out/{name}/ckpt.pt", map_location="cpu")
weights = {key.removeprefix("_orig_mod."): value.to(torch.bfloat16) for key, value in checkpoint["model"].items()}
kept = {"model_args": checkpoint["model_args"], "model": weights, "config": config}
os.makedirs(f"models/{name}", exist_ok=True)
torch.save(kept, f"models/{name}/ckpt.pt")
os.remove(f"out/{name}/ckpt.pt")
PY
done
stamp done
