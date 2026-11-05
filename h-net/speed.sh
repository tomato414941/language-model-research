# Times H-Net's training updates on a rented GPU, from the repository's root after rent.py send has put the code
# there, rent.py send-commit an earlier version in before/ if it is to be timed too, and rent.py send-data research's
# BPE dataset in data/fineweb if H-Net is to be timed on it. speed_check.py writes results/speed-NAME.json and a log
# for each of these, on random bytes with a chunk at every 6 bytes unless named otherwise:
#   before-32x16     the version in before/, in its passes of 32 rows
#   unfused-32x16    this version with Mamba-2 on its unfused path, as before
#   fused-ROWSxN     this version, in each of PASSES
# and then in the config's passes of 32 rows: eager-32x16 with nothing compiled (which every later run takes too if
# the compiled model fails), profile-32x16, chunks3-32x16 and chunks2-32x16 with a chunk at every 3 and 2 bytes, the
# second bundle on random tokens, and last fineweb-32x16: 200 updates on the BPE dataset's bytes with the routing's
# own chunks.
# Usage: [PASSES="32x16 64x8 128x4"] bash h-net/speed.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
# Otherwise Mamba-2's kernels time their configurations again in every run.
export TRITON_CACHE_AUTOTUNING=1
stamp() { date -u +"%H:%M:%S $*"; }
root=$(pwd)
passes=(${PASSES:-32x16 64x8 128x4})
# models/ only so that rent.py fetch finds every directory it fetches.
mkdir -p results models

stamp packages
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
[ "${used:-0}" -lt 2000 ] || { echo "the GPU already holds ${used} MiB"; exit 1; }
# As run.sh installs Mamba: without its own requirements, which would replace the image's PyTorch, or its CUDA build.
pip install -q --break-system-packages einops packaging huggingface_hub tokenizers pyarrow || exit 1
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -q --break-system-packages --no-deps --no-build-isolation "mamba-ssm==2.3.2.post1" || exit 1
python -c 'import torch, triton; print("torch", torch.__version__, "triton", triton.__version__)'

stamp check
python h-net/check_gpu.py mamba2 || exit 1

stamp data
python h-net/speed_check.py data data/hnet-bytes || exit 1
python h-net/speed_check.py tokens data/tokens || exit 1

# NAME [--profile=FIRST:LAST] [--unfused] CONFIG [TRAIN_OPTION...]: UPDATES updates and one more, in DIR.
time_way() {
    local name=$1
    shift
    stamp "$name"
    (cd "${DIR:-.}" && python "$root/h-net/speed_check.py" run "$root/results/speed-$name.json" "$@" \
        --max_iters="${UPDATES:-15}" --eval_interval=1000000 --eval_iters=1 --log_interval=1 \
        --out_dir="out/speed/$name") > "results/speed-$name.log" 2>&1 || tail -n 3 "results/speed-$name.log"
    python - "$name" <<'PY'
import json, sys
name = sys.argv[1]
try:
    summary = json.load(open(f"results/speed-{name}.json"))
except FileNotFoundError:
    raise SystemExit(f"{name}: no update finished")
print(f"{name}: {summary['seconds_per_update']} s per update, {summary.get('peak_allocated_gb', 0):.1f} GB at most,"
      f" {summary.get('alloc_retries')} allocator retries, profiled {summary.get('profiled')}")
PY
}

hnet="h-net/configs/hnet_mamba2.py --fixed_chunks=6"
pass_options() { echo "--batch_size=${1%x*} --gradient_accumulation_steps=${1#*x}"; }

if [ -d before ]; then
    ln -sfn ../data before/data
    DIR=before UPDATES=12 time_way before-32x16 --profile=10:11 $hnet $(pass_options 32x16)
fi
UPDATES=12 time_way unfused-32x16 --unfused $hnet $(pass_options 32x16)
for pass in "${passes[@]}"; do
    time_way "fused-$pass" $hnet $(pass_options "$pass")
done
train=$(pass_options 32x16)
time_way eager-32x16 $hnet $train --compile_main=False
compiled=""
[ -f results/speed-fused-32x16.json ] || { echo "the compiled model did not run; going on uncompiled"; compiled="--compile_main=False"; }
UPDATES=12 time_way profile-32x16 --profile=10:12 $hnet $train $compiled
time_way chunks3-32x16 h-net/configs/hnet_mamba2.py --fixed_chunks=3 $train $compiled
time_way chunks2-32x16 h-net/configs/hnet_mamba2.py --fixed_chunks=2 $train $compiled
time_way bundle2 configs/bundle2.py --dataset=tokens --attention=fa2 --stream_bfloat16=True

if [ -f data/fineweb/train.bin ]; then
    stamp bytes
    mv data/hnet-bytes data/hnet-random
    python h-net/bytes_from_tokens.py data/fineweb data/hnet-bytes || exit 1
    UPDATES=200 time_way fineweb-32x16 h-net/configs/hnet_mamba2.py $train $compiled
    grep "^h-net:" results/speed-fineweb-32x16.log
fi
stamp done
