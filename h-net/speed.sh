# Times H-Net's training updates on a rented GPU, from the repository's root after rent.py send has put the code
# there, rent.py send-commit an earlier version in before/ if it is to be timed too, and rent.py send-data research's
# BPE dataset in data/fineweb if H-Net is to be timed on it. speed_check.py writes results/speed-NAME.json and a log
# for each of these, on random bytes with a chunk at every 6 bytes unless named otherwise:
#   before-32x16     the version in before/, in its passes of 32 rows
#   unfused-32x16    this version with Mamba-2 on its unfused path, as before
#   fused-ROWSxN     this version, in each of PASSES
# and for the fastest of those: eager-* with nothing compiled (or, if the compiled model fails, every pass that way),
# profile-*, chunks3-* with a chunk at every 3 bytes and chunks2-* at every 2, from the fastest pass down to the first
# that fits; then the second bundle on random tokens, and last fineweb-*: 200 updates on the BPE dataset's bytes, with
# the routing's own chunks, in the pass that fitted chunks of 2 bytes.
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
# The fastest of PREFIX-ROWSxN over PASSES that finished, or nothing.
fastest() {
    python - "$1" "${passes[@]}" <<'PY'
import json, sys
prefix, timed = sys.argv[1], {}
for way in sys.argv[2:]:
    try:
        seconds = json.load(open(f"results/speed-{prefix}-{way}.json"))["seconds_per_update"]
    except FileNotFoundError:
        continue
    if seconds is not None:
        timed[way] = seconds
print(min(timed, key=timed.get) if timed else "")
PY
}

for pass in "${passes[@]}"; do
    time_way "fused-$pass" $hnet $(pass_options "$pass")
done
best=$(fastest fused)
compiled=""
if [ -n "$best" ]; then
    time_way "eager-$best" $hnet $(pass_options "$best") --compile_main=False
else
    echo "the compiled model did not run; timing it uncompiled"
    compiled="--compile_main=False"
    for pass in "${passes[@]}"; do
        time_way "eager-$pass" $hnet $(pass_options "$pass") $compiled
    done
    best=$(fastest eager)
    [ -n "$best" ] || { echo "the fused path did not run"; exit 1; }
fi
UPDATES=12 time_way "profile-$best" --profile=10:12 $hnet $(pass_options "$best") $compiled
time_way "chunks3-$best" h-net/configs/hnet_mamba2.py --fixed_chunks=3 $(pass_options "$best") $compiled
# Early in training the routing begins a chunk at about every other byte: the largest pass that holds that.
fits=""
for pass in $(printf '%s\n' "${passes[@]}" | sort -t x -k1,1nr); do
    [ "${pass%x*}" -le "${best%x*}" ] || continue
    time_way "chunks2-$pass" h-net/configs/hnet_mamba2.py --fixed_chunks=2 $(pass_options "$pass") $compiled
    [ -f "results/speed-chunks2-$pass.json" ] && { fits=$pass; break; }
done
time_way bundle2 configs/bundle2.py --dataset=tokens --attention=fa2 --stream_bfloat16=True

if [ -f data/fineweb/train.bin ] && [ -n "$fits" ]; then
    stamp bytes
    mv data/hnet-bytes data/hnet-random
    python h-net/bytes_from_tokens.py data/fineweb data/hnet-bytes || exit 1
    UPDATES=200 time_way "fineweb-$fits" h-net/configs/hnet_mamba2.py $(pass_options "$fits") $compiled
    grep "^h-net:" "results/speed-fineweb-$fits.log"
fi
stamp done
