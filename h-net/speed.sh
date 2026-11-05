# Times H-Net's training updates on a rented GPU, on random bytes rather than research's data, from the repository's
# root after rent.py send has put the code there: install what Mamba-2 needs, check the GPU, and time the second
# bundle, H-Net as it ran before causal_conv.py, and H-Net in each of PASSES (ROWSxPASSES per update) with
# speed_check.py, which writes results/speed-NAME.json and its log. The fastest is then profiled and timed with finer
# chunks. Every H-Net way begins a chunk at every 6 bytes unless named otherwise.
# Usage: [PASSES="32x16 64x8 128x4"] bash h-net/speed.sh
set -uo pipefail
export PYTHONUNBUFFERED=1
# Otherwise Mamba-2's kernels time their configurations again in every run.
export TRITON_CACHE_AUTOTUNING=1
stamp() { date -u +"%H:%M:%S $*"; }
passes=(${PASSES:-32x16 64x8 128x4})
mkdir -p results

stamp packages
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,power.limit --format=csv,noheader
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)
[ "${used:-0}" -lt 2000 ] || { echo "the GPU already holds ${used} MiB"; exit 1; }
# As run.sh installs Mamba: without its own requirements, which would replace the image's PyTorch, or its CUDA build.
pip install -q --break-system-packages einops packaging huggingface_hub || exit 1
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -q --break-system-packages --no-deps --no-build-isolation "mamba-ssm==2.3.2.post1" || exit 1
python -c 'import torch, triton; print("torch", torch.__version__, "triton", triton.__version__)'

stamp check
python h-net/check_gpu.py mamba2 || exit 1

stamp data
python h-net/speed_check.py data data/hnet-bytes || exit 1
python h-net/speed_check.py tokens data/tokens || exit 1

# NAME [--profile=FIRST:LAST] [--unfused] CONFIG [TRAIN_OPTION...], timing UPDATES updates and one more.
time_way() {
    local name=$1
    shift
    stamp "$name"
    python h-net/speed_check.py run "results/speed-$name.json" "$@" --max_iters="${UPDATES:-15}" \
        --eval_interval=1000000 --eval_iters=1 --log_interval=1 --out_dir="out/speed/$name" \
        > "results/speed-$name.log" 2>&1 || tail -n 3 "results/speed-$name.log"
    python - "$name" <<'PY'
import json, sys
name = sys.argv[1]
try:
    summary = json.load(open(f"results/speed-{name}.json"))
except FileNotFoundError:
    raise SystemExit(f"{name}: no update finished")
print(f"{name}: {summary['seconds_per_update']} s per update, {summary.get('peak_allocated_gb', 0):.1f} GB at most,"
      f" {summary.get('alloc_retries')} allocator retries")
PY
}

hnet="h-net/configs/hnet_mamba2.py --fixed_chunks=6"
pass_options() { echo "--batch_size=${1%x*} --gradient_accumulation_steps=${1#*x}"; }

time_way bundle2 configs/bundle2.py --dataset=tokens --attention=fa2 --stream_bfloat16=True
UPDATES=12 time_way "unfused-${passes[0]}" --unfused --profile=10:11 $hnet $(pass_options "${passes[0]}")
for pass in "${passes[@]}"; do
    time_way "fused-$pass" $hnet $(pass_options "$pass")
done

best=$(python - "${passes[@]}" <<'PY'
import json, sys
timed = {}
for way in sys.argv[1:]:
    try:
        timed[way] = json.load(open(f"results/speed-fused-{way}.json"))["seconds_per_update"]
    except FileNotFoundError:
        pass
timed = {way: seconds for way, seconds in timed.items() if seconds is not None}
print(min(timed, key=timed.get) if timed else "")
PY
)
[ -n "$best" ] || { echo "no way of running the fused path was timed"; exit 1; }
UPDATES=12 time_way "profile-$best" --profile=10:12 $hnet $(pass_options "$best")
time_way "chunks3-$best" h-net/configs/hnet_mamba2.py --fixed_chunks=3 $(pass_options "$best")
time_way "chunks2-$best" h-net/configs/hnet_mamba2.py --fixed_chunks=2 $(pass_options "$best")
stamp done
