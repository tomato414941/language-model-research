#!/bin/bash
# Measures on a rented machine what each way of running the first bundle costs, then trains the bundle
# in the cheapest way for the money of the baseline, scores it and keeps its weights.
# Usage: bundle.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./rented.sh

prepare
# FlashAttention-3 and 4 are only two of the ways to measure, so the run goes on without them.
# kernels needs a newer cryptography than the image's Debian package, which pip cannot replace.
pip install -q --break-system-packages --ignore-installed "kernels>=0.18,<0.19" || echo "kernels could not be installed"
pip install -q --break-system-packages "flash-attn-4[cu13]==4.0.0b37" || echo "FlashAttention-4 could not be installed"
# The first load downloads the kernel, which should not count as the time of a measured run.
stamp kernel
python -c 'from model import load_flash_attention; print("FlashAttention-3 loaded:", load_flash_attention() is not None)'
stamp speed
options=$(python speed.py)
echo "options: $options"
train_score_keep bundle configs/bundle.py $options
stamp finished
