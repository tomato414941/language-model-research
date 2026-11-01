#!/bin/bash
# Trains the baseline once for each seed on a rented machine and scores each.
# Usage: baseline.sh SEED...
set -euo pipefail
cd "$(dirname "$0")"
source ./rented.sh

prepare
for seed in "$@"; do
    train_score_keep "seed$seed" configs/baseline.py --seed="$seed"
done
stamp finished
