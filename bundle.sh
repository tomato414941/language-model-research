#!/bin/bash
# Measures on a rented machine what each way of running the third bundle costs, then trains the second and the
# third bundle, each in that way for the cost of the baseline, and scores and keeps their last weights and the
# third's mixed ones. The third goes in if it beats the second.
# Usage: bundle.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./rented.sh

prepare
stamp speed
python speed.py --also=configs/bundle2.py > results/speed-options.txt
third=$(sed -n 1p results/speed-options.txt)
second=$(sed -n 2p results/speed-options.txt)
echo "third bundle: $third"
echo "second bundle: $second"
train_score_keep bundle2 configs/bundle2.py $second
train_score_keep bundle3 configs/bundle3.py $third
stamp finished
