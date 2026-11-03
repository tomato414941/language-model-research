#!/bin/bash
# Measures on a rented machine what each way of running the third bundle costs, then trains the bundle
# in the cheapest way for the cost of the baseline, scores it and keeps its weights.
# Usage: bundle.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./rented.sh

prepare
stamp speed
options=$(python speed.py)
echo "options: $options"
train_score_keep bundle3 configs/bundle3.py $options
stamp finished
