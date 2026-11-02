#!/bin/bash
# Trains one model on a rented machine, with the ways of running and the updates its configuration gives,
# scores it and keeps its weights.
# Usage: train_one.sh NAME CONFIG [TRAIN_OPTION...]
set -euo pipefail
cd "$(dirname "$0")"
source ./rented.sh

prepare
train_score_keep "$@"
stamp finished
