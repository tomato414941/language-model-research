#!/bin/bash
# Scores saved models in full on a rented machine, as when a new way of scoring reaches models trained before
# it. The data directory holds each model's directory beside what rental.py sends by default.
# Usage: score_saved.sh MODEL...
set -euo pipefail
cd "$(dirname "$0")"
source ./rented.sh

prepare
for name in "$@"; do
    score_in_full "$name" "data/$name"
done
stamp finished
