#!/bin/bash
# Scores saved models in full with the LM Evaluation Harness on a rented machine, first the benchmarks
# of each language's mean, then MMLU in its authors' form, which takes longest. The data directory holds
# each model's directory, prepare.py's tokenizer.json, the datasets cache of the harness's questions, and
# constraints.txt, the versions that a try without a GPU ran with. Nothing is fetched from the Hugging
# Face Hub. SCORING_OPTIONS go to the harness, as in rented.sh.
# Usage: milestone.sh MODEL...
set -euo pipefail
cd "$(dirname "$0")"
source ./rented.sh

prepare
stamp harness packages
pip install -q --break-system-packages -c data/constraints.txt "lm-eval[hf]==0.4.13"
export HF_DATASETS_CACHE="$PWD/data/datasets" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

# A smaller batch than the harness's default, whose logits alone would nearly fill 24 GB.
score() {
    local name=$1 result=$2
    shift 2
    python harness.py "results/$result.json" --model "data/$name" --tokenizer data/tokenizer.json \
        --batch-tokens 65536 "$@" ${SCORING_OPTIONS:-} > "results/$result.log" 2>&1
}

for name in "$@"; do
    stamp "harness $name"
    score "$name" "$name"
done
for name in "$@"; do
    stamp "authors' MMLU $name"
    score "$name" "$name-authors-mmlu" --authors-mmlu
done
stamp finished
