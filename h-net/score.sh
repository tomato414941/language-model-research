# Scores kept H-Net weights (models/NAME) on any GPU, from the repository's root after rent.py send has put them there
# with research's harness data (questions.jsonl, datasets, constraints.txt) and the BPE dataset whose validation rows
# scoring.py reads. Usage: bash h-net/score.sh DATASET NAME...
set -uo pipefail
source ./rented.sh
dataset=$1
shift
prepare
pip install -q --break-system-packages "flash-linear-attention==0.6.0" || exit 1
MAMBA_SKIP_CUDA_BUILD=TRUE pip install -q --break-system-packages --no-deps --no-build-isolation "mamba-ssm==2.3.2.post1" || exit 1
for name in "$@"; do
    stamp "score $name"
    python h-net/score_hnet.py "results/$name.json" "models/$name" --tokens "data/$dataset" ${SCORE_OPTIONS:-} \
        > "results/score-$name.log" 2>&1 || tail -n 20 "results/score-$name.log"
done
stamp done
