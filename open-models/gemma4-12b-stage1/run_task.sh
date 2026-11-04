#!/bin/bash
# Run one evaluation task against the local vLLM server and record tokens and time.
# Arguments: run tag, config yaml, task id, concurrent calls, optional max samples.
TAG=$1; CFG=$2; TASK=$3; CONC=$4; MAXS=${5:-}
OUT=/root/out/$TAG
mkdir -p "$OUT"
cd /root/harness
export HF_HOME=/root/hf LITELLM_CONCURRENT_CALLS=$CONC REQUEST_TIMEOUT=7200 LITELLM_REQUEST_TIMEOUT=7200 TOKENIZERS_PARALLELISM=false
tokens() { curl -s -m 5 http://127.0.0.1:8000/metrics | awk '/^vllm:generation_tokens_total/ {s+=$2} END {printf "%d", s}'; }
preempt() { curl -s -m 5 http://127.0.0.1:8000/metrics | awk '/^vllm:num_preemptions_total/ {s+=$2} END {printf "%d", s}'; }
T0=$(date +%s); G0=$(tokens); P0=$(preempt)
EXTRA=(); test -n "$MAXS" && EXTRA=(--max-samples "$MAXS")
.venv/bin/lighteval endpoint litellm "$CFG" "$TASK|0|0" --use-chat-template --output-dir "$OUT" --save-details "${EXTRA[@]}" > "$OUT/log_$(echo "$TASK" | tr '|:' '__').log" 2>&1
RC=$?
T1=$(date +%s); G1=$(tokens); P1=$(preempt)
printf '%s\t%s\trc=%s\tsec=%s\ttokens=%s\tpreempt=%s\ttok_per_s=%s\tmax=%s\n' "$(date +%T)" "$TASK" "$RC" "$((T1-T0))" "$((G1-G0))" "$((P1-P0))" "$(( (G1-G0) / ((T1-T0)>0 ? (T1-T0) : 1) ))" "$MAXS" | tee -a "$OUT/usage.tsv"
