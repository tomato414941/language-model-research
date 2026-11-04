#!/bin/bash
# Run the 50 English tau2 telecom tasks for one served model through the logging proxy.
# Arguments: proxy condition (tb or tv), served model id.
# The customer is played by the base Gemma 4 12B, the same for every agent model. SIM_URL is its server
# (port 8002 by default; the base model's own server when the agent is the base model too).
TAG=$1; MODEL=$2
SIM_URL=${SIM_URL:-http://127.0.0.1:8002/v1}
OUT=/root/out/tau2_$TAG
mkdir -p "$OUT"
note() { echo "$(date +%T) $*" | tee -a /root/out/stage.log; }
if [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "${SIM_URL%/v1}/health")" != 200 ]; then
  note "SIM_UNAVAILABLE $TAG"
  exit 1
fi
cd /root/tau2
T0=$(date +%s)
/root/tau2-env/bin/tau2 run --benchmark text --language en --domain telecom \
  --agent-llm "hosted_vllm/$MODEL" \
  --agent-llm-args "{\"api_base\": \"http://127.0.0.1:8001/$TAG/v1\", \"api_key\": \"none\"}" \
  --user-llm "hosted_vllm/google/gemma-4-12B-it" \
  --user-llm-args "{\"api_base\": \"$SIM_URL\", \"api_key\": \"none\", \"temperature\": 0.0}" \
  --workers 2 --max-concurrency 16 --timeout 1200 --save-to "stage1_$TAG" > "$OUT/run.log" 2>&1
RC=$?
cp -r "/root/tau2/data/simulations/stage1_$TAG" "$OUT/" 2>/dev/null
note "TAU2_RUN_DONE $TAG rc=$RC sec=$(( $(date +%s) - T0 ))"
