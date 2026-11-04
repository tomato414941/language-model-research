#!/bin/bash
# Rerun the base model's tau2 conversations that its server crash cut off, with the settings of run_tau2.sh,
# then stop the server. The task ids are read from retry_ids.txt, one per line.
cd /root/run
BASE=google/gemma-4-12B-it
note() { echo "$(date +%T) $*" | tee -a /root/out/stage.log; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
mapfile -t IDS < /root/run/retry_ids.txt
# Also removes the distilled model's API process, which never finished shutting down.
pkill -9 -f '[v]llm serve'
pkill -9 -f '[V]LLM::'
for i in $(seq 1 30); do test "$(gpu_used)" -lt 1500 && break; sleep 5; done
note "RETRY_START tb: ${#IDS[@]} tasks"
PORT_VLLM=8000 GPU_UTIL=0.92 setsid nohup bash serve.sh "$BASE" gemma4 tools > serve_base_tools2.log 2>&1 < /dev/null &
sleep 10
ok=0
for i in $(seq 1 120); do
  test "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = 200 && { ok=1; break; }
  pgrep -f '[v]llm serve' > /dev/null || break
  sleep 5
done
if [ "$ok" = 1 ]; then
  note "server ready base_tools2"
  cd /root/tau2
  T0=$(date +%s)
  /root/tau2-env/bin/tau2 run --benchmark text --language en --domain telecom \
    --agent-llm "hosted_vllm/$BASE" \
    --agent-llm-args '{"api_base": "http://127.0.0.1:8001/tb/v1", "api_key": "none"}' \
    --user-llm "hosted_vllm/$BASE" \
    --user-llm-args '{"api_base": "http://127.0.0.1:8000/v1", "api_key": "none", "temperature": 0.0}' \
    --workers 2 --max-concurrency 16 --timeout 1200 --task-ids "${IDS[@]}" --save-to stage1_tb_retry \
    > /root/out/tau2_tb/retry.log 2>&1
  note "TAU2_RETRY_DONE tb rc=$? sec=$(( $(date +%s) - T0 ))"
  grep -q -E "EngineDeadError|CUDA error" /root/run/serve_base_tools2.log && note "SERVER_CRASH during tau2 tb retry"
else
  note "server failed base_tools2"
fi
pkill -f '[v]llm serve'
note RETRY_DONE
