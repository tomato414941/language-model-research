#!/bin/bash
# Take over from stage3.sh after the customer server crashed during the distilled model's tau2 run:
# bring the customer server back so the last conversation can finish, then run tau2 for the base model.
# The base model plays both the agent and the customer on one server, since both are the same model.
cd /root/run
BASE=google/gemma-4-12B-it
note() { echo "$(date +%T) $*" | tee -a /root/out/stage.log; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
# Arguments: model id, log tag, port, GPU share, optional "tools".
start_server() {
  if pkill -f "[v]llm serve .*--port $3 "; then
    for i in $(seq 1 12); do pgrep -f "[v]llm serve .*--port $3 " > /dev/null || break; sleep 5; done
    sleep 20
  fi
  PORT_VLLM=$3 GPU_UTIL=$4 setsid nohup bash serve.sh "$1" gemma4 ${5:-} > "serve_$2.log" 2>&1 < /dev/null &
  sleep 10
  for i in $(seq 1 120); do
    test "$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$3/health)" = 200 && { note "server ready $2"; return 0; }
    pgrep -f "[v]llm serve .*--port $3 " > /dev/null || break
    sleep 5
  done
  note "server failed $2"
  return 1
}
stop_all() {
  pkill -f '[v]llm serve' || true
  for i in $(seq 1 30); do test "$(gpu_used)" -lt 1500 && break; sleep 5; done
}
crashed() { grep -q -E "EngineDeadError|CUDA error" "serve_$1.log"; }

note "STAGE4_START: customer server crashed at 11:00:36 (CUDA error); restarting it"
start_server "$BASE" sim2 8002 0.45 tools
while pgrep -f '[r]un_tau2.sh tv' > /dev/null; do sleep 15; done
crashed dist_tools && note "SERVER_CRASH during tau2 tv"
crashed sim2 && note "SERVER_CRASH in the restarted customer server"
stop_all
if start_server "$BASE" base_tools 8000 0.92 tools; then
  SIM_URL=http://127.0.0.1:8000/v1 bash run_tau2.sh tb "$BASE"
  crashed base_tools && note "SERVER_CRASH during tau2 tb"
fi
stop_all
note STAGE_DONE
