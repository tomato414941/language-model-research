#!/bin/bash
# Take over from stage2.sh while its c4 run is still going: wait for c4, run the two base-model conditions,
# then tau2 with the base Gemma 4 12B as the customer on a second server that shares the GPU.
cd /root/run
BASE=google/gemma-4-12B-it
DIST=yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2
note() { echo "$(date +%T) $*" | tee -a /root/out/stage.log; }
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
# Arguments: model id, log tag, port, GPU share, optional "tools".
start_server() {
  if pkill -f "[v]llm serve .*--port $3 "; then
    # Wait for the old server on this port to exit and its engine process to release GPU memory.
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

note STAGE3_START
# stage2.sh started c4 before it was stopped; its run_cond.sh keeps going on its own.
while pgrep -f '[r]un_cond.sh c4' > /dev/null; do sleep 15; done
crashed dist && note "SERVER_CRASH during c4"
note PHASE_B_DONE

stop_all
start_server "$BASE" base2 8000 0.92 && bash run_cond.sh c2 light
TAG=base2
if crashed base2; then
  note "SERVER_CRASH during c2"
  stop_all
  start_server "$BASE" base3 8000 0.92
  TAG=base3
fi
# The first c3 run lost answers when the server crashed, so it is run again from the start.
test -d /root/out/c3 && mv /root/out/c3 "/root/out/c3_crashed_$(date +%H%M)"
bash run_cond.sh c3 full
crashed "$TAG" && note "SERVER_CRASH during c3"
note PHASE_C_DONE

if ! grep -q TAU2_DONE /root/run/setup.log; then
  note "tau2 skipped: install failed"
else
  stop_all
  # The customer server starts first so that each server finds its share of the GPU free.
  if start_server "$BASE" sim 8002 0.45 tools; then
    start_server "$DIST" dist_tools 8000 0.45 tools && bash run_tau2.sh tv "$DIST"
    crashed dist_tools && note "SERVER_CRASH during tau2 tv"
    start_server "$BASE" base_tools 8000 0.45 tools && bash run_tau2.sh tb "$BASE"
    crashed base_tools && note "SERVER_CRASH during tau2 tb"
    crashed sim && note "SERVER_CRASH in the customer server"
  fi
fi
stop_all
note STAGE_DONE
