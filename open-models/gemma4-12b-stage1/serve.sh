#!/bin/bash
# Serve one model with vLLM. Arguments: model id, reasoning parser, optional "tools" for tau2 runs.
# PORT_VLLM and GPU_UTIL let two servers share the GPU during tau2 runs.
export HF_HOME=/root/hf HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0
PORT_VLLM=${PORT_VLLM:-8000}
GPU_UTIL=${GPU_UTIL:-0.92}
EXTRA=(--max-model-len 32768)
# Benchmark runs keep the measured settings; only tau2 runs parse tool calls and allow longer conversations.
if [ "${3:-}" = tools ]; then
  EXTRA=(--max-model-len 65536 --enable-auto-tool-choice --tool-call-parser gemma4)
fi
exec /root/vllm-env/bin/vllm serve "$1" --host 127.0.0.1 --port "$PORT_VLLM" "${EXTRA[@]}" --reasoning-parser "$2" --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs 90 --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'
