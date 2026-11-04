#!/bin/bash
# Install the evaluator and the serving stack, then download the base model; the distilled model follows in the background.
set -euo pipefail
export HF_HOME=/root/hf UV_CACHE_DIR=/root/uvcache
cd /root
test -d harness || git clone --quiet --depth 1 -b v202604 https://github.com/swallow-llm/swallow-evaluation-instruct.git harness
echo "[$(date +%T)] clone done"
cd harness
uv python install 3.11.10 >/dev/null 2>&1
uv python pin 3.11.10 >/dev/null
uv sync --locked --extra lighteval >/root/run/sync.log 2>&1
echo "[$(date +%T)] evaluator installed"
cd /root
uv venv --python 3.12 /root/vllm-env >/dev/null 2>&1
VIRTUAL_ENV=/root/vllm-env uv pip install "vllm==0.32.0" >/root/run/vllm_install.log 2>&1
echo "[$(date +%T)] vllm installed"
/root/vllm-env/bin/hf download google/gemma-4-12B-it --quiet >/dev/null
echo "[$(date +%T)] downloaded base model"
echo SETUP_DONE
/root/vllm-env/bin/hf download yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2 --revision af644ab1c593a95a994257343b04b99546f431b2 --quiet >/dev/null
echo "[$(date +%T)] downloaded distilled model"
echo DISTILLED_DONE
# tau2 at the commit that was reviewed; a failure here must not stop the benchmark runs.
if { test -d /root/tau2 || git clone --quiet https://github.com/sierra-research/tau2-bench.git /root/tau2; } \
  && git -C /root/tau2 checkout --quiet 19f13ea \
  && uv venv --python 3.12 /root/tau2-env >/dev/null 2>&1 \
  && VIRTUAL_ENV=/root/tau2-env uv pip install -e /root/tau2 >/root/run/tau2_install.log 2>&1; then
  echo "[$(date +%T)] tau2 installed"
  echo TAU2_DONE
else
  echo TAU2_FAILED
fi
