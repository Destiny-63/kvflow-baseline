#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
gpu=${1:-0}
container=${W2_CONTAINER:-kvflow-baseline}
if [[ -n "${W2_CPUSET_CPUS:-}" && -n "${W2_CPUSET_MEMS:-}" ]]; then
  default_cpus=$W2_CPUSET_CPUS
  default_mems=$W2_CPUSET_MEMS
else
case "$gpu" in
  0) default_cpus=0,2,4,6,8,10; default_mems=0 ;;
  1) default_cpus=1,3,5,7,9,11; default_mems=1 ;;
  *) echo "Select GPU 0 or 1 on the W2 host" >&2; exit 1 ;;
esac
fi
memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "$memory_used" -gt 100 ]]; then
  echo "GPU $gpu is occupied ($memory_used MiB); select an idle allocated GPU." >&2
  exit 1
fi
mkdir -p w2/logs w2/results w2/cache
image=lmsysorg/sglang@sha256:fce5585fa8da175224ead70727a4660f719e34adf0638aebf6dd27930f03e1b6
docker run -d --name "$container" --gpus "device=$gpu" \
  --user "$(id -u):$(id -g)" --shm-size 2g --ulimit memlock=-1 \
  --cpuset-cpus "${W2_CPUSET_CPUS:-$default_cpus}" --cpuset-mems "${W2_CPUSET_MEMS:-$default_mems}" \
  -p 127.0.0.1:30126:30126 \
  -v "$project_dir:/workspace" -w /workspace \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -e HF_HOME=/workspace/w2/cache/huggingface \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/w2/cache/torchinductor \
  -e FLASHINFER_WORKSPACE_BASE=/workspace/w2/cache \
  -e SGLANG_LOGGING_CONFIG_PATH=/workspace/w2/logging.json \
  -e XDG_CACHE_HOME=/workspace/w2/cache -e TRITON_CACHE_DIR=/workspace/w2/cache/triton \
  -e PYTHONPATH=/workspace/w2/vendor/KVFlow/python:/workspace/w2/scripts \
  -e W2_CACHE_EVENTS=/workspace/w2/logs/cache_events.jsonl \
  "$image" python3 w2/scripts/launch_instrumented.py \
  --model-path /workspace/w2/models/Qwen3-8B --dtype bfloat16 --tp 1 \
  --host 0.0.0.0 --port 30126 --context-length 4096 --max-total-tokens 4096 \
  --max-running-requests 2 --mem-fraction-static 0.88 \
  --chunked-prefill-size 2048 --disable-cuda-graph \
  --disable-prefetch \
  --enable-hierarchical-cache --hicache-ratio 8 \
  --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout layer_first
