#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export W2_CONTAINER=${W2_CONTAINER:-kvflow-baseline-a100}
mkdir -p w2/results w2/logs
nvidia-smi > w2/logs/hardware_before.txt
nvidia-smi topo -m > w2/logs/topology.txt
python3 w2/scripts/verify_model.py > w2/logs/model_verification.log
bash w2/scripts/start_server.sh "${1:-0}"
cleanup() {
  docker logs "$W2_CONTAINER" > w2/logs/server.log 2>&1 || true
  docker inspect "$W2_CONTAINER" > w2/logs/container.json || true
  docker stop -t 30 "$W2_CONTAINER" || true
  nvidia-smi > w2/logs/hardware_after.txt
}
trap cleanup EXIT
ready=0
for ((attempt=0; attempt<180; attempt++)); do
  if curl -fsS --max-time 5 http://127.0.0.1:30126/health > /dev/null; then
    ready=1
    break
  fi
  [[ $(docker inspect -f '{{.State.Running}}' "$W2_CONTAINER") == true ]] || exit 1
  sleep 5
done
[[ $ready == 1 ]] || exit 1
docker exec "$W2_CONTAINER" python3 w2/scripts/inspect_environment.py > w2/logs/environment.log
docker exec "$W2_CONTAINER" python3 w2/scripts/probe_requests.py \
  --model-config w2/models/Qwen3-8B/config.json --events w2/logs/cache_events.jsonl \
  --output w2/results/cache_and_prefill.json > w2/logs/probe.log
docker exec "$W2_CONTAINER" python3 w2/scripts/bench_reload.py \
  --model-config w2/models/Qwen3-8B/config.json --output w2/results/reload.json > w2/logs/reload.log
docker exec "$W2_CONTAINER" python3 w2/scripts/bench_cache_e2e.py \
  --events w2/logs/cache_events.jsonl --output w2/results/cache_e2e.json > w2/logs/cache_e2e.log
docker exec "$W2_CONTAINER" python3 w2/scripts/summarize_results.py > w2/logs/summary.log
echo 'Checks and profiling completed.'
