#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_dir"
model_dir=w2/models/Qwen3-8B
revision=b968826d9c46dd6066d109eabc6255188de91218
endpoint=${W2_HF_ENDPOINT:-https://huggingface.co}
jobs=${W2_DOWNLOAD_JOBS:-3}
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "W2_DOWNLOAD_JOBS must be positive" >&2; exit 1; }
mkdir -p "$model_dir"
download_one() {
  local file=$1 url
  if [[ ! -s "$model_dir/$file" ]]; then
    url="$endpoint/Qwen/Qwen3-8B/resolve/$revision/$file"
    if [[ -n "${W2_ARIA2:-}" ]]; then
      "$W2_ARIA2" --continue=true --max-connection-per-server=16 --split=16 \
        --min-split-size=1M --file-allocation=none --auto-file-renaming=false \
        --allow-overwrite=true --summary-interval=30 --console-log-level=warn \
        --show-console-readout=false --enable-color=false \
        --max-tries=5 --retry-wait=5 --connect-timeout=20 \
        --dir="$model_dir" --out="$file.partial" "$url"
    else
      curl -fL --retry 3 --connect-timeout 20 --speed-limit 1024 --speed-time 60 -C - \
        "$url" -o "$model_dir/$file.partial"
    fi
    mv "$model_dir/$file.partial" "$model_dir/$file"
  fi
}
for file in config.json generation_config.json merges.txt model.safetensors.index.json \
  tokenizer.json tokenizer_config.json vocab.json LICENSE; do
  download_one "$file"
done
pids=()
failed=0
for shard in 1 2 3 4 5; do
  printf -v file 'model-%05d-of-00005.safetensors' "$shard"
  download_one "$file" &
  pids+=("$!")
  if (( ${#pids[@]} >= jobs )); then
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
du -sh "$model_dir"
