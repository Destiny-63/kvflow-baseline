# KVFlow baseline

This repository runs the public [KVFlow](https://github.com/PanZaifeng/KVFlow) fork on real GPUs and checks hierarchical KV cache behavior: a cold prefill, a GPU prefix hit, a CPU backup, and a reload from CPU after eviction. After the reload, the output matches the cold run, and the request recomputes only the tokens that missed.

The run uses the authors' public fork at commit [`7ef897e`](https://github.com/PanZaifeng/KVFlow/commit/7ef897eaedc16626a281ccfe0d453fae80251e8f), source version 0.5.2rc1. The paper prototype uses SGLang v0.4.4, so these numbers are the cache-path checks and base latencies of this fork. They are the environment baseline for later experiments.

The engine source stays in the upstream repository. Its license is Apache-2.0. Model weights and the Docker image are not included here.

## Setup

- Model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, BF16, TP=1
- Image: `lmsysorg/sglang:v0.5.2-cu126`, digest `sha256:fce5585fa8da175224ead70727a4660f719e34adf0638aebf6dd27930f03e1b6`
- GPU KV pool: 4096 tokens (about 576 MiB)
- CPU KV pool: 32768 tokens (host ratio 8, about 4.5 GiB)
- Context length 4096, at most 2 running requests, chunked prefill 2048, CUDA graph off, active prefetch off
- HiCache: write-through, kernel backend, layer-first, FCFS scheduling
- Bytes per KV token: `2 × 36 × 8 × 128 × 2 = 147456`

The same configuration was measured on an RTX 4090 and on an A100 80GB. The two machines differ in CPU, NUMA placement, and background load, so the two sets of numbers stay separate.

## Two cache fixes

The diff is `w2/patches/w2-combined.patch`, applied on the commit above.

1. `RadixCache.reset()` drops agent references to radix nodes that flush has discarded.
2. After a host reload completes, the scheduler matches the device prefix again. Before the fix, a request recomputed all 1024 tokens even though all 36 layers had been loaded back. After the fix it hits 1023 tokens and forwards 1. The pre-fix record is `w2/results/cache_and_prefill_before_reload_fix.json`.

## Results

Request time includes HTTP, scheduling, sampling, and the HiCache write. Forward time is a CUDA event span and includes attention metadata and stream waits. Each length is warmed up, the cache is flushed, and the length is measured 5 times with 1 generated token. The tables report medians.

### RTX 4090 cold prefill

| Input tokens | Request | Forward |
|---:|---:|---:|
| 256 | 73.42 ms | 41.88 ms |
| 1024 | 141.02 ms | 106.51 ms |
| 2048 | 240.56 ms | 202.58 ms |
| 3072 | 352.93 ms | 310.36 ms |

The CPU-to-GPU reload microbenchmark checks every K and V value bit for bit. For contiguous indices, layer-by-layer synchronization, and 1024 tokens (144 MiB), the median is **9.34 ms (16.16 GB/s)**. That is pool-transfer time. It leaves out queuing and the full forward.

Warmed-up 1024-token end-to-end runs, 5 each: cold request **139.58 ms**, GPU hit **71.80 ms**, CPU revisit **112.46 ms**. The GPU hit and the CPU revisit each forward 1 token.

### A100 80GB cold prefill

| Input tokens | Request | Forward |
|---:|---:|---:|
| 256 | 55.27 ms | 26.76 ms |
| 1024 | 110.25 ms | 78.17 ms |
| 2048 | 185.38 ms | 148.77 ms |
| 3072 | 277.38 ms | 241.74 ms |

The same 1024-token reload microbenchmark has median **13.45 ms (11.23 GB/s)**. End to end: cold request **109.89 ms**, GPU hit **64.44 ms**, CPU revisit **103.89 ms**.

On the A100 there is also a two-call task. The first call emits an aggregation request in a fixed tool format, a local script aggregates the records, and the second call reads the tool result and reports the total for each region. All three conditions return the same totals: North 1320, South 1415, East 1405, West 1460. Cold cache takes 1206.89 ms, a repeat run on the GPU takes 1092.77 ms, and a run that first evicts the GPU copy and reloads it from CPU takes 1221.18 ms. Each condition is a single sample.

Raw records:

- 4090: `w2/results/`
- A100: `w2/a100/results/`

## Run

This needs an NVIDIA GPU, Docker, and access to the image and `Qwen/Qwen3-8B` above. Run the commands from the repository root.

```bash
git clone https://github.com/PanZaifeng/KVFlow.git w2/vendor/KVFlow
git -C w2/vendor/KVFlow checkout 7ef897eaedc16626a281ccfe0d453fae80251e8f
git -C w2/vendor/KVFlow apply ../../patches/w2-combined.patch
bash w2/scripts/download_model.sh
python3 w2/scripts/verify_model.py
bash w2/scripts/start_server.sh 0
```

`start_server.sh` refuses to start when the chosen GPU already has more than 100 MiB in use. GPU 0 defaults to CPUs `0,2,4,6,8,10` and NUMA node 0. GPU 1 uses the other set. On another machine, override them with `W2_CPUSET_CPUS` and `W2_CPUSET_MEMS`.

After the server is healthy:

```bash
docker exec kvflow-baseline python3 w2/scripts/inspect_environment.py
docker exec kvflow-baseline python3 w2/scripts/probe_requests.py \
  --model-config w2/models/Qwen3-8B/config.json \
  --events w2/logs/cache_events.jsonl \
  --output w2/results/cache_and_prefill.json
docker exec kvflow-baseline python3 w2/scripts/bench_reload.py \
  --model-config w2/models/Qwen3-8B/config.json \
  --output w2/results/reload.json
docker exec kvflow-baseline python3 w2/scripts/bench_cache_e2e.py \
  --events w2/logs/cache_events.jsonl \
  --output w2/results/cache_e2e.json
docker exec kvflow-baseline python3 w2/scripts/summarize_results.py
```

`bash w2/scripts/run_a100.sh 0` runs the full check sequence. The two-call task:

```bash
docker exec kvflow-baseline python3 w2/scripts/run_sales_task.py \
  --events w2/logs/cache_events.jsonl \
  --output w2/a100/results/sales_task.json
```

These tests flush the KV cache of the running server. Stop it with `docker stop kvflow-baseline`.
