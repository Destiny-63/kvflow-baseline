# KVFlow 基线

把公开的 [KVFlow](https://github.com/PanZaifeng/KVFlow) 跑在真实 GPU 上，核对分层 KV cache：冷计算、GPU 命中、CPU 备份，以及淘汰之后从 CPU 搬回 GPU。搬回之后的输出和冷计算一致，并且只重新计算未命中的 token。

这里用的是作者公开 fork，commit [`7ef897e`](https://github.com/PanZaifeng/KVFlow/commit/7ef897eaedc16626a281ccfe0d453fae80251e8f)，源码版本 0.5.2rc1。论文里的原型是 SGLang v0.4.4，所以这组数字是当前 fork 的缓存路径和基础耗时，用来作为后续实验的环境基线。

引擎源码用上游仓库，不放进本仓库。许可证是 Apache-2.0。模型权重和 Docker 镜像也不在这里。

## 环境

- 模型：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`，BF16，TP=1
- 镜像：`lmsysorg/sglang:v0.5.2-cu126`，digest `sha256:fce5585fa8da175224ead70727a4660f719e34adf0638aebf6dd27930f03e1b6`
- GPU KV pool：4096 tokens（约 576 MiB）
- CPU KV pool：32768 tokens（host ratio 8，约 4.5 GiB）
- context 4096，最多 2 个并发请求，chunked prefill 2048，关闭 CUDA graph，关闭主动 prefetch
- HiCache：write-through、kernel backend、layer-first，调度为 FCFS
- 每个 KV token：`2 × 36 × 8 × 128 × 2 = 147456` bytes

同一套配置分别在 RTX 4090 和 A100 80GB 上跑过。两台机器的 CPU、NUMA 和后台负载不同，数字要分开看。

## 两处缓存修复

对应 diff 在 `w2/patches/w2-combined.patch`，打在上面的 commit 上。

1. `RadixCache.reset()` 在 flush 时清掉 agent 对旧 radix 节点的引用。
2. host reload 完成后重新匹配 device prefix。修复前即使 36 层都已搬回，请求仍会把 1024 个 token 全部重算；修复后命中 1023，只 forward 1 个 token。修复前的记录留在 `w2/results/cache_and_prefill_before_reload_fix.json`。

## 结果

请求总耗时包含 HTTP、调度、采样和 HiCache 写入。forward 时间是 CUDA event span，包含 attention metadata 和 stream wait。每个长度先预热，flush 之后测 5 次，生成 1 个 token，下表是中位数。

### RTX 4090 冷 prefill

| 输入 tokens | 请求总耗时 | forward |
|---:|---:|---:|
| 256 | 73.42 ms | 41.88 ms |
| 1024 | 141.02 ms | 106.51 ms |
| 2048 | 240.56 ms | 202.58 ms |
| 3072 | 352.93 ms | 310.36 ms |

CPU→GPU reload 微基准对全部 K/V 做逐位校验。连续索引、按层同步、1024 tokens（144 MiB）的中位数是 **9.34 ms（16.16 GB/s）**。这是 pool 搬运时间，不含排队和完整前向。

预热后的 1024-token 端到端，各 5 次：冷请求 **139.58 ms**，GPU 命中 **71.80 ms**，CPU 重访 **112.46 ms**。GPU 命中和 CPU 重访都只 forward 1 个 token。

### A100 80GB 冷 prefill

| 输入 tokens | 请求总耗时 | forward |
|---:|---:|---:|
| 256 | 55.27 ms | 26.76 ms |
| 1024 | 110.25 ms | 78.17 ms |
| 2048 | 185.38 ms | 148.77 ms |
| 3072 | 277.38 ms | 241.74 ms |

同样的 1024-token reload 微基准中位数是 **13.45 ms（11.23 GB/s）**。端到端：冷请求 **109.89 ms**，GPU 命中 **64.44 ms**，CPU 重访 **103.89 ms**。

A100 上还跑了一个两步调用的小任务：第一次让模型按协议发出聚合请求，脚本做本地聚合，第二次读入工具结果并给出各地区总额。三种条件的总额都正确（North 1320、South 1415、East 1405、West 1460）。冷缓存 1206.89 ms，GPU 上重复跑 1092.77 ms，先挤出 GPU 缓存再从 CPU 搬回是 1221.18 ms。每种条件只有一个样本。

原始记录：

- 4090：`w2/results/`
- A100：`w2/a100/results/`

## 运行

需要 NVIDIA GPU、Docker，以及能拉取上面的镜像和 `Qwen/Qwen3-8B`。在仓库根目录执行。

```bash
git clone https://github.com/PanZaifeng/KVFlow.git w2/vendor/KVFlow
git -C w2/vendor/KVFlow checkout 7ef897eaedc16626a281ccfe0d453fae80251e8f
git -C w2/vendor/KVFlow apply ../../patches/w2-combined.patch
bash w2/scripts/download_model.sh
python3 w2/scripts/verify_model.py
bash w2/scripts/start_server.sh 0
```

`start_server.sh` 在目标 GPU 已用显存超过 100 MiB 时会拒绝启动。GPU 0 默认绑定 CPU `0,2,4,6,8,10` 和 NUMA 0，GPU 1 绑定另一组。其他机器用 `W2_CPUSET_CPUS` 和 `W2_CPUSET_MEMS` 覆盖。

服务健康之后：

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

A100 上的一整串检查可以用 `bash w2/scripts/run_a100.sh 0`。两步调用任务：

```bash
docker exec kvflow-baseline python3 w2/scripts/run_sales_task.py \
  --events w2/logs/cache_events.jsonl \
  --output w2/a100/results/sales_task.json
```

这些测试会清空当前服务的 KV cache。测完后 `docker stop kvflow-baseline`。
