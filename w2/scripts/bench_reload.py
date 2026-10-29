"""Measure and verify the same host-pool kernels used by HiCache."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import MHATokenToKVPoolHost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    config = json.loads(args.model_config.read_text())
    layers = config["num_hidden_layers"]
    kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    torch.manual_seed(42)
    pool = MHATokenToKVPool(4096, 1, torch.bfloat16, kv_heads, head_dim, layers, "cuda", False)
    host = MHATokenToKVPoolHost(pool, 2, 0, 1, "layer_first")
    for layer in range(layers):
        pool.k_buffer[layer].normal_()
        pool.v_buffer[layer].normal_()
    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "dtype": "bfloat16",
              "layers": layers, "kv_heads": kv_heads, "head_dim": head_dim, "bytes_per_token": host.size_per_token,
              "backend": "kernel", "layout": "layer_first", "page_size": 1, "samples": []}
    for pattern in ["contiguous", "permuted"]:
        for tokens in [256, 1024, 2048, 4096]:
            cpu_indices = torch.arange(tokens, dtype=torch.int64)
            host_indices = cpu_indices.cuda()
            device_indices = torch.arange(1, tokens + 1, device="cuda", dtype=torch.int64)
            host.backup_from_device_all_layer(pool, host_indices, device_indices, "kernel")
            torch.cuda.synchronize()
            # Verify every K/V element after backup before overwriting the GPU destination.
            backup_ok = all(torch.equal(host.k_buffer[layer, :tokens], pool.k_buffer[layer][1:tokens + 1].cpu())
                            and torch.equal(host.v_buffer[layer, :tokens], pool.v_buffer[layer][1:tokens + 1].cpu())
                            for layer in range(layers))
            if pattern == "permuted":
                cpu_indices = torch.randperm(tokens, dtype=torch.int64)
                host_indices = cpu_indices.cuda()
            for layer in range(layers):
                pool.k_buffer[layer][1:tokens + 1].zero_()
                pool.v_buffer[layer][1:tokens + 1].zero_()

            def reload(sync_per_layer=False):
                for layer in range(layers):
                    host.load_to_device_per_layer(pool, host_indices, device_indices, layer, "kernel")
                    if sync_per_layer:
                        torch.cuda.current_stream().synchronize()

            reload()
            torch.cuda.synchronize()
            reload_ok = all(torch.equal(host.k_buffer[layer, cpu_indices], pool.k_buffer[layer][1:tokens + 1].cpu())
                            and torch.equal(host.v_buffer[layer, cpu_indices], pool.v_buffer[layer][1:tokens + 1].cpu())
                            for layer in range(layers))
            assert backup_ok and reload_ok, (pattern, tokens, backup_ok, reload_ok)
            for mode in ["enqueue_all_layers", "controller_layer_sync"]:
                sync = mode == "controller_layer_sync"
                for _ in range(3):
                    reload(sync)
                torch.cuda.synchronize()
                cuda_ms, wall_ms = [], []
                for _ in range(args.repeats):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    start.record()
                    reload(sync)
                    end.record()
                    end.synchronize()
                    cuda_ms.append(start.elapsed_time(end))
                    wall_ms.append((time.perf_counter() - wall_start) * 1000)
                nbytes = tokens * host.size_per_token
                median = statistics.median(cuda_ms)
                result["samples"].append({
                    "pattern": pattern, "mode": mode, "tokens": tokens, "bytes": nbytes,
                    "backup_bitwise_verified": backup_ok, "reload_bitwise_verified": reload_ok,
                    "cuda_ms": cuda_ms, "wall_ms": wall_ms, "median_cuda_ms": median,
                    "median_wall_ms": statistics.median(wall_ms),
                    "median_GB_per_second": nbytes / (median / 1000) / 1e9,
                    "median_GiB_per_second": nbytes / (median / 1000) / (1024 ** 3),
                })
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "samples": [{k: v for k, v in row.items() if k not in ["cuda_ms", "wall_ms"]}
                                           for row in result["samples"]]}, indent=2))


if __name__ == "__main__":
    main()
