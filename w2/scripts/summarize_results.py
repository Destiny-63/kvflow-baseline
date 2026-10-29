"""Produce measured lookup parameters for the W4 screening simulator."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("w2/results"))
    args = parser.parse_args()
    prefill = json.loads((args.results / "cache_and_prefill.json").read_text())
    reload = json.loads((args.results / "reload.json").read_text())
    cache_e2e = json.loads((args.results / "cache_e2e.json").read_text())
    assert all(prefill["checks"].values()), prefill["checks"]
    assert all(cache_e2e["checks"].values()), cache_e2e["checks"]
    assert all(row["backup_bitwise_verified"] and row["reload_bitwise_verified"]
               for row in reload["samples"])
    result = {
        "model": "Qwen/Qwen3-8B",
        "gpu": reload["gpu"],
        "dtype": reload["dtype"],
        "tp": 1,
        "bytes_per_kv_token": reload["bytes_per_token"],
        "gpu_kv_tokens": 4096,
        "host_kv_tokens": 32768,
        "cache_e2e": {
            "input_tokens": cache_e2e["input_tokens"],
            "summary": cache_e2e["summary"],
            "semantics": cache_e2e["semantics"],
        },
        "prefill": [
            {key: group[key] for key in (
                "input_tokens", "median_wall_ms", "median_forward_cuda_ms",
                "median_input_tokens_per_second", "median_forward_tokens_per_second"
            )} for group in prefill["prefill"]
        ],
        "reload": [
            {key: row[key] for key in (
                "pattern", "mode", "tokens", "bytes", "median_wall_ms", "median_cuda_ms"
            )} | {"wall_GB_per_second": row["bytes"] / row["median_wall_ms"] / 1e6}
            for row in reload["samples"]
        ],
        "usage": {
            "prefill": "Length-dependent lookup; forward span and cold request wall time have different semantics. Neither predicts decode or queueing.",
            "reload": "Prefer controller_layer_sync wall time for pool transfer cost in this fork. This excludes queueing, radix bookkeeping and contention; cache_e2e records full request observations separately. Do not treat the pool time as full exposed wait.",
            "interpolation": "Interpolate only within measured lengths as a screening assumption; validate against online runs.",
            "hardware": "Reprofile after changing GPU, NUMA affinity, backend, layout, capacity, or concurrency."
        }
    }
    (args.results / "profiling_parameters.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
