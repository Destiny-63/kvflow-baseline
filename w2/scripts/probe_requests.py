"""Exercise GPU hits and forced host reload, then measure cold request latency."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


def call(base, route, body=None, expect_json=True):
    data = json.dumps(body).encode() if body is not None else None
    request = Request(base + route, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response) if expect_json else response.read().decode()


def token_ids(seed, length):
    rng = random.Random(seed)
    return [rng.randrange(100, 20000) for _ in range(length)]


def generate(base, label, ids):
    start_ns = time.time_ns()
    start = time.perf_counter()
    response = call(base, "/generate", {
        "input_ids": ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
    })
    return {
        "label": label, "start_ns": start_ns, "end_ns": time.time_ns(),
        "wall_ms": (time.perf_counter() - start) * 1000,
        "meta_info": response["meta_info"], "text": response.get("text"),
        "output_ids": response.get("output_ids"),
    }


def read_events(path, start_ns, end_ns):
    if not path.exists():
        return []
    return [record for line in path.read_text().splitlines()
            if start_ns <= (record := json.loads(line))["time_ns"] <= end_ns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30126")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    args = parser.parse_args()
    layers = json.loads(args.model_config.read_text())["num_hidden_layers"]
    result = {"server_info": call(args.base, "/get_server_info"), "cache_probe": [], "prefill": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    call(args.base, "/flush_cache", expect_json=False)
    prefix = token_ids(42, 1024)
    result["cache_probe"].append(generate(args.base, "cold", prefix))
    result["cache_probe"].append(generate(args.base, "gpu_warm", prefix))
    for index in range(6):
        result["cache_probe"].append(generate(args.base, f"pressure_{index}", token_ids(100 + index, 1024)))
        time.sleep(0.15)
    time.sleep(0.5)
    revisit = generate(args.base, "host_revisit", prefix)
    time.sleep(0.5)
    revisit["events"] = read_events(args.events, revisit["start_ns"], time.time_ns())
    result["cache_probe"].append(revisit)
    cold, warm = result["cache_probe"][:2]
    checks = {
        "cold_cached_tokens_zero": cold["meta_info"]["cached_tokens"] == 0,
        "gpu_prefix_hit": warm["meta_info"]["cached_tokens"] >= 1023,
        "host_reload_enqueued": any(e["event"] == "reload_enqueued" for e in revisit["events"]),
        "all_model_reload_layers_submitted": len({e["layer"] for e in revisit["events"]
                                                    if e["event"] == "reload_layer_submitted"}) == layers,
        "host_revisit_cached_tokens": revisit["meta_info"]["cached_tokens"] >= 1023,
        "deterministic_output_preserved": cold["text"] == warm["text"] == revisit["text"],
    }
    if cold["output_ids"] is not None:
        checks["deterministic_token_ids_preserved"] = cold["output_ids"] == warm["output_ids"] == revisit["output_ids"]
    events = read_events(args.events, cold["start_ns"], time.time_ns())
    checks["cpu_backup_submitted"] = any(e["event"] == "backup_kernel_submitted" for e in events)
    checks["gpu_eviction_kept_host_copy"] = any(e["event"] == "gpu_evicted_host_retained" for e in events)
    result["checks"] = checks
    save()
    # One untimed warm-up per length keeps cold KV measurements free of first-use compilation.
    for length in [256, 1024, 2048, 3072]:
        generate(args.base, f"warmup_{length}", token_ids(9000 + length, length))
        rows = []
        for repeat in range(5):
            call(args.base, "/flush_cache", expect_json=False)
            row = generate(args.base, f"cold_{length}_{repeat}", token_ids(20000 + length + repeat, length))
            row["input_tokens"] = length
            row["input_tokens_per_second"] = length / (row["wall_ms"] / 1000)
            assert row["meta_info"]["cached_tokens"] == 0, row
            rows.append(row)
        result["prefill"].append({
            "input_tokens": length, "samples": rows,
            "median_wall_ms": statistics.median(row["wall_ms"] for row in rows),
            "median_input_tokens_per_second": statistics.median(row["input_tokens_per_second"] for row in rows),
            "semantics": "single cold request, max_new_tokens=1; includes HTTP, queue, prefill, sampling, and HiCache write-through",
        })
        save()
    time.sleep(0.5)
    generate(args.base, "drain_cuda_events", token_ids(800000, 128))
    all_events = read_events(args.events, cold["start_ns"], time.time_ns())
    for row in [cold, warm, revisit]:
        timings = [e for e in all_events if e["event"] == "prefill_forward_completed"
                   and row["start_ns"] <= e["forward_start_ns"] <= row["end_ns"]]
        row["forward_input_tokens"] = sum(e["tokens"] for e in timings)
    checks["host_revisit_only_one_prefill_token"] = revisit["forward_input_tokens"] == 1
    for group in result["prefill"]:
        for row in group["samples"]:
            timings = [e for e in all_events if e["event"] == "prefill_forward_completed"
                       and row["start_ns"] <= e["forward_start_ns"] <= row["end_ns"]]
            assert sum(e["tokens"] for e in timings) == row["input_tokens"], (row, timings)
            row["forward_cuda_ms"] = sum(e["cuda_ms"] for e in timings)
            row["forward_tokens_per_second"] = row["input_tokens"] / (row["forward_cuda_ms"] / 1000)
        group["median_forward_cuda_ms"] = statistics.median(row["forward_cuda_ms"] for row in group["samples"])
        group["median_forward_tokens_per_second"] = statistics.median(row["forward_tokens_per_second"] for row in group["samples"])
        group["forward_semantics"] = "CUDA event span around forward_extend, includes attention metadata and stream waits; chunk spans summed; async collection without added synchronization"
    save()
    print(json.dumps({"checks": checks, "prefill": [{k: v for k, v in r.items() if k != "samples"}
                                                      for r in result["prefill"]]}, indent=2))
    if not all(checks.values()):
        raise SystemExit("Cache validation failed; inspect events and raw output")


if __name__ == "__main__":
    main()
