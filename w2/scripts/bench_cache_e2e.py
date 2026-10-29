"""Measure warm-process cache paths separately from the pool reload benchmark."""

import argparse
import json
import statistics
import time
from pathlib import Path

from probe_requests import call, generate, read_events, token_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30126")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    result = {"input_tokens": 1024, "samples": [], "checks": {}, "summary": {}}
    generate(args.base, "e2e_warmup_cold", token_ids(500000, 1024))
    generate(args.base, "e2e_warmup_hit", token_ids(500000, 1024))
    start_ns = time.time_ns()
    for repeat in range(args.repeats):
        call(args.base, "/flush_cache", expect_json=False)
        prefix = token_ids(500001 + repeat, 1024)
        rows = {
            "cold": generate(args.base, f"e2e_cold_{repeat}", prefix),
            "gpu_hit": generate(args.base, f"e2e_gpu_hit_{repeat}", prefix),
        }
        for index in range(6):
            generate(args.base, f"e2e_pressure_{repeat}_{index}",
                     token_ids(600000 + repeat * 6 + index, 1024))
            time.sleep(0.15)
        time.sleep(0.5)
        rows["host_revisit"] = generate(args.base, f"e2e_host_{repeat}", prefix)
        result["samples"].append(rows)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    time.sleep(0.5)
    generate(args.base, "e2e_drain_cuda_events", token_ids(700000, 128))
    events = read_events(args.events, start_ns, time.time_ns())
    for rows in result["samples"]:
        for row in rows.values():
            timings = [event for event in events if event["event"] == "prefill_forward_completed"
                       and row["start_ns"] <= event["forward_start_ns"] <= row["end_ns"]]
            row["forward_input_tokens"] = sum(event["tokens"] for event in timings)
            row["forward_cuda_ms"] = sum(event["cuda_ms"] for event in timings)
        revisit = rows["host_revisit"]
        revisit["reload_enqueued"] = any(event["event"] == "reload_enqueued"
                                         and revisit["start_ns"] <= event["time_ns"] <= revisit["end_ns"]
                                         for event in events)
    result["checks"] = {
        "all_cold_uncached": all(rows["cold"]["meta_info"]["cached_tokens"] == 0
                                 and rows["cold"]["forward_input_tokens"] == 1024
                                 for rows in result["samples"]),
        "all_gpu_hits": all(rows["gpu_hit"]["meta_info"]["cached_tokens"] == 1023
                            and rows["gpu_hit"]["forward_input_tokens"] == 1
                            for rows in result["samples"]),
        "all_host_reloads_reused": all(rows["host_revisit"]["reload_enqueued"]
                                      and rows["host_revisit"]["meta_info"]["cached_tokens"] == 1023
                                      and rows["host_revisit"]["forward_input_tokens"] == 1
                                      for rows in result["samples"]),
        "all_outputs_equal": all(rows["cold"]["text"] == rows["gpu_hit"]["text"] == rows["host_revisit"]["text"]
                                 and rows["cold"]["output_ids"] == rows["gpu_hit"]["output_ids"] == rows["host_revisit"]["output_ids"]
                                 for rows in result["samples"]),
    }
    for path in ["cold", "gpu_hit", "host_revisit"]:
        result["summary"][path] = {
            "wall_ms": [rows[path]["wall_ms"] for rows in result["samples"]],
            "median_wall_ms": statistics.median(rows[path]["wall_ms"] for rows in result["samples"]),
            "median_forward_cuda_ms": statistics.median(rows[path]["forward_cuda_ms"] for rows in result["samples"]),
        }
    result["semantics"] = "Warm process, serial 1024-token requests, one generated token; host revisit follows six cache-pressure requests. Includes HTTP, scheduler, radix/controller and cache bookkeeping. Five samples do not establish tail latency."
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    if not all(result["checks"].values()):
        raise SystemExit("End-to-end cache path validation failed")


if __name__ == "__main__":
    main()
