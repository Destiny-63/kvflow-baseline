"""Run a two-call sales analysis workflow and validate its tool and final answer."""

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from probe_requests import call, generate, read_events, token_ids


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30126")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("w2/models/Qwen3-8B")
    regions = ["North", "South", "East", "West"]
    records = [
        {"order": i + 1, "region": regions[i % 4], "units": 1 + i % 7,
         "unit_price": 20 + 5 * (i % 5)} for i in range(48)
    ]
    totals = {region: 0 for region in regions}
    for record in records:
        totals[record["region"]] += record["units"] * record["unit_price"]
    expected = {"totals": totals, "best_region": max(totals, key=totals.get)}
    messages = [
        {"role": "system", "content": "You are a sales analyst. Return only a JSON object, without explanations or markdown. The available tool aggregate_sales computes sum(units * unit_price) grouped by region."},
        {"role": "user", "content": "Analyze these orders and find the region with the highest revenue. First request the available tool by returning exactly {\"tool\":\"aggregate_sales\",\"group_by\":\"region\"}. Orders:\n" + json.dumps(records)},
    ]
    result = {"task": "sales analysis with a local aggregation tool", "records": records,
              "expected": expected, "runs": [], "checks": {},
              "semantics": "Deterministic local task fixture, not a public benchmark. Cache-pressure calls use synthetic token IDs. No time-based prefetch or eviction policy is enabled."}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def llm(stage, chat):
        ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False)
        if len(ids) + 192 > 4096:
            raise ValueError("Task exceeds the configured context")
        start_ns = time.time_ns()
        started = time.perf_counter()
        response = call(args.base, "/generate", {
            "input_ids": ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 192},
        })
        end_ns = time.time_ns()
        return {"stage": stage, "start_ns": start_ns, "end_ns": end_ns,
                "wall_ms": (time.perf_counter() - started) * 1000,
                "input_tokens": len(ids), "response": response,
                "parsed": parse_json(response["text"])}

    call(args.base, "/flush_cache", expect_json=False)
    for mode in ["cold", "gpu_reuse", "host_revisit"]:
        if mode == "host_revisit":
            for index in range(6):
                generate(args.base, f"sales_pressure_{index}", token_ids(810000 + index, 1024))
                time.sleep(0.15)
            time.sleep(0.5)
        workflow_start = time.perf_counter()
        first = llm("select_tool", messages)
        assert first["parsed"] == {"tool": "aggregate_sales", "group_by": "region"}, first
        tool_start = time.perf_counter()
        tool_totals = {region: 0 for region in regions}
        for record in records:
            tool_totals[record["region"]] += record["units"] * record["unit_price"]
        tool_ms = (time.perf_counter() - tool_start) * 1000
        final_messages = messages + [
            {"role": "assistant", "content": first["response"]["text"]},
            {"role": "user", "content": "Tool aggregate_sales returned: " + json.dumps(tool_totals)
             + '. Return {"totals": <the tool totals>, "best_region": <region with largest total>}.'},
        ]
        second = llm("final_answer", final_messages)
        run = {"mode": mode, "calls": [first, second], "tool_ms": tool_ms,
               "workflow_wall_ms": (time.perf_counter() - workflow_start) * 1000,
               "answer_correct": second["parsed"] == expected}
        result["runs"].append(run)
        save()
        assert run["answer_correct"], run
    time.sleep(0.5)
    generate(args.base, "sales_drain_events", token_ids(820000, 128))
    events = read_events(args.events, result["runs"][0]["calls"][0]["start_ns"], time.time_ns())
    for run in result["runs"]:
        for row in run["calls"]:
            row["reload_enqueued"] = any(e["event"] == "reload_enqueued"
                                         and row["start_ns"] <= e["time_ns"] <= row["end_ns"]
                                         for e in events)
    cold, warm, host = [run["calls"][0] for run in result["runs"]]
    result["checks"] = {
        "all_answers_correct": all(run["answer_correct"] for run in result["runs"]),
        "cold_first_call_uncached": cold["response"]["meta_info"]["cached_tokens"] == 0,
        "warm_first_call_prefix_reused": warm["response"]["meta_info"]["cached_tokens"] >= warm["input_tokens"] - 1,
        "host_first_call_reloaded": host["reload_enqueued"],
        "host_first_call_prefix_reused": host["response"]["meta_info"]["cached_tokens"] >= host["input_tokens"] - 1,
    }
    save()
    print(json.dumps(result, indent=2))
    assert all(result["checks"].values()), result["checks"]


if __name__ == "__main__":
    main()
