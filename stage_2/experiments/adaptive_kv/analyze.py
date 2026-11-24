"""Recompute adaptive-cache metrics from the saved requests and CUDA trace."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from bench import CANDIDATES, ROOT, digest, read, write

sys.path.insert(0, str(ROOT / "src"))
from vlmbench.metrics import quality


def summarize(path, seeds):
    buckets = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        buckets[row["scene"], row["config"]].append(row)
    per_scene = defaultdict(dict)
    for (scene, config), rows in buckets.items():
        controls = {(r["seed"], r["question_id"]): r for r in buckets[scene, "vision"]}
        assert len(rows) == len(controls) and set(r["seed"] for r in rows) == set(seeds)
        assert len({(r["seed"], r["question_id"]) for r in rows}) == len(rows)
        stat = {"n": len(rows), "anls": statistics.mean(quality(r["answer"], r["answers"])["anls"] for r in rows)}
        for key, value in [("ttft", "ttft_s"), ("request", "generation_s")]:
            times = [r[value] * 1000 for r in rows]
            stat[f"{key}_mean_ms"] = statistics.mean(times)
            stat[f"{key}_p95_ms"] = float(np.percentile(times, 95))
            stat[f"{key}_by_block_ms"] = [statistics.mean(r[value] * 1000 for r in rows if r["seed"] == seed)
                                          for seed in seeds]
        stat["token_equal"] = sum(r["token_ids"] == controls[r["seed"], r["question_id"]]["token_ids"] for r in rows)
        for key in ("vision_hit", "prefix_hit", "encoder_calls", "promotion"):
            label = key + "s" if key.endswith(("hit", "promotion")) else key
            stat[f"{label}_per_trace"] = sum(r[key] for r in rows) / len(seeds)
        stat["evictions_per_trace"] = sum(max(r["evictions"] for r in rows if r["seed"] == seed)
                                           for seed in seeds) / len(seeds)
        stat["peak_payload_mib"] = max(r["persistent_payload_bytes"] for r in rows) / 2**20
        stat["peak_allocated_gib"] = max(r["peak_allocated_bytes"] for r in rows) / 2**30
        per_scene[scene][config] = stat
    return dict(per_scene)


def summarize_hardware(output):
    raw = read(output / "hardware.json")
    grouped = defaultdict(list)
    for row in raw["records"]:
        assert row["decode_steps"] == 8 and row["prefix_hit"]
        grouped[row["hardware"]].append(row)
    summary = {}
    for hardware, rows in grouped.items():
        trace = raw["traces"][hardware]
        with gzip.open(output / f"trace-{hardware}.json.gz", "rt") as handle:
            events = json.load(handle)["traceEvents"]
        counts = Counter(event["name"] for event in events if event.get("cat") == "kernel")
        assert dict(counts) == trace["gpu_kernel_counts"]
        assert sum(counts.values()) == trace["gpu_kernels"]
        # Individual kernel durations are collected inside the profiled graph.
        append_events = [event for event in events if event.get("cat") == "kernel" and (
            "_append" in event["name"] if hardware == "fused" else "index_copy" in event["name"])]
        summary[hardware] = {"n": len(rows), "decode_per_token_ms": statistics.mean(r["decode_s"] * 1000 / 8 for r in rows),
                             "ttft_mean_ms": statistics.mean(r["ttft_s"] * 1000 for r in rows),
                             "request_mean_ms": statistics.mean(r["generation_s"] * 1000 for r in rows),
                             "decode_trial_means_ms": {trial: statistics.mean(r["decode_s"] * 1000 / 8
                                for r in rows if r["trial"] == trial) for trial in range(8)},
                             "traced_gpu_kernels_per_step": trace["gpu_kernels"] / trace["steps"],
                             "traced_graph_submissions": trace["host_graph_launches"],
                             "traced_append_or_index_copy_events_per_step": len(append_events) / trace["steps"]}
    return summary


def validate(output, stage, protocol, expected_configs):
    schedules = protocol["ordered_questions"][stage]
    groups = defaultdict(list)
    for line in (output / f"{stage}-requests.jsonl").read_text().splitlines():
        row = json.loads(line)
        groups[row["scene"], row["seed"], row["config"]].append(row)
    assert set(groups) == {(scene, seed, cfg) for scene in schedules for seed in protocol["seeds"][stage]
                           for cfg in expected_configs}
    for (scene, seed, config), rows in groups.items():
        assert [r["question_id"] for r in rows] == schedules[scene]
        assert all(r["hardware"] == "fused" and r["fixed_steps"] is None for r in rows)
        assert all(r["persistent_payload_bytes"] <= protocol["cache_budget_mib"] * 2**20 for r in rows)


def main(output):
    protocol = read(output / "protocol.json")
    decision = read(output / "decision.json")
    assert digest(output / "dev-requests.jsonl") == decision["development_sha256"]
    assert digest(output / "protocol.json") == decision["protocol_sha256"]
    assert decision["configuration"] in CANDIDATES and decision["rule_configuration"] in CANDIDATES
    assert decision["choice"] == max(decision["raw_choice_logits"], key=decision["raw_choice_logits"].get)
    validate(output, "dev", protocol, protocol["configs"])
    validate(output, "eval", protocol, {"vision", "prefix", "cost-default",
                                          decision["configuration"], decision["rule_configuration"]})
    dev = summarize(output / "dev-requests.jsonl", protocol["seeds"]["dev"])
    eval_ = summarize(output / "eval-requests.jsonl", protocol["seeds"]["eval"])
    hardware = summarize_hardware(output)
    report = {"development": dev, "evaluation": eval_, "hardware": hardware,
              "ai_choice": decision["configuration"], "rule_choice": decision["rule_configuration"],
              "ai_decision_s": decision["decision_s"], "ai_total_s": decision["model_and_decision_s"],
              "calibration": {k: v for k, v in read(output / "calibration.json").items() if k != "samples"},
              "hardware_checks": read(output / "hardware.json")["checks"],
              "measurement_records": sum(g["n"] for scene in [*dev.values(), *eval_.values()] for g in scene.values())}
    write(output / "summary.json", report)
    print(json.dumps({"ai": report["ai_choice"], "rule": report["rule_choice"],
                      "request_records": report["measurement_records"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
