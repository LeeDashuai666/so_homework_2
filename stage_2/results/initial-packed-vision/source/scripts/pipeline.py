#!/usr/bin/env python3
"""Resumable, sequential single-GPU development and pre-registered experiments."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "scripts")]
from experiment import source_hashes


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch(name, arguments, protocol=None):
    output = ROOT / "results" / name
    if (output / "summary.json").is_file():
        manifest, summary = read(output / "manifest.json"), read(output / "summary.json")
        expected = sha(protocol) if protocol else None
        if manifest["source_sha256"] != source_hashes() or manifest["protocol_sha256"] != expected:
            raise RuntimeError(f"Refusing to reuse stale output {output}; preserve it under another run name")
        if summary["errors"] or summary["failed"]:
            raise RuntimeError(f"Failed run requires inspection: {output}")
        print("RESUME", name, flush=True)
        return summary
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Incomplete run: preserve {output} before a retry")
    command = [sys.executable, "-u", str(ROOT / "scripts/experiment.py"), "unit",
               "--output", str(output), *map(str, arguments)]
    if protocol:
        command += ["--protocol", str(protocol)]
    log = ROOT / "logs" / (name.replace("/", "--") + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print("START", name, datetime.now(timezone.utc).isoformat(), flush=True)
    with log.open("w") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
    if completed.returncode:
        raise RuntimeError(f"Run {name} exited {completed.returncode}; inspect {log}")
    summary = read(output / "summary.json")
    if summary["failed"] or summary["errors"]:
        raise RuntimeError(f"Invalid run {name}: failed requests")
    print("DONE", name, "qps", round(summary["completed_qps"], 4),
          "goodput", round(summary["goodput_qps"], 4), "ANLS", round(summary["anls"], 5), flush=True)
    return summary


def calibrate():
    """Only development data are read before choosing any protocol values."""
    cfg = read(ROOT / "configs/experiment.json")
    diagnostics = ROOT / "results/diagnostics"
    if not (diagnostics / "profile-hit.json").is_file():
        with (ROOT / "logs/diagnostics.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", str(ROOT / "scripts/diagnose.py")],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
    isolated = read(diagnostics / "calibration.json")
    candidates = []
    for batch in [1, 2]:
        name = f"development/native-b{batch}"
        result = launch(name, ["--variant", "B0", "--batch-size", batch, "--count", 192,
                               "--seed", 1700, "--order", "random"])
        candidates.append({"name": name, "variant": "B0", "batch": batch, "gpu_mib": 0, "summary": result})
    for capacity in [128, 256, 512]:
        name = f"development/lru-{capacity}"
        result = launch(name, ["--variant", "V1", "--gpu-cache-mib", capacity,
                               "--batch-size", 2, "--count", 192, "--seed", 1700, "--order", "random"])
        candidates.append({"name": name, "variant": "V1", "batch": 2, "gpu_mib": capacity, "summary": result})
    # One common batch limit: choose the fastest native baseline. Cache capacity
    # is tuned on the LRU baseline, never chosen to make the optimized policy win.
    native = max((row for row in candidates if row["variant"] == "B0"), key=lambda row: row["summary"]["completed_qps"])
    cfg["batch_size"] = native["batch"]
    if native["batch"] != 2:
        raise RuntimeError("Development selected batch=1; add matching batch=1 cache calibration before freezing")
    caches = [row for row in candidates if row["variant"] == "V1"]
    fastest = max(row["summary"]["completed_qps"] for row in caches)
    selected = min((row for row in caches if row["summary"]["completed_qps"] >= fastest * 0.99), key=lambda row: row["gpu_mib"])
    cfg["gpu_cache_mib"] = selected["gpu_mib"]
    cfg["scheduler_vision_cost_per_token_s"] = isolated["vision_s_per_token"]
    cfg["scheduler_prefill_cost_per_token_s"] = isolated["prefill_s_per_token"]
    cfg["note"] = "Frozen from development only; all variants share input, GPU/CPU limits and per-batch KV/workspace guard."
    write(ROOT / "configs/experiment-calibrated.json", cfg)
    dev = {}
    for variant in ["B0", "FULL"]:
        dev[variant] = launch(f"development/quality-{variant}", ["--variant", variant,
                        "--config", ROOT / "configs/experiment-calibrated.json", "--count", 282,
                        "--seed", 1700, "--order", "random"])
    if dev["B0"]["anls"] - dev["FULL"]["anls"] > cfg["quality_max_anls_drop"]:
        raise RuntimeError("Development quality gate failed; fix implementation before formal evaluation")
    runtime = read(ROOT / cfg["runtime"])
    runtime["dataset_path"] = cfg["dataset_path"]
    capacity = selected["summary"]["completed_qps"]
    protocol = {
        "schema_version": 1, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Real DocVQA questions; synthetic Poisson arrival times; local validation split, not official hidden test",
        "experiment": cfg, "runtime": runtime,
        "source_sha256": source_hashes(),
        "data_manifest_sha256": sha(ROOT / cfg["dataset_path"] / "manifest.json"),
        "asset_lock_sha256": sha(ROOT / "configs/experiment-assets.lock.json"),
        "environment_sha256": sha(ROOT / "env/vlm-environment.json"),
        "calibration": {"candidates": candidates, "selected_lru_run": selected["name"],
                        "native_common_batch": native["name"], "development_quality": dev,
                        "isolated": isolated, "reference_capacity_qps": capacity},
        "slo": {"ttft_s": math.ceil(isolated["ttft_s"]["p95"] * 2 * 100) / 100,
                "e2e_s": math.ceil(isolated["latency_s"]["p95"] * 2 * 100) / 100,
                "rule": "2x native isolated development p95, rounded upward to 10 ms"},
        "rates_qps": {str(factor): capacity * factor for factor in cfg["rate_multipliers"]},
        "quality_rule": {"max_absolute_anls_drop": cfg["quality_max_anls_drop"], "all_558_evaluation_questions": True,
                         "normalization": "NFKC + Unicode casefold + collapse whitespace; local ANLS convention"},
        "statistics": "Three paired independent process runs per load; mean and paired seed intervals; raw per-run p95 retained; no stable p99 claim",
        "auxiliary": {"budgets_mib": [128, 256, 512], "orders": ["session", "interleaved", "random"],
                      "cold_pages": 128, "offline_count": 192, "budget_scope": "Same upper GPU cache cap for LRU and O3; unchanged model/input"},
    }
    destination = ROOT / "configs/protocol.lock.json"
    if destination.exists() and read(destination)["source_sha256"] != source_hashes():
        raise RuntimeError("An older frozen protocol exists; preserve it explicitly before replacing")
    write(destination, protocol)
    # Trace files are frozen before launching any evaluation model process.
    from vlmbench.runner import make_trace
    evaluation = [json.loads(row) for row in (ROOT / cfg["dataset_path"] / "evaluation.jsonl").read_text().splitlines()]
    trace_hashes = {}
    for multiplier, rate in protocol["rates_qps"].items():
        for seed in cfg["seeds"]:
            path = ROOT / "data/traces" / f"load-{multiplier}-seed-{seed}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = make_trace(evaluation, cfg["trace_order"], seed, duration=cfg["main_window_s"], rate=rate)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            trace_hashes[str(path.relative_to(ROOT))] = sha(path)
    write(ROOT / "data/traces/manifest.json", {"protocol_sha256": sha(destination), "traces": trace_hashes})
    print("PROTOCOL FROZEN", destination, "SLO", protocol["slo"], flush=True)


def formal():
    protocol_path = ROOT / "configs/protocol.lock.json"
    protocol = read(protocol_path)
    cfg, slo = protocol["experiment"], protocol["slo"]
    if protocol["source_sha256"] != source_hashes():
        raise RuntimeError("Frozen inference source changed")
    for variant in cfg["variants"]:
        launch(f"quality/{variant}", ["--variant", variant, "--split", "evaluation", "--count", 558,
                    "--order", "random", "--seed", 1701], protocol_path)
    baseline = read(ROOT / "results/quality/B0/summary.json")
    gate = {variant: read(ROOT / "results/quality" / variant / "summary.json")["anls"] - baseline["anls"] for variant in cfg["variants"]}
    write(ROOT / "results/quality/gate.json", {"anls_change_vs_B0": gate, "threshold": -cfg["quality_max_anls_drop"],
                                                "passed": all(delta >= -cfg["quality_max_anls_drop"] for delta in gate.values())})
    if any(delta < -cfg["quality_max_anls_drop"] for delta in gate.values()):
        raise RuntimeError("Evaluation quality gate failed; preserve results and inspect correctness")
    blocks = [(factor, seed) for factor in protocol["rates_qps"] for seed in cfg["seeds"]]
    random.Random(1700).shuffle(blocks)
    schedule = []
    for factor, seed in blocks:
        variants = list(cfg["variants"])
        random.Random(seed + int(float(factor) * 100)).shuffle(variants)
        schedule.extend({"factor": factor, "seed": seed, "variant": variant} for variant in variants)
    write(ROOT / "results/main/schedule.json", {"protocol_sha256": sha(protocol_path), "units": schedule})
    for index, item in enumerate(schedule):
        factor, seed, variant = item["factor"], item["seed"], item["variant"]
        trace = ROOT / "data/traces" / f"load-{factor}-seed-{seed}.jsonl"
        expected = read(ROOT / "data/traces/manifest.json")["traces"][str(trace.relative_to(ROOT))]
        assert sha(trace) == expected
        launch(f"main/load-{factor}-seed-{seed}/{variant}", ["--variant", variant,
               "--split", "evaluation", "--mode", "online", "--duration", cfg["main_window_s"],
               "--trace", trace, "--seed", seed, "--ttft-slo", slo["ttft_s"], "--e2e-slo", slo["e2e_s"]], protocol_path)
        write(ROOT / "results/progress.json", {"phase": "formal", "completed": index + 1, "total": len(schedule),
                                               "last": item, "updated_at": datetime.now(timezone.utc).isoformat()})
    print("FORMAL MATRIX COMPLETE", flush=True)


def auxiliary():
    protocol_path = ROOT / "configs/protocol.lock.json"
    cfg = read(protocol_path)["experiment"]
    for budget in [128, 256, 512]:
        for variant in ["V1", "V1+M"]:
            launch(f"auxiliary/budget-{budget}/{variant}", ["--variant", variant, "--split", "evaluation",
                   "--gpu-cache-mib", budget, "--count", 192, "--seed", 1701, "--order", "random"], protocol_path)
    for order in ["session", "interleaved"]:
        for variant in ["B0", "V1", "FULL"]:
            launch(f"auxiliary/order-{order}/{variant}", ["--variant", variant, "--split", "evaluation",
                    "--count", 192, "--seed", 1701, "--order", order], protocol_path)
    # Exactly one real question from each evaluation page: no possible reuse.
    data = [json.loads(row) for row in (ROOT / cfg["dataset_path"] / "evaluation.jsonl").read_text().splitlines()]
    first = {}
    for row in data:
        first.setdefault(row["page_id"], row["question_id"])
    cold = ROOT / "data/traces/cold-128.jsonl"
    cold.write_text("".join(json.dumps({"request_id": f"r{i:06d}", "offset_s": 0.0, "question_id": q}) + "\n"
                            for i, q in enumerate(first.values())))
    for variant in ["B0", "V1", "FULL"]:
        launch(f"auxiliary/cold/{variant}", ["--variant", variant, "--split", "evaluation", "--trace", cold], protocol_path)
    print("AUXILIARY COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["calibrate", "formal", "auxiliary", "full"])
    parser.add_argument("--refreeze", action="store_true", help="Refresh development-only registration before any evaluation run exists")
    args = parser.parse_args()
    if args.refreeze and (list((ROOT / "results/quality").glob("*/manifest.json")) or list((ROOT / "results/main").glob("load-*/*/manifest.json"))):
        parser.error("Cannot change a registration after evaluation has begun")
    if args.command in ("calibrate", "full") and (args.refreeze or not (ROOT / "configs/protocol.lock.json").exists()):
        calibrate()
    if args.command in ("formal", "full"):
        formal()
    if args.command in ("auxiliary", "full"):
        auxiliary()


if __name__ == "__main__":
    main()
