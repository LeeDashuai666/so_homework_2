"""Paired real DocVQA inference with a single cache-payload quota."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data/processed/docvqa"
BUDGET = 320
CONFIGS = {
    "vision": ("vision", None),
    "prefix": ("prefix", None),
    "cost-default": ("cost", dict(gap_scale=6, prefix_fraction=.55, min_ms_per_mib=.65)),
    "cost-local": ("cost", dict(gap_scale=14, prefix_fraction=.55, min_ms_per_mib=.65)),
    "cost-wide": ("cost", dict(gap_scale=20, prefix_fraction=.80, min_ms_per_mib=.60)),
}
CANDIDATES = list(CONFIGS)[2:]
SEEDS = {"dev": [6401], "eval": [6501, 6502]}
SPLITS = {"dev": "development", "eval": "evaluation"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def traces(split):
    pages = defaultdict(list)
    rows = [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]
    for row in rows:
        pages[row["page_id"]].append(row)
    order = sorted(pages, key=lambda page: hashlib.sha256(f"docvqa-{split}-{page}".encode()).digest())
    hot, cold, sources = [], [], set()
    for target, count, questions in [(hot, 4, 5), (cold, 12, 2)]:
        for page in order:
            source = pages[page][0]["source_document_id"]
            if source in sources or len(pages[page]) < questions:
                continue
            sources.add(source)
            target.append(pages[page][:questions])
            if len(target) == count:
                break
        assert len(target) == count
    mixed = []
    # Rotate short-gap revisits among four active pages, with a stream of
    # other pages whose second request returns later in the trace.
    for round_index in range(5):
        mixed.extend(page[round_index] for page in hot)
        if round_index < 4:
            mixed.extend(page[0] for page in cold[3 * round_index:3 * (round_index + 1)])
    mixed.extend(page[1] for page in cold)
    session = [row for page in hot + cold for row in page]
    assert {r["question_id"] for r in mixed} == {r["question_id"] for r in session}
    return {"mixed": mixed, "session": session}


def timed(runtime, request, **kwargs):
    result = runtime.generate(request, graph=True, **kwargs)
    result["persistent_payload_bytes"] = result.pop("cache_used")
    result.pop("prefix")
    return result


def measure_cost(runtime, pairs):
    samples = []
    for first, again in pairs:
        for mode in ("vision", "prefix"):
            runtime.reset_trace(mode)
            cold = timed(runtime, first, fixed_steps=1)
            warm = timed(runtime, again, fixed_steps=1)
            samples.append({"mode": mode, "cold": cold, "warm": warm})
    groups = {mode: [r for r in samples if r["mode"] == mode] for mode in ("vision", "prefix")}
    gain = statistics.median((v["warm"]["ttft_s"] - p["warm"]["ttft_s"]) * 1000
                             for v, p in zip(groups["vision"], groups["prefix"]))
    promotion = statistics.median((p["cold"]["ttft_s"] - v["cold"]["ttft_s"]) * 1000
                                  for v, p in zip(groups["vision"], groups["prefix"]))
    return {"gain_ms": max(gain, 0), "promotion_ms": max(promotion, 0), "samples": samples}


def hardware_checks(runtime, requests, output):
    import torch
    from torch.profiler import ProfilerActivity, profile
    from fused import check_arena

    checks = check_arena(runtime.arena)
    rows, profiles, paired = [], {}, {}
    for hardware in ("graph", "packed", "fused"):
        runtime.choose_hardware(hardware)
        runtime.reset_trace("prefix")
        for request in requests:
            timed(runtime, request, fixed_steps=9)
    for trial in range(8):
        order = ["graph", "packed", "fused"]
        random.Random(7200 + trial).shuffle(order)
        for hardware in order:
            runtime.choose_hardware(hardware)
            runtime.reset_trace("prefix")
            for request in requests:
                timed(runtime, request, fixed_steps=1)
                result = timed(runtime, request, fixed_steps=9)
                rows.append(dict(result, trial=trial, order=order))
                reference = paired.setdefault((trial, request.record["question_id"]), result["token_ids"])
                assert reference == result["token_ids"]
    for hardware in ("graph", "packed", "fused"):
        runtime.choose_hardware(hardware)
        runtime.reset_trace("prefix")
        request = requests[0]
        timed(runtime, request, fixed_steps=1)
        with torch.inference_mode():
            runtime._prefill(request, True)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(2):
                    runtime.graph.replay()
                torch.cuda.synchronize()
        name = output / f"trace-{hardware}.json"
        prof.export_chrome_trace(str(name))
        trace = read(name)
        kernels = Counter(event["name"] for event in trace["traceEvents"] if event.get("cat") == "kernel")
        api = Counter(event["name"] for event in trace["traceEvents"] if event.get("cat") == "cuda_runtime")
        profiles[hardware] = {"steps": 2, "gpu_kernels": sum(kernels.values()), "host_graph_launches": api["cudaGraphLaunch"],
                              "gpu_kernel_counts": kernels, "runtime_counts": api}
        with gzip.open(name.with_suffix(".json.gz"), "wb", compresslevel=6) as stream:
            stream.write(name.read_bytes())
        name.unlink()
    checks["paired_hardware_token_comparisons"] = len(rows) - len(paired)
    write(output / "hardware.json", {"records": rows, "traces": profiles, "checks": checks})


def run(stage, output):
    import torch
    import triton
    import transformers
    from engine import AdaptiveRuntime

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    protocol = read(output / "protocol.json")
    assert protocol == make_protocol()
    rows = traces(SPLITS[stage])
    assert set(protocol["sources"]["dev"]).isdisjoint(protocol["sources"]["eval"])
    runtime = AdaptiveRuntime(BUDGET)
    model_start = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.strip()
    built = {}
    calibration = read(output / "calibration.json") if stage == "eval" else None
    decision = read(output / "decision.json") if stage == "eval" else None
    if decision:
        assert decision["development_sha256"] == digest(output / "dev-requests.jsonl")
        assert decision["protocol_sha256"] == digest(output / "protocol.json")
    first = runtime.prepare(rows["mixed"][0])
    for hardware in (("fused",) if stage == "dev" else ("graph", "packed", "fused")):
        built[hardware] = runtime.build(first, hardware)
        print("graph ready", hardware, round(built[hardware], 2), flush=True)
    runtime.choose_hardware("fused")
    if stage == "dev":
        by_page = defaultdict(list)
        for row in rows["session"]:
            by_page[row["page_id"]].append(row)
        pairs = [(runtime.prepare(group[0]), runtime.prepare(group[1])) for group in list(by_page.values())[:8]]
        # Warm the multi-token variants before measuring the save/restore tradeoff.
        for pair in pairs[:2]:
            measure_cost(runtime, [pair])
        calibration = measure_cost(runtime, pairs)
        del pairs
        write(output / "calibration.json", calibration)
        print("cost calibration", {k: v for k, v in calibration.items() if k != "samples"}, flush=True)
    runtime.cost = {key: calibration[key] for key in ("gain_ms", "promotion_ms")}
    if stage == "eval":
        mechanism_requests = [runtime.prepare(row) for row in rows["mixed"][:3]]
        hardware_checks(runtime, mechanism_requests, output)
        del mechanism_requests
    names = list(CONFIGS) if stage == "dev" else list(dict.fromkeys([
        "vision", "prefix", "cost-default", decision["configuration"], decision["rule_configuration"]]))
    with (output / f"{stage}-requests.jsonl").open("x") as log:
        for seed in SEEDS[stage]:
            schedule = [(scene, config) for scene in rows for config in names]
            random.Random(seed).shuffle(schedule)
            for order, (scene, config) in enumerate(schedule):
                runtime.choose_hardware("fused")
                runtime.reset_trace(*CONFIGS[config])
                torch.cuda.reset_peak_memory_stats()
                for position, row in enumerate(rows[scene]):
                    request = runtime.prepare(row)
                    record = timed(runtime, request)
                    assert record["persistent_payload_bytes"] <= BUDGET * 2**20
                    record.update(seed=seed, scene=scene, config=config, order=order, position=position,
                                  boundary=request.boundary, answers=row["answers"],
                                  source_document_id=row["source_document_id"])
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(stage, seed, scene, config, "completed", flush=True)
    write(output / f"{stage}-environment.json", {"torch": str(torch.__version__),
          "torch_cuda": torch.version.cuda, "triton": triton.__version__,
          "transformers": transformers.__version__, "gpu_smi_after_model_load": model_start,
          "model_revision": read(ROOT / "configs/vlm-assets.lock.json")["model"]["revision"],
          "kernel": __import__("platform").release(), "graph_build_s": built})


def make_protocol():
    selected = {stage: traces(split) for stage, split in SPLITS.items()}
    return {
        "cache_budget_mib": BUDGET, "configs": {k: list(v) for k, v in CONFIGS.items()}, "seeds": SEEDS,
        "sources": {stage: sorted({r["source_document_id"] for r in sets["mixed"]})
                    for stage, sets in selected.items()},
        "page_pixels": {stage: sorted({r["pixel_sha256"] for r in sets["mixed"]})
                        for stage, sets in selected.items()},
        "ordered_questions": {stage: {scene: [r["question_id"] for r in rows] for scene, rows in sets.items()}
                              for stage, sets in selected.items()},
        "files_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                         [*(HERE / name for name in ("bench.py", "engine.py", "fused.py", "analyze.py", "select_ai.py")),
                          HERE.parent / "cross_layer_v2/runtime.py",
                          *(DATA / f"{split}.jsonl" for split in SPLITS.values()),
                          ROOT / "configs/protocol.lock.json", ROOT / "configs/vlm-assets.lock.json",
                          HERE.parent / "ai_cache_tuning/policy.py", HERE.parent / "ai_cache_tuning/policy-model.lock.json"]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "eval"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.stage == "dev":
        args.output.mkdir(parents=True)
        protocol = make_protocol()
        for key in ("sources", "page_pixels"):
            assert set(protocol[key]["dev"]).isdisjoint(protocol[key]["eval"])
        write(args.output / "protocol.json", protocol)
    run(args.stage, args.output)
