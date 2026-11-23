#!/usr/bin/env python3
"""Development-only quality gate + randomized paired timing + real CUDA trace.

The original frozen 558-question evaluation is never read by this pilot.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
from runtime import ROOT, Runtime
from vlmbench.metrics import quality, quantiles
from vlmbench.runner import question_order

HERE = Path(__file__).resolve().parent
VARIANTS = ["NATIVE", "VISUAL_HF", "STATIC_EAGER", "PREFIX", "GRAPH", "BOTH"]


def call(engine, request, variant):
    if variant in ("NATIVE", "VISUAL_HF"):
        return engine.native(request, visual_cache=variant == "VISUAL_HF")
    return engine.generate(request, prefix=variant in ("PREFIX", "BOTH"), graph=variant in ("GRAPH", "BOTH"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/pilot")
    parser.add_argument("--quality-count", type=int, default=282)
    parser.add_argument("--timing-pages", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-cache-mib", type=int, default=1536)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new result directory; old data are preserved")
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    data = [json.loads(line) for line in (ROOT / "data/processed/docvqa/development.jsonl").read_text().splitlines()]
    grouped = defaultdict(list)
    for row in data:
        grouped[row["page_id"]].append(row)
    # Timing uses the first 16 eligible pages in the original deterministic,
    # document-hash-selected development split, and every real later question.
    pages = [values for values in grouped.values() if len(values) >= 2][:args.timing_pages]
    timing_questions = [row for page in pages for row in page[1:]]
    manifest = {"recorded_at":datetime.now(timezone.utc).isoformat(), "scope":"DEVELOPMENT pilot, not formal held-out experiment",
                "read_split":"development only", "variants":VARIANTS,
                "quality_count":args.quality_count,"timing_pages":len(pages),"timing_questions":len(timing_questions),
                "timing_rule":"Hot pages: warm each page with its first different real question, measure remaining independent questions",
                "cache_quota_mib":args.gpu_cache_mib,"batch_size":1,"static_kv_capacity":1024,
                "max_new_tokens":64,"do_sample":False,"repeats":args.repeats,
                "quality_max_anls_drop":0.005,
                "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}}
    write(args.output / "manifest.json", manifest)
    engine = Runtime(gpu_cache_mib=args.gpu_cache_mib)
    first = engine.prepare(data[0])
    engine.build_graph(first)
    write(args.output / "graph-build.json", {"seconds":engine.graph_build_s,
        "scope":"Includes first full prefill, warmup, snapshot/restore and capture; charged once at startup"})
    print("CAPTURE OK",engine.graph_build_s,flush=True)
    quality_rows, quality_summary = [], {}
    order = question_order(data, "session", 2901)[:args.quality_count]
    for variant in VARIANTS:
        engine.store.clear()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        selected = []
        for index, item in enumerate(order):
            request = engine.prepare(item)
            result = call(engine,request,variant)
            result.update(variant=variant, **quality(result["answer"],item["answers"]))
            selected.append(result)
            if (index + 1) % 64 == 0:
                print("QUALITY",variant,index+1,"/",len(order),flush=True)
        quality_rows.extend(selected)
        quality_summary[variant] = {"questions":len(selected),"anls":statistics.mean(row["anls"] for row in selected),
                 "em":statistics.mean(row["em"] for row in selected),"sum_generation_s":sum(row["generation_s"] for row in selected),
                 "ttft_s":quantiles(row["ttft_s"] for row in selected),
                 "measured_total_s_including_common_preprocessing":time.perf_counter()-started,
                 "prefix_hits":sum(row.get("prefix_hit",False) for row in selected),
                 "peak_allocated_bytes":torch.cuda.max_memory_allocated()}
        (args.output / "quality-requests.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in quality_rows))
        write(args.output / "quality-summary.json",quality_summary)
        print("QUALITY DONE",variant,json.dumps(quality_summary[variant]),flush=True)
    reference = {row["question_id"]:row for row in quality_rows if row["variant"]=="NATIVE"}
    gate = {}
    for variant in VARIANTS:
        selected = [row for row in quality_rows if row["variant"]==variant]
        gate[variant] = {"anls_delta_vs_native":quality_summary[variant]["anls"]-quality_summary["NATIVE"]["anls"],
                         "identical_tokens_vs_native":sum(row["token_ids"]==reference[row["question_id"]]["token_ids"] for row in selected)}
    passed = all(row["anls_delta_vs_native"] >= -0.005 for row in gate.values())
    write(args.output / "quality-gate.json",{"passed":passed,"threshold":-0.005,"variants":gate})
    if not passed:
        raise RuntimeError("Development quality gate failed. Keep data and fix before adopting new directions.")
    # Paired blocks, fixed hot-page membership, randomized variant execution.
    timing_rows, timing_blocks = [], []
    for seed in range(2901,2901+args.repeats):
        variants = list(VARIANTS)
        random.Random(seed).shuffle(variants)
        requests_order = question_order(timing_questions,"random",seed)
        for variant in variants:
            engine.store.clear()
            for page in pages:
                call(engine,engine.prepare(page[0]),variant)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            selected = []
            for item in requests_order:
                result = call(engine,engine.prepare(item),variant)
                result.update(variant=variant,seed=seed)
                selected.append(result)
            timing_rows.extend(selected)
            block={"seed":seed,"variant":variant,"questions":len(selected),
                   "mean_generation_s":statistics.mean(row["generation_s"] for row in selected),
                   "mean_ttft_s":statistics.mean(row["ttft_s"] for row in selected),
                   "total_decode_s":sum(row["decode_s"] for row in selected),
                   "total_decode_steps":sum(row["decode_steps"] for row in selected),
                   "prefix_hit_rate":statistics.mean(row.get("prefix_hit",False) for row in selected),
                   "peak_allocated_bytes":torch.cuda.max_memory_allocated()}
            timing_blocks.append(block)
            (args.output / "timing-requests.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in timing_rows))
            write(args.output / "timing-blocks.json",timing_blocks)
            print("TIMING",json.dumps(block),flush=True)
    # Actual CUDA runtime calls, isolated decode-only work, identical step count.
    engine.store.clear()
    request=engine.prepare(data[0])
    profiler_summary={}
    with torch.inference_mode():
        for graph in [False,True]:
            engine._prefill(request,False)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(8):
                    engine.graph.replay() if graph else engine._decode()
                    int(engine.token.item())
            name="graph" if graph else "eager"
            prof.export_chrome_trace(str(args.output / f"decode-{name}-trace.json"))
            calls={}
            for event in prof.events():
                if "cudaGraphLaunch" in event.name or "cudaLaunchKernel" in event.name:
                    calls[event.name]=calls.get(event.name,0)+1
            profiler_summary[name]={"fixed_decode_steps":8,"runtime_launch_calls":calls}
    write(args.output / "launch-counts.json",profiler_summary)
    write(args.output / "complete.json",{"complete":True,"quality_gate_passed":True,"timing_blocks":len(timing_blocks)})
    print("DEVELOPMENT PILOT COMPLETE",flush=True)


if __name__=="__main__":
    main()
