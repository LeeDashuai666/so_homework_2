#!/usr/bin/env python3
"""CLI for actual inference runs; orchestration is added without changing the engine."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]


def source_hashes():
    paths = [*sorted((ROOT / "src").rglob("*.py")), ROOT / "tools/vlm_runtime.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def unit(args):
    import torch
    from vlmbench.engine import InferenceEngine
    from vlmbench.runner import make_trace, run, warmup

    if args.protocol:
        protocol = json.loads(args.protocol.read_text())
        if protocol["source_sha256"] != source_hashes():
            raise RuntimeError("Inference source differs from the frozen protocol")
        cfg, runtime = protocol["experiment"], protocol["runtime"]
    else:
        cfg = json.loads(args.config.read_text())
        runtime = json.loads((ROOT / cfg["runtime"]).read_text())
    if args.gpu_cache_mib is not None:
        cfg["gpu_cache_mib"] = args.gpu_cache_mib
    if args.cpu_cache_mib is not None:
        cfg["cpu_cache_mib"] = args.cpu_cache_mib
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    runtime["dataset_path"] = cfg["dataset_path"]
    torch.set_num_threads(runtime["cpu_threads"])
    torch.set_num_interop_threads(1)
    questions = [json.loads(row) for row in (ROOT / cfg["dataset_path"] / f"{args.split}.jsonl").read_text().splitlines()]
    if args.limit:
        questions = questions[:args.limit]
    development = [json.loads(row) for row in (ROOT / cfg["dataset_path"] / "development.jsonl").read_text().splitlines()]
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Preserve existing results and choose a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
                "variant": args.variant, "split": args.split, "seed": args.seed,
                "runtime": runtime, "experiment": cfg, "source_sha256": source_hashes(),
                "data_manifest_sha256": hashlib.sha256((ROOT / cfg["dataset_path"] / "manifest.json").read_bytes()).hexdigest(),
                "invocation": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest() if args.protocol else None}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    engine = InferenceEngine(runtime, cfg, args.variant)
    warm = warmup(engine, development, 2 if args.quick else cfg["main_warmup_requests"],
                  0 if args.quick else cfg["main_warmup_min_s"])
    (args.output / "warmup.json").write_text(json.dumps(warm, indent=2) + "\n")
    trace = [json.loads(row) for row in args.trace.read_text().splitlines()] if args.trace else make_trace(
        questions, args.order or cfg["trace_order"], args.seed,
        duration=args.duration if args.mode == "online" else None,
        rate=args.rate, count=args.count or len(questions))
    result = run(engine, questions, trace, args.output,
                 duration=args.duration if args.mode == "online" else None,
                 ttft_slo=args.ttft_slo, e2e_slo=args.e2e_slo)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("unit")
    run.add_argument("--variant", choices=["B0", "V1", "V1+S", "V1+M", "FULL"], required=True)
    run.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    run.add_argument("--protocol", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", choices=["development", "evaluation"], default="development")
    run.add_argument("--mode", choices=["offline", "online"], default="offline")
    run.add_argument("--duration", type=float, default=120)
    run.add_argument("--rate", type=float)
    run.add_argument("--count", type=int)
    run.add_argument("--limit", type=int)
    run.add_argument("--seed", type=int, default=1701)
    run.add_argument("--order", choices=["random", "session", "interleaved"])
    run.add_argument("--trace", type=Path)
    run.add_argument("--ttft-slo", type=float, default=1.0)
    run.add_argument("--e2e-slo", type=float, default=3.0)
    run.add_argument("--gpu-cache-mib", type=int)
    run.add_argument("--cpu-cache-mib", type=int)
    run.add_argument("--batch-size", type=int, choices=[1, 2])
    run.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.command == "unit":
        if args.mode == "online" and (args.rate is None or args.rate <= 0) and args.trace is None:
            parser.error("Online mode requires a positive rate or a frozen trace")
        unit(args)


if __name__ == "__main__":
    main()
