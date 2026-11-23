#!/usr/bin/env python3
"""Check environment capabilities with local assets; this is not a benchmark."""

import argparse
from datetime import datetime
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import torch


ROOT = Path(__file__).resolve().parents[1]
PAIRS = [
    ("What is a giant panda?", "The giant panda is a bear native to China."),
    ("What is a giant panda?", "A computer executes instructions stored in memory."),
]


def dependencies():
    packages = {
        "torch": "torch", "transformers": "transformers", "triton": "triton",
        "sentence-transformers": "sentence_transformers", "datasets": "datasets",
        "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
        "scipy": "scipy", "scikit-learn": "sklearn", "sentencepiece": "sentencepiece",
        "fastapi": "fastapi", "uvicorn": "uvicorn", "rank-bm25": "rank_bm25",
        "z3-solver": "z3", "safetensors": "safetensors", "psutil": "psutil",
    }
    result = {}
    for package, module in packages.items():
        imported = importlib.import_module(module)
        result[package] = {"version": importlib.metadata.version(package), "path": imported.__file__}
    return result


def cuda_transfer():
    assert torch.cuda.is_available(), "CUDA unavailable"
    host = torch.arange(256, dtype=torch.float32, pin_memory=True)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gpu = host.to("cuda", non_blocking=True)
        output = gpu * 2
    stream.synchronize()
    torch.testing.assert_close(output.cpu(), host * 2)
    left = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    assert torch.isfinite(left @ left).all().item()
    return {"device": torch.cuda.get_device_name(0), "runtime": torch.version.cuda,
            "pinned_host_memory": host.is_pinned(), "async_transfer": True,
            "compute_capability": list(torch.cuda.get_device_capability(0))}


def compiler():
    import torch._inductor.config

    torch._inductor.config.compile_threads = 1
    def operation(value):
        return torch.sin(value) * 2 + torch.relu(value)
    data = torch.randn(256, device="cuda")
    expected = operation(data)
    compiled = torch.compile(operation, backend="inductor", mode="reduce-overhead", fullgraph=True)
    actual = compiled(data)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(compiled(data + 1), operation(data + 1))
    return {"backend": "inductor", "mode": "reduce-overhead", "scope": "small GPU tensor function"}


def profiler():
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    data = torch.randn(64, 64, device="cuda")
    with torch.profiler.profile(activities=activities) as profile:
        with torch.profiler.record_function("stage2_environment_smoke"):
            result = data @ data
            torch.cuda.synchronize()
    trace = ROOT / "env/profiler-smoke.json"
    profile.export_chrome_trace(str(trace))
    gpu_events = sum("CUDA" in str(event.device_type) for event in profile.events())
    assert gpu_events > 0, "No CUDA kernel events captured"
    return {"trace": str(trace), "cuda_events": gpu_events, "finite": bool(torch.isfinite(result).all())}


def cpu_perf():
    events = ["cycles:u", "instructions:u", "branches:u", "branch-misses:u"]
    result = subprocess.run(
        [str(ROOT / "tools/perf.sh"), "stat", "-x", ",", "-e", ",".join(events),
         "--", sys.executable, "-c", "sum(range(1000000))"],
        capture_output=True, text=True, timeout=20,
    )
    (ROOT / "env/perf-smoke.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    counts = {}
    for line in result.stderr.splitlines():
        columns = line.split(",")
        if len(columns) >= 3 and columns[2] in events:
            counts[columns[2]] = int(columns[0])
    assert all(counts.get(event, 0) > 0 for event in events), counts
    return {"counts": counts, "scope": "interface smoke only, not a workload benchmark"}


def dataset():
    from rank_bm25 import BM25Okapi

    folder = ROOT / "data/raw/scifact"
    corpus = [json.loads(line) for line in (folder / "corpus.jsonl").read_text().splitlines()]
    queries = {q["_id"]: q["text"] for q in map(json.loads, (folder / "queries.jsonl").read_text().splitlines())}
    rows = (folder / "qrels/test.tsv").read_text().splitlines()[1:]
    qrels = [line.split("\t") for line in rows]
    corpus_ids = {d["_id"] for d in corpus}
    assert all(q in queries and d in corpus_ids for q, d, _ in qrels)
    bm25 = BM25Okapi([(d["title"] + " " + d["text"]).lower().split() for d in corpus[:100]])
    scores = bm25.get_scores(queries[qrels[0][0]].lower().split())
    assert len(scores) == 100
    return {"corpus_documents": len(corpus), "test_queries": len({q[0] for q in qrels}),
            "qrels_valid": True, "bm25_smoke_documents": len(scores)}


def http_and_solver():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import z3

    app = FastAPI()
    @app.get("/health")
    def health():
        return {"ok": True}
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200 and response.json() == {"ok": True}
    value = z3.Int("environment_smoke_value")
    solver = z3.Solver()
    solver.add(value > 0, value < 0)
    assert solver.check() == z3.unsat
    return {"http_testclient": "passed", "z3": z3.get_version_string(),
            "solver_smoke": "unsat", "scope": "tool availability, not project formal verification"}


@torch.inference_mode()
def model_graphs():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = ROOT / "models/bge-reranker-base"
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path), local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
    ).eval().to("cuda")
    checks, keepalive = [], []
    for length in (64, 128):
        batch = tokenizer(PAIRS, padding="max_length", truncation=True, max_length=length,
                          return_tensors="pt").to("cuda")
        static = {name: value.clone() for name, value in batch.items()}
        expected = model(**batch).logits.flatten()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model(**static)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = model(**static).logits.flatten()
        graph.replay()
        torch.cuda.synchronize()
        error = float((output - expected).abs().max())
        torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-3)
        for name in static:
            static[name].copy_(batch[name].flip(0))
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected.flip(0), atol=1e-2, rtol=1e-3)
        assert torch.isfinite(output).all().item()
        checks.append({"batch_size": 2, "sequence_length": length, "max_abs_error": error,
                       "updated_input_replays": 3, "scores": expected.cpu().tolist()})
        keepalive.append((graph, static, output))
    result = {"model": "BAAI/bge-reranker-base", "dtype": "float16",
              "attention_backend": model.config._attn_implementation, "graphs": checks,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "scope": "correctness and capability smoke; no performance claim"}
    return result


@torch.inference_mode()
def sdk():
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(str(ROOT / "models/bge-reranker-base"), device="cuda",
                         local_files_only=True, max_length=512,
                         model_kwargs={"dtype": "float16", "attn_implementation": "sdpa"})
    scores = model.predict(PAIRS, batch_size=2, show_progress_bar=False,
                           activation_fn=torch.nn.Identity(), convert_to_tensor=True)
    assert scores.numel() == 2 and torch.isfinite(scores).all().item()
    return {"scores": scores.cpu().tolist(), "pair_count": 2, "scope": "standard SDK baseline available"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--checks", help="Comma-separated check names; default: all checks")
    parser.add_argument("--output", type=Path, default=ROOT / "env/smoke-latest.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checks = [("dependencies", dependencies), ("cuda_and_transfers", cuda_transfer)]
    if not args.skip_compile:
        checks.append(("torch_compile", compiler))
    checks.extend([("cuda_profiler", profiler), ("cpu_perf", cpu_perf), ("dataset", dataset),
                   ("http_and_solver", http_and_solver), ("model_cuda_graphs", model_graphs),
                   ("crossencoder", sdk)])
    if args.checks:
        selected = set(args.checks.split(","))
        unknown = selected - {name for name, _ in checks}
        if unknown:
            parser.error("Unknown checks: " + ", ".join(sorted(unknown)))
        checks = [(name, function) for name, function in checks if name in selected]
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
              "scope": "environment smoke only", "checks": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, function in checks:
        print(f"Checking {name} ...", flush=True)
        start = time.monotonic()
        try:
            details = function()
            entry = {"passed": True, "details": details}
        except Exception as error:
            entry = {"passed": False, "error": f"{type(error).__name__}: {error}",
                     "traceback": traceback.format_exc()}
        entry["elapsed_seconds"] = time.monotonic() - start
        report["checks"][name] = entry
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {'PASS' if entry['passed'] else 'FAIL'}: {name}", flush=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["passed"] = all(value["passed"] for value in report["checks"].values())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {args.output}", flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
