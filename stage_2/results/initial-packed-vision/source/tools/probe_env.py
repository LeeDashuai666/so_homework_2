#!/usr/bin/env python3
"""Record the project environment; optionally check basic CUDA graph support."""

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": str(error)}


def package_versions():
    result = {}
    for name in (
        "torch", "transformers", "sentence-transformers", "datasets", "numpy",
        "pandas", "matplotlib", "safetensors", "onnxruntime-gpu", "z3-solver",
        "triton", "scipy", "scikit-learn", "sentencepiece", "rank-bm25", "fastapi", "uvicorn",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def cuda_probe():
    try:
        import torch

        result = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        }
        if not result["cuda_available"]:
            return result
        device = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        result.update(
            device_name=device.name,
            compute_capability=[device.major, device.minor],
            total_memory_bytes=total_bytes,
            free_memory_bytes_at_probe=free_bytes,
        )
        static_input = torch.ones(16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = static_input + 1
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = static_input + 1
        graph.replay()
        torch.cuda.synchronize()
        result["cuda_graph_smoke_passed"] = bool(
            torch.equal(output, torch.full((16,), 2.0, device="cuda"))
        )
        static_input.fill_(3)
        graph.replay()
        torch.cuda.synchronize()
        result["cuda_graph_updated_input_passed"] = bool(
            torch.equal(output, torch.full((16,), 4.0, device="cuda"))
        )
        result["scope"] = "Tensor addition only; reranker capture and performance are not tested."
        return result
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    smi = shutil.which("nvidia-smi")
    if smi is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        smi = "/usr/lib/wsl/lib/nvidia-smi"
    result = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(Path.cwd()),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "os_release": command(["cat", "/etc/os-release"]),
        "cpu": command(["lscpu"]),
        "memory": command(["free", "-b"]),
        "disk": command(["df", "-hT", "."]),
        "perf_event_paranoid": command(["cat", "/proc/sys/kernel/perf_event_paranoid"]),
        "packages": package_versions(),
        "gpu": command([
            smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"
        ]) if smi else {"error": "nvidia-smi not found"},
        "cuda": cuda_probe() if args.cuda_smoke else {"checked": False},
    }
    content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content, end="")


if __name__ == "__main__":
    main()
