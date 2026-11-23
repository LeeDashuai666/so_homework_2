#!/usr/bin/env python3
"""Download pinned model files over resumable HTTP when Xet is unreliable."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import constants, hf_hub_download
import requests


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download_file(model, name, expected, workers):
    repo_type = model.get("repo_type", "model")
    cache_prefix = "datasets--" if repo_type == "dataset" else "models--"
    repo = Path(constants.HF_HUB_CACHE) / (cache_prefix + model["repo_id"].replace("/", "--"))
    blobs = repo / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    destination = blobs / expected["sha256"]
    size = expected["size"]
    if destination.exists():
        if destination.stat().st_size != size or sha256(destination) != expected["sha256"]:
            raise RuntimeError(f"Cached file checksum mismatch: {name}")
    else:
        partial = destination.with_suffix(".stage2.part")
        state_path = destination.with_suffix(".stage2.progress.json")
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "prefix_bytes": partial.stat().st_size if partial.exists() else 0, "done": []
        }
        prefix = state["prefix_bytes"]
        if prefix > size:
            raise RuntimeError(f"Unexpected partial file size: {name}")
        descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(descriptor, size)
        chunk = 8 * 1024 * 1024
        ranges = [(start, min(size - 1, start + chunk - 1)) for start in range(prefix, size, chunk)]
        done = set(state["done"])
        lock = threading.Lock()
        def save_state():
            state["done"] = sorted(done)
            temp = state_path.with_suffix(".tmp")
            temp.write_text(json.dumps(state))
            os.replace(temp, state_path)
        save_state()
        print(f"HTTP {name}: preserving {prefix / 2**20:.1f} MiB; {len(ranges)} ranges", flush=True)

        def fetch(index, start, end):
            position = start
            for attempt in range(8):
                try:
                    hub_prefix = "datasets/" if repo_type == "dataset" else ""
                    url = (f"https://huggingface.co/{hub_prefix}{model['repo_id']}/resolve/{model['revision']}/{name}"
                           f"?download=true&stage2_range={position}-{end}")
                    with requests.get(url, headers={"Range": f"bytes={position}-{end}"},
                                      stream=True, timeout=(20, 45)) as response:
                        response.raise_for_status()
                        match = re.match(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                        if response.status_code != 206 or not match or int(match[1]) != position or int(match[3]) != size:
                            raise RuntimeError("HTTP server did not honor the requested range")
                        for block in response.iter_content(256 * 1024):
                            if position + len(block) > end + 1:
                                raise RuntimeError("HTTP response exceeded the requested range")
                            written = os.pwrite(descriptor, block, position)
                            if written != len(block):
                                raise RuntimeError("Short local write")
                            position += written
                    if position == end + 1:
                        with lock:
                            done.add(index)
                            save_state()
                            complete = prefix + sum(ranges[i][1] - ranges[i][0] + 1 for i in done)
                            print(f"HTTP {name}: {complete / 2**20:.1f}/{size / 2**20:.1f} MiB", flush=True)
                        return
                except Exception as error:
                    print(f"HTTP {name}: range {index} retry {attempt + 1} ({type(error).__name__})", flush=True)
                time.sleep(min(2 ** attempt, 10))
            raise RuntimeError(f"HTTP retries exhausted for {name}, range {index}")

        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [pool.submit(fetch, i, start, end) for i, (start, end) in enumerate(ranges) if i not in done]
                for job in as_completed(jobs):
                    job.result()
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if sha256(partial) != expected["sha256"]:
            raise RuntimeError(f"Downloaded file checksum mismatch: {name}; partial retained for inspection")
        os.replace(partial, destination)
        state_path.unlink(missing_ok=True)
    target = repo / "snapshots" / model["revision"] / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(os.path.relpath(destination, target.parent))
    print(f"Verified: {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/assets.lock.json")
    parser.add_argument("--section", default="model")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be between 1 and 16")
    model = json.loads(args.lock.read_text())[args.section]
    with ThreadPoolExecutor(max_workers=2 if model.get("repo_type") == "dataset" else 4) as pool:
        jobs = []
        for name, item in model["files"].items():
            if item.get("sha256"):
                jobs.append(pool.submit(download_file, model, name, item,
                                        args.workers if name.endswith(".safetensors") else 2))
            else:
                jobs.append(pool.submit(hf_hub_download, model["repo_id"], name,
                                        revision=model["revision"], repo_type=model.get("repo_type", "model")))
        for job in as_completed(jobs):
            job.result()


if __name__ == "__main__":
    main()
