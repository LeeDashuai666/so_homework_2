#!/usr/bin/env python3
"""Prepare the pinned model and SciFact data, reusing the Hugging Face cache."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from huggingface_hub import snapshot_download
import requests


ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--http", action="store_true", help="Use resumable HTTP instead of Xet")
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/assets.lock.json").read_text())
    model_spec = lock["model"]
    if args.http:
        if args.offline:
            parser.error("--http and --offline cannot be combined")
        subprocess.run([sys.executable, str(ROOT / "tools/download_model_http.py")], check=True)
    print(f"Model: {model_spec['repo_id']} @ {model_spec['revision']}", flush=True)
    snapshot = Path(snapshot_download(
        repo_id=model_spec["repo_id"], revision=model_spec["revision"],
        allow_patterns=list(model_spec["files"]), max_workers=2,
        local_files_only=args.offline,
    ))
    model_files = {}
    for name, expected in model_spec["files"].items():
        path = snapshot / name
        if path.stat().st_size != expected["size"]:
            raise RuntimeError(f"Model file size mismatch: {name}")
        sha256 = digest(path)
        if expected.get("sha256") and sha256 != expected["sha256"]:
            raise RuntimeError(f"Model file hash mismatch: {name}")
        model_files[name] = {"bytes": path.stat().st_size, "sha256": sha256}
    link = ROOT / "models/bge-reranker-base"
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != snapshot.resolve():
            raise RuntimeError(f"Existing model link points to another revision: {link}")
    elif link.exists():
        raise RuntimeError(f"Model path already exists and is not a managed link: {link}")
    else:
        link.symlink_to(snapshot, target_is_directory=True)
    print(f"Model verified; shared cache linked at {link}", flush=True)

    data_spec = lock["dataset"]
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "scifact.zip"
    if not archive.exists():
        if args.offline:
            raise FileNotFoundError(f"Offline dataset archive is missing: {archive}")
        print(f"Downloading dataset: {data_spec['url']}", flush=True)
        temporary = archive.with_suffix(".zip.part")
        with requests.get(data_spec["url"], stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
        if digest(temporary, "md5") != data_spec["archive_md5"]:
            raise RuntimeError("Dataset archive MD5 does not match the upstream release")
        os.replace(temporary, archive)
    if digest(archive, "md5") != data_spec["archive_md5"]:
        raise RuntimeError("Dataset archive MD5 does not match the upstream release")
    dataset = raw / "scifact"
    if not dataset.exists():
        with tempfile.TemporaryDirectory(prefix=".scifact-", dir=raw) as folder:
            target = Path(folder)
            with zipfile.ZipFile(archive) as zipped:
                for member in zipped.infolist():
                    if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                        raise RuntimeError("Unexpected path in dataset archive")
                zipped.extractall(target)
            os.replace(target / "scifact", dataset)
    required = ["corpus.jsonl", "queries.jsonl", "qrels/test.tsv", "qrels/train.tsv"]
    for name in required:
        if not (dataset / name).is_file():
            raise RuntimeError(f"Dataset file missing: {name}")
    data_files = {
        str(path.relative_to(dataset)): {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in sorted(dataset.rglob("*")) if path.is_file()
    }
    manifest_path = ROOT / "env/assets-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": {"repo_id": model_spec["repo_id"], "revision": model_spec["revision"],
                  "snapshot_path": str(snapshot), "project_path": str(link), "files": model_files},
        "dataset": {"name": "scifact", "source": data_spec["url"],
                    "archive_md5": data_spec["archive_md5"], "archive_sha256": digest(archive),
                    "path": str(dataset), "files": data_files},
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["model"]["files"] != model_files or previous["dataset"]["files"] != data_files:
            raise RuntimeError("Prepared asset contents differ from the saved manifest")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Dataset and model verified; manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
