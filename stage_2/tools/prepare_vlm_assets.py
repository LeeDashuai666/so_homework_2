#!/usr/bin/env python3
"""Prepare the pinned Qwen3-VL model and a verified real DocVQA smoke sample."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from huggingface_hub import snapshot_download

from prepare_docvqa_smoke import digest, prepare


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model-only", action="store_true", help="Prepare only pinned model files; existing experiment images need no viewer sample")
    args = parser.parse_args()
    if args.offline and args.http:
        parser.error("--offline and --http cannot be combined")
    lock_path = ROOT / "configs/vlm-assets.lock.json"
    lock = json.loads(lock_path.read_text())
    model = lock["model"]
    if args.http:
        subprocess.run([sys.executable, str(ROOT / "tools/download_model_http.py"),
                        "--lock", str(lock_path), "--workers", str(args.workers)], check=True)
    snapshot = Path(snapshot_download(model["repo_id"], revision=model["revision"],
                                     allow_patterns=list(model["files"]),
                                     local_files_only=args.offline or args.http, max_workers=2))
    files = {}
    for name, expected in model["files"].items():
        path = snapshot / name
        if path.stat().st_size != expected["size"]:
            raise RuntimeError(f"VLM model file size mismatch: {name}")
        checksum = digest(path)
        if expected.get("sha256") and checksum != expected["sha256"]:
            raise RuntimeError(f"VLM model file checksum mismatch: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": checksum}
    target = ROOT / model["project_link"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != snapshot.resolve():
            raise RuntimeError("The project model link points to another revision")
    elif target.exists():
        raise RuntimeError("The project model target is not a managed symbolic link")
    else:
        target.symlink_to(snapshot, target_is_directory=True)
    if args.model_only:
        manifest_path = ROOT / "env/vlm-model-manifest.json"
        manifest_path.write_text(json.dumps({
            "recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
            "lock_sha256": digest(lock_path),
            "model": {"repo_id": model["repo_id"], "revision": model["revision"],
                      "snapshot": str(snapshot), "project_link": str(target), "files": files},
            "scope": "Model-only preparation; no dataset viewer dependency for frozen experiments"},
            ensure_ascii=False, indent=2) + "\n")
        print(f"Pinned model verified: {manifest_path}", flush=True)
        return
    dataset = prepare(offline=args.offline)
    manifest_path = ROOT / "env/vlm-assets-manifest.json"
    manifest = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "lock_sha256": digest(lock_path),
        "model": {"repo_id": model["repo_id"], "revision": model["revision"],
                  "snapshot": str(snapshot), "project_link": str(target), "files": files},
        "dataset": dataset,
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["model"]["files"] != files or old["dataset"]["files"] != dataset["files"]:
            raise RuntimeError("Prepared VLM assets differ from the previous manifest")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"VLM assets verified: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
