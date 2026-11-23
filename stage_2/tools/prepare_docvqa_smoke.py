#!/usr/bin/env python3
"""Prepare a small, pinned real DocVQA sample without fetching the full corpus."""

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from PIL import Image
import requests


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(folder, manifest):
    for name, entry in manifest["files"].items():
        path = folder / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise RuntimeError(f"DocVQA sample checksum mismatch: {name}")
    return manifest


def prepare(offline=False, row_count=32, page_count=3):
    spec = json.loads((ROOT / "configs/vlm-assets.lock.json").read_text())["dataset"]
    folder = ROOT / "data/raw/docvqa-smoke"
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["revision"] != spec["revision"]:
            raise RuntimeError("DocVQA sample belongs to another locked revision")
        verify(folder, manifest)
        print(f"Reused verified DocVQA sample: {manifest['question_count']} questions", flush=True)
        return manifest
    if offline:
        raise FileNotFoundError("DocVQA smoke sample is not prepared")
    folder.mkdir(parents=True, exist_ok=True)
    images = folder / "images"
    images.mkdir(exist_ok=True)
    print("Fetching real DocVQA rows for a small environment sample ...", flush=True)
    response = requests.get(
        "https://datasets-server.huggingface.co/rows",
        params={"dataset": spec["repo_id"], "config": spec["config"], "split": spec["split"],
                "offset": 0, "length": row_count}, timeout=(20, 45),
    )
    response.raise_for_status()
    payload = response.json()
    groups = OrderedDict()
    for item in payload["rows"]:
        row = item["row"]
        source = row["image"]["src"]
        if f"/--/{spec['revision']}/--/" not in source:
            raise RuntimeError("Dataset viewer revision changed; use the locked Parquet source instead")
        groups.setdefault(str(row["docId"]), []).append((item["row_idx"], row))
    selected = [(key, rows[:3]) for key, rows in groups.items() if len(rows) >= 2][:page_count]
    if len(selected) < 2:
        raise RuntimeError("The fetched sample does not contain two pages with multiple real questions")
    questions, image_metadata = [], {}
    for page_id, rows in selected:
        relative = f"images/page-{page_id}.jpg"
        target = folder / relative
        source = rows[0][1]["image"]["src"]
        if not target.exists():
            temp = target.with_suffix(".part")
            with requests.get(source, stream=True, timeout=(20, 60)) as download:
                download.raise_for_status()
                with temp.open("wb") as output:
                    for block in download.iter_content(1024 * 1024):
                        output.write(block)
            with Image.open(temp) as image:
                image.load()
            os.replace(temp, target)
        with Image.open(target) as image:
            image.load()
            image_metadata[page_id] = {"path": relative, "width": image.width, "height": image.height,
                                       "mode": image.mode, "format": image.format}
        for row_index, row in rows:
            questions.append({
                "question_id": str(row["questionId"]), "question": row["question"],
                "answers": row["answers"], "page_id": page_id,
                "source_document_id": row["ucsf_document_id"],
                "source_page_number": row["ucsf_document_page_no"],
                "image_path": relative, "source_row_index": row_index,
            })
    question_path = folder / "questions.jsonl"
    question_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions))
    manifest = {
        "repo_id": spec["repo_id"], "revision": spec["revision"], "config": spec["config"],
        "split": spec["split"], "source_method": "HF dataset viewer cached image export",
        "scope": "small real-image environment smoke sample; not the formal evaluation split",
        "official_primary_source": spec["primary_source"],
        "viewer_total_rows": payload.get("num_rows_total"), "requested_first_rows": row_count,
        "page_count": len(selected), "question_count": len(questions), "images": image_metadata,
        "files": {str(path.relative_to(folder)): {"bytes": path.stat().st_size, "sha256": digest(path)}
                  for path in [question_path, *(folder / image["path"] for image in image_metadata.values())]},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    verify(folder, manifest)
    print(f"DocVQA sample ready: {len(selected)} pages, {len(questions)} real questions", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    prepare(offline=args.offline)


if __name__ == "__main__":
    main()
