#!/usr/bin/env python3
"""Export a document-disjoint, pinned DocVQA evaluation workload."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq
from PIL import Image
from huggingface_hub import constants

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from prepare_docvqa_smoke import digest
from vlm_runtime import load_processor, prepare_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-pages", type=int, default=64)
    parser.add_argument("--eval-pages", type=int, default=128)
    args = parser.parse_args()
    lock = json.loads((ROOT / "configs/experiment-assets.lock.json").read_text())
    spec = lock["dataset"]
    source = Path(constants.HF_HUB_CACHE) / ("datasets--" + spec["repo_id"].replace("/", "--")) / "snapshots" / spec["revision"]
    files = [source / name for name in sorted(spec["files"])]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("The six pinned validation Parquet shards must be prepared first")
    output = ROOT / "data/processed/docvqa"
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    metadata = []
    columns = ["questionId", "question", "answers", "docId", "ucsf_document_id", "ucsf_document_page_no"]
    for path in files:
        metadata.extend(pq.read_table(path, columns=columns).to_pylist())
    assert len({str(row["questionId"]) for row in metadata}) == len(metadata)
    page_rows = defaultdict(list)
    for row in metadata:
        page_rows[str(row["docId"])].append(row)
    source_pages = defaultdict(set)
    for page, rows in page_rows.items():
        source_pages[str(rows[0]["ucsf_document_id"])].add(page)
    rank = lambda text: hashlib.sha256(("docvqa-system-v1\0" + str(text)).encode()).hexdigest()
    sources = sorted(source_pages, key=rank)
    development, evaluation, dev_sources = [], [], set()
    for doc in sources:
        pages = sorted(source_pages[doc], key=rank)
        if len(development) < args.dev_pages:
            development.extend(pages[:args.dev_pages - len(development)])
            dev_sources.add(doc)
        elif len(evaluation) < args.eval_pages:
            evaluation.extend(pages[:args.eval_pages - len(evaluation)])
    assert len(development) == args.dev_pages and len(evaluation) == args.eval_pages
    selected = set(development + evaluation)
    images = {}
    print(f"Exporting {len(selected)} pages from {len(metadata)} actual questions ...", flush=True)
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=["docId", "image"]):
            for row in batch.to_pylist():
                page = str(row["docId"])
                if page not in selected or page in images:
                    continue
                value = row["image"]
                data = value["bytes"]
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    rgb = image.convert("RGB")
                    pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                    extension = ".png" if image.format == "PNG" else ".jpg"
                    relative = f"images/page-{page}{extension}"
                    (output / relative).write_bytes(data)
                    images[page] = {"path": relative, "width": rgb.width, "height": rgb.height,
                                    "pixel_sha256": pixel_hash, "sha256": hashlib.sha256(data).hexdigest(),
                                    "bytes": len(data), "source_document_id": str(page_rows[page][0]["ucsf_document_id"])}
    assert set(images) == selected
    dev_hashes = {images[page]["pixel_sha256"] for page in development}
    duplicates = [page for page in evaluation if images[page]["pixel_sha256"] in dev_hashes]
    evaluation = [page for page in evaluation if page not in duplicates]
    assert not ({images[p]["source_document_id"] for p in development} & {images[p]["source_document_id"] for p in evaluation})
    cfg = json.loads((ROOT / "configs/vlm-runtime.json").read_text())
    cfg["dataset_path"] = "data/processed/docvqa"
    processor = load_processor(cfg)
    exported = {}
    for split, pages in [("development", development), ("evaluation", evaluation)]:
        questions = []
        for page in pages:
            for row in sorted(page_rows[page], key=lambda x: str(x["questionId"])):
                item = {"question_id": str(row["questionId"]), "page_id": page,
                        "question": row["question"], "answers": row["answers"],
                        "image_path": images[page]["path"], "pixel_sha256": images[page]["pixel_sha256"],
                        "source_document_id": str(row["ucsf_document_id"]),
                        "source_page_number": str(row["ucsf_document_page_no"]), "split": split}
                inputs, detail = prepare_inputs(processor, [item], cfg)
                item["visual_tokens"] = detail["visual_tokens"][0]
                item["input_tokens"] = detail["input_lengths"][0]
                item["grid_thw"] = detail["grid_thw"][0]
                item["feature_bytes_estimate"] = item["visual_tokens"] * 2048 * 2 * 4
                questions.append(item)
                del inputs
        path = output / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions))
        exported[split] = {"pages": len(pages), "questions": len(questions),
                           "source_documents": len({images[p]["source_document_id"] for p in pages}),
                           "question_counts_per_page": dict(Counter(Counter(q["page_id"] for q in questions).values())),
                           "visual_tokens_min": min(q["visual_tokens"] for q in questions),
                           "visual_tokens_max": max(q["visual_tokens"] for q in questions),
                           "file": str(path.relative_to(ROOT)), "sha256": digest(path)}
        print(split, exported[split], flush=True)
    manifest = {"recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
                "source": spec, "source_rows": len(metadata), "source_unique_pages": len(page_rows),
                "scope": "document-disjoint local split of official DocVQA validation; not official hidden test",
                "split_rule": "SHA256(docvqa-system-v1 NUL source_document_id), then page ID hash",
                "cross_split_duplicate_pages_excluded": duplicates,
                "source_document_overlap": 0, "pixel_hash_overlap": 0,
                "splits": exported, "images": images, "runtime_config": cfg,
                "source_file_sha256": {str(p.relative_to(source)): digest(p) for p in files}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print("Prepared reproducible workload:", output, flush=True)


if __name__ == "__main__":
    main()
