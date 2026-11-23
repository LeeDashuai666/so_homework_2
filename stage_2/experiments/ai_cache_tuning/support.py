"""Shared paths and deterministic selection of development pages."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMPARE = HERE.parent / "framework_compare"
DEFAULT_MODEL = Path("/home/lxy18/infer-lab/models/Qwen3-0.6B")
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def jsonl(path):
    return list(map(json.loads, path.read_text().splitlines()))


def hash_order(value):
    return hashlib.sha256(f"ai-cache-v1:{value}".encode()).digest()


def select(rows, panels):
    pages = defaultdict(list)
    for row in rows:
        pages[row["page_id"]].append(row)
    used_docs, chosen = set(), []
    for page in sorted(pages, key=hash_order):
        group = pages[page]
        source = group[0]["source_document_id"]
        if len(group) >= 3 and source not in used_docs:
            used_docs.add(source)
            chosen.append(sorted(group, key=lambda r: hash_order(r["question_id"]))[:3])
        if len(chosen) == 8 * panels:
            break
    assert len(chosen) == 8 * panels
    workloads = []
    for index in range(panels):
        panel = chosen[index * 8:(index + 1) * 8]
        order = list(range(8))
        random.Random(5221 + index).shuffle(order)
        workloads.append({
            "id": index,
            "warmup": [str(panel[i][0]["question_id"]) for i in order],
            "measured": [str(panel[i][round_]["question_id"]) for round_ in [1, 2] for i in order],
            "source_documents": sorted(page[0]["source_document_id"] for page in panel),
            "page_visual_tokens": [panel[i][0]["visual_tokens"] for i in order],
        })
    return workloads


