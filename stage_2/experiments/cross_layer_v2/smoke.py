#!/usr/bin/env python3
"""Real development-data smoke before registering any new formal experiment."""
from collections import defaultdict
import json
from pathlib import Path
import time
import traceback

import torch
from runtime import ROOT, Runtime

OUT = Path(__file__).resolve().parent / "results/smoke"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
data = [json.loads(line) for line in (ROOT / "data/processed/docvqa/development.jsonl").read_text().splitlines()]
pages = defaultdict(list)
for row in data:
    pages[row["page_id"]].append(row)
selected = [row for group in list(pages.values())[:3] for row in group[:2]]
results = []
try:
    engine = Runtime()
    first = engine.prepare(selected[0])
    engine.build_graph(first)
    print("Graph captured", engine.graph_build_s, flush=True)
    for item in selected:
        request = engine.prepare(item)
        native = engine.native(request)
        candidates = [engine.generate(request, prefix=p, graph=g) for p, g in [(False,False),(False,True),(True,False),(True,True)]]
        # A second question shares only the fixed image prefix, not question KV.
        results.append({"question_id": item["question_id"], "page_id": item["page_id"], "boundary": request.boundary,
                        "native": native, "candidates": candidates})
        print(item["question_id"], "native:", native["answer"],
              "candidates:", [(r["prefix"],r["graph"],r["answer"],round(r["generation_s"],4)) for r in candidates], flush=True)
        (OUT / "answers.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    (OUT / "capture.json").write_text(json.dumps({"graph_build_s":engine.graph_build_s,"success":True},indent=2)+"\n")
except Exception:
    (OUT / "error.txt").write_text(traceback.format_exc())
    raise
