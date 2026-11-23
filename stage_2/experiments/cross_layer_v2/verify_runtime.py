#!/usr/bin/env python3
"""GPU state-isolation and graph-address checks; synthetic length case is not a benchmark."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from runtime import ROOT, Runtime, SharedLRU

HERE = Path(__file__).resolve().parent


def addresses(engine):
    return [t.data_ptr() for layer in engine.work.layers for t in (layer.keys, layer.values)] + [
        engine.token.data_ptr(), engine.delta.data_ptr(), engine.columns.data_ptr()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results/runtime-check.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; preserve earlier checks")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows = [json.loads(line) for line in (ROOT / "data/processed/docvqa/development.jsonl").read_text().splitlines()]
    pages = defaultdict(list)
    for row in rows:
        pages[row["page_id"]].append(row)
    first, second = [group for group in pages.values() if len(group) >= 2][:2]
    engine = Runtime()
    a, b, other = [engine.prepare(row) for row in (first[0], first[1], second[0])]
    assert a.key == b.key and a.key != other.key
    assert a.boundary == b.boundary
    assert not torch.equal(a.inputs["input_ids"][:, a.boundary:], b.inputs["input_ids"][:, b.boundary:])
    engine.build_graph(a)
    pointers = addresses(engine)
    with torch.inference_mode():
        engine.store.clear()
        assert not engine.generate(a, prefix=True)["prefix_hit"]
        entry = engine.store.get(("prefix", a.key))
        reference = tuple((k.clone(), v.clone()) for k, v in entry)
        for layer, (key, value) in zip(engine.work.layers, entry):
            assert layer.keys.untyped_storage().data_ptr() != key.untyped_storage().data_ptr()
            assert layer.values.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
        checks = []
        for request in (b, other, a):
            eager = engine.generate(request)
            graph = engine.generate(request, graph=True)
            prefix = engine.generate(request, prefix=True)
            both = engine.generate(request, prefix=True, graph=True)
            assert eager["token_ids"] == graph["token_ids"]
            assert prefix["token_ids"] == both["token_ids"]
            assert both["prefix_hit"]
            assert addresses(engine) == pointers
            assert all(torch.equal(k, kr) and torch.equal(v, vr)
                       for (k, v), (kr, vr) in zip(entry, reference))
            checks.append({"question_id":request.record["question_id"],
                           "prompt_tokens":request.inputs["input_ids"].shape[1],
                           "graph_matches_eager":True,"both_matches_prefix":True,
                           "persistent_prefix_unchanged":True,"working_addresses_unchanged":True})
        # Find the longest valid suffix with the real page; used only to exercise
        # static-capacity indexing. No synthetic question enters quality/timing data.
        low, high, longest = 0, 400, b
        while low <= high:
            middle = (low + high) // 2
            row = dict(first[1], question=first[1]["question"] + "\n" + "data " * middle)
            try:
                candidate = engine.prepare(row)
            except AssertionError:
                high = middle - 1
                continue
            longest = candidate
            low = middle + 1
        length = longest.inputs["input_ids"].shape[1]
        assert length >= engine.max_length - engine.max_new_tokens - 2
        assert longest.key == a.key
        eager = engine.generate(longest, prefix=True, fixed_steps=64)
        graph = engine.generate(longest, prefix=True, graph=True, fixed_steps=64)
        assert eager["token_ids"] == graph["token_ids"]
        assert len(graph["token_ids"]) == 64
        assert all(int(layer.cumulative_length.item()) == length + 63 for layer in engine.work.layers)
        assert addresses(engine) == pointers
        assert all(torch.equal(k, kr) and torch.equal(v, vr)
                   for (k, v), (kr, vr) in zip(entry, reference))
        # Admit either real prefix, but force eviction when the other page arrives.
        payload_a = engine.store.get(("prefix", a.key))
        payload_other = engine.store.get(("prefix", other.key))
        size = lambda payload: sum(t.numel()*t.element_size() for pair in payload for t in pair)
        quota = max(size(payload_a), size(payload_other)) + 1024
        engine.store = SharedLRU(quota)
        miss1 = engine.generate(a, prefix=True)
        miss2 = engine.generate(other, prefix=True)
        assert ("prefix", a.key) not in engine.store.entries
        miss3 = engine.generate(a, prefix=True, graph=True)
        assert not any(row["prefix_hit"] for row in (miss1, miss2, miss3))
        assert engine.store.used <= quota
        assert addresses(engine) == pointers
        torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    result = {"passed":True,"recorded_at":datetime.now(timezone.utc).isoformat(),
              "scope":"Development GPU invariants; synthetic long suffix is only a correctness check",
              "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [HERE / "runtime.py", Path(__file__)]},
              "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
              "shared_prefix_key_for_different_questions":True,"different_image_misses":True,
              "working_kv_disjoint_from_persistent_prefix":True,"cases":checks,
              "capacity_edge":{"prompt_tokens":length,"output_tokens":64,"final_cache_length":length+63,
                               "capacity":engine.max_length,"graph_tokens_equal":True},
              "eviction":{"quota_bytes":quota,"remaining_bytes":engine.store.used,"readmission_missed":True},
              "memory":{"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                         "final_driver_free_bytes":free,"driver_total_bytes":total}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
