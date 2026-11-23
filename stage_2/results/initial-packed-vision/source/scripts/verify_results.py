#!/usr/bin/env python3
"""Independently recompute raw result scores, SLOs, ledgers and paired traces."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "scripts")]
from experiment import source_hashes
from vlmbench.metrics import quality, summarize


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(require_full=True):
    protocol_path = ROOT / "configs/protocol.lock.json"
    protocol = read(protocol_path)
    cfg, slo = protocol["experiment"], protocol["slo"]
    assert source_hashes() == protocol["source_sha256"], "Frozen inference sources changed"
    data_path = ROOT / cfg["dataset_path"]
    assert sha(data_path / "manifest.json") == protocol["data_manifest_sha256"]
    assert sha(ROOT / "configs/experiment-assets.lock.json") == protocol["asset_lock_sha256"]
    data_manifest = read(data_path / "manifest.json")
    data = {name: rows(data_path / f"{name}.jsonl") for name in ["development", "evaluation"]}
    assert not ({q["source_document_id"] for q in data["development"]} & {q["source_document_id"] for q in data["evaluation"]})
    assert not ({q["pixel_sha256"] for q in data["development"]} & {q["pixel_sha256"] for q in data["evaluation"]})
    for name in data:
        assert sha(data_path / f"{name}.jsonl") == data_manifest["splits"][name]["sha256"]
    for image in data_manifest["images"].values():
        path = data_path / image["path"]
        assert sha(path) == image["sha256"]
        with Image.open(path) as value:
            rgb = value.convert("RGB")
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == image["pixel_sha256"]
            assert (rgb.width, rgb.height) == (image["width"], image["height"])
    summary_files = sorted((ROOT / "results/main").glob("load-*/*/summary.json"))
    if require_full:
        assert len(summary_files) == 45, f"Expected 45 formal runs, found {len(summary_files)}"
    checked_requests, checked_batches, maximum_sender_lag = 0, 0, 0.0
    for path in summary_files:
        folder = path.parent
        summary, manifest = read(path), read(folder / "manifest.json")
        assert manifest["protocol_sha256"] == sha(protocol_path)
        assert manifest["source_sha256"] == protocol["source_sha256"]
        assert manifest["experiment"] == cfg
        assert manifest["runtime"] == protocol["runtime"]
        trace, requests, batches = rows(folder / "trace.jsonl"), rows(folder / "requests.jsonl"), rows(folder / "batches.jsonl")
        assert Counter(q["request_id"] for q in trace) == Counter(q["request_id"] for q in requests)
        assert all(count == 1 for count in Counter(q["request_id"] for q in requests).values())
        native_trace = ROOT / "data/traces" / (folder.parent.name + ".jsonl")
        assert sha(folder / "trace.jsonl") == sha(native_trace), "Paired arrival traces differ"
        lookup = {q["question_id"]: q for q in data["evaluation"]}
        for row in requests:
            if row["status"] != "completed":
                continue
            reference = lookup[row["question_id"]]
            assert row["page_id"] == reference["page_id"]
            expected = quality(row["answer"], reference["answers"])
            assert abs(row["anls"] - expected["anls"]) < 1e-12 and row["em"] == expected["em"]
            assert abs(row["ttft_s"] - row["first_token"] + row["scheduled"]) < 1e-8
            assert abs(row["latency_s"] - row["completed"] + row["scheduled"]) < 1e-8
            assert row["scheduled"] <= row["received"] + 1e-6 <= row["prep_start"] + 1e-6 <= row["ready"] + 1e-6 <= row["started"] + 1e-6 <= row["first_token"] + 1e-6 <= row["completed"] + 1e-6
            assert row["input_tokens"] == reference["input_tokens"] and row["visual_tokens"] == reference["visual_tokens"]
            maximum_sender_lag = max(maximum_sender_lag, row["sender_lag_s"])
        recalculated = summarize(requests, cfg["main_window_s"], slo["ttft_s"], slo["e2e_s"])
        for key in ["arrivals", "completed", "rejected", "failed", "good_requests", "goodput_qps", "slo_attainment", "anls", "em"]:
            assert abs(recalculated[key] - summary[key]) < 1e-10, (path, key)
        assert not summary["errors"] and not summary["failed"]
        completed = [row["request_id"] for row in requests if row["status"] == "completed"]
        assert Counter(completed) == Counter(request for batch in batches for request in batch["request_ids"])
        for batch in batches:
            assert batch["real_batch_size"] <= cfg["batch_size"]
            assert sum(batch["visual_tokens"]) <= cfg["max_batch_visual_tokens"]
            cache = batch["cache"]
            if cache:
                assert cache["gpu_payload_bytes"] <= cache["gpu_limit"] <= cfg["gpu_cache_mib"] * 1024**2
                assert cache["cpu_payload_bytes"] <= cache["cpu_limit"] <= cfg["cpu_cache_mib"] * 1024**2
                assert cache["active_leases"] == 0
            else:
                assert batch["vision_images"] == batch["real_batch_size"]
        checked_requests += len(requests)
        checked_batches += len(batches)
    qualities = {}
    baseline = None
    baseline_tokens = None
    for variant in cfg["variants"]:
        folder = ROOT / "results/quality" / variant
        if not (folder / "summary.json").is_file():
            if require_full:
                raise AssertionError(f"Missing full quality run {variant}")
            continue
        requests = rows(folder / "requests.jsonl")
        manifest = read(folder / "manifest.json")
        assert manifest["source_sha256"] == protocol["source_sha256"]
        assert manifest["protocol_sha256"] == sha(protocol_path)
        assert len(requests) == len(data["evaluation"]) == 558
        assert {r["question_id"] for r in requests} == {q["question_id"] for q in data["evaluation"]}
        assert all(r["status"] == "completed" for r in requests)
        values = {q["question_id"]: q for q in data["evaluation"]}
        for row in requests:
            assert row["anls"] == quality(row["answer"], values[row["question_id"]]["answers"])["anls"]
        anls = sum(r["anls"] for r in requests) / len(requests)
        token_map = {r["question_id"]: r["token_ids"] for r in requests}
        if baseline is None:
            baseline, baseline_tokens = anls, token_map
        delta = anls - baseline
        assert delta >= -cfg["quality_max_anls_drop"], (variant, delta)
        qualities[variant] = {"questions": len(requests), "anls": anls, "em": sum(r["em"] for r in requests) / len(requests),
                              "anls_delta_vs_B0": delta,
                              "identical_token_sequences_vs_B0": sum(tokens == baseline_tokens[q] for q, tokens in token_map.items()),
                              "output_tokens": sum(r["output_tokens"] for r in requests),
                              "truncated": sum(not r["eos_seen"] for r in requests)}
    auxiliary_files = sorted((ROOT / "results/auxiliary").glob("*/*/summary.json"))
    if require_full:
        assert len(auxiliary_files) == 15, f"Expected 15 auxiliary runs, found {len(auxiliary_files)}"
    auxiliary_traces = {}
    for path in auxiliary_files:
        folder = path.parent
        group = folder.parent.name
        summary, manifest = read(path), read(folder / "manifest.json")
        assert manifest["source_sha256"] == protocol["source_sha256"] and manifest["protocol_sha256"] == sha(protocol_path)
        assert not summary["failed"] and not summary["errors"] and not summary["rejected"]
        expected_count = 128 if group == "cold" else protocol["auxiliary"]["offline_count"]
        assert summary["completed"] == summary["arrivals"] == expected_count
        trace_hash = sha(folder / "trace.jsonl")
        if group in auxiliary_traces:
            assert auxiliary_traces[group] == trace_hash
        auxiliary_traces[group] = trace_hash
        requests, batches = rows(folder / "requests.jsonl"), rows(folder / "batches.jsonl")
        assert Counter(r["request_id"] for r in rows(folder / "trace.jsonl")) == Counter(r["request_id"] for r in requests)
        lookup = {q["question_id"]: q for q in data["evaluation"]}
        for row in requests:
            assert row["anls"] == quality(row["answer"], lookup[row["question_id"]]["answers"])["anls"]
        for batch in batches:
            cache = batch["cache"]
            if cache:
                assert cache["gpu_payload_bytes"] <= cache["gpu_limit"] <= manifest["experiment"]["gpu_cache_mib"] * 1024**2
                assert cache["cpu_payload_bytes"] <= cache["cpu_limit"] and cache["active_leases"] == 0
        if group == "cold":
            assert len({r["page_id"] for r in requests}) == 128 and summary["vision_images_encoded"] == 128
    result = {"passed": True, "formal_runs": len(summary_files), "checked_requests": checked_requests,
              "auxiliary_runs": len(auxiliary_files),
              "checked_batches": checked_batches, "checked_original_images": len(data_manifest["images"]),
              "source_document_overlap": 0, "pixel_hash_overlap": 0,
              "maximum_completed_request_sender_lag_s": maximum_sender_lag,
              "quality": qualities, "protocol_sha256": sha(protocol_path),
              "checks": ["frozen source and data hashes", "original file and RGB pixel integrity", "source document split isolation",
                         "paired trace equality", "complete unique request ledger", "timestamps and live token timing",
                         "reference answer rescoring", "fixed-window SLO recomputation", "per-batch memory and lease bounds"]}
    destination = ROOT / "results/verification/results-integrity.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(not args.partial), indent=2))
