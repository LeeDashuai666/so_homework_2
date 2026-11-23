#!/usr/bin/env python3
"""Run a local AI decision and score it on the separately archived vLLM traces."""
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
import time

from support import COMPARE, DEFAULT_MODEL, HERE, MODEL_REVISION, ROOT, digest, jsonl, read, select, write

ARCHIVE = COMPARE / "results"
CHOICES = {1024: "V11", 16384: "V11E"}
SEEDS = (4101, 4102, 4103)
LIMIT = 6.75
MAX_ANLS_DROP = 0.005
MODEL_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json")


def observation():
    data = ROOT / "data/processed/docvqa/development.jsonl"
    trial = ARCHIVE / "development/V11/quality-requests.jsonl"
    rows = jsonl(data)
    pages = {r["page_id"]: r for r in rows}
    # The synthetic access pattern uses real DocVQA page sizes and changes only the access order.
    panels = select(rows, 2)
    token_sizes = [tokens for panel in panels for tokens in panel["page_visual_tokens"]]
    measured = jsonl(trial)
    assert len(measured) == len(rows) == 282
    assert {r["page_id"] for r in measured} == set(pages)
    seen = set()
    revisits = []
    for record in measured:
        page = record["page_id"]
        if page in seen:
            revisits.append(record)
        seen.add(page)
    return {
        "page_visual_tokens": token_sizes,
        "active_working_set_visual_tokens": sum(token_sizes),
        "baseline": {
            "capacity": 1024, "development_access_order": "questions grouped by page",
            "questions": len(measured), "total_encoder_calls": sum(r["encoder_calls"] for r in measured),
            "page_revisits": len(revisits), "revisit_encoder_calls": sum(r["encoder_calls"] for r in revisits),
            "mean_revisit_ttft_ms": round(1000 * mean(r["ttft_s"] for r in revisits), 3),
            "peak_allocated_gib": round(max(r["peak_allocated_bytes"] for r in measured) / 1024**3, 4),
        },
        "target_access_order": "sixteen warm pages, interleaved revisits",
        "peak_allocated_limit_gib": LIMIT, "candidates": list(CHOICES),
    }


def evaluate(decision):
    dataset = jsonl(COMPARE / "data/evaluation.jsonl")
    by_id = {str(row["question_id"]): row for row in dataset}
    dev = jsonl(ROOT / "data/processed/docvqa/development.jsonl")
    assert not ({r["source_document_id"] for r in dev} & {r["source_document_id"] for r in dataset})
    dev_ids = {str(row["question_id"]) for row in dev}
    sys.path.insert(0, str(ROOT / "src"))
    from vlmbench.metrics import quality
    # Include the other measured configuration as a post-selection comparison.
    options = {"fixed": 1024, "ai": decision["capacity"], "reference_16384": 16384}
    parts = []
    evidence = {}
    for seed in SEEDS:
        fields = {}
        traces = {}
        for label, capacity in options.items():
            folder = ARCHIVE / f"formal/seed-{seed}-{CHOICES[capacity]}"
            manifest = read(folder / "manifest.json")
            env = manifest["environment"]
            assert env["resolved_encoder_cache_tokens"] == capacity
            assert env["engine_args"]["enable_prefix_caching"] is True
            assert env["engine_args"]["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
            assert env["engine_core_client"] == "InprocClient"
            path = folder / "hot-requests.jsonl"
            traces[label] = records = jsonl(path)
            assert len(records) == 72 and all(r["seed"] == seed and r["scenario"] == "hot" for r in records)
            identifiers = [r["question_id"] for r in records]
            assert not dev_ids.intersection(identifiers)
            assert len(set(identifiers)) == len(identifiers)
            peak = max(r["peak_allocated_bytes"] for r in records) / 1024**3
            anls = mean(quality(r["answer"], by_id[r["question_id"]]["answers"])["anls"] for r in records)
            assert abs(anls - mean(r["anls"] for r in records)) < 1e-9
            fields[label] = {
                "capacity": capacity, "requests": len(records),
                "ttft_ms": 1000 * mean(r["ttft_s"] for r in records),
                "e2e_ms": 1000 * mean(r["e2e_s"] for r in records),
                "encoder_calls": sum(r["encoder_calls"] for r in records),
                "allocated_gib": peak, "anls": anls,
            }
            evidence[str(path.relative_to(ROOT))] = digest(path)
        pairs = list(zip(traces["fixed"], traces["ai"], strict=True))
        for reference in (traces["ai"], traces["reference_16384"]):
            assert all(f["question_id"] == a["question_id"] and f["prompt_sha256"] == a["prompt_sha256"]
                       for f, a in zip(traces["fixed"], reference, strict=True))
        quality_pass = fields["ai"]["anls"] >= fields["fixed"]["anls"] - MAX_ANLS_DROP
        memory_pass = max(fields[role]["allocated_gib"] for role in options) <= LIMIT
        parts.append({"seed": seed, **fields, "token_matches": sum(f["token_ids"] == a["token_ids"] for f, a in pairs),
                      "quality_pass": quality_pass, "memory_pass": memory_pass})
    measures = ("ttft_ms", "e2e_ms", "encoder_calls", "anls")
    aggregate = {role: {"capacity": capacity, **{key: mean(p[role][key] for p in parts) for key in measures},
                        "peak_allocated_gib": max(p[role]["allocated_gib"] for p in parts)}
                 for role, capacity in options.items()}
    return {
        "method": "Offline AI capacity selection, evaluated using existing measured vLLM GPU traces",
        "new_gpu_runs": 0, "development_questions": 282,
        "heldout_requests_per_arm": sum(p["fixed"]["requests"] for p in parts),
        "unique_heldout_questions": parts[0]["fixed"]["requests"],
        "memory_limit_gib": LIMIT, "max_anls_drop": MAX_ANLS_DROP,
        "quality_and_memory_pass": all(p["quality_pass"] and p["memory_pass"] for p in parts),
        "token_matches": sum(p["token_matches"] for p in parts),
        "per_seed": parts, "aggregate": aggregate, "eval_trace_sha256": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", action="store_true", help="reuse the saved model decision")
    args = parser.parse_args()
    out = args.output.resolve()
    observed = observation()
    sources = [HERE / name for name in ["replay.py", "policy.py", "support.py", "policy-model.lock.json"]]
    sources.extend([ROOT / "data/processed/docvqa/development.jsonl",
                    ARCHIVE / "development/V11/quality-requests.jsonl", COMPARE / "protocol.lock.json"])
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    if args.report:
        protocol = read(out / "protocol.json")
        assert protocol["source_sha256"] == hashes
        assert observed == read(out / "observation.json")
        assert digest(out / "decision.json") == read(out / "decision.sha256.json")["sha256"]
    else:
        if out.exists():
            parser.error("Choose a new output directory")
        out.mkdir(parents=True)
        model_hashes = {name: digest(args.model / name) for name in MODEL_FILES}
        model_lock = {"name": "Qwen/Qwen3-0.6B", "revision": MODEL_REVISION, "sha256": model_hashes}
        asset_lock = HERE / "policy-model.lock.json"
        if asset_lock.exists():
            assert model_lock == read(asset_lock)
        write(out / "protocol.json", {
            "AI_on_CPU": True, "model": model_lock, "source_sha256": hashes, "fixed_capacity": 1024,
            "selection": "One decision per policy. Fixed policy retains 1024. AI gets only the archived V11 development observations and development-page sizes.",
            "evaluation": "Only after decision recording, load matching hot request measurements for seeds 4101–4103.",
            "memory_limit_gib": LIMIT, "max_anls_drop": MAX_ANLS_DROP,
        })
        write(out / "observation.json", observed)
        t = time.perf_counter()
        subprocess.run([sys.executable, str(HERE / "policy.py"), "--model", str(args.model),
                        "--observation", str(out / "observation.json"), "--output", str(out / "decision.json")],
                       check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        write(out / "decision.sha256.json", {"sha256": digest(out / "decision.json"),
                                            "policy_subprocess_wall_s": time.perf_counter() - t})
    decision = read(out / "decision.json")
    assert decision["observation"] == observed
    assert max(decision["raw_choice_logits"], key=decision["raw_choice_logits"].get) == decision["choice"]
    assert decision["choices"][decision["choice"]] == decision["capacity"] in CHOICES
    result = evaluate(decision)
    result["ai_decision"] = {k: decision[k] for k in ("decision_s", "model_load_s", "model_and_decision_s", "process_peak_rss_kib")}
    result["ai_decision"].update(read(out / "decision.sha256.json"))
    summary = out / "summary.json"
    if args.report:
        assert result == read(summary), "Replayed aggregate differs from the archived result"
    else:
        write(summary, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
