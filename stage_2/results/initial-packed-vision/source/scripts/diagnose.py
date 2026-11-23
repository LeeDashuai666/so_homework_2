#!/usr/bin/env python3
"""Real GPU equivalence, isolated phase calibration and profiler evidence."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]


def main():
    import torch
    from vlmbench.engine import InferenceEngine
    from vlmbench.runner import question_order, warmup
    from vlmbench.metrics import quality, quantiles

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/diagnostics")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    runtime = json.loads((ROOT / cfg["runtime"]).read_text())
    runtime["dataset_path"] = cfg["dataset_path"]
    torch.set_num_threads(runtime["cpu_threads"])
    torch.set_num_interop_threads(1)
    data = [json.loads(row) for row in (ROOT / cfg["dataset_path"] / "development.jsonl").read_text().splitlines()]
    engine = InferenceEngine(runtime, cfg, "B0")
    warmup(engine, data, 8, 5)
    isolated, phases = [], []
    for index, record in enumerate(question_order(data, "random", 1700)[:32]):
        now = time.perf_counter()
        request = engine.prepare(record, f"isolated-{index}", now, now)
        rows, batch = engine.execute([request], f"isolated-{index}")
        rows[0].update(quality(rows[0]["answer"], record["answers"]))
        isolated.extend(rows)
        phases.append(batch)
    (args.output / "isolated-requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in isolated))
    (args.output / "isolated-batches.jsonl").write_text("".join(json.dumps(row) + "\n" for row in phases))
    calibration = {"sample_count": len(isolated), "ttft_s": quantiles(r["ttft_s"] for r in isolated),
                   "latency_s": quantiles(r["latency_s"] for r in isolated),
                   "parameter_bytes": engine.parameter_bytes,
                   "vision_s_per_token": sum(b["vision_s"] for b in phases) / sum(sum(b["visual_tokens"]) for b in phases),
                   "prefill_s_per_token": sum(b["prefill_s"] for b in phases) / sum(sum(b["input_tokens"]) for b in phases)}
    (args.output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print("Isolated calibration", json.dumps(calibration), flush=True)
    grouped = defaultdict(list)
    for row in data:
        grouped[row["page_id"]].append(row)
    pages = [rows for rows in grouped.values() if len(rows) >= 2][:3]
    combinations = [[pages[0][0]], pages[0][:2], [pages[0][0], pages[1][0]], [pages[1][1], pages[2][0]]]
    checks = []
    for index, records in enumerate(combinations):
        now = time.perf_counter()
        requests = [engine.prepare(row, f"check-{index}-{i}", now, now) for i, row in enumerate(records)]
        engine.set_variant("B0")
        normal, normal_batch = engine.execute(requests, "native")
        engine.set_variant("V1")
        miss, miss_batch = engine.execute(requests, "miss")
        hit, hit_batch = engine.execute(requests, "hit")
        saved_features = {key: tuple(t.detach().cpu().clone() for t in value.tensors)
                          for key, value in engine.cache.gpu.items()}
        # Force all complete cached tensors through actual device->host->device.
        engine.cache.resize(0)
        assert not engine.cache.gpu and engine.cache.cpu
        engine.cache.resize(engine.cache.gpu_upper)
        cpu, cpu_batch = engine.execute(requests, "cpu-hit")
        transport_equal = all(all(torch.equal(left, right.detach().cpu()) for left, right in
                              zip(saved_features[key], payload.tensors)) for key, payload in engine.cache.gpu.items())
        token_equal = all(a["token_ids"] == b["token_ids"] == c["token_ids"] == d["token_ids"]
                          for a, b, c, d in zip(normal, miss, hit, cpu))
        # Match the exact batch and compare first-step logits, including DeepStack.
        packed = engine.collate(requests)
        engine.adapter.cache = None
        engine.adapter.begin(requests)
        with torch.inference_mode():
            original = engine.model(**packed, use_cache=False, logits_to_keep=1).logits.detach().cpu()
        engine.adapter.finish()
        engine.adapter.cache = engine.cache
        engine.adapter.begin(requests)
        with torch.inference_mode():
            cached = engine.model(**packed, use_cache=False, logits_to_keep=1).logits.detach().cpu()
        engine.adapter.finish()
        max_error = float((original.float() - cached.float()).abs().max())
        close = torch.allclose(original, cached, atol=runtime["smoke"]["logit_atol"], rtol=runtime["smoke"]["logit_rtol"])
        features_equal = True
        feature_max_error = 0.0
        native_batch_vs_single_max_error = 0.0
        with torch.inference_mode():
            native = engine.adapter.original(packed["pixel_values"], packed["image_grid_thw"], return_dict=True)
            chunks = [value.split([p.shape[0] for p in native.pooler_output]) for value in native.deepstack_features]
            for i, request in enumerate(requests):
                payload = engine.cache.acquire(request.key)
                features_equal &= torch.equal(native.pooler_output[i], payload.pooled)
                features_equal &= all(torch.equal(values[i], tensor) for values, tensor in zip(chunks, payload.deepstack))
                feature_max_error = max(feature_max_error, float((native.pooler_output[i].float() - payload.pooled.float()).abs().max()))
                engine.cache.release([request.key])
                pixel_chunks = packed["pixel_values"].split(packed["image_grid_thw"].prod(-1).tolist())
                single = engine.adapter.original(pixel_chunks[i], packed["image_grid_thw"][i:i + 1], return_dict=True)
                native_batch_vs_single_max_error = max(native_batch_vs_single_max_error,
                    float((native.pooler_output[i].float() - single.pooler_output[0].float()).abs().max()))
        result = {"case": index, "page_ids": [r["page_id"] for r in records],
                  "question_ids": [r["question_id"] for r in records],
                  "tokens_equal_native_miss_gpu_cpu": token_equal,
                  "features_bitwise_equal": features_equal, "first_logits_max_abs_error": max_error,
                  "feature_max_abs_error": feature_max_error,
                  "native_batch_vs_single_feature_max_abs_error": native_batch_vs_single_max_error,
                  "cpu_gpu_roundtrip_bitwise_equal": transport_equal,
                  "first_logits_within_tolerance": bool(close),
                  "native_images": normal_batch["vision_images"], "miss_images": miss_batch["vision_images"],
                  "gpu_hit_images_encoded": hit_batch["vision_images"], "cpu_hit_images_encoded": cpu_batch["vision_images"],
                  "answers": [r["answer"] for r in normal], "cache": engine.cache.snapshot()}
        checks.append(result)
        (args.output / "equivalence.json").write_text(json.dumps(checks, indent=2) + "\n")
        # Exact transport must hold for every batch. Packed BF16 vision kernels
        # can change rounding when the encoded image count changes; this is
        # measured explicitly and is subject to the full-dataset quality gate.
        assert token_equal and transport_equal and hit_batch["vision_images"] == 0 and cpu_batch["vision_images"] == 0
        if len(records) == 1:
            assert features_equal and close and max_error == 0
        print("Equivalence case", index, "passed", flush=True)
    # Native profiler includes actual kernels; not used as the timed benchmark.
    engine.set_variant("V1")
    now = time.perf_counter()
    requests = [engine.prepare(pages[0][0], "profile", now, now)]
    for state in ["miss", "hit"]:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            with torch.profiler.record_function(f"vlm_{state}"):
                engine.execute(requests, state)
        prof.export_chrome_trace(str(args.output / f"profile-{state}.json"))
        (args.output / f"profile-{state}.txt").write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=35))
    print("Real GPU diagnostics and profiler complete", flush=True)


if __name__ == "__main__":
    main()
