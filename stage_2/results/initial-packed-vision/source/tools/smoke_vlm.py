#!/usr/bin/env python3
"""Validate real VLM inference and the visual-feature boundary, without benchmarking."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from zoneinfo import ZoneInfo

import psutil
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from prepare_docvqa_smoke import prepare
from vlm_runtime import (ROOT, configuration, feature_identity, generate_answers, load_model, load_processor,
                         prepare_inputs, sample_questions)


@contextmanager
def replace_visual_boundary(core, replacement):
    own = core.__dict__.get("get_image_features")
    core.get_image_features = replacement
    try:
        yield
    finally:
        if own is None:
            del core.get_image_features
        else:
            core.get_image_features = own


def generate(model, processor, inputs, config):
    result = generate_answers(model, processor, inputs, config)
    if not all(answer.strip() for answer in result["answers"]):
        raise AssertionError("The model generated an empty decoded answer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "env/vlm-smoke-latest.json")
    args = parser.parse_args()
    config = configuration()
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    state = {}
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
              "scope": "real-image environment and feature-interface smoke; no performance or full-dataset quality claim",
              "config": config, "checks": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def check(name, function, requires=()):
        print(f"Checking {name} ...", flush=True)
        if any(not report["checks"].get(key, {}).get("passed") for key in requires):
            report["checks"][name] = {"passed": False, "skipped": True,
                                      "reason": "A required environment check failed"}
            save()
            return
        start = time.monotonic()
        try:
            with torch.inference_mode():
                details = function()
            entry = {"passed": True, "details": details}
        except Exception as error:
            entry = {"passed": False, "error": f"{type(error).__name__}: {error}",
                     "traceback": traceback.format_exc()}
        entry["elapsed_seconds"] = time.monotonic() - start
        report["checks"][name] = entry
        print(f"  {'PASS' if entry['passed'] else 'FAIL'}: {name}", flush=True)
        save()

    def assets():
        manifest = json.loads((ROOT / "env/vlm-assets-manifest.json").read_text())
        lock_path = ROOT / "configs/vlm-assets.lock.json"
        assert hashlib.sha256(lock_path.read_bytes()).hexdigest() == manifest["lock_sha256"]
        assert (ROOT / config["model_path"]).resolve() == Path(manifest["model"]["snapshot"]).resolve()
        sample = prepare(offline=True)
        state["revision"] = manifest["model"]["revision"]
        return {"model_revision": state["revision"], "pages": sample["page_count"],
                "questions": sample["question_count"], "cuda": torch.cuda.is_available(),
                "packages": {name: importlib.metadata.version(name) for name in
                             ["torch", "transformers", "torchvision", "pillow", "accelerate", "datasets"]}}

    def processor():
        processor = load_processor(config)
        questions = sample_questions(config)
        first = questions[0]
        second = next(q for q in questions if q["page_id"] == first["page_id"] and q["question_id"] != first["question_id"])
        other = next(q for q in questions if q["page_id"] != first["page_id"])
        one, detail = prepare_inputs(processor, [first], config)
        two, _ = prepare_inputs(processor, [second], config)
        different, _ = prepare_inputs(processor, [other], config)
        pair, pair_detail = prepare_inputs(processor, [first, other], config)
        identities = [feature_identity(value, config, state["revision"]) for value in [one, two, different]]
        assert identities[0] == identities[1] and identities[0] != identities[2]
        state.update(processor=processor, first=first, second=second, other=other,
                     cpu_inputs=one, cpu_second=two, cpu_other=different, cpu_batch=pair)
        return {"processor": type(processor).__name__, "image_size_policy": processor.image_processor.size,
                "single": detail, "batch2": pair_detail,
                "same_page_key_equal": True, "different_page_key_equal": False}

    def model_load():
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
        state["free_before_model"] = torch.cuda.mem_get_info()[0]
        model = load_model(config)
        state["model"] = model
        for key in ["inputs", "second_inputs", "other_inputs", "batch_inputs"]:
            cpu_key = {"inputs": "cpu_inputs", "second_inputs": "cpu_second",
                       "other_inputs": "cpu_other", "batch_inputs": "cpu_batch"}[key]
            state[key] = state[cpu_key].to(config["device"])
        core = model.model
        state["core"] = core
        state["vision_calls"] = 0
        def vision_called(module, inputs):
            state["vision_calls"] += 1
        state["vision_hook"] = core.visual.register_forward_pre_hook(vision_called)
        return {"device": torch.cuda.get_device_name(0), "dtype": str(next(model.parameters()).dtype),
                "all_parameters_on_gpu": all(p.device.type == "cuda" for p in model.parameters()),
                "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
                "cpu_rss_bytes": psutil.Process().memory_info().rss}

    def first_answer():
        before = state["vision_calls"]
        generated = generate(state["model"], state["processor"], state["inputs"], config)
        assert state["vision_calls"] - before == 1
        state["first_reference"] = generated
        return {"question_id": state["first"]["question_id"], "question": state["first"]["question"],
                "reference_answers": state["first"]["answers"], "generated": generated,
                "vision_encoder_calls": state["vision_calls"] - before,
                "quality_scope": "answer inspection only; not an accuracy benchmark"}

    def features():
        core, model, inputs = state["core"], state["model"], state["inputs"]
        original = core.get_image_features
        captured = {}
        def capture(*args, **kwargs):
            output = original(*args, **kwargs)
            captured["pool"] = tuple(t.detach().cpu().clone() for t in output.pooler_output)
            captured["deepstack"] = [t.detach().cpu().clone() for t in output.deepstack_features]
            return output
        with replace_visual_boundary(core, capture):
            result = model(**inputs, use_cache=False, logits_to_keep=1)
        reference = result.logits[:, -1, :].float().cpu()
        del result
        assert torch.isfinite(reference).all()
        assert len(captured["deepstack"]) == len(model.config.vision_config.deepstack_visual_indexes)
        pool = tuple(value.to(config["device"]) for value in captured["pool"])
        deepstack = [value.to(config["device"]) for value in captured["deepstack"]]
        assert all(torch.equal(cpu, gpu.cpu()) for cpu, gpu in zip(captured["pool"], pool))
        assert all(torch.equal(cpu, gpu.cpu()) for cpu, gpu in zip(captured["deepstack"], deepstack))
        state["cache_counters"] = Counter()
        def cached(pixel_values, image_grid_thw=None, **kwargs):
            same = torch.equal(pixel_values, inputs["pixel_values"]) and torch.equal(image_grid_thw, inputs["image_grid_thw"])
            if not same:
                state["cache_counters"]["miss"] += 1
                return original(pixel_values, image_grid_thw, **kwargs)
            state["cache_counters"]["hit"] += 1
            return BaseModelOutputWithDeepstackFeatures(pooler_output=pool, deepstack_features=deepstack)
        before = state["vision_calls"]
        with replace_visual_boundary(core, cached):
            result = model(**inputs, use_cache=False, logits_to_keep=1)
        actual = result.logits[:, -1, :].float().cpu()
        del result
        torch.testing.assert_close(actual, reference, atol=config["smoke"]["logit_atol"], rtol=config["smoke"]["logit_rtol"])
        assert state["vision_calls"] == before
        state["cached_boundary"] = cached
        all_features = [*pool, *deepstack]
        return {"pooled_shapes": [list(x.shape) for x in pool],
                "deepstack_shapes": [list(x.shape) for x in deepstack],
                "cached_feature_bytes": sum(x.numel() * x.element_size() for x in all_features),
                "cpu_gpu_roundtrip_bitwise_equal": True,
                "prefill_logits_max_abs_error": float((reference - actual).abs().max()),
                "replayed_vision_encoder_calls": state["vision_calls"] - before,
                "scope": "temporary diagnostic visual-boundary replacement, not a production cache"}

    def multiquestion():
        records = []
        for key, question, reference_key in [("inputs", state["first"], "first_reference"),
                                             ("second_inputs", state["second"], None)]:
            inputs = state[key]
            reference = state[reference_key] if reference_key else generate(state["model"], state["processor"], inputs, config)
            before = state["vision_calls"]
            with replace_visual_boundary(state["core"], state["cached_boundary"]):
                actual = generate(state["model"], state["processor"], inputs, config)
            assert actual["token_ids"] == reference["token_ids"]
            assert state["vision_calls"] == before
            records.append({"question_id": question["question_id"], "question": question["question"],
                            "reference_answers": question["answers"], "generated": actual,
                            "generated_tokens_identical": True, "replayed_vision_encoder_calls": 0})
        return {"questions": records, "language_kv_reused_between_questions": False}

    def other_page():
        before = state["vision_calls"]
        with replace_visual_boundary(state["core"], state["cached_boundary"]):
            generated = generate(state["model"], state["processor"], state["other_inputs"], config)
        assert state["vision_calls"] - before == 1
        return {"question_id": state["other"]["question_id"], "page_id": state["other"]["page_id"],
                "generated": generated, "vision_encoder_calls": 1, "cache_miss_correct": True}

    def batch_two():
        generated = generate(state["model"], state["processor"], state["batch_inputs"], config)
        assert len(generated["answers"]) == 2
        return {"batch_size": 2, "answers": generated["answers"],
                "page_ids": [state["first"]["page_id"], state["other"]["page_id"]]}

    def memory():
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.max_memory_reserved()
        margin = config["smoke"]["reserved_gpu_margin_mib"] * 1024**2
        assert free >= margin, "Observed free GPU memory is below the configured smoke margin"
        assert state["free_before_model"] - peak >= margin, "Peak reserved memory consumed the initial smoke margin"
        return {"device_total_bytes": total, "free_after_smoke_bytes": free,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": peak,
                "free_before_model_bytes": state["free_before_model"], "required_margin_bytes": margin,
                "cpu_rss_bytes": psutil.Process().memory_info().rss,
                "scope": "observed batch-1/2 samples at the configured image budget; not an arbitrary-input guarantee"}

    check("pinned_assets_and_dependencies", assets)
    check("real_image_processor", processor, ["pinned_assets_and_dependencies"])
    check("gpu_model_load", model_load, ["real_image_processor"])
    check("real_image_generation", first_answer, ["gpu_model_load"])
    check("deepstack_feature_roundtrip_and_logits", features, ["real_image_generation"])
    check("same_page_multiple_questions", multiquestion, ["deepstack_feature_roundtrip_and_logits"])
    check("different_page_cache_miss", other_page, ["deepstack_feature_roundtrip_and_logits"])
    check("batch_two_generation", batch_two, ["gpu_model_load"])
    check("gpu_memory_margin", memory, ["gpu_model_load"])
    if state.get("vision_hook"):
        state["vision_hook"].remove()
    report["passed"] = all(value["passed"] for value in report["checks"].values())
    save()
    print(f"Report: {args.output}", flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
