"""Single-model VLM inference with exact visual reuse and observable phase timing."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import time

import torch
from transformers import AutoTokenizer
from transformers.generation.streamers import BaseStreamer
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from vlm_runtime import ROOT, load_model, load_processor, prepare_inputs
from .cache import FeatureCache, Payload


VARIANTS = {
    "B0": {"cache": False, "scheduler": "fifo", "policy": "lru"},
    "V1": {"cache": True, "scheduler": "fifo", "policy": "lru"},
    "V1+S": {"cache": True, "scheduler": "affinity", "policy": "lru"},
    "V1+M": {"cache": True, "scheduler": "fifo", "policy": "value"},
    "FULL": {"cache": True, "scheduler": "affinity", "policy": "value"},
}


@dataclass
class ReadyRequest:
    record: dict
    inputs: object
    metadata: dict
    key: str
    scheduled: float
    received: float
    prep_start: float
    ready: float
    request_id: str


class AnswerStreamer(BaseStreamer):
    """Timestamp actual CPU-visible tokens and publish decoded answers at EOS."""
    def __init__(self, tokenizer, batch_size, eos):
        self.tokenizer = tokenizer
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.prompt = True
        self.tokens = [[] for _ in range(batch_size)]
        self.first = [None] * batch_size
        self.finish = [None] * batch_size
        self.answers = [None] * batch_size
        self.eos_seen = [False] * batch_size

    def put(self, value):
        if self.prompt:
            self.prompt = False
            return
        values = value.reshape(-1).tolist()
        now = time.perf_counter()
        for index, token in enumerate(values):
            if self.finish[index] is not None:
                continue
            if self.first[index] is None:
                self.first[index] = now
            self.tokens[index].append(int(token))
            if token in self.eos:
                self.eos_seen[index] = True
                self._finish(index)

    def _finish(self, index):
        self.answers[index] = self.tokenizer.decode(self.tokens[index], skip_special_tokens=True)
        self.finish[index] = time.perf_counter()

    def end(self):
        for index in range(len(self.tokens)):
            if self.finish[index] is None:
                self._finish(index)


class VisualAdapter:
    def __init__(self, model, cache, vision_microbatch=1):
        self.model, self.core, self.cache = model, model.model, cache
        self.original = self.core.get_image_features
        self.context = None
        self.vision_microbatch = vision_microbatch
        self.core.get_image_features = self.forward

    def begin(self, requests):
        self.context = {"requests": requests, "leases": [], "handled": False,
                        "vision_s": 0.0, "vision_images": 0, "assembly_s": 0.0,
                        "lookup_s": 0.0, "batch_shared_images": 0}

    def finish(self):
        context = self.context
        if self.cache is not None and context["leases"]:
            self.cache.release(context["leases"])
        self.context = None
        return {key: value for key, value in context.items() if key not in ("requests", "leases", "handled")}

    def encode(self, pixels, grid, kwargs):
        # Use the same per-image visual execution shape in every variant.
        # BF16 packed-image GEMMs otherwise make a cached page depend on the
        # number/shapes of the other images that happened to encode with it.
        if self.vision_microbatch == 1 and len(grid) > 1:
            chunks = pixels.split(grid.prod(-1).tolist())
            values = [self._encode_native(chunk, grid[i:i + 1], kwargs) for i, chunk in enumerate(chunks)]
            outputs = [value[0] for value in values]
            combined = BaseModelOutputWithDeepstackFeatures(
                pooler_output=tuple(tensor for output in outputs for tensor in output.pooler_output),
                deepstack_features=[torch.cat([output.deepstack_features[i] for output in outputs], dim=0)
                                    for i in range(len(outputs[0].deepstack_features))])
            return combined, sum(value[1] for value in values)
        return self._encode_native(pixels, grid, kwargs)

    def _encode_native(self, pixels, grid, kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = self.original(pixels, grid, return_dict=True,
                               **{k: v for k, v in kwargs.items() if k != "return_dict"})
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0
        self.context["vision_s"] += elapsed
        self.context["vision_images"] += grid.shape[0]
        return output, elapsed

    def forward(self, pixel_values, image_grid_thw=None, **kwargs):
        if self.context is None:
            return self.original(pixel_values, image_grid_thw, **kwargs)
        if self.context["handled"]:
            raise RuntimeError("The visual branch was unexpectedly executed twice in one generation")
        self.context["handled"] = True
        requests = self.context["requests"]
        if len(requests) != image_grid_thw.shape[0]:
            raise RuntimeError("Visual features do not correspond to the current request batch")
        if self.cache is None:
            return self.encode(pixel_values, image_grid_thw, kwargs)[0]
        keys = [request.key for request in requests]
        unique = list(dict.fromkeys(keys))
        counts = Counter(keys)
        for key in unique:
            self.cache.touch(key, counts[key])
        payloads = {}
        before = time.perf_counter()
        for key in unique:
            payload = self.cache.acquire(key)
            if payload is not None:
                payloads[key] = payload
                self.context["leases"].append(key)
        self.context["lookup_s"] += time.perf_counter() - before
        missing = [key for key in unique if key not in payloads]
        if missing:
            patch_counts = image_grid_thw.prod(-1).tolist()
            pixel_chunks = pixel_values.split(patch_counts)
            indices = [keys.index(key) for key in missing]
            pixels = torch.cat([pixel_chunks[index] for index in indices], dim=0)
            grid = image_grid_thw[indices]
            output, elapsed = self.encode(pixels, grid, kwargs)
            token_counts = [value.shape[0] for value in output.pooler_output]
            deep_chunks = [value.split(token_counts) for value in output.deepstack_features]
            for position, key in enumerate(missing):
                # Clone each slice: cached pages must not retain a whole batch's storage.
                payload = Payload(output.pooler_output[position].detach().clone(),
                                  tuple(layer[position].detach().clone() for layer in deep_chunks))
                portion = elapsed * token_counts[position] / sum(token_counts)
                payloads[key] = self.cache.insert_acquire(key, payload, portion)
                self.context["leases"].append(key)
        before = time.perf_counter()
        pooled = tuple(payloads[key].pooled for key in keys)
        deep = [torch.cat([payloads[key].deepstack[layer] for key in keys], dim=0)
                for layer in range(len(payloads[keys[0]].deepstack))]
        self.context["assembly_s"] += time.perf_counter() - before
        self.context["batch_shared_images"] = len(keys) - len(unique)
        return BaseModelOutputWithDeepstackFeatures(pooler_output=pooled, deepstack_features=deep)


class InferenceEngine:
    def __init__(self, runtime, experiment, variant):
        self.runtime, self.experiment = runtime, experiment
        self.device = runtime["device"]
        self.processor = load_processor(runtime)
        self.decoder = AutoTokenizer.from_pretrained(str(ROOT / runtime["model_path"]), local_files_only=True)
        self.model = load_model(runtime)
        self.parameter_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        assets = json.loads((ROOT / "configs/vlm-assets.lock.json").read_text())
        identity = {"revision": assets["model"]["revision"], "processor": runtime["processor"],
                    "dtype": runtime["dtype"], "backend": runtime["attention_backend"],
                    "vision_microbatch": experiment["vision_microbatch"],
                    "transformers": __import__("transformers").__version__}
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.adapter = VisualAdapter(self.model, None, experiment["vision_microbatch"])
        self.current = None
        self.prefill_start = None
        self._pre = self.model.model.language_model.register_forward_pre_hook(self._language_pre)
        self._post = self.model.model.language_model.register_forward_hook(self._language_post)
        self.set_variant(variant)

    def set_variant(self, variant):
        self.variant, self.options = variant, VARIANTS[variant]
        self.cache = FeatureCache(self.experiment["gpu_cache_mib"] * 1024**2,
                                  self.experiment["cpu_cache_mib"] * 1024**2,
                                  policy=self.options["policy"], device=self.device) if self.options["cache"] else None
        self.adapter.cache = self.cache

    def _language_pre(self, module, args):
        if self.current is not None and "prefill_s" not in self.current:
            self.prefill_start = torch.cuda.Event(enable_timing=True)
            self.prefill_start.record()

    def _language_post(self, module, args, output):
        if self.current is not None and "prefill_s" not in self.current:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            self.current["prefill_s"] = self.prefill_start.elapsed_time(end) / 1000.0

    def prepare(self, record, request_id, scheduled, received):
        start = time.perf_counter()
        inputs, metadata = prepare_inputs(self.processor, [record], self.runtime)
        identity = {"pixels": record["pixel_sha256"], "grid": metadata["grid_thw"][0],
                    "runtime": self.identity}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ready = time.perf_counter()
        return ReadyRequest(record, inputs, metadata, key, scheduled, received, start, ready, request_id)

    def collate(self, requests):
        maximum = max(request.inputs["input_ids"].shape[1] for request in requests)
        result = {}
        for name in ["input_ids", "attention_mask", "mm_token_type_ids"]:
            fill = self.processor.tokenizer.pad_token_id if name == "input_ids" else 0
            tensor = torch.full((len(requests), maximum), fill, dtype=requests[0].inputs[name].dtype)
            for index, request in enumerate(requests):
                value = request.inputs[name]
                tensor[index, -value.shape[1]:] = value[0]
            result[name] = tensor.to(self.device)
        for name in ["pixel_values", "image_grid_thw"]:
            result[name] = torch.cat([request.inputs[name] for request in requests], dim=0).to(self.device)
        return result

    def budget(self, requests):
        text = self.model.config.text_config
        sequence = max(request.metadata["input_lengths"][0] for request in requests) + self.runtime["generation"]["max_new_tokens"]
        kv = len(requests) * sequence * 2 * text.num_hidden_layers * text.num_key_value_heads * text.head_dim * 2
        reserve = self.experiment["workspace_reserve_mib"] * 1024**2
        limit = self.experiment["process_budget_mib"] * 1024**2 - self.parameter_bytes - kv - reserve
        # All policies obey the same per-batch safety budget. O3 changes which
        # complete features survive, rather than giving one method more memory.
        if self.cache:
            self.cache.resize(limit)
        return {"estimated_kv_bytes": kv, "cache_admission_budget_bytes": max(0, limit)}

    @torch.inference_mode()
    def execute(self, requests, batch_id):
        start = time.perf_counter()
        budget = self.budget(requests)
        tiers = [self.cache.peek(request.key) if self.cache else "disabled" for request in requests]
        inputs = self.collate(requests)
        packed = time.perf_counter()
        streamer = AnswerStreamer(self.decoder, len(requests), self.model.generation_config.eos_token_id)
        self.current = {}
        self.adapter.begin(requests)
        try:
            self.model.generate(**inputs, **self.runtime["generation"], streamer=streamer)
            torch.cuda.synchronize()
            end = time.perf_counter()
        finally:
            visual = self.adapter.finish()
        batch = {"batch_id": batch_id, "request_ids": [r.request_id for r in requests],
                 "page_ids": [r.record["page_id"] for r in requests], "real_batch_size": len(requests),
                 "input_tokens": [r.metadata["input_lengths"][0] for r in requests],
                 "visual_tokens": [r.metadata["visual_tokens"][0] for r in requests],
                 "cache_tiers": tiers, "pack_h2d_s": packed - start, "service_s": end - start,
                 **visual, **self.current, **budget,
                 "allocated_bytes": torch.cuda.memory_allocated(),
                 "reserved_bytes": torch.cuda.memory_reserved(),
                 "cache": self.cache.snapshot() if self.cache else {}}
        self.current = None
        results = []
        for index, request in enumerate(requests):
            finish = streamer.finish[index] or end
            first = streamer.first[index] or finish
            results.append({
                "request_id": request.request_id, "question_id": request.record["question_id"],
                "page_id": request.record["page_id"], "status": "completed", "batch_id": batch_id,
                "scheduled": request.scheduled, "received": request.received,
                "prep_start": request.prep_start, "ready": request.ready,
                "started": start, "first_token": first, "completed": finish,
                "sender_lag_s": request.received - request.scheduled,
                "preprocess_s": request.ready - request.prep_start,
                "queue_s": start - request.ready, "ttft_s": first - request.scheduled,
                "latency_s": finish - request.scheduled, "service_s": finish - start,
                "answer": streamer.answers[index], "token_ids": streamer.tokens[index],
                "output_tokens": len(streamer.tokens[index]), "eos_seen": streamer.eos_seen[index],
                "cache_tier": tiers[index], "input_tokens": request.metadata["input_lengths"][0],
                "visual_tokens": request.metadata["visual_tokens"][0],
            })
        return results, batch

    def reset(self):
        if self.cache:
            self.cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
