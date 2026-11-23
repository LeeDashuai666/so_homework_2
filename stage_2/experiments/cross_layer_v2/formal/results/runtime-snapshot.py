"""Development-only cross-layer VLM prototype. Original frozen engine is untouched.

Software: reuse only the immutable token prefix ending at <vision_end>.
GPU execution: replay a static-cache, single-token decode CUDA graph.
"""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from transformers import StaticCache
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "src")]
from vlm_runtime import load_model, load_processor, prepare_inputs
from vlmbench.engine import AnswerStreamer


@dataclass
class Prepared:
    record: dict
    inputs: dict
    positions: torch.Tensor
    delta: torch.Tensor
    boundary: int
    key: str
    metadata: dict


class SharedLRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.used = 0

    def get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def put(self, key, payload, tensors):
        size = sum(t.numel() * t.element_size() for t in tensors)
        if size > self.capacity:
            return
        if key in self.entries:
            self.used -= self.entries.pop(key)[1]
        while self.entries and self.used + size > self.capacity:
            self.used -= self.entries.popitem(last=False)[1][1]
        self.entries[key] = (payload, size)
        self.used += size

    def drop(self, key):
        if key in self.entries:
            self.used -= self.entries.pop(key)[1]

    def clear(self):
        self.entries.clear()
        self.used = 0


class Runtime:
    def __init__(self, gpu_cache_mib=1536, max_length=1024):
        protocol = json.loads((ROOT / "configs/protocol.lock.json").read_text())
        self.cfg = protocol["runtime"]
        self.processor = load_processor(self.cfg)
        self.model = load_model(self.cfg)
        self.core = self.model.model
        self.native_visual = self.core.get_image_features
        self.lm = self.core.language_model
        self.max_length = max_length
        self.store = SharedLRU(gpu_cache_mib * 1024**2)
        self.work = StaticCache(config=self.model.config.text_config, max_cache_len=max_length)
        text = self.model.config.text_config
        self.work.early_initialization(1, text.num_key_value_heads, text.head_dim, torch.bfloat16, torch.device("cuda:0"))
        self.token = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        self.delta = torch.zeros((1,), dtype=torch.long, device="cuda")
        self.columns = torch.arange(max_length, device="cuda").view(1, 1, 1, -1)
        self.graph = None
        self.graph_build_s = None
        self.eos = set(self.model.generation_config.eos_token_id)
        self.max_new_tokens = self.cfg["generation"]["max_new_tokens"]
        self.identity = hashlib.sha256(json.dumps({"model": json.loads((ROOT / "configs/vlm-assets.lock.json").read_text())["model"]["revision"],
            "processor": self.cfg["processor"], "dtype": self.cfg["dtype"], "attention": self.cfg["attention_backend"],
            "transformers": __import__("transformers").__version__, "cache_boundary": "inclusive first vision_end; no question tokens"}, sort_keys=True).encode()).hexdigest()

    def prepare(self, row):
        cpu, metadata = prepare_inputs(self.processor, [row], self.cfg)
        locations = (cpu["input_ids"][0] == self.model.config.vision_end_token_id).nonzero().flatten()
        assert locations.numel() == 1
        boundary = int(locations[0]) + 1
        assert boundary < cpu["input_ids"].shape[1]
        assert not cpu["mm_token_type_ids"][:, boundary:].any()
        assert cpu["input_ids"].shape[1] + self.max_new_tokens <= self.max_length
        data = {key: value.to("cuda") for key, value in cpu.items()}
        vision, delta = self.core.get_rope_index(data["input_ids"], mm_token_type_ids=data["mm_token_type_ids"],
                        image_grid_thw=data["image_grid_thw"], attention_mask=data["attention_mask"])
        plain = torch.arange(data["input_ids"].shape[1], device="cuda").view(1, 1, -1)
        positions = torch.cat([plain, vision], dim=0)
        h = hashlib.sha256((self.identity + row["pixel_sha256"] + json.dumps(metadata["grid_thw"])).encode())
        h.update(cpu["input_ids"][:, :boundary].contiguous().numpy().tobytes())
        h.update(positions[:, :, :boundary].contiguous().cpu().numpy().tobytes())
        return Prepared(row, data, positions, delta, boundary, h.hexdigest(), metadata)

    def _visual(self, request):
        key = ("visual", request.key)
        result = self.store.get(key)
        if result is not None:
            return result, True
        output = self.native_visual(request.inputs["pixel_values"], request.inputs["image_grid_thw"], return_dict=True)
        # Single-image encoding is canonical in all paths.
        result = (output.pooler_output[0].detach().clone(), tuple(t.detach().clone() for t in output.deepstack_features))
        self.store.put(key, result, [result[0], *result[1]])
        return result, False

    def _language(self, request, start=0, end=None, visual=None):
        end = end or request.inputs["input_ids"].shape[1]
        kwargs = {"attention_mask": request.inputs["attention_mask"][:, :end],
                  "position_ids": request.positions[:, :, start:end],
                  "past_key_values": self.work, "use_cache": True}
        ids = request.inputs["input_ids"][:, start:end]
        if visual is None:
            return self.lm(input_ids=ids, **kwargs)
        embedding = self.model.get_input_embeddings()(ids)
        mask = (ids == self.model.config.image_token_id)
        embedding = embedding.masked_scatter(mask.unsqueeze(-1), visual[0])
        return self.lm(inputs_embeds=embedding, visual_pos_masks=mask,
                       deepstack_visual_embeds=list(visual[1]), **kwargs)

    def _save(self, length):
        return tuple((layer.keys[:, :, :length].clone(), layer.values[:, :, :length].clone()) for layer in self.work.layers)

    def _restore(self, payload, length):
        for layer, (key, value) in zip(self.work.layers, payload):
            layer.keys[:, :, :length].copy_(key)
            layer.values[:, :, :length].copy_(value)
            layer.cumulative_length.fill_(length)

    def _prefill(self, request, prefix):
        self.work.reset()
        vision_hit, prefix_hit = False, False
        if prefix:
            entry = self.store.get(("prefix", request.key))
            prefix_hit = entry is not None
            if entry is not None:
                self._restore(entry, request.boundary)
            else:
                visual, vision_hit = self._visual(request)
                self._language(request, end=request.boundary, visual=visual)
                entry = self._save(request.boundary)
                # Prefix KV is a replacement representation, not an additional
                # copy of the same page's visual payload in the cache quota.
                self.store.drop(("visual", request.key))
                self.store.put(("prefix", request.key), entry, [tensor for pair in entry for tensor in pair])
            output = self._language(request, start=request.boundary)
        else:
            visual, vision_hit = self._visual(request)
            output = self._language(request, visual=visual)
        logits = self.model.lm_head(output.last_hidden_state[:, -1:, :])
        self.token.copy_(logits[:, -1].argmax(-1).view(1, 1))
        self.delta.copy_(request.delta.reshape(1))
        return {"prefix_hit": prefix_hit, "vision_hit": vision_hit,
                "language_prefill_tokens": request.inputs["input_ids"].shape[1] - (request.boundary if prefix_hit else 0)}

    def _decode(self):
        cursor = self.work.layers[0].cumulative_length
        positions = (cursor + self.delta).view(1, 1, 1).expand(3, 1, 1)
        visible = self.columns <= cursor
        mask = torch.where(visible, 0.0, torch.finfo(torch.bfloat16).min).to(torch.bfloat16)
        output = self.lm(input_ids=self.token, position_ids=positions, attention_mask=mask,
                         past_key_values=self.work, use_cache=True)
        logits = self.model.lm_head(output.last_hidden_state[:, -1:, :])
        self.token.copy_(logits[:, -1].argmax(-1).view(1, 1))
        return logits

    @torch.inference_mode()
    def build_graph(self, request):
        started = time.perf_counter()
        self._prefill(request, False)
        length = request.inputs["input_ids"].shape[1]
        snapshot, first = self._save(length), self.token.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._decode()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self._restore(snapshot, length)
        self.token.copy_(first)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.graph_logits = self._decode()
        torch.cuda.synchronize()
        self._restore(snapshot, length)
        self.token.copy_(first)
        self.graph_build_s = time.perf_counter() - started
        self.store.clear()

    @torch.inference_mode()
    def generate(self, request, prefix=False, graph=False, fixed_steps=None):
        torch.cuda.synchronize()
        started = time.perf_counter()
        stats = self._prefill(request, prefix)
        first = int(self.token.item())
        first_time = time.perf_counter()
        tokens = [first]
        decode_steps = (fixed_steps or self.max_new_tokens) - 1
        for _ in range(decode_steps):
            if fixed_steps is None and tokens[-1] in self.eos:
                break
            if graph:
                self.graph.replay()
            else:
                self._decode()
            tokens.append(int(self.token.item()))
        torch.cuda.synchronize()
        loop_ended = time.perf_counter()
        answer = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        ended = time.perf_counter()
        return {"question_id": request.record["question_id"], "page_id": request.record["page_id"],
                "prefix": prefix, "graph": graph, "fixed_steps": fixed_steps,
                "ttft_s": first_time - started, "generation_s": ended - started,
                "decode_s": loop_ended - first_time, "answer_decode_s": ended - loop_ended, "decode_steps": len(tokens) - 1,
                "answer": answer, "token_ids": tokens, **stats,
                "persistent_payload_bytes": self.store.used,
                "allocated_bytes": torch.cuda.memory_allocated(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()}

    @torch.inference_mode()
    def native(self, request, visual_cache=False):
        torch.cuda.synchronize()
        clock = AnswerStreamer(self.processor.tokenizer, 1, list(self.eos))
        if visual_cache:
            def reuse(*args, **kwargs):
                payload, _ = self._visual(request)
                return BaseModelOutputWithDeepstackFeatures(pooler_output=(payload[0],), deepstack_features=list(payload[1]))
            self.core.get_image_features = reuse
        started = time.perf_counter()
        try:
            self.model.generate(**request.inputs, **self.cfg["generation"], streamer=clock)
            torch.cuda.synchronize()
        finally:
            self.core.get_image_features = self.native_visual
        ended = time.perf_counter()
        tokens = clock.tokens[0]
        return {"question_id": request.record["question_id"], "answer": self.processor.tokenizer.decode(tokens, skip_special_tokens=True),
                "token_ids": tokens, "generation_s": ended - started,
                "ttft_s": clock.first[0] - started, "decode_s": ended - clock.first[0],
                "decode_steps": len(tokens)-1, "persistent_payload_bytes":self.store.used,
                "allocated_bytes":torch.cuda.memory_allocated(),"peak_allocated_bytes":torch.cuda.max_memory_allocated()}
