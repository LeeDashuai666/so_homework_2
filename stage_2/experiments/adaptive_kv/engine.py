"""Cost-based representation selection on the original document-VQA runtime."""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

from fused import Arena, bind_fused_updates

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cross_layer_v2"))
from runtime import Runtime


@dataclass
class Entry:
    kind: str
    payload: object
    size: int


class Cache:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.prefix_bytes = 0
        self.entries = OrderedDict()
        self.evictions = 0

    def remove(self, key):
        old = self.entries.pop(key, None)
        if old:
            self.used -= old.size
            if old.kind == "prefix":
                self.prefix_bytes -= old.size

    def reset(self):
        self.entries.clear()
        self.used = self.prefix_bytes = self.evictions = 0

    def get(self, key):
        entry = self.entries.get(key)
        if entry:
            self.entries.move_to_end(key)
        return entry

    def put(self, key, entry):
        if entry.size > self.limit:
            return False
        self.remove(key)
        while self.used + entry.size > self.limit:
            self.remove(next(iter(self.entries)))
            self.evictions += 1
        self.entries[key] = entry
        self.used += entry.size
        if entry.kind == "prefix":
            self.prefix_bytes += entry.size
        return True


def nbytes(tensor):
    return tensor.numel() * tensor.element_size()


class AdaptiveRuntime(Runtime):
    def __init__(self, budget_mib):
        super().__init__(gpu_cache_mib=budget_mib)
        self.reference_work = self.work
        self.packed_work = type(self.work)(config=self.model.config.text_config,
                                          max_cache_len=self.max_length)
        self.packed_work.early_initialization(1, self.model.config.text_config.num_key_value_heads,
                                              self.model.config.text_config.head_dim, torch.bfloat16,
                                              torch.device("cuda:0"))
        self.arena = Arena(self.packed_work)
        bind_fused_updates(self.packed_work)
        self.shared = Cache(budget_mib * 2**20)
        self.graphs = {}
        self.hardware = "graph"
        self.policy = "prefix"
        self.tuning = {"gap_scale": 8, "prefix_fraction": 0.6, "min_ms_per_mib": 0.5}
        self.cost = {"gain_ms": 35.0, "promotion_ms": 2.0}

    def choose_hardware(self, hardware):
        self.hardware = hardware
        self.work = self.reference_work if hardware == "graph" else self.packed_work
        for layer in self.packed_work.layers:
            layer.fuse_append = hardware == "fused"
        self.graph = self.graphs.get(hardware)

    def reset_work(self):
        if self.hardware == "graph":
            self.work.reset()
        else:
            self.arena.reset()

    def reset_trace(self, policy, tuning=None):
        self.policy = policy
        if tuning is not None:
            self.tuning = tuning
        self.shared.reset()
        self.position = 0
        self.previous = {}

    def _save(self, length):
        return (super()._save(length) if self.hardware == "graph"
                else self.arena.save(length))

    def _restore(self, payload, length):
        if self.hardware == "graph":
            super()._restore(payload, length)
        else:
            self.arena.restore(payload, length)

    def save_size(self, length):
        first = self.work.layers[0].keys
        return 2 * len(self.work.layers) * first.shape[1] * length * first.shape[-1] * first.element_size()

    def _promote(self, entry, gap, size):
        if self.policy == "prefix":
            return True
        if not entry or entry.kind != "visual" or gap is None:
            return False
        if self.policy == "second":
            return True
        if self.policy != "cost":
            return False
        projected_reuses = self.tuning["gap_scale"] / max(1, gap)
        benefit = projected_reuses * self.cost["gain_ms"] - self.cost["promotion_ms"]
        extra_mib = max(1, size - entry.size) / 2**20
        allowance = self.tuning["prefix_fraction"] * self.shared.limit
        return (benefit / extra_mib >= self.tuning["min_ms_per_mib"]
                and self.shared.prefix_bytes + size <= allowance)

    def _prefill(self, request, prefix):
        """Use the same split-prefill operation on all cache misses for equivalence."""
        key, boundary = request.key, request.boundary
        old_at = self.previous.get(key)
        self.position += 1
        self.previous[key] = self.position
        entry = self.shared.get(key)
        prefix_hit = entry is not None and entry.kind == "prefix"
        visual_hit = entry is not None and entry.kind == "visual"
        promoted = False
        encoded = False

        if prefix_hit:
            if self.hardware == "graph":
                self.reset_work()
            self._restore(entry.payload, boundary)
        else:
            self.reset_work()
            if visual_hit:
                visual = entry.payload
            else:
                output = self.native_visual(request.inputs["pixel_values"], request.inputs["image_grid_thw"],
                                            return_dict=True)
                visual = (output.pooler_output[0].detach().clone(),
                          tuple(x.detach().clone() for x in output.deepstack_features))
                encoded = True
            self._language(request, end=boundary, visual=visual)
            size = self.save_size(boundary)
            gap = self.position - old_at if old_at is not None else None
            if size <= self.shared.limit and self._promote(entry, gap, size):
                promoted = self.shared.put(key, Entry("prefix", self._save(boundary), size))
            elif not visual_hit:
                visual_size = nbytes(visual[0]) + sum(map(nbytes, visual[1]))
                self.shared.put(key, Entry("visual", visual, visual_size))
        output = self._language(request, start=boundary)
        logits = self.model.lm_head(output.last_hidden_state[:, -1:, :])
        self.token.copy_(logits[:, -1].argmax(-1).view(1, 1))
        self.delta.copy_(request.delta.reshape(1))
        return {"prefix_hit": prefix_hit, "vision_hit": visual_hit, "encoder_calls": int(encoded),
                "promotion": int(promoted), "cache_used": self.shared.used,
                "prefix_bytes": self.shared.prefix_bytes, "evictions": self.shared.evictions,
                "hardware": self.hardware, "policy": self.policy}

    def build(self, request, hardware):
        self.choose_hardware(hardware)
        self.reset_trace("vision")
        self.build_graph(request)
        self.graphs[hardware] = self.graph
        return self.graph_build_s
