"""Bounded, exact two-tier visual-feature cache with explicit inference leases."""

from collections import Counter, OrderedDict
from dataclasses import dataclass
import math
import time

import torch


@dataclass
class Payload:
    pooled: torch.Tensor
    deepstack: tuple

    @property
    def tensors(self):
        return (self.pooled, *self.deepstack)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)

    def to(self, device):
        tensors = tuple(t.detach().to(device) for t in self.tensors)
        return Payload(tensors[0], tensors[1:])


class FeatureCache:
    def __init__(self, gpu_bytes, cpu_bytes, policy="lru", device="cuda:0"):
        self.gpu_limit = int(gpu_bytes)
        self.gpu_upper = int(gpu_bytes)
        self.cpu_limit = int(cpu_bytes)
        self.policy, self.device = policy, device
        self.gpu, self.cpu = OrderedDict(), OrderedDict()
        self.pins = Counter()
        self.history = {}
        self.tick = 0
        self.stats = Counter()

    @property
    def gpu_bytes(self):
        return sum(v.nbytes for v in self.gpu.values())

    @property
    def cpu_bytes(self):
        return sum(v.nbytes for v in self.cpu.values())

    def peek(self, key):
        return "gpu" if key in self.gpu else "cpu" if key in self.cpu else "miss"

    def touch(self, key, count=1):
        self.tick += count
        item = self.history.setdefault(key, {"frequency": 0.0, "last": self.tick,
                                              "vision_s": 0.04, "bytes": 1})
        item["frequency"] *= 0.5 ** ((self.tick - item["last"]) / 64.0)
        item["frequency"] += count
        item["last"] = self.tick

    def value(self, key):
        item = self.history.get(key)
        if not item:
            return 0.0
        frequency = item["frequency"] * 0.5 ** ((self.tick - item["last"]) / 64.0)
        return (1.0 + frequency) * item["vision_s"] / max(1, item["bytes"])

    def note_cost(self, key, vision_s, nbytes):
        item = self.history.setdefault(key, {"frequency": 1.0, "last": self.tick})
        item["vision_s"] = max(vision_s, 1e-6)
        item["bytes"] = nbytes

    def victim(self, entries, protected=True):
        keys = [key for key in entries if not protected or not self.pins[key]]
        if not keys:
            return None
        if self.policy == "value":
            return min(keys, key=lambda key: (self.value(key), list(entries).index(key)))
        return keys[0]

    def _cpu_store(self, key, payload):
        if payload.nbytes > self.cpu_limit:
            self.stats["cpu_bypass"] += 1
            return
        if key in self.cpu:
            self.cpu.pop(key)
        while self.cpu and self.cpu_bytes + payload.nbytes > self.cpu_limit:
            victim = self.victim(self.cpu, protected=False)
            self.cpu.pop(victim)
            self.stats["cpu_evictions"] += 1
        start = time.perf_counter()
        self.cpu[key] = payload.to("cpu")
        self.stats["d2h_bytes"] += payload.nbytes
        self.stats["d2h_s"] += time.perf_counter() - start

    def _evict_gpu(self, key):
        if self.pins[key]:
            raise RuntimeError("Attempted eviction of a leased visual feature")
        payload = self.gpu.pop(key)
        self._cpu_store(key, payload)
        self.stats["gpu_evictions"] += 1

    def resize(self, limit):
        self.gpu_limit = max(0, min(self.gpu_upper, int(limit)))
        while self.gpu_bytes > self.gpu_limit:
            victim = self.victim(self.gpu)
            if victim is None:
                break
            self._evict_gpu(victim)

    def _admit_gpu(self, key, payload):
        if key in self.gpu:
            self.gpu.move_to_end(key)
            return
        while self.gpu_bytes + payload.nbytes > self.gpu_limit:
            victim = self.victim(self.gpu)
            if victim is None:
                self._cpu_store(key, payload)
                self.stats["gpu_bypass"] += 1
                return
            if self.policy == "value" and self.value(key) < self.value(victim):
                self._cpu_store(key, payload)
                self.stats["gpu_value_bypass"] += 1
                return
            self._evict_gpu(victim)
        self.gpu[key] = payload
        self.stats["gpu_peak_payload_bytes"] = max(self.stats["gpu_peak_payload_bytes"], self.gpu_bytes)

    def acquire(self, key):
        if key in self.gpu:
            self.stats["gpu_hits"] += 1
            self.gpu.move_to_end(key)
            self.pins[key] += 1
            return self.gpu[key]
        if key in self.cpu:
            self.stats["cpu_hits"] += 1
            self.pins[key] += 1
            start = time.perf_counter()
            payload = self.cpu.pop(key).to(self.device)
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
            self.stats["h2d_s"] += time.perf_counter() - start
            self.stats["h2d_bytes"] += payload.nbytes
            self._admit_gpu(key, payload)
            return payload
        self.stats["misses"] += 1
        return None

    def insert_acquire(self, key, payload, vision_s):
        self.note_cost(key, vision_s, payload.nbytes)
        self.pins[key] += 1
        self._admit_gpu(key, payload)
        return payload

    def release(self, keys):
        for key in keys:
            if self.pins[key] <= 0:
                raise RuntimeError("Unbalanced visual-feature release")
            self.pins[key] -= 1
            if self.pins[key] == 0:
                del self.pins[key]
        self.resize(self.gpu_limit)

    def clear(self):
        if any(self.pins.values()):
            raise RuntimeError("Cannot clear a cache with active inference leases")
        self.gpu.clear()
        self.cpu.clear()
        self.history.clear()
        self.stats.clear()
        self.tick = 0

    def snapshot(self):
        return {**dict(self.stats), "gpu_payload_bytes": self.gpu_bytes,
                "cpu_payload_bytes": self.cpu_bytes, "gpu_limit": self.gpu_limit,
                "cpu_limit": self.cpu_limit, "gpu_entries": len(self.gpu),
                "cpu_entries": len(self.cpu), "active_leases": sum(self.pins.values())}
