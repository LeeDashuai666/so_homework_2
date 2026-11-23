"""Meaningful invariants for indexing, latency accounting, eviction and scheduling."""

from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from vlmbench.cache import FeatureCache, Payload
from vlmbench.metrics import quality, summarize
from vlmbench.runner import make_trace
from vlmbench.scheduler import select


def payload(size=4):
    return Payload(torch.ones(size), (torch.ones(size),))


def test_cache_lease_prevents_eviction_and_balances():
    cache = FeatureCache(32, 128, device="cpu")
    cache.touch("a")
    cache.insert_acquire("a", payload(), 0.01)
    cache.touch("b")
    cache.insert_acquire("b", payload(), 0.02)
    assert cache.peek("a") == "gpu" and cache.pins["a"] == 1
    assert cache.peek("b") == "cpu"  # current batch owns the transient GPU payload
    cache.release(["a", "b"])
    assert not cache.pins
    assert cache.acquire("b") is not None
    assert cache.peek("a") == "cpu" and cache.peek("b") == "gpu"
    cache.release(["b"])
    assert cache.gpu_bytes <= cache.gpu_limit
    with pytest.raises(RuntimeError):
        cache.release(["b"])


def test_quality_uses_normalized_denominator_and_all_references():
    assert quality("  SAN  DIEGO ", ["San Diego", "elsewhere"])["anls"] == 1
    assert quality("abc", ["xyz"])["anls"] == 0
    assert quality("", ["answer"])["anls"] == 0
    assert quality("abcdef", ["abcdeg"])["anls"] == pytest.approx(5 / 6)


def test_goodput_counts_rejections_and_slo_misses():
    rows = [{"status": "completed", "ttft_s": 0.1, "latency_s": 0.2, "anls": 1, "em": 1},
            {"status": "completed", "ttft_s": 2, "latency_s": 3, "anls": 1, "em": 1},
            {"status": "rejected"}]
    result = summarize(rows, 10, 1, 2)
    assert result["goodput_qps"] == 0.1
    assert result["slo_attainment"] == pytest.approx(1 / 3)
    assert result["rejected"] == 1


def test_frozen_trace_is_deterministic_and_stops_at_window():
    data = [{"question_id": str(i), "page_id": str(i // 2)} for i in range(12)]
    one = make_trace(data, "random", 1701, duration=10, rate=3)
    assert one == make_trace(data, "random", 1701, duration=10, rate=3)
    assert all(row["offset_s"] < 10 for row in one)
    assert len({row["request_id"] for row in one}) == len(one)


def test_affinity_is_bounded_and_age_protects_oldest():
    cfg = {"lookahead": 3, "batch_size": 2, "max_batch_visual_tokens": 1536,
           "age_guard_slo_fraction": 0.5, "scheduler_vision_cost_per_token_s": 0.001,
           "scheduler_prefill_cost_per_token_s": 0.0001, "cpu_transfer_cost_per_mib_s": 0.001}
    engine = SimpleNamespace(experiment=cfg, cache=None, options={"scheduler": "affinity"})
    def item(key, length, scheduled=0.9):
        return SimpleNamespace(key=key, scheduled=scheduled, record={},
                               metadata={"visual_tokens": [length], "input_lengths": [length + 20]})
    old, a, b, future = item("old", 700, 0), item("same", 600), item("same", 600), item("future", 1)
    chosen, reason = select([old, a, b, future], engine, 1, 1)
    assert chosen[0] is old and reason == "age_guard"
    assert all(value is not future for value in chosen)
    old.scheduled = 0.9
    chosen, reason = select([old, a, b, future], engine, 1, 1)
    assert {id(v) for v in chosen} == {id(a), id(b)}
