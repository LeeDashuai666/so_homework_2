"""Actual open-loop arrival, CPU preparation and one GPU-worker experiment runner."""

from collections import defaultdict, deque
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import random
import subprocess
import threading
import time
import traceback
from zoneinfo import ZoneInfo

import psutil
import torch

from .metrics import quality, summarize
from .scheduler import select


def question_order(questions, order, seed):
    rng = random.Random(seed)
    if order == "random":
        result = list(questions)
        rng.shuffle(result)
        return result
    grouped = defaultdict(list)
    for row in questions:
        grouped[row["page_id"]].append(row)
    pages = list(grouped)
    rng.shuffle(pages)
    for values in grouped.values():
        rng.shuffle(values)
    if order == "session":
        return [q for page in pages for q in grouped[page]]
    # Independent page sessions, round-robin interleaved in windows of 16 pages.
    result = []
    for start in range(0, len(pages), 16):
        active = [deque(grouped[page]) for page in pages[start:start + 16]]
        while any(active):
            for values in active:
                if values:
                    result.append(values.popleft())
    return result


def make_trace(questions, order, seed, duration=None, rate=None, count=None):
    rng = random.Random(seed)
    cycle = question_order(questions, order, seed)
    rows, arrival, index = [], 0.0, 0
    while True:
        if duration is not None:
            arrival += rng.expovariate(rate)
            if arrival >= duration:
                break
        elif count is not None and index >= count:
            break
        if index and index % len(cycle) == 0:
            cycle = question_order(questions, order, seed + index // len(cycle))
        rows.append({"request_id": f"r{index:06d}", "offset_s": arrival,
                     "question_id": cycle[index % len(cycle)]["question_id"]})
        index += 1
    return rows


def warmup(engine, questions, minimum_requests, minimum_seconds):
    started, done = time.perf_counter(), 0
    while done < minimum_requests or time.perf_counter() - started < minimum_seconds:
        batch = []
        for index in range(engine.experiment["batch_size"]):
            row = questions[(done + index) % len(questions)]
            now = time.perf_counter()
            batch.append(engine.prepare(row, f"warm-{done + index}", now, now))
        engine.execute(batch, f"warm-{done}")
        done += len(batch)
    engine.reset()
    return {"requests": done, "seconds": time.perf_counter() - started}


def run(engine, questions, trace, output, duration=None, ttft_slo=1.0, e2e_slo=3.0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lookup = {q["question_id"]: q for q in questions}
    (output / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace))
    origin = time.perf_counter() + 0.1
    incoming = queue.Queue()
    ready = []
    rows, batches, monitoring = [], [], []
    condition = threading.Condition()
    stop = threading.Event()
    producer_done = threading.Event()
    prepare_done = threading.Event()
    outstanding = 0
    accepted = {}
    errors = []
    batch_id = 0

    def reject(item, now, status="rejected", error=None):
        record = {"request_id": item["request_id"], "question_id": item["question_id"],
                  "page_id": lookup[item["question_id"]]["page_id"], "status": status,
                  "scheduled": origin + item["offset_s"], "received": now,
                  "sender_lag_s": now - origin - item["offset_s"]}
        if error:
            record["error"] = error
        rows.append(record)

    def produce():
        nonlocal outstanding
        try:
            stop.wait(max(0.0, origin - time.perf_counter()))
            for item in trace:
                if stop.is_set():
                    break
                scheduled = origin + item["offset_s"]
                if duration is not None:
                    stop.wait(max(0.0, scheduled - time.perf_counter()))
                now = time.perf_counter()
                with condition:
                    if duration is None:
                        condition.wait_for(lambda: outstanding < engine.experiment["queue_limit"] or stop.is_set())
                    if stop.is_set():
                        break
                    if outstanding >= engine.experiment["queue_limit"]:
                        reject(item, now)
                        continue
                    outstanding += 1
                    accepted[item["request_id"]] = item
                incoming.put((item, scheduled, now))
        except Exception:
            errors.append(traceback.format_exc())
        finally:
            producer_done.set()
            incoming.put(None)

    def preprocess():
        nonlocal outstanding
        try:
            while not stop.is_set():
                message = incoming.get()
                if message is None:
                    break
                item, scheduled, received = message
                try:
                    prepared = engine.prepare(lookup[item["question_id"]], item["request_id"], scheduled, received)
                    with condition:
                        ready.append(prepared)
                        condition.notify_all()
                except Exception:
                    with condition:
                        reject(item, received, "preprocess_error", traceback.format_exc())
                        outstanding -= 1
                        accepted.pop(item["request_id"], None)
                        condition.notify_all()
        except Exception:
            errors.append(traceback.format_exc())
        finally:
            prepare_done.set()
            with condition:
                condition.notify_all()

    def monitor():
        while not stop.wait(5.0):
            entry = {"offset_s": time.perf_counter() - origin,
                     "cpu_rss_bytes": psutil.Process().memory_info().rss}
            try:
                completed = subprocess.run(["/usr/lib/wsl/lib/nvidia-smi",
                    "--query-gpu=temperature.gpu,clocks.current.sm,power.draw,memory.used",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                entry["gpu_monitor_raw"] = completed.stdout.strip()
                entry["returncode"] = completed.returncode
            except Exception as error:
                entry["error"] = type(error).__name__
            monitoring.append(entry)

    threads = [threading.Thread(target=f, daemon=True) for f in [produce, preprocess, monitor]]
    for thread in threads:
        thread.start()
    try:
        while True:
            now = time.perf_counter()
            if duration is not None and now > origin + duration + engine.experiment["drain_timeout_s"]:
                stop.set()
                break
            with condition:
                if not ready:
                    if producer_done.is_set() and prepare_done.is_set() and outstanding == 0:
                        break
                    condition.wait(timeout=0.01)
                    continue
                if len(ready) < engine.experiment["batch_size"] and not prepare_done.is_set():
                    condition.wait(timeout=engine.experiment["batch_wait_ms"] / 1000)
                selection_start = time.perf_counter()
                selected, reason = select(ready, engine, selection_start, ttft_slo)
                ready[:] = [request for request in ready if all(request is not item for item in selected)]
                selection_s = time.perf_counter() - selection_start
            result, batch = engine.execute(selected, f"b{batch_id:06d}")
            batch_id += 1
            batch.update(selection_reason=reason, scheduling_s=selection_s,
                         offset_s=selection_start - origin)
            batches.append(batch)
            with condition:
                for row, request in zip(result, selected):
                    rows.append(row)
                    accepted.pop(request.request_id, None)
                outstanding -= len(selected)
                condition.notify_all()
    except Exception:
        errors.append(traceback.format_exc())
        stop.set()
    finally:
        stop.set()
        with condition:
            condition.notify_all()
        incoming.put(None)
        for thread in threads:
            thread.join(timeout=10)
    end = time.perf_counter()
    for row in rows:
        if row["status"] == "completed":
            row.update(quality(row["answer"], lookup[row["question_id"]]["answers"]))
    with condition:
        for item in accepted.values():
            reject(item, end, "unfinished")
    existing = {row["request_id"] for row in rows}
    for item in trace:
        if item["request_id"] not in existing:
            reject(item, end, "not_delivered")
    rows.sort(key=lambda r: r["request_id"])
    assert len(rows) == len(trace) and len({row["request_id"] for row in rows}) == len(trace)
    elapsed = max(1e-9, end - origin)
    summary = summarize(rows, duration if duration is not None else elapsed, ttft_slo, e2e_slo)
    summary.update(variant=engine.variant, mode="open_loop" if duration is not None else "offline",
                   measured_elapsed_s=elapsed, batch_count=len(batches),
                   vision_images_encoded=sum(b["vision_images"] for b in batches),
                   vision_s=sum(b["vision_s"] for b in batches),
                   prefill_s=sum(b.get("prefill_s", 0) for b in batches),
                   preprocessing_s=sum(row.get("preprocess_s", 0) for row in rows),
                   scheduling_s=sum(b["scheduling_s"] for b in batches),
                   batch_shared_images=sum(b["batch_shared_images"] for b in batches),
                   cache=engine.cache.snapshot() if engine.cache else {},
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   cpu_rss_bytes=psutil.Process().memory_info().rss,
                   errors=errors, recorded_at=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat())
    for name, values in [("requests", rows), ("batches", batches), ("monitor", monitoring)]:
        (output / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"Experiment failed; inspect {output / 'summary.json'}")
    return summary
