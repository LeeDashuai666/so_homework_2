"""Answer quality and request-level experiment statistics."""

import math
import unicodedata
import numpy as np


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def edit_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def quality(prediction, references):
    prediction = normalize(prediction)
    scores = []
    for answer in references:
        answer = normalize(answer)
        denominator = max(len(answer), len(prediction))
        distance = edit_distance(prediction, answer) / denominator if denominator else 0.0
        scores.append(1.0 - distance if distance < 0.5 else 0.0)
    return {"anls": max(scores, default=0.0),
            "em": float(any(prediction == normalize(answer) for answer in references))}


def quantiles(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {f"p{p}": float(np.percentile(values, p)) if values else None for p in [50, 95, 99]}


def summarize(rows, seconds, ttft_slo, e2e_slo):
    completed = [r for r in rows if r["status"] == "completed"]
    good = [r for r in completed if r["ttft_s"] <= ttft_slo and r["latency_s"] <= e2e_slo]
    return {
        "arrivals": len(rows), "completed": len(completed),
        "rejected": sum(r["status"] == "rejected" for r in rows),
        "failed": sum(r["status"] not in ("completed", "rejected") for r in rows),
        "good_requests": len(good), "window_s": seconds,
        "goodput_qps": len(good) / seconds if seconds else None,
        "completed_qps": len(completed) / seconds if seconds else None,
        "slo_attainment": len(good) / len(rows) if rows else 0.0,
        "ttft_s": quantiles(r.get("ttft_s") for r in completed),
        "latency_s": quantiles(r.get("latency_s") for r in completed),
        "queue_s": quantiles(r.get("queue_s") for r in completed),
        "sender_lag_s": quantiles(r.get("sender_lag_s") for r in rows),
        "anls": float(np.mean([r["anls"] for r in completed])) if completed else None,
        "em": float(np.mean([r["em"] for r in completed])) if completed else None,
        "output_tokens": sum(r.get("output_tokens", 0) for r in completed),
        "empty_answers": sum(not r.get("answer", "").strip() for r in completed),
    }
