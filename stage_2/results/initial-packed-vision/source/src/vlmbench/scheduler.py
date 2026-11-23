"""Bounded-lookahead visual-cost and image-affinity scheduling."""


def estimate(request, engine):
    cfg = engine.experiment
    tier = engine.cache.peek(request.key) if engine.cache else "miss"
    visual = request.metadata["visual_tokens"][0]
    text = request.metadata["input_lengths"][0]
    vision = visual * cfg["scheduler_vision_cost_per_token_s"] if tier == "miss" else 0.0
    transfer = request.record.get("feature_bytes_estimate", visual * 2048 * 8) / 1024**2 * cfg["cpu_transfer_cost_per_mib_s"] if tier == "cpu" else 0.0
    return text * cfg["scheduler_prefill_cost_per_token_s"] + vision + transfer


def select(ready, engine, now, ttft_slo):
    """Only ready, already-arrived requests are visible; no future trace access."""
    cfg = engine.experiment
    window = list(ready[:cfg["lookahead"]])
    if not window:
        return [], "empty"
    affinity = engine.options["scheduler"] == "affinity"
    aged = now - window[0].scheduled >= cfg["age_guard_slo_fraction"] * ttft_slo
    first = window[0] if not affinity or aged else min(window, key=lambda r: (estimate(r, engine), r.scheduled))
    selected = [first]
    reason = "age_guard" if affinity and aged else "affinity_cost" if affinity else "fifo"
    while len(selected) < cfg["batch_size"]:
        remaining = [r for r in window if all(r is not old for old in selected)]
        remaining = [r for r in remaining if sum(x.metadata["visual_tokens"][0] for x in selected) + r.metadata["visual_tokens"][0] <= cfg["max_batch_visual_tokens"]]
        if not remaining:
            break
        if not affinity:
            candidate = remaining[0]
        else:
            keys = {r.key for r in selected}
            candidate = min(remaining, key=lambda r: (r.key not in keys,
                                                      estimate(r, engine) + abs(r.metadata["input_lengths"][0] - first.metadata["input_lengths"][0]) * cfg["scheduler_prefill_cost_per_token_s"],
                                                      r.scheduled))
        selected.append(candidate)
    return selected, reason
