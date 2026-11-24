"""Freeze one local Qwen cache policy decision from development observations."""
import argparse
import importlib.util
from pathlib import Path

from bench import CANDIDATES, HERE, digest, read, write
from analyze import summarize

BASE = HERE.parent / "ai_cache_tuning"
SYSTEM = (
    "Select one of the three cost-based cache configurations for Qwen3-VL document question answering. "
    "The target is mixed revisits and sequential sessions with equal weight. "
    "All candidates use the same 320 MiB persistent payload limit; visual cache preserves cheaper features, "
    "and prefix KV skips more work when revisits are close. "
    "Use the reported development measurements. Choose the smallest value of mean_scene_ttft_ms when "
    "both scenes have identical output tokens to the visual control. Prefer the smaller p95 on ties. "
    "Reply with the single letter assigned to the chosen configuration."
)


def main(output, model):
    protocol = read(output / "protocol.json")
    lock = read(BASE / "policy-model.lock.json")
    for name, sha in lock["sha256"].items():
        assert digest(model / name) == sha, name
    dev = summarize(output / "dev-requests.jsonl", protocol["seeds"]["dev"])
    by_config = {}
    for name in CANDIDATES:
        groups = [dev[scene][name] for scene in ("mixed", "session")]
        by_config[name] = {
            "mean_scene_ttft_ms": round(sum(g["ttft_mean_ms"] for g in groups) / 2, 2),
            "mean_scene_p95_ms": round(sum(g["ttft_p95_ms"] for g in groups) / 2, 2),
            "identical_to_visual": all(g["token_equal"] == g["n"] for g in groups),
            "prefix_hits_by_scene": [g["prefix_hits_per_trace"] for g in groups],
        }
    observation = {"candidates": CANDIDATES, "dev_only": by_config, "source_documents": len(protocol["sources"]["dev"]),
                   "budget_mib": protocol["cache_budget_mib"], "scenario_order": ["mixed", "session"]}
    write(output / "observation.json", observation)
    spec = importlib.util.spec_from_file_location("policy_model", BASE / "policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SYSTEM = SYSTEM
    result = module.choose(model, observation)
    result["configuration"] = result.pop("capacity")
    eligible = {name: g for name, g in by_config.items() if g["identical_to_visual"]}
    assert eligible, "No candidate passed the token check on development questions."
    result["rule_configuration"] = min(eligible, key=lambda name: (eligible[name]["mean_scene_ttft_ms"],
                                                                    eligible[name]["mean_scene_p95_ms"]))
    result["model_lock"] = lock
    result["development_sha256"] = digest(output / "dev-requests.jsonl")
    result["protocol_sha256"] = digest(output / "protocol.json")
    assert not (output / "decision.json").exists()
    write(output / "decision.json", result)
    print("AI decision:", result["configuration"], "deterministic development rule:", result["rule_configuration"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    main(args.output, args.model)
