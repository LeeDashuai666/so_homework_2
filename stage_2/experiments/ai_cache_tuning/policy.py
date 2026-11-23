#!/usr/bin/env python3
"""Select one encoder-cache trial on CPU with a pinned, local language model."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent

SYSTEM = (
    "You select one experiment for vLLM's multimodal encoder cache. "
    "The visual encoder results are cached in units of visual tokens. "
    "Requests to the pages in the supplied working set alternate across pages. "
    "A revisit can avoid encoding when its encoder result remains resident or APC skips the image segment. "
    "Use the baseline's encoder calls, TTFT and memory and the distribution of page sizes. "
    "Choose the smallest offered capacity that is likely to hold the active visual working set "
    "within the peak allocated memory limit and remove repeat encoder work. "
    "Only one trial remains. Answer using the corresponding single letter."
)


def choose(model_path, observation):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).eval()
    loaded = time.perf_counter()
    options = {chr(65 + i): capacity for i, capacity in enumerate(observation["candidates"])}
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps({**observation, "choices": options}, sort_keys=True)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    token_ids = {label: tokenizer.encode(label, add_special_tokens=False) for label in options}
    if any(len(tokens) != 1 for tokens in token_ids.values()):
        raise RuntimeError("The frozen policy requires one token per candidate label")
    with torch.inference_mode():
        inference_started = time.perf_counter()
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        scores = {label: float(logits[tokens[0]]) for label, tokens in token_ids.items()}
        inference_s = time.perf_counter() - inference_started
    label = max(scores, key=scores.__getitem__)
    return {
        "prompt": prompt, "observation": observation, "prompt_tokens": ids.shape[1],
        "choices": options, "raw_choice_logits": scores, "candidate_token_ids": token_ids,
        "choice": label, "capacity": options[label], "model_load_s": loaded - model_started,
        "decision_s": inference_s, "model_and_decision_s": time.perf_counter() - model_started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "device": "cpu", "dtype": "bfloat16", "threads": 2,
                        "decoding": "one greedy token constrained to candidate letters", "thinking": False},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--observation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = choose(args.model, json.loads(args.observation.read_text()))
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("AI selected", result["capacity"], "tokens in", round(result["decision_s"], 3), "s", flush=True)
