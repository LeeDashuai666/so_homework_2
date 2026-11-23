#!/usr/bin/env python3
"""Run a local document-image question or one of the verified DocVQA samples."""

import argparse
import json
from pathlib import Path

import torch

from vlm_runtime import (configuration, generate_answers, load_model, load_processor,
                         prepare_inputs, sample_questions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()
    if bool(args.image) != bool(args.question):
        parser.error("--image and --question must be supplied together")
    config = configuration()
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    if args.image:
        if not args.image.is_file():
            parser.error("The input image does not exist")
        question = {"image_path": str(args.image.resolve()), "question": args.question,
                    "question_id": "local-user-input", "answers": []}
    else:
        samples = sample_questions(config)
        if not 0 <= args.sample < len(samples):
            parser.error(f"sample must be between 0 and {len(samples) - 1}")
        question = samples[args.sample]
    processor = load_processor(config)
    inputs, prepared = prepare_inputs(processor, [question], config)
    model = load_model(config)
    result = generate_answers(model, processor, inputs.to(config["device"]), config)
    print(json.dumps({"question": question["question"], "answer": result["answers"][0],
                      "input": prepared, "dtype": config["dtype"],
                      "attention_backend": config["attention_backend"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
