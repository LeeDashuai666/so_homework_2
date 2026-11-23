#!/usr/bin/env python3
"""Shared local-only configuration and input preparation for VLM diagnostics."""

import hashlib
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT / "configs/vlm-runtime.json").read_text())


def sample_questions(config):
    path = ROOT / config["dataset_path"] / "questions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_processor(config, source=None):
    processor = AutoProcessor.from_pretrained(
        str(source or ROOT / config["model_path"]), local_files_only=True
    )
    image = processor.image_processor
    factor = image.patch_size * image.merge_size
    if factor != config["processor"]["spatial_factor"]:
        raise RuntimeError("Processor spatial factor does not match the runtime configuration")
    image.size = {"shortest_edge": config["processor"]["min_pixels"],
                  "longest_edge": config["processor"]["max_pixels"]}
    processor.tokenizer.padding_side = config["processor"]["padding_side"]
    if not processor.chat_template:
        raise RuntimeError("The model chat template is missing")
    return processor


def load_model(config):
    import torch
    from transformers import Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("The configured VLM runtime requires CUDA")
    if config["dtype"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured bfloat16 is not supported on this device")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(ROOT / config["model_path"]), local_files_only=True,
        dtype=getattr(torch, config["dtype"]), attn_implementation=config["attention_backend"],
        device_map={"": config["device"]}, low_cpu_mem_usage=True,
    ).eval()
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("The smoke profile requires all model parameters on the GPU")
    return model


def prepare_inputs(processor, questions, config):
    images, texts = [], []
    folder = ROOT / config["dataset_path"]
    for item in questions:
        with Image.open(folder / item["image_path"]) as image:
            images.append(image.convert("RGB"))
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": item["question"] + "\n" + config["smoke"]["prompt_suffix"]},
        ]}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    lengths = inputs["attention_mask"].sum(-1).tolist()
    tokens = (inputs["image_grid_thw"].prod(-1) // processor.image_processor.merge_size**2).tolist()
    if any(length > config["processor"]["max_input_tokens"] for length in lengths):
        raise RuntimeError("Prepared input exceeds the configured token guard")
    if any(count > config["processor"]["max_image_tokens"] for count in tokens):
        raise RuntimeError("Processor exceeded the configured visual token guard")
    if "mm_token_type_ids" not in inputs:
        raise RuntimeError("Processor did not provide multimodal token type IDs")
    return inputs, {"input_lengths": lengths, "visual_tokens": tokens,
                    "grid_thw": inputs["image_grid_thw"].tolist(),
                    "pixel_shape": list(inputs["pixel_values"].shape)}


def feature_identity(inputs, config, revision):
    pixels = inputs["pixel_values"].detach().cpu().contiguous()
    details = {
        "pixel_sha256": hashlib.sha256(pixels.numpy().tobytes()).hexdigest(),
        "pixel_shape": list(pixels.shape), "pixel_dtype": str(pixels.dtype),
        "grid_thw": inputs["image_grid_thw"].tolist(), "processor": config["processor"],
        "model_revision": revision, "dtype": config["dtype"], "backend": config["attention_backend"],
    }
    return hashlib.sha256(json.dumps(details, sort_keys=True).encode()).hexdigest()


def generate_answers(model, processor, inputs, config):
    import torch

    with torch.inference_mode():
        output = model.generate(**inputs, **config["generation"])
    torch.cuda.synchronize()
    tokens = output[:, inputs["input_ids"].shape[1]:].cpu()
    return {"answers": processor.batch_decode(tokens, skip_special_tokens=True),
            "token_ids": tokens.tolist()}
