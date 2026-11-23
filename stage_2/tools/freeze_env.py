#!/usr/bin/env python3
"""Record the effective environment and lock the project's dependency closure."""

from datetime import datetime
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
DIRECT = ["torch", "torchvision", "pillow", "transformers", "datasets", "numpy", "pandas",
          "matplotlib", "safetensors", "accelerate", "fastapi", "uvicorn",
          "z3-solver", "sentencepiece", "psutil", "pytest", "ninja"]


def main():
    pending = [(name, frozenset()) for name in DIRECT]
    visited, versions = set(), {}
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        if (name, extras) in visited:
            continue
        visited.add((name, extras))
        distribution = metadata.distribution(name)
        versions[name] = distribution.version
        for text in distribution.requires or []:
            dependency = Requirement(text)
            if dependency.marker and not any(dependency.marker.evaluate({"extra": extra})
                                             for extra in {"", *extras}):
                continue
            pending.append((dependency.name, frozenset(dependency.extras)))
    contents = ["# Python 3.12 / Linux x86_64; generated from the tested project environment.",
                "# Exact versions, not a portable binary or driver image.",
                "--extra-index-url https://download.pytorch.org/whl/cu130"]
    contents.extend(f"{name}=={version}" for name, version in sorted(versions.items()))
    (ROOT / "requirements.lock").write_text("\n".join(contents) + "\n")
    packages = {}
    for name in sorted(versions):
        distribution = metadata.distribution(name)
        packages[name] = {"version": distribution.version, "location": str(distribution.locate_file(""))}
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
              "python": sys.executable, "direct_requirements": DIRECT, "packages": packages}
    (ROOT / "env/effective-packages.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Locked {len(packages)} required packages; unrelated base extras excluded.")


if __name__ == "__main__":
    main()
