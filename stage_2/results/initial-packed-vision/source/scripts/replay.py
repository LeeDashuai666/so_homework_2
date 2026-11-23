#!/usr/bin/env python3
"""Start a clean result tree while reusing the existing model, data and venv."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "replays" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        parser.error("The output directory must be new; existing raw results are preserved")
    destination.mkdir(parents=True)
    for name in ["configs", "src", "tools", "scripts", "tests", "env"]:
        shutil.copytree(ROOT / name, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ["run.sh", "env.sh", "requirements.lock", "REPRODUCE.md"]:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, destination / name)
    for name in [".venv", "models"]:
        (destination / name).symlink_to(ROOT / name, target_is_directory=True)
    (destination / "data").mkdir()
    for name in ["processed", "raw"]:
        (destination / "data" / name).symlink_to(ROOT / "data" / name, target_is_directory=True)
    shutil.copytree(ROOT / "data/traces", destination / "data/traces")
    (destination / "logs").mkdir()
    (destination / "replay-origin.json").write_text(json.dumps({"source": str(ROOT),
                        "mode": "fresh result tree; shared immutable model/data and existing Python environment"}, indent=2) + "\n")
    print("Fresh replay workspace:", destination, flush=True)
    if not args.prepare_only:
        subprocess.run(["bash", str(destination / "run.sh"), "experiments"], check=True)
        subprocess.run(["bash", str(destination / "run.sh"), "test"], check=True)
        subprocess.run(["bash", str(destination / "run.sh"), "report"], check=True)


if __name__ == "__main__":
    main()
