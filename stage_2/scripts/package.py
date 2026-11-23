#!/usr/bin/env python3
"""Package the reviewable experiment, excluding reusable binary/model caches."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert (ROOT / "report/研究报告.html").is_file(), "Generate the verified report first"
    assert json.loads((ROOT / "results/verification/results-integrity.json").read_text())["formal_runs"] == 45
    integrity = json.loads((ROOT / "results/verification/results-integrity.json").read_text())
    assert integrity["auxiliary_runs"] == 18
    subprocess.run([sys.executable, str(ROOT / "tools/prepare_vlm_assets.py"), "--offline", "--model-only"], check=True)
    status = subprocess.check_output([sys.executable, str(ROOT / "scripts/status.py")], text=True)
    (ROOT / "logs/status-latest.json").write_text(status)
    (ROOT / "env/experiment-delivery.json").write_text(json.dumps({
        "recorded_at": datetime.now(timezone.utc).isoformat(), "complete": True,
        "main_runs": integrity["formal_runs"], "auxiliary_runs": integrity["auxiliary_runs"],
        "quality_questions_per_variant": 558, "quality_gate_passed": True,
        "full_integrity_passed": integrity["passed"], "protocol_sha256": integrity["protocol_sha256"],
        "optimization_layers": ["model visual computation graph", "service scheduling", "GPU/CPU feature storage"],
        "visual_execution": "canonical per-image BF16; language batches up to two",
        "initial_quality_failure_preserved": "results/initial-packed-vision",
        "environment": "unchanged existing CUDA/PyTorch stack with project overlay",
        "report": "report/研究报告.html", "reproduction": "REPRODUCE.md"}, ensure_ascii=False, indent=2) + "\n")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    files = []
    for name in ["src", "scripts", "tools", "tests", "configs", "env", "report", "results", "logs", "data/processed/docvqa", "data/traces", "plan"]:
        for path in sorted((ROOT / name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append(path)
    for name in ["README.md", "REPRODUCE.md", "requirements.lock", "run.sh", "env.sh"]:
        files.append(ROOT / name)
    files = sorted(set(files))
    records = {str(path.relative_to(ROOT)): {"bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files}
    inventory = dist / "submission-manifest.json"
    inventory.write_text(json.dumps({"files": records, "excluded": [".venv", "models", "shared HF cache", "replays", "legacy RAG datasets"],
            "rebuild_assets": "Use configs/experiment-assets.lock.json and ./stage_2/run.sh prepare-data"}, ensure_ascii=False, indent=2) + "\n")
    archive = dist / "多模态推理系统优化-实验提交包.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=5) as bundle:
        for path in files:
            bundle.add(path, arcname=str(Path("stage_2") / path.relative_to(ROOT)), recursive=False)
        bundle.add(inventory, arcname="stage_2/submission-manifest.json", recursive=False)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dist / "SHA256SUMS").write_text(f"{checksum}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": checksum,
                      "files": len(files)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
