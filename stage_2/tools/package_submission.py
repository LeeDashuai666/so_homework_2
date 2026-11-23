#!/usr/bin/env python3
"""Package all project rounds and verify every archived payload by SHA256."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    audit = json.loads((ROOT / "results/submission-audit/checks.json").read_text())
    assert audit["passed"], "Run tools/check_submission.py before packaging"
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    selected = {}
    trees = ["src", "scripts", "tools", "tests", "configs", "env", "report", "results",
             "logs", "plan", "docs", "experiments/cross_layer_v2", "experiments/framework_compare",
             "experiments/ai_cache_tuning", "data/processed/docvqa", "data/traces", "data/raw/docvqa-smoke"]
    omitted = {"cache", ".cache", "__pycache__", ".pytest_cache", ".venv"}
    for directory in trees:
        for path in sorted((ROOT / directory).rglob("*")):
            relative = path.relative_to(ROOT)
            if path.is_file() and not (set(relative.parts) & omitted) and path.suffix != ".pyc":
                selected[(Path("stage_2") / relative).as_posix()] = path
    for name in ["README.md", "REPRODUCE.md", "requirements.lock", "run.sh", "env.sh"]:
        selected["stage_2/" + name] = ROOT / name
    # The document renderer uses this exact style reference; the rest of stage_1 is excluded.
    reference = ROOT.parent / "stage_1/docs/设计文档.docx"
    if reference.is_file():
        selected["stage_1/docs/设计文档.docx"] = reference
    required = audit["checks"]["required-files"]["files"]
    assert all("stage_2/" + name in selected for name in required)
    records = {name: {"bytes": path.stat().st_size, "sha256": digest(path)}
               for name, path in sorted(selected.items())}
    inventory = dist / "current-submission-manifest.json"
    inventory.write_text(json.dumps({
        "created_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "scope": "Current design, second-round ablation/framework evaluation, plus preserved first-round evidence",
        "files": records,
        "excluded": [".venv", "model weights", "Docker image binaries", "shared HF cache/full Parquet shards",
                     "compile caches", "dist", "replays", "unrelated legacy RAG datasets"],
        "reproduction": "stage_2/REPRODUCE.md",
        "first_project_reference": "Optional local stage_1/docs/设计文档.docx serves only as the original styling reference",
        "manifest_itself": "Archived as stage_2/submission-manifest.json; not self-hashed",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = dist / "多模态推理系统优化-最终提交包.tar.gz"
    temporary = dist / (archive.name + ".partial")
    with tarfile.open(temporary, "w:gz", compresslevel=5) as bundle:
        for name, path in sorted(selected.items()):
            bundle.add(path, arcname=name, recursive=False)
        bundle.add(inventory, arcname="stage_2/submission-manifest.json", recursive=False)
    seen = set()
    with tarfile.open(temporary, "r:gz") as bundle:
        for member in bundle:
            assert member.isfile(), "Archive contains a non-file entry: " + member.name
            assert member.name not in seen and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
            seen.add(member.name)
            with bundle.extractfile(member) as stream:
                value = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            expected = ({"bytes": inventory.stat().st_size, "sha256": digest(inventory)}
                        if member.name == "stage_2/submission-manifest.json" else records[member.name])
            assert member.size == expected["bytes"] and value.hexdigest() == expected["sha256"], member.name
    assert seen == set(records) | {"stage_2/submission-manifest.json"}
    temporary.replace(archive)
    checksum = digest(archive)
    (dist / "最终提交包.SHA256SUMS").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    result = {"passed": True, "archive": archive.name, "sha256": checksum,
              "archive_bytes": archive.stat().st_size, "payload_files": len(records),
              "archive_members": len(seen), "all_payload_sizes_and_sha256_verified": True,
              "original_first_round_archive_preserved": (dist / "多模态推理系统优化-实验提交包.tar.gz").is_file()}
    (dist / "current-archive-check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
