#!/usr/bin/env python3
"""Audit deliverables using raw-record checks, CPU tests and existing SMT2 files."""
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/submission-audit"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    commands = [
        ("first-round-integrity", [sys.executable, str(ROOT / "scripts/verify_results.py")]),
        ("development-analysis", [sys.executable, str(ROOT / "experiments/cross_layer_v2/analyze_pilot.py")]),
        ("framework-analysis", [sys.executable, str(ROOT / "tools/analyze_framework_compare.py")]),
        ("ai-cache-replay", [sys.executable, str(ROOT / "experiments/ai_cache_tuning/replay.py"),
                             "--report", "--output", str(ROOT / "experiments/ai_cache_tuning/results/replay")]),
        ("cross-layer-formal", [sys.executable, str(ROOT / "experiments/cross_layer_v2/formal/verify.py"), "--replay"]),
        ("cpu-tests", [sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")]),
    ]
    checks = {}
    for name, argv in commands:
        logfile = OUT / (name + ".log")
        with logfile.open("w", encoding="utf-8") as stream:
            result = subprocess.run(argv, cwd=ROOT.parent, stdout=stream, stderr=subprocess.STDOUT)
        checks[name] = {"passed": result.returncode == 0, "returncode": result.returncode,
                        "argv": argv, "log": str(logfile.relative_to(ROOT))}
        print(name, "PASS" if result.returncode == 0 else "FAIL", flush=True)

    import z3
    replay = {}
    for filename, expected in [("leases-safe.smt2", "unsat"), ("leases-unsafe.smt2", "sat")]:
        path = ROOT / "results/verification" / filename
        solver = z3.Solver()
        solver.from_file(str(path))
        actual = str(solver.check())
        replay[filename] = {"actual": actual, "expected": expected,
                            "passed": actual == expected, "sha256": sha(path)}
    checks["first-round-smt-replay"] = {
        "passed": all(item["passed"] for item in replay.values()),
        "z3_solver_package": importlib.metadata.version("z3-solver"), "results": replay,
        "scope": "First-round bounded feature-lease model; current prototype proof is checked separately",
    }
    document = read(ROOT / "docs/设计文档.检查.json")
    hashes = {f"docs/设计文档.{ext}": sha(ROOT / f"docs/设计文档.{ext}")
              for ext in ["md", "docx"]}
    checks["current-document"] = {
        "passed": (hashes["docs/设计文档.md"] == document["source_sha256"]
                   and hashes["docs/设计文档.docx"] == document["docx_sha256"]
                   and sha(ROOT / "experiments/framework_compare/results/analysis/analysis.json") == document["framework_analysis_sha256"]
                   and document["pdf_text_matches_docx"]
                   and document["numerical_tables_match_audited_data"]),
        "sha256": hashes, "pages": document["pdf_pages"],
        "text_and_layout_checks": "Markdown and Word hashes match the audited document pair",
    }
    runtime_path = ROOT / "experiments/cross_layer_v2/results/runtime-check.json"
    runtime = read(runtime_path)
    checks["gpu-invariants-record"] = {
        "passed": runtime["passed"] and all(sha(ROOT / "experiments/cross_layer_v2" / name) == digest for name, digest in runtime["source_sha256"].items()),
        "evidence": str(runtime_path.relative_to(ROOT)),
        "scope": "Previously measured GPU invariants; source hashes rechecked without rerunning GPU inference",
    }
    required = [
        "docs/设计文档.docx", "docs/设计文档.md",
        "docs/AI使用说明.md", "REPRODUCE.md", "requirements.lock", "env.sh",
        "docs/课程要求自查.md", "data/README.md", "tools/package_submission.py",
        "experiments/cross_layer_v2/formal/verify.py", "experiments/cross_layer_v2/formal/results/summary.json",
        "experiments/cross_layer_v2/formal/results/形式化验证报告.md",
        "experiments/cross_layer_v2/runtime.py", "experiments/cross_layer_v2/run.sh",
        "experiments/cross_layer_v2/results/pilot/timing-requests.jsonl",
        "experiments/framework_compare/README.md", "experiments/framework_compare/run.sh",
        "experiments/framework_compare/protocol.lock.json", "experiments/framework_compare/options.lock.json",
        "experiments/framework_compare/environment-host-effective.txt",
        "experiments/framework_compare/environment-container.txt",
        "experiments/framework_compare/hardware-at-formal-start.json",
        "experiments/framework_compare/data/evaluation.jsonl",
        "experiments/framework_compare/框架对比实验报告.md",
        "experiments/framework_compare/results/analysis/means.csv",
        "experiments/ai_cache_tuning/run.sh", "experiments/ai_cache_tuning/README.md",
        "experiments/ai_cache_tuning/results/replay/decision.json",
        "experiments/ai_cache_tuning/results/replay/summary.json",
    ]
    checks["required-files"] = {"passed": all((ROOT / name).is_file() for name in required),
                                "files": {name: (ROOT / name).is_file() for name in required}}
    client = {}
    for name, argv in [("codex", ["codex", "--version"]), ("node", ["node", "--version"])]:
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=10)
            client[name] = result.stdout.strip() if result.returncode == 0 else "unavailable"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            client[name] = "unavailable (not required for experiment replay)"
    analysis = read(ROOT / "experiments/framework_compare/results/analysis/analysis.json")
    formal = read(ROOT / "experiments/cross_layer_v2/formal/results/summary.json")
    result = {
        "checked_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "audit_script_sha256": sha(Path(__file__).resolve()),
        "passed": all(item["passed"] for item in checks.values()), "checks": checks,
        "current_measurement": {name: analysis[name] for name in ["trials", "scene_blocks", "requests", "pages", "questions", "formal_quality_passed"]},
        "bonus": {"automated_execution": True, "ai_assistance_documented": True,
                  "formal_verification": "current prefix/KV/CUDA Graph state contracts, plus first-round lease model"},
        "current_formal_verification": {key: formal[key] for key in ["passed", "safe_unsat", "negative_sat", "nonvacuity_sat_checks", "scope"]},
        "ai_development_tools_at_audit": client,
        "not_executed_in_this_audit": ["fresh GPU matrix", "fresh-machine environment installation", "Word/PDF regeneration"],
    }
    (OUT / "checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": len(checks), "record": str(OUT / "checks.json")}, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
