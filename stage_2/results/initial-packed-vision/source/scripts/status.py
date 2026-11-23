#!/usr/bin/env python3
"""Read actual completed units and the active GPU run without touching it."""

import json
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
values = {}
for phase, pattern, total in [("quality", "*/summary.json", 5), ("main", "load-*/*/summary.json", 45),
                               ("auxiliary", "*/*/summary.json", 15)]:
    completed = []
    for path in sorted((ROOT / "results" / phase).glob(pattern)):
        summary = json.loads(path.read_text())
        completed.append({"unit": str(path.parent.relative_to(ROOT / "results")),
                          "completed": summary["completed"], "failed": summary["failed"],
                          "goodput_qps": summary["goodput_qps"], "anls": summary["anls"]})
    values[phase] = {"completed_units": len(completed), "planned_units": total, "units": completed}
active = []
for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
    try:
        command = process.info["cmdline"] or []
        if any(str(ROOT / "scripts/experiment.py") == item for item in command):
            output = command[command.index("--output") + 1] if "--output" in command else None
            active.append({"pid": process.pid, "output": output, "created_at_unix": process.info["create_time"]})
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
values["active_gpu_units"] = active
print(json.dumps(values, ensure_ascii=False, indent=2))
