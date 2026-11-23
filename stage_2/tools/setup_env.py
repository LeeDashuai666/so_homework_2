#!/usr/bin/env python3
"""Build a small project overlay without installing or changing the base stack."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]


def run(argv, **kwargs):
    print("Running:", " ".join(str(value) for value in argv), flush=True)
    return subprocess.run([str(value) for value in argv], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-python", type=Path)
    parser.add_argument("--profile", choices=("vlm", "rerank"), default="vlm")
    parser.add_argument("--offline", action="store_true", help="Use existing pip wheel cache only")
    args = parser.parse_args()
    requirements = ROOT / "env" / ("extra-requirements.lock" if args.profile == "vlm"
                                   else "extra-requirements-rerank.lock")
    inventory = json.loads((ROOT / "env/base-packages.json").read_text())
    base_python = args.base_python or Path(os.environ.get("STAGE2_BASE_PYTHON", inventory["python"]))
    inspect = """import sys,sysconfig,json,importlib.metadata as m
print(json.dumps({'python':sys.executable,'version':list(sys.version_info[:3]),
'site_packages':sysconfig.get_path('purelib'),
'packages':{d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name']}}))"""
    details = json.loads(subprocess.check_output([str(base_python), "-c", inspect], text=True))
    if details["version"][:2] != [3, 12]:
        raise RuntimeError("The recorded stack requires Python 3.12")
    changes = {
        name: {"expected": version, "found": details["packages"].get(name)}
        for name, version in inventory["packages"].items()
        if details["packages"].get(name) != version
    }
    if changes:
        raise RuntimeError("The reusable base changed; inspect before rebuilding: " + json.dumps(changes))
    target = ROOT / ".venv"
    if not target.exists():
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("Run this setup with Python 3.12")
        venv.EnvBuilder(with_pip=False).create(target)
    python = target / "bin/python"
    purelib = Path(subprocess.check_output(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip())
    if not purelib.resolve().is_relative_to(target.resolve()):
        raise RuntimeError("Refusing to install outside the project virtual environment")
    base_site = Path(details["site_packages"])
    if not base_site.is_dir() or base_site.resolve().is_relative_to(target.resolve()):
        raise RuntimeError("Invalid reusable base site-packages")
    (purelib / "stage2_reuse_base.pth").write_text(str(base_site) + "\n")
    pip_entry = (f"#!{python}\nimport sys\nfrom pip._internal.cli.main import main\n"
                 "if __name__ == '__main__':\n    sys.exit(main())\n")
    for name in ("pip", "pip3", "pip3.12"):
        entry = target / "bin" / name
        entry.write_text(pip_entry)
        entry.chmod(0o755)
    ninja_source = base_python.parent / "ninja"
    ninja_target = target / "bin/ninja"
    if ninja_source.exists() and not ninja_target.exists():
        ninja_target.symlink_to(ninja_source)
    perf_source = ROOT.parent / "stage_1/.tools/bin/perf"
    perf_target = target / "bin/perf"
    if perf_source.exists() and not perf_target.exists():
        perf_target.symlink_to(ROOT / "tools/perf.sh")
    uv = shutil.which("uv")
    if uv and not args.offline:
        cache_root = Path(os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache/uv")))
        cached = {path.name.removesuffix(".dist-info").lower().replace("_", "-")
                  for path in (cache_root / "archive-v0").glob("*/*.dist-info")}
        for line in requirements.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            spec = line.split()[0]
            name, version = spec.split("==")
            if f"{name.lower().replace('_', '-')}-{version}" not in cached:
                continue
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as requirement:
                requirement.write(line + "\n")
                requirement.flush()
                run([uv, "pip", "install", "--python", python, "--offline", "--no-deps",
                     "--require-hashes", "--link-mode", "hardlink", "-r", requirement.name])
    install = [python, "-m", "pip", "install", "--no-deps", "--require-hashes",
               "--only-binary=:all:", "--requirement", requirements]
    if args.offline:
        wheels = ROOT / "env/wheelhouse"
        if not wheels.is_dir():
            raise RuntimeError("Offline rebuild requires env/wheelhouse; see env/README.md")
        install.extend(["--no-index", "--find-links", wheels])
    run(install)
    run([python, "-m", "pip", "check"])
    manifest = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "mode": "project overlay with a referenced, unchanged base site-packages",
        "project_python": str(python), "base_python": str(base_python),
        "base_site_packages": str(base_site), "python_version": details["version"],
        "base_package_count": len(inventory["packages"]),
        "base_versions_verified": True,
        "profile": args.profile,
        "extra_requirements": str(requirements),
        "note": "Keep the referenced base environment; the project venv is not self-contained.",
    }
    (ROOT / "env/environment.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Ready. Activate with: source {ROOT / 'env.sh'}", flush=True)


if __name__ == "__main__":
    main()
