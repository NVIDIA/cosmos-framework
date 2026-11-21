# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Install/verify the isolated source-checkout runtime for both native action families."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
from pathlib import Path

PROFILE = Path(__file__).resolve().parent
CHECKOUT = PROFILE.parents[2]
PYTHON_VERSION = "3.12.13"
UV_VERSION = "0.11.29"
SOURCES = {
    "Isaac-GR00T": {
        "url": "https://github.com/NVIDIA/Isaac-GR00T.git",
        "revision": "51d4c89f72fda44cbf77285c6a8114b52676b8a1",
        "package": "gr00t",
        "directories": ["model", "configs", "data"],
        "sha256": "819b113eb4b47483d759127c6ffbf9f397241fc479070daea2d929fd371d7b35",
        "pythonpath": ".",
    },
    "alpamayo": {
        "url": "https://github.com/NVlabs/alpamayo.git",
        "revision": "11a0e01c13a5622377c45ee37d653351453ec43b",
        "package": "src/alpamayo_r1",
        "directories": ["."],
        "sha256": "4363fee75791d3c465e7b1374451fafe5ea9b93a1af582ebe1251d60fb6371eb",
        "pythonpath": "src",
    },
}
CUDA_PACKAGES = {"torch", "torchcodec", "torchvision"}


def run(argv, *, capture=False, env=None):
    return subprocess.run(argv, check=True, text=True, capture_output=capture, env=env)


def requirements(path=PROFILE / "requirements.txt"):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if match is None:
            raise ValueError(f"native action closure requires exact version pins: {line!r}")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in result or name.startswith("cosmos-framework"):
            raise ValueError(f"duplicate or private framework distribution in native action closure: {name}")
        result[name] = version
    return result


def verify_source(root: Path, spec):
    """Check exact commit and package bytes before adding a public source to Python's path."""
    revision = run(["git", "-C", str(root), "rev-parse", "HEAD"], capture=True).stdout.strip()
    if revision != spec["revision"]:
        raise ValueError(f"native action source revision differs: {root}")
    if run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], capture=True).stdout.strip():
        raise ValueError(f"native action source checkout is modified: {root}")
    package = root / spec["package"]
    files = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in spec["directories"]
        for path in (package / directory).rglob("*.py")
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if not files or digest != spec["sha256"]:
        raise ValueError(f"native action source bytes differ: {root}; use the pinned unmodified public source")
    return {"revision": revision, "sha256": digest}


def source_paths(venv: Path):
    return [
        str(CHECKOUT),
        *[str(venv / "native-sources" / name / spec["pythonpath"]) for name, spec in SOURCES.items()],
    ]


def verify_environment(venv: Path):
    pins = requirements()
    source_records = {name: verify_source(venv / "native-sources" / name, spec) for name, spec in SOURCES.items()}
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(source_paths(venv)), "PYTHONNOUSERSITE": "1"}
    script = """
import importlib.metadata as m, importlib.util, json, platform, sys
pins=json.loads(sys.argv[1]); actual={name:m.version(name) for name in pins}
wrong={name:[expected,actual[name]] for name,expected in pins.items() if actual[name]!=expected}
if platform.python_version()!=sys.argv[2] or wrong:
 raise RuntimeError(f'native action runtime version mismatch: python={platform.python_version()}, packages={wrong}')
expected_paths=json.loads(sys.argv[3])
for name in ('cosmos_framework','gr00t','alpamayo_r1'):
 spec=importlib.util.find_spec(name)
 if spec is None: raise RuntimeError(f'missing native source package {name}')
 if expected_paths[name] not in list(spec.submodule_search_locations or []):
  raise RuntimeError(f'native package {name} resolved outside the declared source checkout')
print(json.dumps({'python':platform.python_version(),'versions':actual},sort_keys=True))
"""
    expected_paths = {
        "cosmos_framework": str(CHECKOUT / "cosmos_framework"),
        "gr00t": str(venv / "native-sources/Isaac-GR00T/gr00t"),
        "alpamayo_r1": str(venv / "native-sources/alpamayo/src/alpamayo_r1"),
    }
    checked = run(
        [str(venv / "bin/python"), "-c", script, json.dumps(pins), PYTHON_VERSION, json.dumps(expected_paths)],
        capture=True,
        env=env,
    )
    record = json.loads(checked.stdout)
    record["sources"] = source_records
    record["framework_checkout"] = str(CHECKOUT)
    record["framework_revision"] = run(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], capture=True).stdout.strip()
    record["requirements_sha256"] = hashlib.sha256((PROFILE / "requirements.txt").read_bytes()).hexdigest()
    return record


def activation_script(venv: Path) -> str:
    return "\n".join(
        [
            "# Native action source profile; does not change scheduler routing or shared libraries.",
            f"source {shlex.quote(str(venv / 'bin/activate'))}",
            f"export PYTHONPATH={shlex.quote(os.pathsep.join(source_paths(venv)))}",
            "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1",
            "",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", required=True, type=Path)
    parser.add_argument("--uv", default="uv", help="Bootstrap uv 0.11.29 executable")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("this native action profile is Linux x86_64/CUDA12.8 only")
    venv = args.venv.expanduser().resolve()
    if args.verify_only:
        print(json.dumps(verify_environment(venv), indent=2, sort_keys=True))
        return
    if venv.exists():
        raise FileExistsError(f"choose a new isolated environment; refusing to mutate {venv}")
    uv = shutil.which(args.uv)
    if uv is None or run([uv, "--version"], capture=True).stdout.split()[1] != UV_VERSION:
        raise ValueError("native action installation requires bootstrap uv==0.11.29 (--uv /path/to/uv)")
    pins = requirements()
    run([uv, "--no-config", "venv", str(venv), "--python", PYTHON_VERSION, "--python-preference", "only-managed"])
    python = str(venv / "bin/python")
    # Source-controlled exact pins make resolution unnecessary. Separate official
    # wheel indexes prevent CUDA wheel selection from changing other packages.
    for names, index in (
        (CUDA_PACKAGES, "https://download.pytorch.org/whl/cu128"),
        (set(pins) - CUDA_PACKAGES, "https://pypi.org/simple"),
    ):
        run(
            [
                uv,
                "--no-config",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                "--index-url",
                index,
                *[f"{name}=={pins[name]}" for name in sorted(names)],
            ]
        )
    for name, spec in SOURCES.items():
        root = venv / "native-sources" / name
        root.mkdir(parents=True)
        run(["git", "init", "--quiet", str(root)])
        run(["git", "-C", str(root), "remote", "add", "origin", spec["url"]])
        run(["git", "-C", str(root), "fetch", "--depth=1", "origin", spec["revision"]])
        run(["git", "-C", str(root), "checkout", "--detach", "--quiet", "FETCH_HEAD"])
        verify_source(root, spec)
    run([uv, "--no-config", "pip", "check", "--python", python])
    record = verify_environment(venv)
    (venv / "native-action-runtime.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (venv / "native-action-env.sh").write_text(activation_script(venv))
    print(f"Native action source environment installed. source {shlex.quote(str(venv / 'native-action-env.sh'))}")


if __name__ == "__main__":
    main()
