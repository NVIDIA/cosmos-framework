# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Install/verify the isolated source-checkout runtime for both native action families."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROFILE = Path(__file__).resolve().parent
CHECKOUT = PROFILE.parents[2]
PYTHON_VERSION = "3.12.13"
UV_VERSION = "0.11.29"


def pinned_constants(relative: str) -> dict[str, str]:
    """Read module-level string pins from a Cosmos source file without importing Cosmos."""
    tree = ast.parse((CHECKOUT / relative).read_text())
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


_GR00T = pinned_constants("cosmos_framework/posttrain/model/gr00t/adaptation.py")
_ALPAMAYO = pinned_constants("cosmos_framework/posttrain/model/alpamayo/sft.py")
SOURCES = {
    "Isaac-GR00T": {
        "url": "https://github.com/NVIDIA/Isaac-GR00T.git",
        "revision": _GR00T["ISAAC_GR00T_REVISION"],
        "pythonpath": ".",
    },
    "alpamayo": {
        "url": "https://github.com/NVlabs/alpamayo.git",
        "revision": _ALPAMAYO["REFERENCE_REVISION"],
        "pythonpath": "src",
    },
}
CUDA_PACKAGES = {"torch", "torchcodec", "torchvision"}


# Generous bound for one package download/install or source fetch; a hung network call fails.
COMMAND_TIMEOUT_SECONDS = 3600


def run(argv, *, capture=False, env=None):
    try:
        return subprocess.run(
            argv, check=True, text=True, capture_output=capture, env=env, timeout=COMMAND_TIMEOUT_SECONDS
        )
    except subprocess.CalledProcessError as error:
        if capture and error.stderr:
            sys.stderr.write(error.stderr)
        raise


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
            raise ValueError(f"duplicate or cosmos-framework distribution in native action requirements: {name}")
        result[name] = version
    return result


def verify_source(root: Path, spec):
    """Check the exact commit and an unmodified checkout before adding a public source to Python's path.

    The native loaders compare only this commit at runtime (cosmos_framework/posttrain/utils/native_source.py).
    """
    revision = run(["git", "-C", str(root), "rev-parse", "HEAD"], capture=True).stdout.strip()
    if revision != spec["revision"]:
        raise ValueError(f"native action source revision differs: {root}")
    if run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], capture=True).stdout.strip():
        raise ValueError(f"native action source checkout is modified: {root}")
    return {"revision": revision}


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
            "# Native action source profile; does not change shared libraries.",
            f"source {shlex.quote(str(venv / 'bin/activate'))}",
            f"export PYTHONPATH={shlex.quote(os.pathsep.join(source_paths(venv)))}",
            "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1",
            "",
        ]
    )


def build(uv: str, venv: Path, pins: dict[str, str]) -> None:
    run(
        [
            uv,
            "--no-config",
            "venv",
            str(venv),
            "--relocatable",
            "--python",
            PYTHON_VERSION,
            "--python-preference",
            "only-managed",
        ]
    )
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
    # Build next to the target and rename on success, so a failed install leaves nothing behind
    # and can simply be rerun. --relocatable keeps scripts and activation valid after the rename.
    staging = venv.with_name(f".{venv.name}.partial-{os.getpid()}")
    try:
        build(uv, staging, pins)
        record = verify_environment(staging)
        (staging / "native-action-runtime.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        (staging / "native-action-env.sh").write_text(activation_script(venv))
        os.replace(staging, venv)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"Native action source environment installed. source {shlex.quote(str(venv / 'native-action-env.sh'))}")


if __name__ == "__main__":
    main()
