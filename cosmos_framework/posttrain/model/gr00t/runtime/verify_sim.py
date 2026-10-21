# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Verify the frozen simulator release assets during image assembly."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    build = Path("/build")
    site = Path("/opt/venv/libero/lib/python3.12/site-packages")
    expected = json.loads((build / "sim-packages.json").read_text())
    actual = {item.metadata["Name"]: item.version for item in importlib.metadata.distributions(path=[str(site)])}
    if actual != expected:
        raise RuntimeError(f"frozen simulator package inventory differs: expected={expected}, actual={actual}")
    files = json.loads((build / "sim-files.json").read_text())

    def verify(item: tuple[str, str]) -> None:
        name, digest = item
        path = site / name
        if not path.resolve().is_relative_to(site):
            raise RuntimeError(f"frozen simulator file escapes its environment: {name}")
        with path.open("rb") as stream:
            actual_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_digest != digest:
            raise RuntimeError(f"frozen simulator file differs: {name}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for _ in tqdm(pool.map(verify, files.items()), total=len(files), desc="Verify frozen simulator"):
            pass
    check = subprocess.run(
        ["/opt/posttrain/bin/uv", "--no-config", "pip", "check", "--python", "/opt/venv/libero/bin/python"],
        capture_output=True,
        text=True,
        check=False,
    )
    output = check.stdout + check.stderr
    conflict = "The package `contourpy` requires `numpy>=2.0`, but `1.26.4` is installed"
    # This release preserves the measured package set, including its one audited metadata conflict.
    if check.returncode != 1 or "Found 1 incompatibility" not in output or conflict not in output:
        raise RuntimeError(f"simulator dependency audit differs from the qualified release:\n{output}")
    print(output, end="")
    Path("/opt/posttrain/sim-audit.json").write_text(
        json.dumps(
            {
                "packages": len(actual),
                "files": len(files),
                "pip_check_passed": False,
                "inherited_conflicts": [conflict],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
