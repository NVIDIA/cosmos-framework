# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Catch regressions in the specific dependency removals covered by the license audit."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VLMEVALKIT = ROOT / ".agents/skills/evaluation/cosmos3/reasoner/vlmevalkit"
REMOVED = {"plyfile", "pyyaml-include", "polygon3", "distance", "levenshtein", "unidecode", "pymupdf"}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    declarations = {"uv.lock": {normalize(package["name"]) for package in lock["package"]}}
    for path in sorted(VLMEVALKIT.rglob("requirements*.txt")):
        names = set()
        for line in path.read_text().splitlines():
            name = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)", line)
            if name:
                names.add(normalize(name.group(1)))
        declarations[str(path.relative_to(ROOT))] = names

    failures = [f"{path}: {name}" for path, names in declarations.items() for name in sorted(names & REMOVED)]
    if failures:
        raise SystemExit("Previously removed dependencies reappeared:\n" + "\n".join(failures))
    print("Previously removed dependencies are absent from the Framework lockfile and VLMEvalKit requirements.")


if __name__ == "__main__":
    main()
