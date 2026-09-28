#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stamp versions.yaml values into skill files, or verify they are in sync.

Skills are standalone: they carry literal container URIs and wheel pins instead
of resolving ``versions.yaml`` at runtime. Each embedded literal is annotated
with the versions.yaml key it came from, so this script can re-stamp every
site on a release bump and CI can verify nothing drifted:

    container_image: cosmos-framework:local  # versions-key: images.containers.pyt
    export COSMOS_DS_IMAGE=cosmos-framework:local  # versions-key: images.containers.data_services

Rules enforced:
  * A line carrying ``# versions-key: <dotted.key>`` must contain exactly the
    value versions.yaml resolves for that key. ``stamp`` rewrites it; ``--check``
    fails on mismatch.
  * The key must exist in versions.yaml (unknown key = error in both modes).
  * Deliberately unmanaged pins are annotated ``# unpinned: <reason>`` (or
    ``<!-- unpinned: <reason> -->`` in markdown) and are skipped by the stray
    scan. Formats that cannot carry a trailing comment (Dockerfile ``ARG``
    lines, YAML values consumed verbatim) may put the annotation on the line
    immediately above the pin.
  * Stray scan: image references with an explicit tag, or nvidia-cosmos-* wheel
    pins, on lines with neither annotation are reported. ``.json`` files are
    exempt (JSON cannot carry annotations; pins there are recorded artifacts,
    not templates). The 7.1.0 stray backlog is cleared: CI runs
    ``--check --strict-strays``, so a new unannotated pin fails the pipeline.
  * Scan scope: model/data/execution/service resources, templates, and scripts (so pins embedded
    in CI helpers and test fixtures are covered too — a stale duplicate there is
    exactly how a marked pin can drift unnoticed). The versions-key tooling
    itself (``stamp_versions.py``, ``migrate-to-version-keys.py``) is skipped
    because its docstrings document the marker format with example pins.

versions.yaml is parsed with a minimal indentation-based reader (2-space
indents, scalar leaves) so this script has no third-party dependencies and can
run in both the GitLab CI image and the Jenkins release pod.

Usage:
    scripts/stamp_versions.py               # rewrite marked lines in place
    scripts/stamp_versions.py --check       # verify only; exit non-zero on drift
    scripts/stamp_versions.py --check --strict-strays
"""

from __future__ import annotations

import argparse
import os
import re
import sys

MARKER_RE = re.compile(r"(?:#|<!--)\s*versions-key:\s*([A-Za-z0-9_.]+)")
UNPINNED_RE = re.compile(r"(?:#|<!--)\s*unpinned:\s*\S")
# An image reference with an explicit tag (registry host / path : tag).
IMAGE_RE = re.compile(r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}/[A-Za-z0-9_./-]+:[A-Za-z0-9][A-Za-z0-9_.-]*")
LOCAL_IMAGE_RE = re.compile(r"(?<![A-Za-z0-9_./-])[A-Za-z0-9][A-Za-z0-9_./-]*:[A-Za-z0-9][A-Za-z0-9_.-]*(?!//)")
# A pinned wheel spec, optionally with extras: name[extra]==1.2.3 / name==1.2.3rc4
WHEEL_RE = re.compile(r"[A-Za-z0-9._-]+(?:\[[A-Za-z0-9_,-]+\])?==[A-Za-z0-9.]+")
# nvidia-cosmos-* wheels are the only release-cadenced wheels; strays scan just those.
STRAY_WHEEL_RE = re.compile(r"nvidia-cosmos-[a-z-]+(?:\[[A-Za-z0-9_,-]+\])?==[A-Za-z0-9.]+")
# Bare dotted key as the whole value (first-time stamping of key-form fields).
DOTTED_KEY_VALUE_RE = re.compile(r"^(\s*[A-Za-z_]+:\s*)([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)(\s*#.*)$")

SCAN_EXTENSIONS = {".md", ".yaml", ".yml", ".sh", ".py", ".config", ".json", ".txt"}

# The versions-key tooling documents the marker format with example image URIs
# and example ``# versions-key:`` comments in its own docstrings. Those examples
# are not live pins, so scanning them would raise spurious drift/stray reports —
# skip these files by basename.
SKIP_BASENAMES = {"stamp_versions.py", "migrate-to-version-keys.py"}


def parse_versions(path: str) -> dict[str, str]:
    """Flatten versions.yaml into {'images.containers.pyt': 'nvcr.io/...', ...}.

    Minimal reader for this file's known shape: nested mappings by 2-space
    indentation with scalar string leaves. Comments and blanks ignored.
    """
    flat: dict[str, str] = {}
    stack: list[tuple[int, str]] = []  # (indent, key)
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.split("#", 1)[0].rstrip()
            if not stripped.strip():
                continue
            indent = len(stripped) - len(stripped.lstrip())
            body = stripped.strip()
            if ":" not in body:
                continue
            key, _, value = body.partition(":")
            key, value = key.strip(), value.strip()
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if value:
                dotted = ".".join([k for _, k in stack] + [key])
                flat[dotted] = value
            else:
                stack.append((indent, key))
    return flat


def iter_skill_files(skills_dir: str):
    for root, dirs, files in os.walk(skills_dir):
        dirs[:] = [d for d in dirs if d != ".git"]
        for fname in files:
            if fname in SKIP_BASENAMES:
                continue
            if os.path.splitext(fname)[1] in SCAN_EXTENSIONS:
                yield os.path.join(root, fname)


def replace_value(line: str, new_value: str) -> tuple[str, bool]:
    """Replace the versioned value token on a marked line. Returns (line, ok)."""
    code = line.split("#", 1)[0]
    m = IMAGE_RE.search(code)
    if m:
        return line[: m.start()] + new_value + line[m.end() :], True
    m = LOCAL_IMAGE_RE.search(code)
    if m:
        return line[: m.start()] + new_value + line[m.end() :], True
    m = WHEEL_RE.search(code)
    if m:
        return line[: m.start()] + new_value + line[m.end() :], True
    m = DOTTED_KEY_VALUE_RE.match(line)
    if m:  # first-time stamp of a bare key-form value
        return f"{m.group(1)}{new_value}{m.group(3)}", True
    return line, False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; do not rewrite")
    ap.add_argument(
        "--strict-strays", action="store_true", help="unannotated image/wheel pins are errors, not warnings"
    )
    ap.add_argument("--versions-file", default="versions.yaml")
    ap.add_argument(
        "--resources-dir",
        "--skills-dir",
        dest="skills_dir",
        default=".",
        help="resource tree to scan; default scans all maintained component directories",
    )
    args = ap.parse_args()

    versions = parse_versions(args.versions_file)
    if not versions:
        print(f"ERROR: no keys parsed from {args.versions_file}", file=sys.stderr)
        return 2

    errors: list[str] = []
    strays: list[str] = []
    stamped = 0

    scan_dirs = (
        [d for d in ("models", "data", "execution", "inference-service", "templates", "scripts") if os.path.isdir(d)]
        if args.skills_dir == "."
        else [args.skills_dir]
    )
    all_files = [p for d in scan_dirs for p in iter_skill_files(d)]
    for path in sorted(all_files):
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()
        except (UnicodeDecodeError, OSError):
            continue

        changed = False
        for i, line in enumerate(lines):
            marker = MARKER_RE.search(line)
            if marker:
                key = marker.group(1)
                if key not in versions:
                    errors.append(f"{path}:{i + 1}: unknown versions-key '{key}'")
                    continue
                expected = versions[key]
                if expected in line.split("#", 1)[0]:
                    continue  # already correct
                if args.check:
                    errors.append(
                        f"{path}:{i + 1}: stale stamp for '{key}' — expected '{expected}'. "
                        f"Edit versions.yaml and run scripts/stamp_versions.py; do not hand-edit."
                    )
                else:
                    new_line, ok = replace_value(line, expected)
                    if not ok:
                        errors.append(f"{path}:{i + 1}: marked line has no recognizable value token to stamp")
                        continue
                    lines[i] = new_line
                    changed = True
                    stamped += 1
            elif not (
                UNPINNED_RE.search(line) or (i > 0 and UNPINNED_RE.search(lines[i - 1])) or path.endswith(".json")
            ):
                code = line.split("#", 1)[0]
                img = IMAGE_RE.search(code)
                if img and "nvcr.io" in img.group(0):
                    strays.append(f"{path}:{i + 1}: unmarked image pin '{img.group(0)}'")
                else:
                    whl = STRAY_WHEEL_RE.search(code)
                    if whl:
                        strays.append(f"{path}:{i + 1}: unmarked wheel pin '{whl.group(0)}'")

        if changed and not args.check:
            with open(path, "w", encoding="utf-8") as fh:
                fh.writelines(lines)

    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    if strays:
        level = "ERROR" if args.strict_strays else "WARN"
        for s in strays:
            print(f"{level}: {s}", file=sys.stderr)
        print(
            f"{level}: {len(strays)} unannotated pin(s). Mark release-managed values with "
            f"'# versions-key: <key>' or annotate one-offs with '# unpinned: <reason>'.",
            file=sys.stderr,
        )

    fatal = len(errors) + (len(strays) if args.strict_strays else 0)
    if args.check:
        if fatal == 0:
            print(f"OK: all versions-key stamps match {args.versions_file} ({len(strays)} stray warning(s)).")
        return 1 if fatal else 0
    print(f"Stamped {stamped} line(s) from {args.versions_file}. {len(strays)} stray warning(s).")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
