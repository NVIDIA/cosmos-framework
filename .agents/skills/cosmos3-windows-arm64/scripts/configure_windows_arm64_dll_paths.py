#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Install the per-venv DLL search-path bootstrap required by TorchCodec.

Python 3.13 on Windows does not reliably use PATH for extension-module DLL
dependencies. This script writes a one-line .pth file into the selected venv's
site-packages directory. It keeps os.add_dll_directory() handles alive for the
lifetime of each interpreter, including worker and console-script processes.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Iterable


PTH_FILENAME = "_win_tao_dlldirs.pth"


def _unique_existing(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    seen: set[pathlib.Path] = set()
    existing: list[pathlib.Path] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        existing.append(resolved)
    return existing


def candidate_directories(venv: pathlib.Path, extra: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    program_files = pathlib.Path(os.environ.get("ProgramFiles", r"C:\\Program Files"))
    cuda_root = program_files / "NVIDIA GPU Computing Toolkit" / "CUDA"
    cuda_paths = sorted(cuda_root.glob("v*/bin/arm64"), reverse=True) if cuda_root.is_dir() else []
    cudnn_paths = (
        sorted(program_files.glob("cudnn-windows-arm64-*_cuda13-archive/bin/arm64"), reverse=True)
        if program_files.is_dir()
        else []
    )
    return _unique_existing(
        [
            venv / "Lib" / "site-packages" / "torch" / "lib",
            *cuda_paths,
            *cudnn_paths,
            program_files / "gstreamer" / "1.0" / "msvc_arm64" / "bin",
            *extra,
        ]
    )


def pth_content(directories: Iterable[pathlib.Path]) -> str:
    literals = ", ".join(repr(str(path)) for path in directories)
    return (
        "import os, sys; globals().setdefault('_win_tao_handles', "
        f"[os.add_dll_directory(x) for x in [{literals}] "
        "if sys.platform == 'win32' and os.path.isdir(x)])\n"
    )


def configure(
    *, venv: pathlib.Path, extra: Iterable[pathlib.Path], force: bool, dry_run: bool
) -> tuple[pathlib.Path, list[pathlib.Path]]:
    venv = venv.expanduser().resolve()
    site_packages = venv / "Lib" / "site-packages"
    if not site_packages.is_dir():
        raise ValueError(f"not a Windows venv site-packages directory: {site_packages}")
    directories = candidate_directories(venv, extra)
    # torch\lib is the only directory every route needs. GStreamer (jpeg8.dll, png16.dll) is needed
    # only by a source-built TorchCodec; the CI-built wheels bundle their image and FFmpeg libraries.
    # This check comes first so that a venv without torch is named as such on every machine: with no
    # CUDA toolkit, cuDNN or GStreamer installed the candidate list is empty.
    torch_lib = (site_packages / "torch" / "lib").resolve()
    if torch_lib not in directories:
        raise ValueError(f"required DLL directory is missing: {torch_lib} (install torch into this venv first)")

    output = site_packages / PTH_FILENAME
    content = pth_content(directories)
    if output.exists() and output.read_text(encoding="utf-8") != content and not force:
        raise ValueError(f"refusing to replace different DLL bootstrap: {output}; use --force")
    if not dry_run:
        output.write_text(content, encoding="utf-8", newline="\n")
    return output, directories


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", required=True, type=pathlib.Path)
    parser.add_argument("--extra-dll-dir", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output, directories = configure(
            venv=args.venv,
            extra=args.extra_dll_dir,
            force=args.force,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as exc:
        print(f"configure_windows_arm64_dll_paths: {exc}", file=sys.stderr)
        return 2
    print(f"{'would write' if args.dry_run else 'wrote'} {output}")
    for directory in directories:
        print(f"  {directory}")
    if not any("gstreamer" in str(directory).lower() for directory in directories):
        print(
            "note: no GStreamer directory registered (optional). A self-contained TorchCodec wheel does not "
            "need it; a source-built TorchCodec does: pass its msvc_arm64\\bin and the FFmpeg bin with "
            "--extra-dll-dir and --force."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
