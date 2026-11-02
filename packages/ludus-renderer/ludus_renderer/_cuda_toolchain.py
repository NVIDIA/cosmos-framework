# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Discover a CUDA build toolchain from the host or NVIDIA Python wheels."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterable

import torch
import torch.utils.cpp_extension


@dataclass(frozen=True)
class CudaToolchain:
    """Paths needed by PyTorch's JIT extension builder."""

    root: Path
    include_dir: Path
    lib_dir: Path
    nvcc: Path
    source: str
    major: int | None

    def library_flags(self, library: str, soname: str | None = None) -> list[str]:
        """Return linker flags for a library living in this toolchain."""
        filename = soname or f"lib{library}.so"
        return [f"-L{self.lib_dir}", f"-l:{filename}", f"-Wl,-rpath,{self.lib_dir}"]


def _distribution_root(distribution_name: str) -> Path | None:
    """Return the shared ``nvidia/cu*`` root installed by a distribution."""
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None

    candidates: set[Path] = set()
    for file in distribution.files or ():
        parts = Path(file).parts
        if len(parts) >= 2 and parts[0] == "nvidia" and re.fullmatch(r"cu\d+", parts[1]):
            candidates.add(Path(distribution.locate_file(Path(*parts[:2]))).resolve())
    return next(iter(candidates), None)


def _cuda_major(version: str | None) -> int | None:
    if not version:
        return None
    match = re.match(r"(\d+)", version)
    return int(match.group(1)) if match else None


def _validate_major(toolchain_major: int | None) -> None:
    torch_major = _cuda_major(torch.version.cuda)
    if toolchain_major is not None and torch_major is not None and toolchain_major != torch_major:
        raise RuntimeError(
            f"CUDA toolchain major {toolchain_major} does not match PyTorch CUDA {torch.version.cuda}. "
            "Install matching NVIDIA CUDA wheels or select a matching CUDA_HOME."
        )


def _candidate_system_roots() -> Iterable[Path]:
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured:
        yield Path(configured).expanduser()

    nvcc = shutil.which("nvcc")
    if nvcc:
        yield Path(nvcc).resolve().parent.parent

    yield Path("/usr/local/cuda")


def _from_root(root: Path, source: str, major: int | None = None) -> CudaToolchain | None:
    root = root.resolve()
    nvcc = root / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    include_dir = root / "include"
    lib_candidates = [root / "lib", root / "lib64", root / "lib" / "x64"]
    lib_dir = next((path for path in lib_candidates if path.is_dir()), lib_candidates[0])
    if not nvcc.is_file() or not include_dir.is_dir() or not lib_dir.is_dir():
        return None
    return CudaToolchain(root, include_dir, lib_dir, nvcc, source, major)


def _pip_toolchain_root(source_root: Path) -> Path:
    """Create a stable CUDA_HOME facade for NVIDIA's shared pip-wheel layout."""
    digest = hashlib.sha256(str(source_root).encode()).hexdigest()[:12]
    root = Path.home() / ".cache" / "cosmos-auto" / "cuda" / digest
    directory_mappings = {
        root / "bin": source_root / "bin",
        root / "include": source_root / "include",
    }
    for destination, source in directory_mappings.items():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source, target_is_directory=True)

    lib_dir = root / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    for source in (source_root / "lib").iterdir():
        destination = lib_dir / source.name
        if not destination.exists():
            destination.symlink_to(source)

    # PyTorch links ``-lcudart`` while the runtime wheel ships only the
    # versioned soname, so expose the conventional development symlink.
    cudart = next((source_root / "lib").glob("libcudart.so.*"), None)
    if cudart is not None:
        destination = lib_dir / "libcudart.so"
        if not destination.exists():
            destination.symlink_to(cudart)
    return root


def discover_cuda_toolchain() -> CudaToolchain:
    """Discover a CUDA toolkit, preferring explicit/system configuration."""
    for root in _candidate_system_roots():
        toolchain = _from_root(root, "system")
        if toolchain is not None:
            _validate_major(toolchain.major)
            return toolchain

    for distribution_name in ("nvidia-cuda-nvcc", "nvidia-cuda-nvcc-cu12", "nvidia-cuda-nvcc-cu11"):
        root = _distribution_root(distribution_name)
        if root is None:
            continue
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            version = None
        pip_root = _pip_toolchain_root(root)
        toolchain = _from_root(pip_root, distribution_name, _cuda_major(version))
        if toolchain is not None:
            _validate_major(toolchain.major)
            return toolchain

    raise RuntimeError(
        "No CUDA compiler toolchain was found. Install the package's CUDA dependencies "
        "or configure CUDA_HOME to a toolkit containing bin/nvcc, include/, and lib/."
    )


def configure_cuda_toolchain() -> CudaToolchain:
    """Configure PyTorch JIT compilation to use the discovered toolkit."""
    toolchain = discover_cuda_toolchain()
    os.environ["CUDA_HOME"] = str(toolchain.root)
    os.environ["CUDA_PATH"] = str(toolchain.root)
    os.environ["PATH"] = f"{toolchain.root / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
    torch.utils.cpp_extension.CUDA_HOME = str(toolchain.root)
    return toolchain


def find_nvidia_library(
    distribution_name: str | tuple[str, ...],
    header: str,
    sonames: tuple[str, ...],
    fallback: CudaToolchain,
    import_package: str | None = None,
) -> tuple[Path, Path, str]:
    """Find headers and a shared library in an NVIDIA wheel or CUDA toolkit."""
    distribution_names = (distribution_name,) if isinstance(distribution_name, str) else distribution_name
    distribution_root = next(
        (root for name in distribution_names if (root := _distribution_root(name)) is not None),
        None,
    )
    import_roots: list[Path] = []
    if import_package is not None:
        try:
            package = __import__(import_package, fromlist=["__file__"])
        except ImportError:
            pass
        else:
            import_roots.append(Path(package.__file__).resolve().parent)

    roots = [distribution_root, *import_roots, fallback.root]
    for root in roots:
        if root is None:
            continue
        include_dir = root / "include"
        lib_dirs = (root, root / "lib", root / "lib64", root / "lib" / "x64")
        for lib_dir in lib_dirs:
            if (include_dir / header).is_file():
                soname = next((name for name in sonames if (lib_dir / name).is_file()), None)
                if soname is not None:
                    return include_dir, lib_dir, soname
    raise RuntimeError(
        f"Could not find {header} and one of {sonames}. Install one of {distribution_names} "
        "or provide them in CUDA_HOME."
    )
