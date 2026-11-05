# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fail-loud runtime preflight for the posttrain NIXL weight plane."""

from __future__ import annotations

import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_RUNTIME_CONTRACTS: dict[str, tuple[dict[str, str], str]] = {
    "cu12": (
        {
            "cuda-bindings": "12.9.4",
            "cuda-python": "12.9.4",
            "cupy-cuda12x": "13.6.0",
            "fastrlock": "0.8.3",
            "nixl": "1.3.2",
            "nixl-cu12": "1.3.2",
            "ray": "2.56.1",
        },
        "12.8",
    ),
    "cu13": (
        {
            "cuda-bindings": "13.0.3",
            "cuda-python": "13.0.3",
            "cupy-cuda13x": "13.6.0",
            "fastrlock": "0.8.3",
            "nixl": "1.3.2",
            "nixl-cu13": "1.3.2",
            "ray": "2.56.1",
        },
        "13.",
    ),
}


def _runtime_contract(cuda_flavor: str) -> tuple[dict[str, str], str]:
    """Return the pinned distributions and Torch CUDA prefix for one image flavor."""
    aliases = {"cu128": "cu12", "cu130": "cu13"}
    normalized_flavor = aliases.get(cuda_flavor, cuda_flavor)
    try:
        return _RUNTIME_CONTRACTS[normalized_flavor]
    except KeyError as exc:
        supported = ", ".join(sorted((*_RUNTIME_CONTRACTS, *aliases)))
        raise ValueError(
            f"unsupported POSTTRAIN_NIXL_CUDA_FLAVOR={cuda_flavor!r}; expected one of: {supported}"
        ) from exc


def _ucx_declaration_failures(ucx_tls: str, ucx_devices: str) -> list[str]:
    """Validate the launch-time UCX declarations; real routing is proven only by a transfer."""
    failures: list[str] = []
    transports = {item.strip() for item in ucx_tls.split(",") if item.strip()}
    if "tcp" not in transports:
        failures.append("UCX_TLS must include tcp for NIXL agent wireup")
    if not transports.intersection({"cuda_copy", "cuda_ipc"}):
        failures.append("UCX_TLS must include a CUDA transport (cuda_copy or cuda_ipc)")
    if not ucx_devices or ucx_devices == "all":
        failures.append("UCX_NET_DEVICES must explicitly select the qualified NICs instead of being unset/all")
    return failures


def _ucx_module_failures(module_dir: str) -> list[str]:
    """Require the selected directory to contain the baked CUDA plugin artifact."""
    if not module_dir:
        return ["UCX_MODULE_DIR must select the baked UCX plugin directory"]
    root = Path(module_dir)
    if not root.is_dir():
        return [f"UCX_MODULE_DIR does not exist: {module_dir}"]
    if not any(root.glob("libuct_cuda.so*")):
        return [f"UCX_MODULE_DIR has no libuct_cuda.so plugin: {module_dir}"]
    return []


def validate_nixl_runtime() -> tuple[list[str], list[str]]:
    """Return a version report and every runtime-contract violation found in this process."""
    report = [f"python={platform.python_version()} executable={sys.executable} machine={platform.machine()}"]
    failures: list[str] = []
    cuda_flavor = os.environ.get("POSTTRAIN_NIXL_CUDA_FLAVOR") or os.environ.get("POSTTRAIN_CUDA_FLAVOR", "cu130")
    report.append(f"cuda_flavor={cuda_flavor}")
    try:
        pinned_distributions, torch_cuda_prefix = _runtime_contract(cuda_flavor)
    except ValueError as exc:
        failures.append(str(exc))
        pinned_distributions = {}
        torch_cuda_prefix = ""

    for distribution, expected in pinned_distributions.items():
        try:
            actual = version(distribution)
        except PackageNotFoundError:
            failures.append(f"{distribution} is not installed")
            continue
        report.append(f"{distribution}={actual}")
        if actual != expected:
            failures.append(f"{distribution}=={actual}, expected the locked version {expected}")

    try:
        import torch

        report.append(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
        if torch_cuda_prefix and not str(torch.version.cuda or "").startswith(torch_cuda_prefix):
            failures.append(f"torch must use CUDA {torch_cuda_prefix}, got torch.version.cuda={torch.version.cuda!r}")
    except Exception as exc:  # noqa: BLE001 -- aggregate every missing/broken runtime component
        failures.append(f"torch is not importable: {exc}")

    try:
        import vllm

        report.append(f"vllm={vllm.__version__}")
    except Exception as exc:  # noqa: BLE001 -- qualification requires the live rollout engine
        failures.append(f"vllm is not importable: {exc}")

    from cosmos_framework.posttrain.workers.nixl import nixl_available

    available, reason = nixl_available()
    if not available:
        failures.append(f"NIXL/Ray RDT capability check failed: {reason}")

    ucx_tls = os.environ.get("UCX_TLS", "")
    ucx_devices = os.environ.get("UCX_NET_DEVICES", "")
    ucx_module_dir = os.environ.get("UCX_MODULE_DIR", "")
    report.extend(
        (
            f"UCX_TLS={ucx_tls or '<unset>'}",
            f"UCX_NET_DEVICES={ucx_devices or '<unset>'}",
            f"UCX_MODULE_DIR={ucx_module_dir or '<unset>'}",
        )
    )
    failures.extend(_ucx_declaration_failures(ucx_tls, ucx_devices))
    failures.extend(_ucx_module_failures(ucx_module_dir))
    return report, failures


def main() -> None:
    """Print the runtime identity and exit nonzero if any prerequisite is missing or mismatched."""
    report, failures = validate_nixl_runtime()
    print("posttrain NIXL runtime:")
    for line in report:
        print(f"  {line}")
    if failures:
        sys.stdout.flush()
        print("runtime preflight failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
