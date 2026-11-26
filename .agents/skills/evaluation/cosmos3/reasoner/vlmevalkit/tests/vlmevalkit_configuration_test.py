# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the toolkit's endpoint and dataset-storage configuration.

The toolkit's runtime dependencies (torch, model wrappers, s3fs) are not needed: the
modules under test are loaded from source with only their heavy imports stubbed.

- Private endpoints: no private-network (RFC 1918) IPv4 literal anywhere in the toolkit tree.
- Endpoint overrides: an explicit `api_base` configures the env-driven model entries even when
  their endpoint variable is unset at config-load time.
- Grounding storage: `VANTAGE_2DGrounding` does not derive paths from the bucket variables;
  a cold cache without explicit paths fails before any S3 access, as documented.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import ipaddress
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]

TOOLKIT = Path(__file__).resolve().parents[1]

# Model entries whose endpoint comes from an environment variable (EXTERNAL_CONFIGURATION.md).
ENDPOINT_VARIABLES = {
    "qwen3_235b_a22b": "VLMEVAL_QWEN3_235B_API_BASE",
    "qwen3_30b_a3b": "VLMEVAL_QWEN3_30B_API_BASE",
    "qwen3_8b": "VLMEVAL_QWEN3_8B_API_BASE",
    "cosmos_reason1_7b": "VLMEVAL_COSMOS_REASON1_API_BASE",
    "LVS_service": "VLMEVAL_LVS_API_BASE",
    "Qwen3-VL-8B--crop--arm_thinker_prompt--sglang": "VLMEVAL_ARM_THINKER_SGLANG_API_BASE",
}
CALLER = "https://caller.example.invalid/v1/chat/completions"
ENVIRONMENT = "https://environment.example.invalid/v1/chat/completions"


# --- No hard-coded private-network endpoints ----------------------------------------------

_IPV4 = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
_PRIVATE_NETWORKS = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")]


def _private_ipv4_literals(text: str) -> list[str]:
    found = []
    for candidate in _IPV4.findall(text):
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if any(address in network for network in _PRIVATE_NETWORKS):
            found.append(candidate)
    return found


def test_private_ipv4_detector() -> None:
    assert _private_ipv4_literals("http://10.1.2.3:8000/v1 and 192.168.0.1") == ["10.1.2.3", "192.168.0.1"]
    assert _private_ipv4_literals("http://0.0.0.0:8000 8.8.8.8 172.32.0.1 v1.10.0.1.2") == []


def test_no_private_network_endpoints() -> None:
    offenders = []
    for path in sorted(TOOLKIT.rglob("*")):
        if not path.is_file() or path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary assets
        for line_number, line in enumerate(text.splitlines(), 1):
            # Report the location only, never the address itself.
            offenders += [f"{path.relative_to(TOOLKIT)}:{line_number}" for _ in _private_ipv4_literals(line)]
    assert not offenders, f"private-network IPv4 literal(s) at: {offenders}"


# --- Explicit api_base without the endpoint environment variable -------------------------


class _RecordingModel:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__getattr__ = lambda attribute: type(attribute, (_RecordingModel,), {})  # type: ignore[method-assign]
    return module


@pytest.fixture
def stub_vlmeval(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A `vlmeval` package whose `api` and `vlm` hand out recording model classes."""
    package = types.ModuleType("vlmeval")
    package.__path__ = []  # type: ignore[attr-defined]
    package.api = _stub_module("vlmeval.api")  # type: ignore[attr-defined]
    package.vlm = _stub_module("vlmeval.vlm")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vlmeval", package)
    monkeypatch.setitem(sys.modules, "vlmeval.api", package.api)
    monkeypatch.setitem(sys.modules, "vlmeval.vlm", package.vlm)
    return package


def _load_supported_vlm() -> dict:
    namespace: dict = {"__name__": "vlmeval.config"}
    source = (TOOLKIT / "vlmeval" / "config.py").read_text()
    exec(compile(source, str(TOOLKIT / "vlmeval" / "config.py"), "exec"), namespace)
    return namespace["supported_VLM"]


def _build_model_from_config(supported_vlm: dict):
    """`run.build_model_from_config`, compiled unchanged from run.py (run.py imports torch)."""
    tree = ast.parse((TOOLKIT / "run.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model_from_config")
    namespace = {"cp": copy, "os": os, "supported_VLM": supported_vlm}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(TOOLKIT / "run.py"), "exec"), namespace)
    return namespace["build_model_from_config"]


@pytest.mark.parametrize("model", ENDPOINT_VARIABLES)
@pytest.mark.parametrize(
    ("environment", "caller", "expected"),
    [
        pytest.param(None, CALLER, CALLER, id="caller-only"),
        pytest.param(ENVIRONMENT, None, ENVIRONMENT, id="environment-only"),
        pytest.param(ENVIRONMENT, CALLER, CALLER, id="caller-over-environment"),
        pytest.param(None, None, None, id="missing-both"),
    ],
)
def test_endpoint_resolution(
    monkeypatch: pytest.MonkeyPatch,
    stub_vlmeval: types.ModuleType,
    model: str,
    environment: str | None,
    caller: str | None,
    expected: str | None,
) -> None:
    for variable in ENDPOINT_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    if environment:
        monkeypatch.setenv(ENDPOINT_VARIABLES[model], environment)
    # The environment is read when config.py loads, so load it after setting it.
    build_model_from_config = _build_model_from_config(_load_supported_vlm())
    config = {model: {"api_base": caller} if caller else {}}

    if expected is None:
        with pytest.raises(ValueError, match=ENDPOINT_VARIABLES[model]):
            build_model_from_config(config, model)
        return
    assert build_model_from_config(config, model).kwargs["api_base"] == expected


# --- VANTAGE_2DGrounding storage: bucket variables are not a path source -----------------


class _S3Reached(AssertionError):
    pass


@pytest.fixture
def grounding_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """`Metropolis2DGroundingDataset` from source, with `LMUData` pointing at an empty tmp dir."""
    for module in ("numpy", "pandas", "PIL", "yaml"):
        pytest.importorskip(module)

    lmu_data = tmp_path / "LMUData"
    lmu_data.mkdir()
    monkeypatch.setenv("LMUData", str(lmu_data))

    package_dir = TOOLKIT / "vlmeval" / "dataset" / "metropolis2d"
    for name in ("vlmeval", "vlmeval.dataset", "vlmeval.dataset.metropolis2d"):
        package = types.ModuleType(name)
        package.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, package)

    image_base = types.ModuleType("vlmeval.dataset.image_base")
    image_base.ImageBaseDataset = type("ImageBaseDataset", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vlmeval.dataset.image_base", image_base)

    smp = types.ModuleType("vlmeval.smp")
    smp.get_logger = lambda name: types.SimpleNamespace(info=print, warning=print)  # type: ignore[attr-defined]
    smp.load = None  # type: ignore[attr-defined]
    smp.LMUDataRoot = lambda: os.environ["LMUData"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vlmeval.smp", smp)

    def reach_s3(*args, **kwargs):
        raise _S3Reached("S3 client constructed")

    s3fs = types.ModuleType("s3fs")
    s3fs.S3FileSystem = reach_s3  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "s3fs", s3fs)

    modules = {}
    for name in ("utils", "grounding_2d_dataset"):
        qualified = f"vlmeval.dataset.metropolis2d.{name}"
        spec = importlib.util.spec_from_file_location(qualified, package_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["grounding_2d_dataset"].Metropolis2DGroundingDataset, lmu_data


def test_grounding_bucket_only_cold_cache_fails_before_s3(
    monkeypatch: pytest.MonkeyPatch, grounding_dataset
) -> None:
    dataset_class, _ = grounding_dataset
    monkeypatch.setenv("VLMEVAL_BENCHMARK_S3_BUCKET", "qa-benchmark")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://storage.example.invalid")
    with pytest.raises(ValueError, match="data_root must be specified"):
        dataset_class(dataset="VANTAGE_2DGrounding")


def _write_llava_fixture(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "images" / "frame.jpg").write_bytes(b"")  # existence is all the loader checks
    annotation = root / "annotations" / "refdrone_test_llava.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            [
                {
                    "media": "images/frame.jpg",
                    "conversations": [
                        {"from": "human", "value": 'Locate "the red car"'},
                        {"from": "gpt", "value": "[[1, 2, 3, 4]]"},
                    ],
                    "_meta": {"sentence": "the red car", "image_width": 8, "image_height": 8},
                }
            ]
        )
    )
    return annotation


def test_grounding_hydrated_cache(grounding_dataset) -> None:
    dataset_class, lmu_data = grounding_dataset
    cached_root = lmu_data / "vantage_2d_grounding"
    annotation = _write_llava_fixture(cached_root)
    dataset = dataset_class(dataset="VANTAGE_2DGrounding")
    assert (dataset.data_root, dataset.annotation_file) == (str(cached_root), str(annotation))
    assert len(dataset.data) == 1


def test_grounding_explicit_local_paths(grounding_dataset, tmp_path: Path) -> None:
    dataset_class, _ = grounding_dataset
    root = tmp_path / "explicit"
    annotation = _write_llava_fixture(root)
    dataset = dataset_class(dataset="VANTAGE_2DGrounding", data_root=str(root), annotation_file=str(annotation))
    assert len(dataset.data) == 1


def test_grounding_does_not_read_bucket_variables() -> None:
    source = (TOOLKIT / "vlmeval" / "dataset" / "metropolis2d" / "grounding_2d_dataset.py").read_text()
    assert "S3_BUCKET" not in source
    documentation = " ".join((TOOLKIT / "EXTERNAL_CONFIGURATION.md").read_text().split())
    assert "`VANTAGE_2DGrounding` does not read the bucket variables." in documentation
