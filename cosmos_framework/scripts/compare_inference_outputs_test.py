# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import struct
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from safetensors.numpy import save_file

import cosmos_framework.scripts.compare_inference_outputs as compare_inference_outputs_module
from cosmos_framework.scripts.compare_inference_outputs import compare_inference_outputs, main

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _write_sample(
    directory: Path,
    *,
    values: tuple[float, ...] = (0.0, 1.0),
    dtype: str = "float32",
    shape: tuple[int, ...] = (1, 2),
    name: str = "video",
    sample: str = "sample.safetensors",
    metadata: dict[str, str] | None = None,
) -> None:
    path = directory / sample
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = np.array(values, dtype=dtype).reshape(shape)  # [*shape]
    save_file({name: tensor}, str(path), metadata=metadata)


def _runs(tmp_path: Path, *, legs: tuple[str, ...] = ("native-i4", "framework")) -> dict[str, list[Path]]:
    runs = {leg: [tmp_path / f"{leg}-{repeat}" for repeat in range(2)] for leg in legs}
    for paths in runs.values():
        for path in paths:
            _write_sample(path)
    return runs


def test_compares_six_runs_and_tensor_payloads_independent_of_headers(tmp_path: Path) -> None:
    runs = _runs(tmp_path, legs=("native-i4", "hf-i4", "framework"))
    _write_sample(runs["framework"][1], metadata={"runtime": "framework", "repeat": "2"})
    for paths in runs.values():
        for path in paths:
            _write_sample(path, sample="nested/another.safetensors", name="latent")

    report = compare_inference_outputs(runs, expected_samples=("sample.safetensors", "nested/another.safetensors"))

    assert report["passed"] is True
    assert report["sample_count"] == 2
    assert report["tensor_count"] == 2
    assert len(report["comparisons"]) == 5
    assert sum(item["kind"] == "repeatability" for item in report["comparisons"]) == 3
    assert report["samples"]["sample.safetensors"]["video"]["dtype"] == "F32"
    assert report["mp4"]["checked"] is False


@pytest.mark.parametrize("leg,repeat", [("native-i4", 1), ("framework", 0)])
def test_rejects_repeatability_or_cross_leg_difference(tmp_path: Path, leg: str, repeat: int) -> None:
    runs = _runs(tmp_path)
    _write_sample(runs[leg][repeat], values=(0.0, 1.0000001))
    if repeat == 0:
        _write_sample(runs[leg][1], values=(0.0, 1.0000001))
    with pytest.raises(ValueError, match="tensor bytes differ"):
        compare_inference_outputs(runs)


def test_rejects_signed_zero_even_when_numerically_equal(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    _write_sample(runs["native-i4"][1], values=(-0.0, 1.0))
    with pytest.raises(ValueError, match="tensor bytes differ"):
        compare_inference_outputs(runs)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_identical_nonfinite_outputs(tmp_path: Path, dtype: str, value: float) -> None:
    runs = _runs(tmp_path)
    for paths in runs.values():
        for path in paths:
            _write_sample(path, values=(value, 1.0), dtype=dtype)
    with pytest.raises(ValueError, match="nonfinite"):
        compare_inference_outputs(runs)


@pytest.mark.parametrize("bits,valid", [(0x3F80, True), (0x7F80, False), (0xFF80, False), (0x7FC0, False)])
def test_checks_bfloat16_without_torch(tmp_path: Path, bits: int, valid: bool) -> None:
    runs = _runs(tmp_path)
    # Build BF16 bytes directly because NumPy does not have a native BF16 dtype.
    header = json.dumps({"video": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    header += b" " * (-len(header) % 8)
    payload = struct.pack("<Q", len(header)) + header + struct.pack("<H", bits)
    for paths in runs.values():
        for path in paths:
            (path / "sample.safetensors").write_bytes(payload)
    if valid:
        assert compare_inference_outputs(runs)["passed"] is True
    else:
        with pytest.raises(ValueError, match="nonfinite"):
            compare_inference_outputs(runs)


@pytest.mark.parametrize("change", ["dtype", "shape", "tensor_names", "sample_names"])
def test_rejects_changed_output_schema(tmp_path: Path, change: str) -> None:
    runs = _runs(tmp_path)
    path = runs["native-i4"][1]
    if change == "dtype":
        _write_sample(path, dtype="float64")
    elif change == "shape":
        _write_sample(path, shape=(2, 1))
    elif change == "tensor_names":
        _write_sample(path, name="different")
    else:
        (path / "sample.safetensors").rename(path / "other.safetensors")
    with pytest.raises(ValueError, match="differ"):
        compare_inference_outputs(runs)


@pytest.mark.parametrize("empty", ["samples", "tensor_names", "tensor_values"])
def test_rejects_vacuous_comparison(tmp_path: Path, empty: str) -> None:
    runs = _runs(tmp_path)
    path = runs["native-i4"][0]
    if empty == "samples":
        (path / "sample.safetensors").unlink()
        (path / "sample.mp4").write_bytes(b"video alone is not evidence")
    elif empty == "tensor_names":
        save_file({}, str(path / "sample.safetensors"))
    else:
        _write_sample(path, values=(), shape=(0,))
    with pytest.raises(ValueError, match="no safetensors|no tensors|empty tensor"):
        compare_inference_outputs(runs)


def test_required_inventory_detects_output_missing_from_every_run(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(ValueError, match="missing required tensors"):
        compare_inference_outputs(runs, required_tensors=("latent",))
    with pytest.raises(ValueError, match="expected samples differ"):
        compare_inference_outputs(runs, expected_samples=("sample.safetensors", "missing.safetensors"))


def test_rejects_reusing_a_directory_as_an_independent_repeat(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    runs["native-i4"][1] = runs["native-i4"][0]
    with pytest.raises(ValueError, match="distinct directory"):
        compare_inference_outputs(runs)


def test_rejects_missing_repeat_or_missing_runtime(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(ValueError, match="two named inference legs"):
        compare_inference_outputs({"only-runtime": runs["native-i4"]})
    runs["native-i4"].pop()
    with pytest.raises(ValueError, match="two independent runs"):
        compare_inference_outputs(runs)


def test_optional_mp4_comparison_cannot_replace_tensor_evidence(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    with pytest.raises(ValueError, match="no MP4 files"):
        compare_inference_outputs(runs, compare_mp4=True)
    for paths in runs.values():
        for path in paths:
            (path / "sample.mp4").write_bytes(b"same encoded bytes")
    assert compare_inference_outputs(runs, compare_mp4=True)["mp4"]["checked"] is True
    (runs["framework"][1] / "sample.mp4").write_bytes(b"different encoded bytes")
    assert compare_inference_outputs(runs)["passed"] is True
    with pytest.raises(ValueError, match="MP4 bytes differ"):
        compare_inference_outputs(runs, compare_mp4=True)


def test_cli_writes_failure_report_for_corrupt_safetensors(tmp_path: Path) -> None:
    runs = _runs(tmp_path)
    (runs["native-i4"][0] / "sample.safetensors").write_bytes(b"not safetensors")
    report_path = tmp_path / "reports" / "parity.json"
    argv = [argument for leg, paths in runs.items() for path in paths for argument in ("--run", f"{leg}={path}")]
    assert main([*argv, "--report", str(report_path)]) == 1
    assert json.loads(report_path.read_text())["passed"] is False


def test_cli_writes_failure_report_for_missing_safetensors_header_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = _runs(tmp_path)
    missing_key = "data_offsets"
    original_safe_open = compare_inference_outputs_module.safe_open

    @contextmanager
    def safe_open_with_inconsistent_header(path: str, *, framework: str) -> Generator[Any, None, None]:
        with original_safe_open(path, framework=framework) as tensors:
            artifact = Path(path)
            payload = artifact.read_bytes()
            header_length = int.from_bytes(payload[:8], "little")
            header = json.loads(payload[8 : 8 + header_length])
            del header["video"][missing_key]
            malformed_header = json.dumps(header, separators=(",", ":")).encode()
            assert len(malformed_header) <= header_length
            malformed_header += b" " * (header_length - len(malformed_header))
            artifact.write_bytes(payload[:8] + malformed_header + payload[8 + header_length :])
            yield tensors

    monkeypatch.setattr(compare_inference_outputs_module, "safe_open", safe_open_with_inconsistent_header)
    report_path = tmp_path / "reports" / "parity.json"
    argv = [argument for leg, paths in runs.items() for path in paths for argument in ("--run", f"{leg}={path}")]

    assert main([*argv, "--report", str(report_path)]) == 1

    report = json.loads(report_path.read_text())
    assert report["passed"] is False
    assert missing_key in report["error"]
