# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Check raw inference artifacts for repeatability and exact runtime parity.

Run each inference leg independently at least twice, with identical inputs,
seed, checkpoint, precision, and device configuration, then compare its saved
safetensors outputs. For example::

    python -m cosmos_framework.scripts.compare_inference_outputs \
        --run native-i4=outputs/native-1 --run native-i4=outputs/native-2 \
        --run hf-i4=outputs/hf-1 --run hf-i4=outputs/hf-2 \
        --run framework=outputs/cf-1 --run framework=outputs/cf-2 \
        --required-tensor vision --report outputs/parity.json

The comparison validates the supplied artifacts, not their generation settings
or provenance. Capture raw tensors before video encoding. Optional MP4 checks
supplement the mandatory tensor checks; encoded video alone cannot pass.
"""

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import SafetensorError, safe_open

_CHUNK_BYTES = 1024 * 1024
_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}
# Inspect the encoded exponent directly, including BF16 without a torch import.
_FLOAT_EXPONENTS: dict[str, tuple[str, int]] = {
    "F16": ("<u2", 0x7C00),
    "BF16": ("<u2", 0x7F80),
    "F32": ("<u4", 0x7F800000),
    "F64": ("<u8", 0x7FF0000000000000),
}


@dataclass(frozen=True)
class TensorArtifact:
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int
    sha256: str


@dataclass(frozen=True)
class RunArtifacts:
    leg: str
    repeat: int
    directory: Path
    samples: dict[str, dict[str, TensorArtifact]]
    videos: dict[str, str]

    @property
    def name(self) -> str:
        return f"{self.leg}[{self.repeat}]"


def _scan_tensor(path: Path, name: str, dtype: str, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            payload = stream.read(min(remaining, _CHUNK_BYTES))
            if not payload:
                raise ValueError(f"{path}:{name}: truncated tensor payload")
            if dtype in _FLOAT_EXPONENTS:
                unsigned_dtype, exponent_mask = _FLOAT_EXPONENTS[dtype]
                bits = np.frombuffer(payload, dtype=unsigned_dtype)  # [N]
                nonfinite = (bits & exponent_mask) == exponent_mask  # [N]
                if np.any(nonfinite):
                    raise ValueError(f"{path}:{name}: nonfinite tensor values")
            digest.update(payload)
            remaining -= len(payload)
    return digest.hexdigest()


def _scan_sample(path: Path, required_tensors: set[str]) -> dict[str, TensorArtifact]:
    # safe_open validates the container, offsets, shapes, and dtype byte counts.
    # Reading payload ranges ourselves preserves signed zeros and BF16 bits.
    with safe_open(str(path), framework="np") as tensors:
        names = set(tensors.keys())
        if not names:
            raise ValueError(f"{path}: no tensors")
        if missing := required_tensors - names:
            raise ValueError(f"{path}: missing required tensors {sorted(missing)}")
        with path.open("rb") as stream:
            header_length = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(header_length))
        result: dict[str, TensorArtifact] = {}
        for name in sorted(names):
            tensor_slice = tensors.get_slice(name)  # [...]
            dtype = tensor_slice.get_dtype()
            shape = tuple(tensor_slice.get_shape())
            if math.prod(shape) == 0:
                raise ValueError(f"{path}:{name}: empty tensor with shape {shape}")
            if dtype not in _DTYPE_BYTES:
                raise ValueError(f"{path}:{name}: unsupported tensor dtype {dtype}; finite checking is unavailable")
            start, end = header[name]["data_offsets"]
            offset = 8 + header_length + start
            length = end - start
            digest = _scan_tensor(path, name, dtype, offset, length)
            result[name] = TensorArtifact(dtype, shape, offset, length, digest)
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(_CHUNK_BYTES):
            digest.update(payload)
    return digest.hexdigest()


def _scan_run(
    leg: str,
    repeat: int,
    directory: Path,
    required_tensors: set[str],
    expected_samples: set[str] | None,
    compare_mp4: bool,
) -> RunArtifacts:
    paths = sorted(directory.rglob("*.safetensors"))
    if not paths:
        raise ValueError(f"{directory}: no safetensors samples")
    samples = {path.relative_to(directory).as_posix(): _scan_sample(path, required_tensors) for path in paths}
    if expected_samples is not None and set(samples) != expected_samples:
        raise ValueError(
            f"{directory}: expected samples differ: missing={sorted(expected_samples - set(samples))}, "
            f"unexpected={sorted(set(samples) - expected_samples)}"
        )
    videos: dict[str, str] = {}
    if compare_mp4:
        for path in sorted(directory.rglob("*.mp4")):
            if path.stat().st_size == 0:
                raise ValueError(f"{path}: empty MP4")
            videos[path.relative_to(directory).as_posix()] = _file_sha256(path)
        if not videos:
            raise ValueError(f"{directory}: MP4 comparison requested but no MP4 files found")
    return RunArtifacts(leg, repeat, directory, samples, videos)


def _equal_bytes(left: Path, right: Path, left_offset: int, right_offset: int, length: int) -> bool:
    """Confirm byte equality after matching SHA-256 digests.

    Digests reject ordinary mismatches without rereading both payloads. This
    direct comparison makes the bit-exact result independent of collision
    resistance, including for deliberately constructed inputs.
    """
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        left_stream.seek(left_offset)
        right_stream.seek(right_offset)
        remaining = length
        while remaining:
            size = min(remaining, _CHUNK_BYTES)
            left_payload = left_stream.read(size)
            right_payload = right_stream.read(size)
            if len(left_payload) != size or left_payload != right_payload:
                return False
            remaining -= size
    return True


def _compare_runs(left: RunArtifacts, right: RunArtifacts) -> None:
    context = f"{left.name} vs {right.name}"
    if left.samples.keys() != right.samples.keys():
        raise ValueError(
            f"{context}: samples differ: missing={sorted(left.samples.keys() - right.samples.keys())}, "
            f"unexpected={sorted(right.samples.keys() - left.samples.keys())}"
        )
    for sample, left_tensors in left.samples.items():
        right_tensors = right.samples[sample]
        if left_tensors.keys() != right_tensors.keys():
            raise ValueError(f"{context}: {sample}: tensor names differ")
        for name, left_tensor in left_tensors.items():
            right_tensor = right_tensors[name]
            if (left_tensor.dtype, left_tensor.shape) != (right_tensor.dtype, right_tensor.shape):
                raise ValueError(
                    f"{context}: {sample}:{name}: dtype/shape differ: "
                    f"{left_tensor.dtype}{left_tensor.shape} vs {right_tensor.dtype}{right_tensor.shape}"
                )
            if left_tensor.sha256 != right_tensor.sha256 or not _equal_bytes(
                left.directory / sample,
                right.directory / sample,
                left_tensor.offset,
                right_tensor.offset,
                left_tensor.length,
            ):
                raise ValueError(f"{context}: {sample}:{name}: tensor bytes differ")
    if left.videos.keys() != right.videos.keys():
        raise ValueError(f"{context}: MP4 files differ")
    for video, digest in left.videos.items():
        left_path, right_path = left.directory / video, right.directory / video
        if (
            digest != right.videos[video]
            or left_path.stat().st_size != right_path.stat().st_size
            or not _equal_bytes(left_path, right_path, 0, 0, left_path.stat().st_size)
        ):
            raise ValueError(f"{context}: {video}: MP4 bytes differ")


def compare_inference_outputs(
    runs: dict[str, list[Path]],
    *,
    required_tensors: tuple[str, ...] = (),
    expected_samples: tuple[str, ...] | None = None,
    compare_mp4: bool = False,
) -> dict[str, Any]:
    """Require self-repeatability in every leg and raw byte parity across legs.

    ``expected_samples`` names relative ``.safetensors`` paths. Supply it to
    detect a sample omitted by every run; otherwise the first run is the sample
    inventory. ``required_tensors`` similarly detects universally missing keys.
    """
    if len(runs) < 2:
        raise ValueError("At least two named inference legs are required")
    if any(not leg.strip() or len(paths) < 2 for leg, paths in runs.items()):
        raise ValueError("Every named leg requires at least two independent runs")
    if expected_samples is not None and not expected_samples:
        raise ValueError("Expected sample inventory must not be empty")
    resolved = [path.resolve(strict=True) for paths in runs.values() for path in paths]
    if any(not path.is_dir() for path in resolved):
        raise ValueError("Every run must be a directory")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Each independent run must have a distinct directory")
    if any(left in right.parents for left in resolved for right in resolved if left != right):
        raise ValueError("Independent run directories must not overlap")

    scanned: list[RunArtifacts] = []
    comparisons: list[dict[str, str]] = []
    for leg, paths in runs.items():
        leg_runs = [
            _scan_run(
                leg,
                repeat,
                path.resolve(),
                set(required_tensors),
                set(expected_samples) if expected_samples is not None else None,
                compare_mp4,
            )
            for repeat, path in enumerate(paths, start=1)
        ]
        for candidate in leg_runs[1:]:
            _compare_runs(leg_runs[0], candidate)
            comparisons.append({"kind": "repeatability", "left": leg_runs[0].name, "right": candidate.name})
        if scanned:
            _compare_runs(scanned[0], leg_runs[0])
            comparisons.append({"kind": "cross_leg", "left": scanned[0].name, "right": leg_runs[0].name})
        scanned.extend(leg_runs)

    reference = scanned[0]
    return {
        "passed": True,
        "comparison": "raw_tensor_bytes",
        "runs": [{"leg": run.leg, "repeat": run.repeat, "directory": str(run.directory)} for run in scanned],
        "comparisons": comparisons,
        "sample_count": len(reference.samples),
        "tensor_count": sum(len(tensors) for tensors in reference.samples.values()),
        "samples": {
            sample: {
                name: {"dtype": tensor.dtype, "shape": list(tensor.shape), "sha256": tensor.sha256}
                for name, tensor in tensors.items()
            }
            for sample, tensors in reference.samples.items()
        },
        "mp4": {"checked": compare_mp4, "sha256": reference.videos},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="LEG=DIRECTORY")
    parser.add_argument("--required-tensor", action="append", default=[], help="Require this key in every sample")
    parser.add_argument(
        "--expected-sample", action="append", help="Exact expected relative safetensors path; repeatable"
    )
    parser.add_argument("--compare-mp4", action="store_true", help="Additionally require matching MP4 files and bytes")
    parser.add_argument("--report", type=Path, help="Write the JSON success or failure report here")
    args = parser.parse_args(argv)
    runs: dict[str, list[Path]] = {}
    for value in args.run:
        leg, separator, directory = value.partition("=")
        if not separator or not leg.strip() or not directory:
            parser.error(f"Invalid --run {value!r}; expected LEG=DIRECTORY")
        runs.setdefault(leg, []).append(Path(directory))
    try:
        report = compare_inference_outputs(
            runs,
            required_tensors=tuple(args.required_tensor),
            expected_samples=tuple(args.expected_sample) if args.expected_sample is not None else None,
            compare_mp4=args.compare_mp4,
        )
    except (KeyError, OSError, ValueError, SafetensorError) as error:
        report = {"passed": False, "error": str(error)}
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized)
    print(serialized, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
