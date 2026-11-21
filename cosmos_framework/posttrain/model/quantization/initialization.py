# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit, cached compensation for a *fixed* native NF4 representation.

Preparation is a separate operation: one BF16 source matrix and its actual NF4
dequantization are used at a time. Model construction only verifies and copies
the recorded FP32 factors. Missing artifacts never trigger decomposition.

The residual objective follows PEFT v0.17.1's ``_loftq_init_new`` (commit
53c25fe4fdd7c6aa4b40db0560815ea570d32303, utils/loftq_utils.py), not its model
wrappers or alternating requantization. We use a bounded randomized reduced
SVD and explicitly divide the factors by sqrt(alpha/rank).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.posttrain.config.model import quantization_config_identity

METHOD = "nf4_residual_svd"
ARTIFACT_PROTOCOL = "cosmos.native-nf4-residual-initialization.v1"
IDENTITY_PROTOCOL = "cosmos.adapter-initialization.v1"
OVERSAMPLING = 16
POWER_ITERATIONS = 2
ERROR_CHUNK_ROWS = 64
ALGORITHM = {
    "name": "randomized_residual_svd_v1",
    "residual": "fp32(source_bf16)-fp32(actual_nf4_dequantized_bf16)",
    "oversampling": OVERSAMPLING,
    "power_iterations": POWER_ITERATIONS,
    "factor_dtype": "float32",
    "scale": "alpha/rank",
    "seed_derivation": "sha256(master_seed,target_name)",
    "upstream": {
        "repository": "https://github.com/huggingface/peft",
        "revision": "53c25fe4fdd7c6aa4b40db0560815ea570d32303",
        "file": "src/peft/utils/loftq_utils.py",
        "file_sha256": "b6f6c426e1d877f9497a8b1d01530889f3ac9b30e0e37207243dde0d85f95614",
    },
}


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def matrix_seed(seed: int, target: str) -> int:
    """A local generator per canonical target; ordering/global RNG is irrelevant."""
    if type(seed) is not int or not 0 <= seed < 2**63 or not isinstance(target, str) or not target:
        raise ValueError("residual initialization requires an explicit seed and target name")
    return int.from_bytes(hashlib.sha256(_json([seed, target])).digest()[:8], "big") % 2**63


@torch.no_grad()
def residual_factors(
    source: torch.Tensor,
    effective: torch.Tensor,
    *,
    rank: int,
    alpha: float,
    seed: int,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return FP32 A/B with (alpha/rank) B A approximating source - effective.

    ``effective`` must come from the actual attached native NF4 state; the pure
    numerical routine is separately CPU-testable. Dense working storage is one
    FP32 residual plus the caller's one BF16 effective matrix. All other SVD
    work is O((rows+columns)*(rank+16)); the error pass uses 64-row chunks.
    """
    if (
        source.is_meta
        or effective.is_meta
        or source.ndim != 2
        or source.shape != effective.shape
        or source.device.type != "cpu"
        or source.dtype != torch.bfloat16
        or effective.dtype != torch.bfloat16
        or type(rank) is not int
        or not 1 <= rank <= min(source.shape)
        or type(alpha) not in (float, int)
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("residual initialization requires matching BF16 matrices, positive rank and alpha")
    local_seed = matrix_seed(seed, target)
    scale = alpha / rank
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid native LoRA alpha/rank scale")
    with torch.autocast(device_type=effective.device.type, enabled=False):
        # copy=True is essential when a CPU fixture already has FP32-like views;
        # neither input nor any native packed parameter is ever modified.
        residual = source.to(device=effective.device, dtype=torch.float32, copy=True)
        residual.sub_(effective)
        if not torch.isfinite(residual).all():
            raise ValueError("nonfinite source or effective quantized weight")
        rows, columns = residual.shape
        width = min(rank + OVERSAMPLING, rows, columns)
        generator = torch.Generator(device=effective.device).manual_seed(local_seed)
        sketch = torch.randn(columns, width, device=effective.device, dtype=torch.float32, generator=generator)
        basis = torch.linalg.qr(residual @ sketch, mode="reduced").Q
        del sketch
        for _ in range(POWER_ITERATIONS):
            dual = torch.linalg.qr(residual.T @ basis, mode="reduced").Q
            basis = torch.linalg.qr(residual @ dual, mode="reduced").Q
            del dual
        projected = basis.T @ residual
        u, singular, vh = torch.linalg.svd(projected, full_matrices=False)
        left = basis @ u[:, :rank]
        # Resolve SVD sign ambiguity without consuming another RNG stream.
        v = vh[:rank]
        pivots = v.abs().argmax(dim=1, keepdim=True)
        signs = torch.where(v.gather(1, pivots) < 0, -1.0, 1.0)
        v = v * signs
        left = left * signs.T
        magnitude = (singular[:rank] / scale).sqrt()
        a = (magnitude[:, None] * v).contiguous()
        b = (left * magnitude[None, :]).contiguous()
        del projected, u, singular, vh, left, v, basis
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite residual adapter factors")
        before, after = 0.0, 0.0
        for offset in range(0, rows, ERROR_CHUNK_ROWS):
            chunk = residual[offset : offset + ERROR_CHUNK_ROWS]
            before += float(chunk.square().sum(dtype=torch.float64).item())
            error = chunk - scale * (b[offset : offset + ERROR_CHUNK_ROWS] @ a)
            after += float(error.square().sum(dtype=torch.float64).item())
        if not math.isfinite(before) or not math.isfinite(after) or after > before * (1 + 2e-6):
            raise RuntimeError("residual factorization failed its reconstruction-error check")
    return (
        a,
        b,
        {
            "before_squared_frobenius": before,
            "after_squared_frobenius": after,
            "relative_squared_error": after / before if before else None,
            "rank": rank,
            "sketch_columns": width,
            "seed": local_seed,
        },
    )


def validate_initialization_record(record: Any) -> None:
    """Validate persisted identity without opening an artifact or allocating tensors."""
    if (
        not isinstance(record, dict)
        or set(record) != {"protocol", "method", "artifact_manifest_sha256", "algorithm", "quantized_base_changed"}
        or record["protocol"] != IDENTITY_PROTOCOL
        or record["method"] != METHOD
        or not _is_sha(record["artifact_manifest_sha256"])
        or record["algorithm"] != ALGORITHM
        or record["quantized_base_changed"] is not False
    ):
        raise ValueError("unsupported or incomplete residual initialization identity")


def _adapter_record(adapter: Any) -> dict[str, Any]:
    from cosmos_framework.posttrain.config.model import native_adapter_config_identity

    result = native_adapter_config_identity(adapter)
    result.pop("initialization_artifact_sha256", None)
    result.pop("initialization_artifact", None)
    result["initialization"] = METHOD
    return result


def _plain(value: Any) -> Any:
    return json.loads(_json(value))


def _runtime_identity() -> dict[str, Any]:
    from cosmos_framework.posttrain.config import model as config_model
    from cosmos_framework.posttrain.model.quantization.loader import runtime_precision_policy
    from cosmos_framework.utils.generator import lora

    sources = {
        name: Path(__file__).with_name(name) for name in ("initialization.py", "bnb.py", "loader.py", "source.py")
    }
    sources["native_lora.py"] = Path(lora.__file__)
    sources["config_model.py"] = Path(config_model.__file__)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "bitsandbytes": version("bitsandbytes"),
        "safetensors": version("safetensors"),
        "hardware": torch.cuda.get_device_name(torch.cuda.current_device()),
        "capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
        "math_precision": runtime_precision_policy(),
        "implementation_sha256": {name: _sha_file(path) for name, path in sources.items()},
        "determinism": {
            "enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "native_build": {
            name: os.environ.get(name) for name in ("COSMOS_TRAINING_CODE_REVISION", "COSMOS_TRAINING_WHEEL_SHA256")
        },
    }


class InitializationArtifact:
    """Strict source-bound factor cache; applying it cannot invoke SVD or quantize."""

    def __init__(
        self,
        path: str | Path,
        expected_sha256: str,
        *,
        adapter: Any,
        source_identity: dict[str, Any],
        quantization: Any,
    ) -> None:
        from cosmos_framework.posttrain.model.quantization.loader import canonical_targets

        if getattr(quantization, "excluded_language_layers", ()):
            raise ValueError("residual initialization with excluded language layers is not supported")

        supplied = Path(path)
        if supplied.is_symlink() or not supplied.is_file() or supplied.name != "manifest.json":
            raise ValueError("residual initialization requires a committed manifest.json artifact")
        self.path = supplied.resolve(strict=True)
        self.root = self.path.parent
        raw = self.path.read_bytes()
        if not _is_sha(expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("residual initialization manifest SHA256 mismatch")
        self.sha256 = expected_sha256

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate initialization manifest key")
                result[key] = value
            return result

        self.manifest = json.loads(raw, object_pairs_hook=unique)
        m = self.manifest
        targets = {name: list(shape) for name, shape in sorted(canonical_targets().items())}
        if (
            not isinstance(m, dict)
            or set(m) != {"protocol", "algorithm", "base", "adapter", "quantization", "targets", "matrices", "runtime"}
            or m["protocol"] != ARTIFACT_PROTOCOL
            or m["algorithm"] != ALGORITHM
            or m["base"] != _plain(source_identity)
            or m["adapter"] != _plain(_adapter_record(adapter))
            or m["adapter"].get("initialization") != METHOD
            or quantization is None
            or m["quantization"] != _plain(quantization_config_identity(quantization))
            or m["targets"] != targets
            or not isinstance(m["matrices"], dict)
            or set(m["matrices"]) != set(targets)
            or not isinstance(m["runtime"], dict)
            or set(m["runtime"]) != {"identity", "elapsed_seconds", "scope"}
            or m["runtime"]["identity"] != _plain(_runtime_identity())
            or type(m["runtime"]["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(m["runtime"]["elapsed_seconds"])
            or m["runtime"]["elapsed_seconds"] < 0
        ):
            raise ValueError("residual initialization source/configuration/target identity mismatch")
        # Reject non-JSON values such as NaN even in diagnostic fields.
        _json(m)
        self.files = {"manifest.json": expected_sha256}
        self.applied: set[str] = set()
        rank = adapter.rank
        for index, (name, shape) in enumerate(targets.items()):
            row = m["matrices"][name]
            if not isinstance(row, dict) or set(row) != {
                "source_tensor",
                "quantized_state",
                "factors",
                "reconstruction",
                "preparation_seconds",
            }:
                raise ValueError("residual initialization matrix record is incomplete")
            if (
                type(row["preparation_seconds"]) not in (int, float)
                or not math.isfinite(row["preparation_seconds"])
                or row["preparation_seconds"] < 0
            ):
                raise ValueError("invalid per-matrix preparation duration")
            factors = row["factors"]
            if (
                not isinstance(factors, dict)
                or set(factors) != {"file", "file_sha256", "lora_A", "lora_B"}
                or factors["file"] != f"{index:03d}.safetensors"
                or not _is_sha(factors["file_sha256"])
            ):
                raise ValueError("residual initialization factor file/name mismatch")
            for key, dimensions in (("lora_A", [rank, shape[1]]), ("lora_B", [shape[0], rank])):
                descriptor = factors[key]
                if (
                    not isinstance(descriptor, dict)
                    or set(descriptor) != {"shape", "dtype", "sha256"}
                    or descriptor["shape"] != dimensions
                    or descriptor["dtype"] != "float32"
                    or not _is_sha(descriptor["sha256"])
                ):
                    raise ValueError("residual initialization factor shape/dtype/hash mismatch")
            source = row["source_tensor"]
            if (
                not isinstance(source, dict)
                or set(source) != {"shape", "dtype", "sha256"}
                or source["shape"] != shape
                or source["dtype"] != "bfloat16"
                or not _is_sha(source["sha256"])
                or not isinstance(row["quantized_state"], dict)
                or not row["quantized_state"]
            ):
                raise ValueError("residual initialization source/quantized fingerprint mismatch")
            error = row["reconstruction"]
            if (
                not isinstance(error, dict)
                or set(error)
                != {
                    "before_squared_frobenius",
                    "after_squared_frobenius",
                    "relative_squared_error",
                    "rank",
                    "sketch_columns",
                    "seed",
                }
                or error["rank"] != rank
                or error["sketch_columns"] != min(rank + OVERSAMPLING, *shape)
                or error["seed"] != matrix_seed(adapter.initialization_seed, name)
                or any(
                    type(error[k]) not in (int, float) or not math.isfinite(error[k]) or error[k] < 0
                    for k in ("before_squared_frobenius", "after_squared_frobenius")
                )
                or error["after_squared_frobenius"] > error["before_squared_frobenius"] * (1 + 2e-6)
                or error["relative_squared_error"]
                != (
                    error["after_squared_frobenius"] / error["before_squared_frobenius"]
                    if error["before_squared_frobenius"]
                    else None
                )
            ):
                raise ValueError("residual initialization reconstruction record differs")
            self.files[factors["file"]] = factors["file_sha256"]
        self._stable()

    def identity(self) -> dict[str, Any]:
        return {
            "protocol": IDENTITY_PROTOCOL,
            "method": METHOD,
            "artifact_manifest_sha256": self.sha256,
            "algorithm": _plain(ALGORITHM),
            "quantized_base_changed": False,
        }

    def _stable(self) -> None:
        actual = {str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file()}
        if actual != set(self.files):
            raise ValueError("residual initialization artifact inventory differs")
        for name, expected in self.files.items():
            path = self.root / name
            if path.is_symlink() or path.resolve().parent != self.root or _sha_file(path) != expected:
                raise ValueError("residual initialization artifact changed or escaped its root")

    @torch.no_grad()
    def apply(self, name: str, module: Any, source: torch.Tensor) -> None:
        from safetensors import safe_open

        from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record, tensor_digest

        if name not in self.manifest["matrices"] or name in self.applied:
            raise ValueError("unexpected or repeated residual initialization target")
        row = self.manifest["matrices"][name]
        if tensor_digest(source) != row["source_tensor"] or quantized_state_record(module) != row["quantized_state"]:
            raise ValueError("actual source or attached NF4 representation differs from initialization artifact")
        factors = row["factors"]
        path = self.root / factors["file"]
        if path.is_symlink() or _sha_file(path) != factors["file_sha256"]:
            raise ValueError("residual initialization factor file changed")
        with safe_open(path, framework="pt", device="cpu") as reader:
            if set(reader.keys()) != {"lora_A", "lora_B"}:
                raise ValueError("residual initialization tensor inventory differs")
            values = {key: reader.get_tensor(key) for key in ("lora_A", "lora_B")}
            for key, value in values.items():
                destination = getattr(module, key).weight
                if (
                    tensor_digest(value) != factors[key]
                    or value.dtype != torch.float32
                    or not torch.isfinite(value).all()
                    or destination.is_meta
                    or destination.dtype != torch.float32
                    or destination.shape != value.shape
                    or not destination.requires_grad
                    or destination.device != module.weight.device
                ):
                    raise ValueError("residual initialization tensor/live parameter differs")
            # Verify both factors before mutating either live Parameter.
            for key, value in values.items():
                getattr(module, key).weight.copy_(value)
        if _sha_file(path) != factors["file_sha256"]:
            raise ValueError("residual initialization factors changed during read")
        self.applied.add(name)

    def assert_complete(self) -> None:
        if self.applied != set(self.manifest["matrices"]):
            raise ValueError("residual initialization did not cover every exact target")
        self._stable()


@torch.no_grad()
def prepare_initialization_artifact(
    cfg: Any, output: str | Path, *, model_config: Any, pdims: Any = None
) -> dict[str, Any]:
    """Explicit one-device preparation; no full model or training is constructed.

    The output directory is exclusively created and the manifest is published
    last with an atomic, non-overwriting hard link. Failed partial outputs have
    no committed manifest and cannot be reused as a successful preparation.
    """
    if cfg.adapter.initialization not in ("kaiming_A_zero_B", METHOD) or cfg.quantized_base is None:
        raise ValueError("residual preparation requires explicit native NF4 residual initialization")
    if cfg.quantized_base.excluded_language_layers:
        raise ValueError("residual preparation with excluded language layers is not supported")
    from safetensors import safe_open
    from safetensors.torch import save_file

    from cosmos_framework.posttrain.checkpoint.io import fsync_directory
    from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit
    from cosmos_framework.posttrain.model.quantization.loader import (
        _safetensor_inventory,
        canonical_targets,
        quantized_state_record,
        tensor_digest,
        validate_runtime,
    )
    from cosmos_framework.posttrain.model.quantization.source import validate_adapter_source

    base = validate_adapter_source(cfg)
    device = validate_runtime(cfg, model_config, pdims)
    if not 1 <= cfg.adapter.rank <= 1024:
        raise ValueError("residual preparation rank must be between 1 and 1024")
    if getattr(cfg, "adapter_compute_dtype", "float32") != "float32":
        raise ValueError("initial residual compensation requires FP32 adapter computation")
    inventory = _safetensor_inventory(Path(cfg.name_or_path))
    targets = dict(sorted(canonical_targets().items()))
    for name, shape in targets.items():
        if name + ".weight" not in inventory or inventory[name + ".weight"][1:] != (shape, "BF16"):
            raise ValueError("residual preparation source target shape/dtype differs")
        if cfg.adapter.rank > min(shape):
            raise ValueError("residual adapter rank exceeds a targeted matrix dimension")
    output = Path(output).absolute()
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    runtime = _runtime_identity()
    matrices = {}
    for index, (name, shape) in enumerate(targets.items()):
        matrix_started = time.monotonic()
        path = inventory[name + ".weight"][0]
        with safe_open(path, framework="pt", device="cpu") as reader:
            source = reader.get_tensor(name + ".weight")
            module = NativeLoraLinear4bit(
                torch.nn.Linear(shape[1], shape[0], bias=False, device="meta"),
                cfg.adapter,
                cfg.quantized_base,
            )
            module.load_dense_weight_once(source, device)
            before = quantized_state_record(module)
            effective = module.effective_weight_for_initialization()
            a, b, error = residual_factors(
                source,
                effective,
                rank=cfg.adapter.rank,
                alpha=cfg.adapter.alpha,
                seed=cfg.adapter.initialization_seed,
                target=name,
            )
            del effective
            if quantized_state_record(module) != before:
                raise RuntimeError("residual preparation changed the fixed NF4 representation")
            factors = {"lora_A": a.cpu(), "lora_B": b.cpu()}
            name_on_disk = f"{index:03d}.safetensors"
            save_file(factors, output / name_on_disk)
            matrices[name] = {
                "source_tensor": tensor_digest(source),
                "quantized_state": before,
                "factors": {
                    "file": name_on_disk,
                    "file_sha256": _sha_file(output / name_on_disk),
                    **{key: tensor_digest(value) for key, value in factors.items()},
                },
                "reconstruction": error,
                "preparation_seconds": time.monotonic() - matrix_started,
            }
            print(
                json.dumps(
                    {
                        "event": "nf4_residual_matrix_prepared",
                        "index": index + 1,
                        "total": len(targets),
                        "target": name,
                        "preparation_seconds": matrices[name]["preparation_seconds"],
                        "reconstruction": error,
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
                flush=True,
            )
            del a, b, factors, source, module
    torch.cuda.synchronize(device)
    if validate_adapter_source(cfg) != base or _runtime_identity() != runtime:
        raise RuntimeError("base source changed during residual artifact preparation")
    if {p.name for p in output.iterdir()} != {row["factors"]["file"] for row in matrices.values()}:
        raise RuntimeError("initialization output changed before manifest publication")
    for row in matrices.values():
        path = output / row["factors"]["file"]
        if path.is_symlink() or _sha_file(path) != row["factors"]["file_sha256"]:
            raise RuntimeError("initialization factors changed before manifest publication")
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    manifest = {
        "protocol": ARTIFACT_PROTOCOL,
        "algorithm": _plain(ALGORITHM),
        "base": base,
        "adapter": _adapter_record(cfg.adapter),
        "quantization": quantization_config_identity(cfg.quantized_base),
        "targets": {name: list(shape) for name, shape in targets.items()},
        "matrices": matrices,
        "runtime": {
            "identity": runtime,
            "elapsed_seconds": time.monotonic() - started,
            "scope": "Explicit per-matrix initialization preparation; no training or task-quality claim.",
        },
    }
    raw = _json(manifest) + b"\n"
    # The temporary manifest lives outside the artifact inventory; a reader can
    # only see the committed name after every factor file is complete.
    fd, temporary = tempfile.mkstemp(prefix=".nf4-initialization-", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output / "manifest.json")
        fsync_directory(str(output))
        fsync_directory(str(output.parent))
    finally:
        Path(temporary).unlink(missing_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    artifact = InitializationArtifact(
        output / "manifest.json", digest, adapter=cfg.adapter, source_identity=base, quantization=cfg.quantized_base
    )
    return {"path": str(artifact.path), "sha256": digest, "identity": artifact.identity()}
