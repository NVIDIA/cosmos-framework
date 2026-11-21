# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Numerical/cache contracts on CPU; these do not qualify an actual NF4 kernel."""

from __future__ import annotations

import copy
import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from cosmos_framework.posttrain.config.model import (
    NativeLoRAConfig,
    QuantizedBaseTrainingConfig,
    quantization_config_identity,
)
from cosmos_framework.posttrain.model.quantization import initialization as init
from cosmos_framework.posttrain.model.quantization import loader
from cosmos_framework.utils.generator.lora import LoraInjectedLinear

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def matrices():
    g = torch.Generator().manual_seed(9)
    source = torch.randn(19, 13, generator=g).to(torch.bfloat16)
    effective = (source.float() * 4).round().div(4).to(torch.bfloat16)
    return source, effective


def test_scaled_factors_reduce_actual_residual_and_are_nonzero_without_mutating_inputs():
    source, effective = matrices()
    before = source.clone(), effective.clone()
    a, b, report = init.residual_factors(source, effective, rank=3, alpha=12, seed=42, target="layer")
    delta = 4 * (b @ a)
    residual = source.float() - effective.float()
    singular = torch.linalg.svdvals(residual)
    assert a.dtype == b.dtype == torch.float32
    assert torch.count_nonzero(a) and torch.count_nonzero(b)
    assert torch.equal(source, before[0]) and torch.equal(effective, before[1])
    assert report["after_squared_frobenius"] < report["before_squared_frobenius"]
    assert torch.allclose((residual - delta).square().sum(), singular[3:].square().sum(), atol=1e-6)
    assert report["after_squared_frobenius"] == pytest.approx(
        (residual - delta).double().square().sum().item(), rel=1e-6
    )
    # Different alpha must change the factors while preserving the scaled update.
    a2, b2, _ = init.residual_factors(source, effective, rank=3, alpha=3, seed=42, target="layer")
    assert torch.allclose(delta, b2 @ a2, atol=1e-6)


def test_explicit_target_seed_is_order_independent_and_does_not_consume_global_rng():
    source, effective = matrices()
    state = torch.random.get_rng_state().clone()
    first = init.residual_factors(source, effective, rank=2, alpha=4, seed=17, target="one")
    assert torch.equal(state, torch.random.get_rng_state())
    torch.rand(39)
    init.residual_factors(source, effective, rank=2, alpha=4, seed=123, target="other")
    second = init.residual_factors(source, effective, rank=2, alpha=4, seed=17, target="one")
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert first[2] == second[2]
    assert init.matrix_seed(17, "one") != init.matrix_seed(17, "other")


@pytest.mark.parametrize("change", ["source_dtype", "effective_dtype", "shape", "rank", "alpha", "seed", "nan"])
def test_invalid_residual_input_is_rejected(change):
    source, effective = matrices()
    options = dict(rank=2, alpha=4, seed=42, target="one")
    if change == "source_dtype":
        source = source.float()
    elif change == "effective_dtype":
        effective = effective.float()
    elif change == "shape":
        effective = effective[:-1]
    elif change == "rank":
        options["rank"] = 14
    elif change == "alpha":
        options["alpha"] = float("inf")
    elif change == "seed":
        options["seed"] = True
    elif change == "nan":
        effective[0, 0] = float("nan")
    with pytest.raises(ValueError):
        init.residual_factors(source, effective, **options)


def test_zero_residual_is_finite_and_does_not_invent_improvement():
    source, _ = matrices()
    a, b, report = init.residual_factors(source, source, rank=2, alpha=4, seed=42, target="one")
    assert not torch.count_nonzero(a) and not torch.count_nonzero(b)
    assert report["before_squared_frobenius"] == report["after_squared_frobenius"] == 0
    assert report["relative_squared_error"] is None


@pytest.fixture
def cached(tmp_path, monkeypatch):
    source, effective = matrices()
    shape = tuple(source.shape)
    monkeypatch.setattr(loader, "canonical_targets", lambda: {"layer": shape})
    monkeypatch.setattr(loader, "quantized_state_record", lambda m: {"weight": loader.tensor_digest(m.weight)})
    monkeypatch.setattr(init, "_runtime_identity", lambda: {"fixture": "cpu"})
    config = NativeLoRAConfig(
        rank=3,
        alpha=12,
        initialization=init.METHOD,
        initialization_artifact=str(tmp_path / "manifest.json"),
        initialization_artifact_sha256="a" * 64,
    )
    quantization = QuantizedBaseTrainingConfig()
    module = LoraInjectedLinear(
        torch.nn.Linear(shape[1], shape[0], bias=False, dtype=torch.bfloat16),
        config.rank,
        config.alpha,
        adapter_dtype=torch.float32,
    )
    module.weight.requires_grad_(False)
    with torch.no_grad():
        module.weight.copy_(effective)
    a, b, error = init.residual_factors(source, effective, rank=3, alpha=12, seed=42, target="layer")
    factors = {"lora_A": a, "lora_B": b}
    save_file(factors, tmp_path / "000.safetensors")
    manifest = {
        "protocol": init.ARTIFACT_PROTOCOL,
        "algorithm": copy.deepcopy(init.ALGORITHM),
        "base": {"revision": "fixed-source"},
        "adapter": init._adapter_record(config),
        "quantization": quantization_config_identity(quantization),
        "targets": {"layer": list(shape)},
        "runtime": {"identity": {"fixture": "cpu"}, "elapsed_seconds": 0.0, "scope": "CPU fixture"},
        "matrices": {
            "layer": {
                "source_tensor": loader.tensor_digest(source),
                "quantized_state": loader.quantized_state_record(module),
                "factors": {
                    "file": "000.safetensors",
                    "file_sha256": init._sha_file(tmp_path / "000.safetensors"),
                    **{k: loader.tensor_digest(v) for k, v in factors.items()},
                },
                "reconstruction": error,
                "preparation_seconds": 0.0,
            }
        },
    }

    def publish():
        raw = init._json(manifest)
        (tmp_path / "manifest.json").write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        config.initialization_artifact_sha256 = digest
        return init.InitializationArtifact(
            tmp_path / "manifest.json",
            digest,
            adapter=config,
            source_identity=manifest["base"],
            quantization=quantization,
        )

    artifact = publish()
    return SimpleNamespace(
        root=tmp_path,
        module=module,
        source=source,
        a=a,
        b=b,
        artifact=artifact,
        manifest=manifest,
        publish=publish,
        config=config,
        quantization=quantization,
    )


def test_cached_factors_apply_without_svd_or_parameter_replacement_and_train_both_factors(cached, monkeypatch):
    old_ids = [id(cached.module.lora_A.weight), id(cached.module.lora_B.weight)]
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("build/resume must not decompose"))
    monkeypatch.setattr(init, "residual_factors", lambda *a, **k: pytest.fail("no implicit preparation"))
    cached.artifact.apply("layer", cached.module, cached.source)
    cached.artifact.assert_complete()
    assert old_ids == [id(cached.module.lora_A.weight), id(cached.module.lora_B.weight)]
    assert torch.equal(cached.module.lora_A.weight, cached.a)
    assert torch.equal(cached.module.lora_B.weight, cached.b)
    x = torch.randn(4, 13, dtype=torch.bfloat16, requires_grad=True)
    base = torch.nn.functional.linear(x, cached.module.weight)
    actual = cached.module(x)
    assert not torch.equal(base, actual)  # Intentionally not zero-LoRA parity.
    actual.float().square().mean().backward()
    assert torch.count_nonzero(cached.module.lora_A.weight.grad)
    assert torch.count_nonzero(cached.module.lora_B.weight.grad)
    assert torch.count_nonzero(x.grad) and cached.module.weight.grad is None
    init.validate_initialization_record(cached.artifact.identity())
    with pytest.raises(ValueError, match="repeated"):
        cached.artifact.apply("layer", cached.module, cached.source)


@pytest.mark.parametrize(
    "change",
    [
        "algorithm",
        "rank",
        "runtime",
        "quantization",
        "source",
        "target",
        "traversal",
        "dtype",
        "factor_shape",
        "duplicate_file",
        "error",
    ],
)
def test_cache_manifest_refuses_changed_contract(cached, change):
    m = cached.manifest
    if change == "algorithm":
        m["algorithm"]["power_iterations"] += 1
    elif change == "rank":
        m["adapter"]["rank"] += 1
    elif change == "runtime":
        m["runtime"]["identity"] = {"fixture": "changed"}
    elif change == "quantization":
        m["quantization"]["nested"] = False
    elif change == "source":
        m["base"] = {"revision": "different"}
    elif change == "target":
        m["targets"] = {"other": [19, 13]}
    elif change == "traversal":
        m["matrices"]["layer"]["factors"]["file"] = "../000.safetensors"
    elif change == "dtype":
        m["matrices"]["layer"]["factors"]["lora_A"]["dtype"] = "bfloat16"
    elif change == "factor_shape":
        m["matrices"]["layer"]["factors"]["lora_B"]["shape"] = [19, 4]
    elif change == "duplicate_file":
        (cached.root / "extra.safetensors").write_bytes(b"extra")
    elif change == "error":
        m["matrices"]["layer"]["reconstruction"]["after_squared_frobenius"] *= 100
    if change == "source":
        raw = init._json(m)
        (cached.root / "manifest.json").write_bytes(raw)
        with pytest.raises(ValueError, match="identity"):
            init.InitializationArtifact(
                cached.root / "manifest.json",
                hashlib.sha256(raw).hexdigest(),
                adapter=cached.config,
                source_identity={"revision": "fixed-source"},
                quantization=cached.quantization,
            )
    else:
        with pytest.raises(ValueError):
            cached.publish()


@pytest.mark.parametrize("change", ["source", "packed", "file_bytes", "factor_nan", "extra_tensor", "missing_tensor"])
def test_cache_apply_rejects_mutation_before_changing_parameters(cached, change):
    before = cached.module.lora_A.weight.clone(), cached.module.lora_B.weight.clone()
    if change == "source":
        cached.source[0, 0] += 1
    elif change == "packed":
        with torch.no_grad():
            cached.module.weight[0, 0] += 1
    elif change == "file_bytes":
        (cached.root / "000.safetensors").write_bytes(b"corrupt")
    else:
        factors = load_file(cached.root / "000.safetensors")
        if change == "factor_nan":
            factors["lora_A"][0, 0] = float("nan")
        elif change == "extra_tensor":
            factors["extra"] = torch.zeros(1)
        else:
            factors.pop("lora_B")
        save_file(factors, cached.root / "000.safetensors")
        record = cached.manifest["matrices"]["layer"]["factors"]
        record["file_sha256"] = init._sha_file(cached.root / "000.safetensors")
        if change == "factor_nan":
            record["lora_A"] = loader.tensor_digest(factors["lora_A"])
        cached.artifact = cached.publish()
    with pytest.raises(ValueError):
        cached.artifact.apply("layer", cached.module, cached.source)
    assert torch.equal(cached.module.lora_A.weight, before[0])
    assert torch.equal(cached.module.lora_B.weight, before[1])


def test_missing_cache_never_prepares_and_incomplete_or_late_changed_cache_fails(cached, monkeypatch):
    monkeypatch.setattr(init, "prepare_initialization_artifact", lambda *a, **k: pytest.fail("no fallback"))
    with pytest.raises(ValueError, match="every exact target"):
        cached.artifact.assert_complete()
    with pytest.raises(ValueError, match="committed"):
        init.InitializationArtifact(
            cached.root / "absent.json",
            "a" * 64,
            adapter=cached.config,
            source_identity=cached.manifest["base"],
            quantization=cached.quantization,
        )
    cached.artifact.apply("layer", cached.module, cached.source)
    (cached.root / "000.safetensors").write_bytes(b"late change")
    with pytest.raises(ValueError, match="changed"):
        cached.artifact.assert_complete()


def test_v3_identity_binds_artifact_and_preserves_legacy_default_policy(cached):
    identity = {
        "format": "cosmos.adapter-model.v3",
        "initialization": cached.artifact.identity(),
        "adapter": {"initialization": init.METHOD, "initialization_artifact_sha256": cached.artifact.sha256},
        "quantization": quantization_config_identity(cached.quantization),
    }
    assert loader.execution_config_from_identity(identity) == loader.execution_config_from_identity(
        {"format": "cosmos.adapter-model.v1"}
    )
    for field in ("initialization", "quantization", "adapter"):
        invalid = copy.deepcopy(identity)
        invalid.pop(field)
        with pytest.raises(ValueError):
            loader.execution_config_from_identity(invalid)
    invalid = copy.deepcopy(identity)
    invalid["adapter"]["initialization_artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        loader.execution_config_from_identity(invalid)
    invalid = copy.deepcopy(identity)
    invalid["quantization"]["format"] = "other"
    with pytest.raises(ValueError, match="NF4"):
        loader.execution_config_from_identity(invalid)
    invalid = copy.deepcopy(identity)
    invalid["execution_policy"] = loader.resolved_execution_policy(
        SimpleNamespace(
            activation_checkpointing="full",
            adapter_compute_dtype="bfloat16",
            sft_logits_mode="full",
            compile_language_model=False,
        )
    )
    with pytest.raises(ValueError, match="FP32"):
        loader.execution_config_from_identity(invalid)
    identity["format"] = "cosmos.adapter-model.v2"
    with pytest.raises(ValueError, match="v3"):
        loader.execution_config_from_identity(identity)


@pytest.mark.parametrize("purpose", ["reload", "resume"])
def test_legacy_residual_precision_compatibility_preserves_artifact_identity(cached, purpose):
    from cosmos_framework.posttrain.tests.test_checkpoint_adapter import _trainer

    trainer = _trainer()
    codec = trainer._adapter_checkpoint_codec()
    codec.identity.update(
        format="cosmos.adapter-model.v3",
        initialization=cached.artifact.identity(),
        adapter={"initialization": init.METHOD, "initialization_artifact_sha256": cached.artifact.sha256},
        quantization=quantization_config_identity(cached.quantization),
        execution_policy=loader.resolved_execution_policy(
            SimpleNamespace(activation_checkpointing="full"), include_defaults=True
        ),
    )
    optimizer = trainer.opt if purpose == "resume" else None
    original = copy.deepcopy(codec.identity)
    legacy = codec.metadata(optimizer)
    legacy["identity"].pop("execution_policy")
    with pytest.warns(UserWarning, match="no recorded runtime math precision"):
        codec.validate_metadata(legacy, optimizer, purpose=purpose)
    assert codec.identity == original
    # A different, structurally valid residual artifact is never a precision-only difference.
    changed = copy.deepcopy(legacy)
    changed["identity"]["initialization"]["artifact_manifest_sha256"] = "b" * 64
    changed["identity"]["adapter"]["initialization_artifact_sha256"] = "b" * 64
    loader.execution_config_from_identity(changed["identity"])
    with pytest.warns(UserWarning, match="no recorded runtime math precision"):
        with pytest.raises(ValueError, match="identity differs"):
            codec.validate_metadata(changed, optimizer, purpose=purpose)


def test_explicit_preparation_uses_per_matrix_backend_and_publishes_cache_last(tmp_path, monkeypatch):
    """Synthetic transport fixture, not bitsandbytes/CUDA execution evidence."""
    from cosmos_framework.posttrain.model.quantization import source as source_module

    source, effective = matrices()
    snapshot = tmp_path / "base"
    snapshot.mkdir()
    save_file({"layer.weight": source}, snapshot / "model.safetensors")
    monkeypatch.setattr(loader, "canonical_targets", lambda: {"layer": tuple(source.shape)})
    monkeypatch.setattr(loader, "validate_runtime", lambda *a: torch.device("cpu"))
    monkeypatch.setattr(loader, "runtime_precision_policy", lambda: {"fixture": "cpu"})
    monkeypatch.setattr(loader, "quantized_state_record", lambda m: {"weight": loader.tensor_digest(m.weight)})
    monkeypatch.setattr(source_module, "validate_adapter_source", lambda cfg: {"revision": "fixture"})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "cpu-fixture")
    monkeypatch.setattr(init, "version", lambda name: "fixture")
    monkeypatch.setattr(init, "_runtime_identity", lambda: {"fixture": "cpu"})
    calls = []

    class Matrix:
        def __init__(self, base, adapter, quantization):
            assert base.weight.is_meta
            calls.append(tuple(base.weight.shape))

        def load_dense_weight_once(self, value, device):
            assert torch.equal(value, source)
            self.weight = effective.clone()

        def effective_weight_for_initialization(self):
            return self.weight.clone()

    monkeypatch.setitem(
        sys.modules, "cosmos_framework.posttrain.model.quantization.bnb", SimpleNamespace(NativeLoraLinear4bit=Matrix)
    )
    cfg = SimpleNamespace(
        adapter=NativeLoRAConfig(rank=3, alpha=12),
        quantized_base=QuantizedBaseTrainingConfig(),
        name_or_path=str(snapshot),
    )
    out = tmp_path / "result"
    result = init.prepare_initialization_artifact(cfg, out, model_config=object())
    assert calls == [tuple(source.shape)] and result["path"] == str(out / "manifest.json")
    assert result["sha256"] == init._sha_file(out / "manifest.json")
    assert set(p.name for p in out.iterdir()) == {"000.safetensors", "manifest.json"}
    with pytest.raises(FileExistsError):
        init.prepare_initialization_artifact(cfg, out, model_config=object())
    assert calls == [tuple(source.shape)]


def test_native_load_finalize_and_checkpoint_rebuild_reuse_all_cached_factors_without_solver(tmp_path, monkeypatch):
    """Full 504-parameter CPU transport/codec fixture; no NF4 kernel qualification."""
    from cosmos_framework.utils.generator.lora import NativeLoRAMixin

    targets = {f"layer_{index}": (2, 2) for index in range(252)}
    source = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    save_file({name + ".weight": source.clone() for name in targets}, snapshot / "model.safetensors")
    cache = tmp_path / "cache"
    cache.mkdir()
    adapter = NativeLoRAConfig(
        rank=1,
        alpha=2,
        initialization=init.METHOD,
        initialization_artifact=str(cache / "manifest.json"),
        initialization_artifact_sha256="a" * 64,
    )
    quant = QuantizedBaseTrainingConfig()
    monkeypatch.setattr(loader, "canonical_targets", lambda: targets)
    monkeypatch.setattr(loader, "quantized_state_record", lambda m: {"packed": loader.tensor_digest(m.weight)})
    monkeypatch.setattr(loader, "version", lambda name: "cpu-fixture")
    monkeypatch.setattr(init, "_runtime_identity", lambda: {"fixture": "cpu"})
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "cpu-fixture")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (0, 0))
    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)
    monkeypatch.setattr(init, "residual_factors", lambda *a, **k: pytest.fail("model construction cannot solve"))
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **k: pytest.fail("resume cannot decompose"))

    class Matrix(NativeLoRAMixin, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.uint8), requires_grad=False)
            self.lora_A = torch.nn.Linear(2, 1, bias=False, dtype=torch.float32)
            self.lora_B = torch.nn.Linear(1, 2, bias=False, dtype=torch.float32)

        def load_dense_weight_once(self, value, device):
            assert device == torch.device("cpu") and torch.equal(value, source)

        def reset_adapter_parameters(self, **kwargs):
            pytest.fail("residual cache must not be overwritten with zero-B initialization")

    factors = {"lora_A": torch.full((1, 2), 0.125), "lora_B": torch.full((2, 1), 0.25)}
    records = {}
    for index, name in enumerate(sorted(targets)):
        filename = f"{index:03d}.safetensors"
        save_file(factors, cache / filename)
        records[name] = {
            "source_tensor": loader.tensor_digest(source),
            "quantized_state": loader.quantized_state_record(Matrix()),
            "factors": {
                "file": filename,
                "file_sha256": init._sha_file(cache / filename),
                **{k: loader.tensor_digest(v) for k, v in factors.items()},
            },
            "reconstruction": {
                "before_squared_frobenius": 1.0,
                "after_squared_frobenius": 0.5,
                "relative_squared_error": 0.5,
                "rank": 1,
                "sketch_columns": 2,
                "seed": init.matrix_seed(42, name),
            },
            "preparation_seconds": 0.0,
        }
    raw = init._json(
        {
            "protocol": init.ARTIFACT_PROTOCOL,
            "algorithm": init.ALGORITHM,
            "base": {"fixture": "source"},
            "adapter": init._adapter_record(adapter),
            "quantization": quantization_config_identity(quant),
            "targets": {k: list(v) for k, v in sorted(targets.items())},
            "matrices": records,
            "runtime": {"identity": {"fixture": "cpu"}, "elapsed_seconds": 0.0, "scope": "CPU fixture"},
        }
    )
    (cache / "manifest.json").write_bytes(raw)
    adapter.initialization_artifact_sha256 = hashlib.sha256(raw).hexdigest()

    def rebuild():
        net = torch.nn.ModuleDict({name: Matrix() for name in targets})
        lifecycle = loader.NativeAdapterLifecycle.__new__(loader.NativeAdapterLifecycle)
        lifecycle.cfg = SimpleNamespace(
            adapter=adapter, quantized_base=quant, name_or_path=str(snapshot), activation_checkpointing="full"
        )
        lifecycle.device, lifecycle.loaded, lifecycle.quantized = torch.device("cpu"), False, True
        lifecycle.targets = dict(net.items())
        lifecycle.quantized_targets = dict(lifecycle.targets)
        lifecycle.inventory = loader._safetensor_inventory(snapshot)
        lifecycle.source_slots = {name + ".weight": loader.TensorSlot(module, "weight") for name, module in net.items()}
        lifecycle.source_identity, lifecycle.buffer_slots = {"fixture": "source"}, {}
        lifecycle.runtime_precision, lifecycle.execution_policy, lifecycle.canary = (
            loader.runtime_precision_policy(),
            None,
            None,
        )
        lifecycle.initialization_artifact = init.InitializationArtifact(
            cache / "manifest.json",
            adapter.initialization_artifact_sha256,
            adapter=adapter,
            source_identity=lifecycle.source_identity,
            quantization=quant,
        )
        lifecycle.load(net, str(snapshot), None, None)
        lifecycle.finalize(net)
        assert len(net._cosmos_adapter_parameters) == 504
        assert net._cosmos_adapter_metadata["format"] == "cosmos.adapter-model.v3"
        assert all(torch.equal(module.lora_B.weight, factors["lora_B"]) for module in net.values())
        return net

    first, restored = rebuild(), rebuild()
    assert first._cosmos_adapter_metadata == restored._cosmos_adapter_metadata
    with torch.no_grad():
        for parameter in first._cosmos_adapter_parameters.values():
            parameter.add_(0.01)
    restored._cosmos_adapter_checkpoint_codec.view.load_state_dict(
        first._cosmos_adapter_checkpoint_codec.view.state_dict()
    )
    for name, value in first._cosmos_adapter_parameters.items():
        assert torch.equal(value, restored._cosmos_adapter_parameters[name])
    restored._cosmos_validate_frozen_base()
