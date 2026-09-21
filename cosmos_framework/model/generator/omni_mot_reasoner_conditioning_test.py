# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import cosmos_framework.model.generator.omni_mot_model as omni_mot_model_module
from cosmos_framework.callbacks.load_pretrained import (
    _warm_start_partially_skips_ema,
    _warm_start_skips_complete_ema,
)
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.model.generator.omni_mot_model import (
    REASONER_FEATURE_BATCH_KEY,
    REASONER_FEATURE_FUTURE_KEY,
    REASONER_SAMPLE_KEYS_KEY,
    OmniMoTModel,
    _create_external_reasoner_provider_fail_closed,
    _reasoner_cache_identity,
    _validate_reasoner_conditioning,
)
from cosmos_framework.model.generator.reasoner_feature_cache import (
    ReasonerFeatureCacheIdentity,
    build_reasoner_feature_requests,
)
from cosmos_framework.model.generator.reasoner_features import (
    CapturingReasonerKVMemoryState,
    ReasonerFeatureBatch,
    ReasonerFeatureSignature,
    ReasonerLayerKV,
    StaticReasonerKVMemoryState,
)


def _config(backend: str, **conditioning_overrides: object) -> SimpleNamespace:
    conditioning = dict(
        backend=backend,
        cache_root=None,
        endpoint=None,
        reasoner_fingerprint=None,
        tokenizer_fingerprint=None,
        framing_fingerprint=None,
        strict_fingerprint=True,
        layerwise_h2d=False,
    )
    conditioning.update(conditioning_overrides)
    return SimpleNamespace(
        reasoner_conditioning=conditioning,
        joint_attn_implementation="two_way",
        parallelism=SimpleNamespace(context_parallel_shard_degree=1),
        video_temporal_causal=False,
        causal_training_strategy="none",
    )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_reasoner_conditioning_validation_fails_closed() -> None:
    assert _validate_reasoner_conditioning(_config("joint")) == "joint"
    assert _validate_reasoner_conditioning(_config("inline")) == "inline"

    with pytest.raises(ValueError, match="requires cache_root"):
        _validate_reasoner_conditioning(_config("offline"))
    with pytest.raises(ValueError, match="requires endpoint"):
        _validate_reasoner_conditioning(_config("remote"))

    with pytest.raises(ValueError, match="strict_fingerprint=true"):
        _validate_reasoner_conditioning(_config("offline", cache_root="/features"))
    with pytest.raises(ValueError, match="requires strict_fingerprint=true"):
        _validate_reasoner_conditioning(
            _config(
                "offline",
                cache_root="/features",
                reasoner_fingerprint="reasoner",
                tokenizer_fingerprint="tokenizer",
                framing_fingerprint="framing",
                strict_fingerprint=False,
            )
        )
    assert (
        _validate_reasoner_conditioning(
            _config(
                "offline",
                cache_root="/features",
                reasoner_fingerprint="reasoner",
                tokenizer_fingerprint="tokenizer",
                framing_fingerprint="framing",
            )
        )
        == "offline"
    )

    config = _config("inline")
    config.parallelism.context_parallel_shard_degree = 2
    with pytest.raises(ValueError, match="context parallel degree 1"):
        _validate_reasoner_conditioning(config)

    config = _config("inline")
    config.causal_training_strategy = "teacher_forcing"
    with pytest.raises(ValueError, match="causal_training_strategy='none'"):
        _validate_reasoner_conditioning(config)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_non_external_backend_ignores_partial_cache_identity() -> None:
    assert _reasoner_cache_identity(_config("joint", reasoner_fingerprint="bookkeeping-only")) is None
    assert _reasoner_cache_identity(_config("inline", reasoner_fingerprint="bookkeeping-only")) is None


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_provider_startup_preserves_non_distributed_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    startup_error = ValueError("identity mismatch")
    create_provider = MagicMock(side_effect=startup_error)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(ValueError) as raised:
        _create_external_reasoner_provider_fail_closed(create_provider)

    assert raised.value is startup_error


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_provider_startup_synchronizes_local_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    startup_error = ValueError("identity mismatch")
    create_provider = MagicMock(side_effect=startup_error)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages: list[str | None], local_message: str | None) -> None:
        assert local_message == "ValueError: identity mismatch"
        messages[:] = [local_message, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="rank 0: ValueError: identity mismatch") as raised:
        _create_external_reasoner_provider_fail_closed(create_provider)

    assert raised.value.__cause__ is startup_error


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_provider_startup_closes_local_provider_on_remote_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages: list[str | None], local_message: str | None) -> None:
        assert local_message is None
        messages[:] = [None, "RuntimeError: GetInfo unavailable"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="rank 1: RuntimeError: GetInfo unavailable"):
        _create_external_reasoner_provider_fail_closed(lambda: provider)

    provider.close.assert_called_once_with()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_provider_startup_returns_provider_when_all_ranks_succeed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages: list[str | None], local_message: str | None) -> None:
        assert local_message is None
        messages[:] = [None, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    assert _create_external_reasoner_provider_fail_closed(lambda: provider) is provider
    provider.close.assert_not_called()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_causal_model_rejects_uncomposed_reasoner_feature_dispatch() -> None:
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    config = SimpleNamespace(reasoner_conditioning={"backend": "offline"})
    with pytest.raises(ValueError, match="OmniMoTCausalModel currently supports only"):
        OmniMoTCausalModel(config)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_non_base_model_subclass_rejects_uncomposed_reasoner_lifecycle() -> None:
    class UnsupportedModel(OmniMoTModel):
        pass

    config = _config(
        "offline",
        cache_root="/features",
        reasoner_fingerprint="reasoner",
        tokenizer_fingerprint="tokenizer",
        framing_fingerprint="framing",
    )
    with pytest.raises(ValueError, match="does not yet compose its overridden lifecycle"):
        UnsupportedModel(config)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_offline_cache_signature_is_checked_before_model_materialization() -> None:
    model = object.__new__(OmniMoTModel)
    provider = SimpleNamespace(signature=ReasonerFeatureSignature(3, 2, 8, torch.bfloat16))
    model.reasoner_feature_provider = provider
    model.config = SimpleNamespace(precision="bfloat16")
    net = SimpleNamespace(num_hidden_layers=3, num_kv_heads=2, head_dim=8)

    model._validate_reasoner_feature_signature(net)
    provider.signature = ReasonerFeatureSignature(2, 2, 8, torch.bfloat16)
    with pytest.raises(ValueError, match="architecture mismatch"):
        model._validate_reasoner_feature_signature(net)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_signature_validation_synchronizes_local_mismatch_before_fsdp(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(OmniMoTModel)
    provider = MagicMock()
    provider.signature = ReasonerFeatureSignature(2, 2, 8, torch.bfloat16)
    model.reasoner_feature_provider = provider
    model.config = SimpleNamespace(precision="bfloat16")
    net = SimpleNamespace(num_hidden_layers=3, num_kv_heads=2, head_dim=8)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages: list[str | None], local_message: str | None) -> None:
        assert local_message is not None and "architecture mismatch" in local_message
        messages[:] = [local_message, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="rank 0: ValueError:.*architecture mismatch") as raised:
        model._validate_reasoner_feature_signature(net)

    assert isinstance(raised.value.__cause__, ValueError)
    provider.close.assert_called_once_with()
    assert model.reasoner_feature_provider is None


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_signature_validation_closes_peer_provider_on_remote_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = object.__new__(OmniMoTModel)
    provider = MagicMock()
    provider.signature = ReasonerFeatureSignature(3, 2, 8, torch.bfloat16)
    model.reasoner_feature_provider = provider
    model.config = SimpleNamespace(precision="bfloat16")
    net = SimpleNamespace(num_hidden_layers=3, num_kv_heads=2, head_dim=8)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages: list[str | None], local_message: str | None) -> None:
        assert local_message is None
        messages[:] = [None, "ValueError: architecture mismatch"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    with pytest.raises(RuntimeError, match="rank 1: ValueError: architecture mismatch"):
        model._validate_reasoner_feature_signature(net)

    provider.close.assert_called_once_with()
    assert model.reasoner_feature_provider is None


class _MemoryBuilder:
    build_memory_state = OmniMoTModel.build_memory_state

    def __init__(self, backend: str) -> None:
        self.reasoner_conditioning_backend = backend
        self.net = SimpleNamespace(
            language_model=SimpleNamespace(model=SimpleNamespace(layers=[object(), object()])),
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_build_memory_state_selects_joint_inline_and_external_modes() -> None:
    packed = SimpleNamespace()

    assert _MemoryBuilder("joint").build_memory_state(packed, {}) is None
    inline = _MemoryBuilder("inline").build_memory_state(packed, {})
    assert isinstance(inline, CapturingReasonerKVMemoryState)

    external = _MemoryBuilder("offline")
    with pytest.raises(RuntimeError, match=REASONER_FEATURE_BATCH_KEY):
        external.build_memory_state(packed, {})

    features = ReasonerFeatureBatch(
        cross_k=(torch.zeros(3, 1, 2), torch.zeros(3, 1, 2)),
        cross_v=(torch.zeros(3, 1, 2), torch.zeros(3, 1, 2)),
        causal_offsets=torch.tensor([0, 3], dtype=torch.int32),
    )
    state = external.build_memory_state(packed, {REASONER_FEATURE_BATCH_KEY: features})
    assert isinstance(state, StaticReasonerKVMemoryState)


class _InlineDenoiser(_MemoryBuilder):
    _denoise_training_with_reasoner_conditioning = OmniMoTModel._denoise_training_with_reasoner_conditioning

    def __init__(self) -> None:
        super().__init__("inline")
        self.calls: list[tuple[bool, bool]] = []

    def denoise(self, *, data_batch_packed: object, memory: object) -> dict[str, str]:
        del data_batch_packed
        assert isinstance(memory, CapturingReasonerKVMemoryState)
        self.calls.append((memory.is_gen_only(), torch.is_grad_enabled()))
        if not memory.is_gen_only():
            memory._causal_offsets = torch.tensor([0, 2], dtype=torch.int32)
            layer = ReasonerLayerKV(torch.zeros(2, 1, 2), torch.zeros(2, 1, 2))
            memory._layers = [layer, layer]
            return {"phase": "capture"}
        return {"phase": "train"}


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_inline_training_runs_capture_then_generator_only() -> None:
    model = _InlineDenoiser()

    output = model._denoise_training_with_reasoner_conditioning(SimpleNamespace(), {})

    assert model.calls == [(False, False), (True, True)]
    assert output == {"phase": "train"}


class _ExternalCheckpointLoader:
    load_pretrained_model_if_needed = OmniMoTModel.load_pretrained_model_if_needed

    def __init__(self, *, copy_from_reasoner: bool) -> None:
        self.reasoner_conditioning_backend = "offline"
        self.net = object()
        self.net_ema = object()
        self.net_ema_worker = MagicMock()
        self.config = SimpleNamespace(
            diffusion_expert_config=SimpleNamespace(load_weights_from_pretrained=copy_from_reasoner),
            ema=SimpleNamespace(enabled=True),
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_backend_requires_generator_checkpoint_for_reasoner_copy() -> None:
    loader = _ExternalCheckpointLoader(copy_from_reasoner=True)

    with pytest.raises(ValueError, match="checkpoint.load_path"):
        loader.load_pretrained_model_if_needed(has_resumable_checkpoint=False, has_load_path=False)

    loader.load_pretrained_model_if_needed(
        has_resumable_checkpoint=False,
        has_load_path=True,
        warm_start_ema_skipped=True,
    )

    loader.net_ema_worker.copy_to.assert_called_once_with(src_model=loader.net, tgt_model=loader.net_ema)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_backend_preserves_loaded_warm_start_ema() -> None:
    loader = _ExternalCheckpointLoader(copy_from_reasoner=True)

    loader.load_pretrained_model_if_needed(
        has_resumable_checkpoint=False,
        has_load_path=True,
        warm_start_ema_skipped=False,
    )

    loader.net_ema_worker.copy_to.assert_not_called()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_warm_start_ema_skip_detection_requires_a_root_wide_pattern() -> None:
    assert _warm_start_skips_complete_ema(["net_ema."])
    assert _warm_start_skips_complete_ema(["ema"])
    assert not _warm_start_skips_complete_ema([])
    assert not _warm_start_skips_complete_ema(["net_ema.language_model.model.layers.0"])
    assert not _warm_start_skips_complete_ema(["model.net_ema."])


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_warm_start_partial_ema_skip_detection_uses_actual_state_leaves() -> None:
    ema_fqns = [
        "net_ema.language_model.layer.0.weight",
        "net_ema.language_model.layer.1.weight",
        "net_ema.vfm.weight",
    ]

    assert _warm_start_partially_skips_ema(["layer.0"], ema_fqns)
    assert _warm_start_partially_skips_ema(["language_model"], ema_fqns)
    assert not _warm_start_partially_skips_ema(["net_ema."], ema_fqns)
    assert _warm_start_skips_complete_ema(["language_model", "vfm"], ema_fqns)
    assert not _warm_start_partially_skips_ema(["missing"], ema_fqns)


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_backend_rejects_partial_ema_warm_start() -> None:
    loader = _ExternalCheckpointLoader(copy_from_reasoner=True)

    with pytest.raises(ValueError, match="matches only part of the 'net_ema.' subtree"):
        loader.load_pretrained_model_if_needed(
            has_resumable_checkpoint=False,
            has_load_path=True,
            warm_start_ema_partially_skipped=True,
            warm_start_strict_resume=True,
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_backend_rejects_ambiguous_non_strict_ema_warm_start() -> None:
    loader = _ExternalCheckpointLoader(copy_from_reasoner=True)

    with pytest.raises(ValueError, match="cannot safely use a non-strict warm start"):
        loader.load_pretrained_model_if_needed(
            has_resumable_checkpoint=False,
            has_load_path=True,
            warm_start_ema_skipped=False,
            warm_start_strict_resume=False,
        )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_backend_allows_explicit_random_generator_initialization() -> None:
    loader = _ExternalCheckpointLoader(copy_from_reasoner=False)

    loader.load_pretrained_model_if_needed(has_resumable_checkpoint=False, has_load_path=False)

    loader.net_ema_worker.copy_to.assert_not_called()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_external_backend_resume_preserves_checkpoint_ema() -> None:
    loader = _ExternalCheckpointLoader(copy_from_reasoner=True)

    # ``load_path`` may remain configured while a latest same-job checkpoint
    # takes precedence. That resume restores EMA and must not reset it from net.
    loader.load_pretrained_model_if_needed(has_resumable_checkpoint=True, has_load_path=True)

    loader.net_ema_worker.copy_to.assert_not_called()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_build_reasoner_requests_uses_framed_text_positions_and_sample_offsets() -> None:
    packed = PackedSequence(
        sample_lens=[5, 5],
        split_lens=[2, 3, 3, 2],
        attn_modes=["causal", "full", "causal", "full"],
        sequence_length=10,
        text_ids=torch.tensor([11, 12, 21, 22, 23]),
        text_indexes=torch.tensor([0, 1, 5, 6, 7]),
        position_ids=torch.arange(30, dtype=torch.float32).reshape(3, 10),
    )
    identity = ReasonerFeatureCacheIdentity("reasoner", "tokenizer", "framing")

    requests = build_reasoner_feature_requests(packed, ("first", "second"), identity)

    assert [request.sample_key for request in requests] == ["first", "second"]
    assert requests[0].token_ids.tolist() == [11, 12]
    assert requests[1].token_ids.tolist() == [21, 22, 23]
    assert requests[0].position_ids.shape == (3, 2)
    assert requests[1].position_ids.shape == (3, 3)
    assert requests[0].causal_offsets.tolist() == [0, 2]
    assert requests[1].causal_offsets.tolist() == [0, 3]
    assert requests[0].fingerprint != requests[1].fingerprint


class _FeatureResolver:
    _resolve_reasoner_feature_batch = OmniMoTModel._resolve_reasoner_feature_batch

    def __init__(self) -> None:
        self.config = SimpleNamespace(reasoner_conditioning={"request_timeout_s": 1.0})


class _FeatureSubmitter(_FeatureResolver):
    pre_noise_memory_hook = OmniMoTModel.pre_noise_memory_hook

    def __init__(self, provider: object) -> None:
        super().__init__()
        self.reasoner_conditioning_backend = "remote"
        self.reasoner_feature_provider = provider
        self.reasoner_cache_identity = ReasonerFeatureCacheIdentity("reasoner", "tokenizer", "framing")


class _MemoryInitializer(_FeatureResolver):
    memory_init_training = OmniMoTModel.memory_init_training
    pre_noise_memory_hook = OmniMoTModel.pre_noise_memory_hook

    def __init__(self, provider: object) -> None:
        super().__init__()
        self.reasoner_conditioning_backend = "remote"
        self.reasoner_feature_provider = provider
        self.reasoner_cache_identity = ReasonerFeatureCacheIdentity("reasoner", "tokenizer", "framing")


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize(
    ("data_batch", "batch_size", "error_match"),
    (
        ({}, 1, "requires a per-sample '__key__' field"),
        ({"__key__": ["only-one"]}, 2, "Expected 2 non-empty Reasoner sample keys"),
    ),
)
def test_sample_key_failure_is_deferred_to_distributed_resolution_gate(
    data_batch: dict[str, object],
    batch_size: int,
    error_match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock()
    model = _MemoryInitializer(provider)
    gen_data_clean = SimpleNamespace(batch_size=batch_size)

    returned, memory_info = model.memory_init_training(gen_data_clean, data_batch, [])

    assert returned is gen_data_clean
    assert REASONER_SAMPLE_KEYS_KEY not in memory_info
    failed_future = memory_info[REASONER_FEATURE_FUTURE_KEY]
    assert isinstance(failed_future, Future)
    assert failed_future.done()
    assert isinstance(failed_future.exception(), ValueError)

    # The pre-noise hook must preserve the failed Future rather than replacing
    # it with a later provider submission.
    assert model.pre_noise_memory_hook(object(), object(), memory_info) is memory_info
    assert memory_info[REASONER_FEATURE_FUTURE_KEY] is failed_future
    provider.submit.assert_not_called()

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages: list[str | None], local_message: str | None) -> None:
        assert local_message is not None and error_match in local_message
        messages[:] = [local_message, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="failed before FSDP forward") as raised:
        model._resolve_reasoner_feature_batch(memory_info)

    assert error_match in str(raised.value)
    assert raised.value.__cause__ is failed_future.exception()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_sample_key_extraction_preserves_base_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MagicMock()
    model = _MemoryInitializer(provider)
    gen_data_clean = SimpleNamespace(batch_size=1)
    monkeypatch.setattr(
        omni_mot_model_module,
        "_reasoner_sample_keys",
        MagicMock(side_effect=KeyboardInterrupt("stop")),
    )

    with pytest.raises(KeyboardInterrupt, match="stop"):
        model.memory_init_training(gen_data_clean, {"__key__": ["sample"]}, [])


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_resolve_reasoner_features_consumes_future_and_surfaces_failure() -> None:
    features = ReasonerFeatureBatch(
        cross_k=(torch.zeros(2, 1, 2),),
        cross_v=(torch.zeros(2, 1, 2),),
        causal_offsets=torch.tensor([0, 2]),
        fingerprints=("fingerprint",),
    )
    successful: Future[ReasonerFeatureBatch] = Future()
    successful.set_result(features)
    memory_info = {REASONER_FEATURE_FUTURE_KEY: successful}

    assert _FeatureResolver()._resolve_reasoner_feature_batch(memory_info) is features
    assert memory_info[REASONER_FEATURE_BATCH_KEY] is features
    assert REASONER_FEATURE_FUTURE_KEY not in memory_info

    failed: Future[ReasonerFeatureBatch] = Future()
    failed.set_exception(KeyError("missing"))
    with pytest.raises(RuntimeError, match="failed before FSDP forward"):
        _FeatureResolver()._resolve_reasoner_feature_batch({REASONER_FEATURE_FUTURE_KEY: failed})


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("failure_stage", ("provider", "identity", "sample_keys", "build", "submit"))
def test_synchronous_reasoner_submission_failure_reaches_distributed_resolution_gate(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock()
    build_requests = MagicMock(return_value=(object(),))
    if failure_stage == "build":
        build_requests.side_effect = ValueError("build failed")
    else:
        provider.submit.side_effect = ValueError("submit failed") if failure_stage == "submit" else None
    monkeypatch.setattr(omni_mot_model_module, "build_reasoner_feature_requests", build_requests)

    submitter = _FeatureSubmitter(provider)
    memory_info = {REASONER_SAMPLE_KEYS_KEY: ("sample",)}
    if failure_stage == "provider":
        submitter.reasoner_feature_provider = None
    elif failure_stage == "identity":
        submitter.reasoner_cache_identity = None
    elif failure_stage == "sample_keys":
        memory_info.clear()
    assert submitter.pre_noise_memory_hook(object(), object(), memory_info) is memory_info
    future = memory_info[REASONER_FEATURE_FUTURE_KEY]
    assert isinstance(future, Future)
    assert future.done()
    deferred_error = future.exception()
    assert isinstance(deferred_error, Exception)
    local_message = f"{type(deferred_error).__name__}: {deferred_error}"

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages: list[str | None], gathered_local_message: str | None) -> None:
        assert gathered_local_message == local_message
        messages[:] = [gathered_local_message, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="failed before FSDP forward") as raised:
        submitter._resolve_reasoner_feature_batch(memory_info)

    assert f"rank 0: {local_message}" in str(raised.value)
    assert raised.value.__cause__ is deferred_error
    if failure_stage in {"provider", "identity", "sample_keys", "build"}:
        provider.submit.assert_not_called()
    else:
        provider.submit.assert_called_once()


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("failure_stage", ("build", "submit"))
def test_reasoner_submission_preserves_base_exception_control_flow(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MagicMock()
    build_requests = MagicMock(return_value=(object(),))
    if failure_stage == "build":
        build_requests.side_effect = KeyboardInterrupt("stop")
    else:
        provider.submit.side_effect = KeyboardInterrupt("stop")
    monkeypatch.setattr(omni_mot_model_module, "build_reasoner_feature_requests", build_requests)

    memory_info = {REASONER_SAMPLE_KEYS_KEY: ("sample",)}
    with pytest.raises(KeyboardInterrupt, match="stop"):
        _FeatureSubmitter(provider).pre_noise_memory_hook(object(), object(), memory_info)

    assert REASONER_FEATURE_FUTURE_KEY not in memory_info
