# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from cosmos_framework.callbacks.load_pretrained import (
    _warm_start_partially_skips_ema,
    _warm_start_skips_complete_ema,
)
from cosmos_framework.data.generator.sequence_packing import PackedSequence
from cosmos_framework.model.generator.omni_mot_model import (
    REASONER_FEATURE_BATCH_KEY,
    REASONER_FEATURE_FUTURE_KEY,
    OmniMoTModel,
    _reasoner_cache_identity,
    _validate_reasoner_conditioning,
)
from cosmos_framework.model.generator.reasoner_feature_cache import (
    OfflineReasonerFeatureProvider,
    ReasonerFeatureCacheIdentity,
    build_reasoner_feature_requests,
)
from cosmos_framework.model.generator.reasoner_features import (
    CapturingReasonerKVMemoryState,
    ReasonerFeatureBatch,
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
    provider = MagicMock(spec=OfflineReasonerFeatureProvider)
    provider.manifest = SimpleNamespace(num_layers=3, num_kv_heads=2, head_dim=8)
    model.reasoner_feature_provider = provider
    net = SimpleNamespace(num_hidden_layers=3, num_kv_heads=2, head_dim=8)

    model._validate_reasoner_feature_signature(net)
    provider.manifest.num_layers = 2
    with pytest.raises(ValueError, match="architecture mismatch"):
        model._validate_reasoner_feature_signature(net)


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
