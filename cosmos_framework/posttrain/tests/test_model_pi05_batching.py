# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Selected-row batching preserves recorded inputs and releases failed packed forwards."""

import gc
import traceback
import weakref
from dataclasses import replace

import pytest
import torch

from cosmos_framework.posttrain.model.pi05 import policy
from cosmos_framework.posttrain.model.pi05.config import PALIGEMMA_VOCAB_SIZE, Pi05ModelConfig
from cosmos_framework.posttrain.model.pi05.ppo import Pi05PPOCondition, Pi05PPOTrace
from cosmos_framework.posttrain.model.pi05.processing import Pi05ReplayInputs
from cosmos_framework.posttrain.model.pi05.records import (
    INPUTS_KEY,
    TRACE_KEY,
    pack_ppo_decisions,
    validate_ppo_decision,
)
from cosmos_framework.posttrain.schema import ObsBundle, Transition
from cosmos_framework.posttrain.tests.helpers.pi05_loading import SmallPi05Model, write_source

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.fixture(autouse=True)
def restore_torch_settings():
    precision, threads = torch.get_float32_matmul_precision(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.set_num_threads(threads)


def _invocation(seed: int, denoise_index: int) -> tuple[Pi05ReplayInputs, Pi05PPOTrace]:
    generator = torch.Generator().manual_seed(seed)
    batch = 2
    inputs = Pi05ReplayInputs(
        tuple(torch.randint(0, 256, (batch, 224, 224, 3), generator=generator, dtype=torch.uint8) for _ in range(2)),
        torch.tensor([[True, True, False]]).expand(batch, -1),
        torch.randint(0, 16, (batch, 200), generator=generator),
        torch.rand((batch, 200), generator=generator) > 0.25,
        torch.randn((batch, 32), generator=generator, dtype=torch.float64),
    )
    trace = Pi05PPOTrace(
        torch.randn((batch, 10, 32), generator=generator, dtype=torch.float32),
        torch.randn((batch, 10, 32), generator=generator, dtype=torch.float32),
        torch.full((batch,), denoise_index, dtype=torch.int64),
    )
    return inputs, trace


def _decision(invocation: tuple[Pi05ReplayInputs, Pi05PPOTrace], row: int) -> Transition:
    condition, trace = invocation
    return Transition(
        obs=ObsBundle(representations={INPUTS_KEY: condition}),
        policy_info={TRACE_KEY: trace, "sample_index": row},
    )


def _leaves(condition: Pi05ReplayInputs | Pi05PPOCondition, trace: Pi05PPOTrace) -> tuple[torch.Tensor, ...]:
    conditioning = (
        (*condition.images, condition.image_masks, condition.tokens, condition.token_mask, condition.state)
        if isinstance(condition, Pi05ReplayInputs)
        else (
            *(tensor for pair in condition.prefix_cache for tensor in pair),
            condition.prefix_mask,
            condition.value_features,
        )
    )
    return (
        *conditioning,
        trace.latent_before,
        trace.latent_after,
        trace.denoise_index,
    )


@pytest.fixture
def loaded_policy(tmp_path, monkeypatch):
    config = write_source(tmp_path)
    monkeypatch.setattr(policy, "Pi05Model", SmallPi05Model)
    loaded = policy.Pi05TrainPolicy(config, device="cpu")
    assert loaded.net.expert is loaded.model.paligemma_with_expert.gemma_expert.model
    return loaded


def test_pack_selected_rows_preserves_mixed_invocations_padding_and_source():
    invocations = (_invocation(19, 0), _invocation(47, 2))
    steps = [_decision(invocations[0], 1), _decision(invocations[1], 0)]
    steps.extend((steps[0], _decision(invocations[1], 1)))
    config = Pi05ModelConfig("checkpoint", "tokenizer", "stats")
    for step in steps:
        validate_ppo_decision(step, config)
    originals = [(tensor, tensor.clone()) for invocation in invocations for tensor in _leaves(*invocation)]
    condition, trace = pack_ppo_decisions(steps, device=torch.device("cpu"))
    assert (trace.sampler_identity, trace.score_horizon, trace.denoise_steps) == (
        invocations[0][1].sampler_identity,
        invocations[0][1].score_horizon,
        invocations[0][1].denoise_steps,
    )
    packed = _leaves(condition, trace)
    for position, step in enumerate(steps):
        original = _leaves(step.obs.representations[INPUTS_KEY], step.policy_info[TRACE_KEY])
        for actual, source in zip(packed, original, strict=True):
            assert actual.dtype == source.dtype and actual.shape[0] == len(steps)
            torch.testing.assert_close(actual[position], source[step.policy_info["sample_index"]], rtol=0, atol=0)
    assert trace.denoise_index.tolist() == [0, 2, 0, 2]
    assert condition.images[0].dtype == torch.uint8
    assert condition.image_masks.dtype == torch.bool
    assert condition.state.dtype == torch.float64
    assert trace.latent_before.dtype == torch.float32
    assert trace.denoise_index.dtype == torch.int64
    # Duplicate padding rows are copies, so mutating a packed row cannot alter its peer or source.
    repeated = trace.latent_before[2].clone()
    trace.latent_before[0].add_(1)
    condition.images[0][0].add_(1)
    torch.testing.assert_close(trace.latent_before[2], repeated, rtol=0, atol=0)
    for actual, before in originals:
        torch.testing.assert_close(actual, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("latent_before", "latent_before"),
        ("latent_after", "latent_after"),
        ("denoise_index", "denoising schedule"),
        ("dtype", "latent_after"),
        ("sampler", "sampler identity"),
        ("sample_index", "original invocation row"),
        ("state", "inputs.state"),
        ("image_masks", "missing third view"),
        ("tokens", "within"),
        ("tokens_upper", "within"),
        ("images", r"inputs.images\[1\]"),
    ],
)
def test_late_invalid_record_is_rejected_before_forward_including_unselected_rows(loaded_policy, corruption, match):
    good, bad = _invocation(29, 1), _invocation(61, 2)
    selected_row, unselected_row = 0, 1
    condition, trace = bad
    if corruption in ("latent_before", "latent_after"):
        value = getattr(trace, corruption).clone()
        value[unselected_row, -1, -1] = float("nan")
        trace = replace(trace, **{corruption: value})
    elif corruption == "denoise_index":
        value = trace.denoise_index.clone()
        value[unselected_row] = loaded_policy.config.denoise_steps
        trace = replace(trace, denoise_index=value)
    elif corruption == "dtype":
        trace = replace(trace, latent_after=trace.latent_after.to(torch.bfloat16))
    elif corruption == "sampler":
        trace = replace(trace, sampler_identity="wrong-sampler")
    elif corruption == "state":
        value = condition.state.clone()
        value[unselected_row, -1] = float("nan")
        condition = replace(condition, state=value)
    elif corruption == "image_masks":
        value = condition.image_masks.clone()
        value[unselected_row, 2] = True
        condition = replace(condition, image_masks=value)
    elif corruption in ("tokens", "tokens_upper"):
        value = condition.tokens.clone()
        value[unselected_row, -1] = -1 if corruption == "tokens" else PALIGEMMA_VOCAB_SIZE
        condition = replace(condition, tokens=value)
    elif corruption == "images":
        condition = replace(condition, images=(condition.images[0], condition.images[1].float()))
    else:
        selected_row = trace.latent_before.shape[0]
    steps = [_decision(good, 1), _decision((condition, trace), selected_row)]

    def unexpected_forward(_module, _inputs):
        raise AssertionError("forward started before validating the complete record batch")

    with loaded_policy.net.register_forward_pre_hook(unexpected_forward), pytest.raises(ValueError, match=match):
        loaded_policy.score_ppo_decisions(steps)
    with loaded_policy.net.register_forward_pre_hook(unexpected_forward), pytest.raises(ValueError, match=match):
        loaded_policy.score_ppo_decision(steps[-1])


def test_cached_prefix_record_is_rejected_and_replay_identity_changes(loaded_policy):
    invocation = _invocation(31, 0)
    step = _decision(invocation, 0)
    step.obs.representations = {"pi05_ppo_condition": loaded_policy.condition_from_inputs(invocation[0])}
    with pytest.raises(TypeError, match="cached-prefix records are unsupported"):
        loaded_policy.score_ppo_decisions([step])
    assert loaded_policy.ppo_identity()["training_replay"] == "preprocessed-uint8-frozen-prefix-fixed-microbatch-v2"


def test_frozen_prefix_is_recomputed_once_for_the_selected_batch_without_gradients(loaded_policy):
    invocations = (_invocation(31, 0), _invocation(71, 2))
    steps = [_decision(invocations[0], 1), _decision(invocations[1], 0)]
    steps += [steps[0]]
    calls = []

    def encoder(_module, inputs):
        calls.append((len(inputs[0]), torch.is_grad_enabled()))

    def heads(_module, inputs):
        condition, trace = inputs
        assert condition.prefix_mask.shape[0] == len(steps)
        assert trace.denoise_index.tolist() == [0, 2, 0]
        assert all(not tensor.requires_grad for tensor in _leaves(condition, trace))
        raise RuntimeError("heads reached after frozen preprocessing")

    with (
        loaded_policy.model.paligemma_with_expert.paligemma.register_forward_pre_hook(encoder),
        loaded_policy.net.register_forward_pre_hook(heads),
        pytest.raises(RuntimeError, match="heads reached"),
    ):
        loaded_policy.score_ppo_decisions(steps)
    assert calls == [(len(steps), False)]


def _retain_forward_failure(loaded_policy, steps):
    references = []

    def fail_forward(_module, inputs):
        condition, trace = inputs
        references.extend(weakref.ref(value) for value in (condition, trace, *_leaves(condition, trace)))
        try:
            raise ValueError("retained forward cause")
        except ValueError as cause:
            raise RuntimeError("injected packed forward failure") from cause

    with (
        loaded_policy.net.register_forward_pre_hook(fail_forward),
        pytest.raises(RuntimeError, match="^injected packed forward failure") as caught,
    ):
        loaded_policy.score_ppo_decisions(steps)
    assert str(caught.value) == "injected packed forward failure"
    return caught.value, references


def test_failed_packed_forward_releases_tensors_with_retained_exception_and_negative_control(
    loaded_policy, monkeypatch
):
    invocations = (_invocation(31, 0), _invocation(71, 2))
    steps = [_decision(invocations[0], 1), _decision(invocations[1], 0)]
    retained, references = _retain_forward_failure(loaded_policy, steps)
    assert any(frame.name == "score_ppo_decisions" for frame in traceback.extract_tb(retained.__traceback__))
    assert isinstance(retained.__cause__, ValueError)
    diagnostic = "".join(traceback.format_exception(retained))
    for marker in (
        "score_ppo_decisions",
        "fail_forward",
        "test_model_pi05_batching.py",
        "retained forward cause",
        "injected packed forward failure",
    ):
        assert marker in diagnostic
    assert len(references) == 2 + 18 * 2 + 2 + 3
    gc.collect()
    assert all(reference() is None for reference in references)

    clear_frames = policy._clear_exception_frames
    with monkeypatch.context() as negative:
        negative.setattr(policy, "_clear_exception_frames", lambda _error: None)
        leaked, leaked_references = _retain_forward_failure(loaded_policy, steps)
    gc.collect()
    assert any(reference() is not None for reference in leaked_references)
    # Release only the deliberately retained negative-control frames after observing the defect.
    clear_frames(leaked)
    gc.collect()
    assert all(reference() is None for reference in leaked_references)


def test_failed_frozen_prefix_releases_materialized_inputs_before_heads(loaded_policy, monkeypatch):
    steps = [_decision(_invocation(83, 1), 0)]
    references, heads = [], []
    materialize = loaded_policy.processor.materialize

    def capture(*args, **kwargs):
        result = materialize(*args, **kwargs)
        references.extend(
            weakref.ref(value)
            for value in (*result.images, result.image_masks, result.tokens, result.token_mask, result.state)
        )
        return result

    def fail_encoder(_module, inputs):
        references.append(weakref.ref(inputs[0]))
        raise RuntimeError("injected frozen-prefix failure")

    def observe_heads(*_args):
        heads.append(True)

    monkeypatch.setattr(loaded_policy.processor, "materialize", capture)
    with (
        loaded_policy.model.paligemma_with_expert.paligemma.register_forward_pre_hook(fail_encoder),
        loaded_policy.net.register_forward_pre_hook(observe_heads),
        pytest.raises(RuntimeError, match="injected frozen-prefix failure") as caught,
    ):
        loaded_policy.score_ppo_decisions(steps)
    retained = caught.value
    del caught  # Retain the public exception, not pytest's independent traceback reference.
    assert "injected frozen-prefix failure" in str(retained)
    assert not heads
    gc.collect()
    assert references and all(reference() is None for reference in references)
    # The same public preparation still works after retaining the failed call's error.
    condition = loaded_policy.condition_from_inputs(steps[0].obs.representations[INPUTS_KEY])
    assert condition.prefix_mask.shape == (2, 968)
