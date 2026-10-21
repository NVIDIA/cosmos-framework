# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Transactional online-WFM source tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import cosmos_framework.posttrain.roles.env.wfm as wfm_env_module
from cosmos_framework.posttrain.data.datasets import NFTPromptImageDataLoader
from cosmos_framework.posttrain.roles.env import WFMDataEnv
from cosmos_framework.posttrain.schema import WFM_CONDITIONING_KEY, ObsBundle, WFMConditioningInput
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.utils.wfm import make_t2v_conditioning_input, repeat_conditioning_batch

_SOURCE_FINGERPRINT = "wfm-test-loader-v1"


class _Loader:
    def __init__(self) -> None:
        self.start = 0
        self.yield_count = 0

    def set_start_iteration(self, iteration: int) -> None:
        self.start = iteration

    def __iter__(self) -> Iterator[dict[str, object]]:
        index = self.start
        while True:
            self.yield_count += 1
            yield {
                "video": [torch.full((1, 2), float(index))],  # one sample [C,T]
                "ai_caption": [f"prompt-{index}"],
                "x": torch.tensor([float(index), 1.0]),  # [D]
            }
            index += 1

    def cursor_fingerprint(self) -> str:
        return _SOURCE_FINGERPRINT


def _conditioning(obs: ObsBundle) -> WFMConditioningInput:
    value = obs.representations[WFM_CONDITIONING_KEY]
    assert isinstance(value, WFMConditioningInput)
    return value


class _MalformedThenValidLoader(_Loader):
    def __iter__(self) -> Iterator[dict[str, object]]:
        self.yield_count += 1
        yield {"video": [], "ai_caption": ["malformed"]}
        self.yield_count += 1
        yield {
            "video": [torch.full((1, 2), 1.0)],  # one sample [C,T]
            "ai_caption": ["prompt-1"],
            "x": torch.tensor([1.0, 1.0]),  # [D]
        }


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_peek_abort_commit_and_restore() -> None:
    source = WFMDataEnv(_Loader())
    first = source.get_samples(1)
    source.abort()
    assert source.get_samples(1) is first
    with pytest.raises(RuntimeError, match="inside an accumulation window"):
        source.state_dict()

    source.commit()
    state = source.state_dict()
    assert state["committed_rows"] == 1
    assert state["optimizer_step"] == 1

    restored = WFMDataEnv(_Loader())
    assert restored.load_state_dict(state)
    replayed = restored.get_samples(1)
    assert replayed[0].id == "wfm_source_00000001"
    replayed_conditioning = _conditioning(replayed[0])
    expected_conditioning = _conditioning(WFMDataEnv(_Loader(), start_iteration=1).get_samples(1)[0])
    assert replayed_conditioning.prompt == "prompt-1"
    assert replayed_conditioning.template_identity == expected_conditioning.template_identity


@pytest.mark.L0
@pytest.mark.CPU
def test_t2i_prompt_source_binds_artifact_and_emits_native_image_schema(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text('{"prompt":"first"}\n{"prompt":"second"}\n')
    loader = NFTPromptImageDataLoader(str(prompt_path), height=32, width=48, expected_rows=2)
    source = WFMDataEnv(loader, modality="image")

    observation = source.get_samples(1)[0]
    conditioning = _conditioning(observation)
    assert conditioning.modality == "image"
    assert conditioning.prompt == "first"
    assert conditioning.template_identity.startswith("wfm-t2i-template-v1:")
    assert conditioning.model_batch["images"][0][0].shape == (3, 32, 48)
    assert torch.count_nonzero(conditioning.model_batch["images"][0][0]) == 0
    repeated = repeat_conditioning_batch(conditioning, 3)
    assert len(repeated["image_size"]) == 3
    assert all(tuple(image_size.shape) == (4,) for image_size in repeated["image_size"])
    assert tuple(repeated["conditioning_fps"].shape) == (3,)
    source.commit()
    state = source.state_dict()

    restored = WFMDataEnv(
        NFTPromptImageDataLoader(str(prompt_path), height=32, width=48, expected_rows=2),
        modality="image",
    )
    assert restored.load_state_dict(state)
    restored_conditioning = _conditioning(restored.get_samples(1)[0])
    assert restored_conditioning.prompt == "second"
    assert restored_conditioning.template_identity != conditioning.template_identity


@pytest.mark.L0
@pytest.mark.CPU
def test_t2i_prompt_source_matches_rank_zero_distributed_sampler_across_epochs(tmp_path: Path) -> None:
    prompts = [f"prompt-{index}" for index in range(16)]
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text("".join(f'{{"prompt":"{prompt}"}}\n' for prompt in prompts))
    loader = NFTPromptImageDataLoader(
        str(prompt_path),
        height=16,
        width=16,
        shuffle=True,
        shuffle_seed=0,
        num_replicas=4,
        source_rank=0,
        drop_last=True,
    )

    expected: list[str] = []
    for epoch in range(2):
        generator = torch.Generator().manual_seed(epoch)
        expected.extend(prompts[index] for index in torch.randperm(len(prompts), generator=generator).tolist()[::4])
    iterator = iter(loader)
    actual = [next(iterator)["ai_caption"] for _ in range(len(expected))]

    assert actual == expected


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_commits_and_rewinds_one_complete_accumulation_window() -> None:
    with pytest.raises(ValueError, match="not an optimizer boundary"):
        WFMDataEnv(_Loader(), start_iteration=1, grad_accum=2)

    loader = _Loader()
    source = WFMDataEnv(loader, grad_accum=2)

    first = source.get_samples(1)[0]
    source.stage()
    second = source.get_samples(1)[0]
    assert (first.id, second.id) == ("wfm_source_00000000", "wfm_source_00000001")
    source.stage()
    with pytest.raises(RuntimeError, match="inside an accumulation window"):
        source.state_dict()
    source.commit_window(2)
    assert source.state_dict() == {
        "version": 1,
        "resume_kind": "optimizer_window_rows",
        "committed_rows": 2,
        "optimizer_step": 1,
        "grad_accum": 2,
        "source_fingerprint": "wfm-test-loader-v1",
    }

    source.get_samples(1)
    source.abort_window()
    assert source.get_samples(1)[0].id == "wfm_source_00000002"
    assert loader.start == 2


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_rejects_invalid_accumulation_transitions_without_committing() -> None:
    pending = WFMDataEnv(_Loader(), grad_accum=2)
    pending.get_samples(1)
    with pytest.raises(RuntimeError, match="with a pending row"):
        pending.commit_window(2)
    assert pending._committed_rows == 0
    pending.abort_window()
    assert pending.state_dict()["committed_rows"] == 0

    incomplete = WFMDataEnv(_Loader(), grad_accum=2)
    incomplete.get_samples(1)
    incomplete.stage()
    with pytest.raises(RuntimeError, match="staged=1"):
        incomplete.commit_window(2)
    assert incomplete._committed_rows == 0
    incomplete.abort_window()
    assert incomplete.state_dict()["committed_rows"] == 0

    mismatched = WFMDataEnv(_Loader(), grad_accum=2)
    for _ in range(2):
        mismatched.get_samples(1)
        mismatched.stage()
    with pytest.raises(RuntimeError, match="expected_rows=1"):
        mismatched.commit_window(1)
    assert mismatched._committed_rows == 0
    mismatched.commit_window(2)
    assert mismatched.state_dict()["committed_rows"] == 2

    over_staged = WFMDataEnv(_Loader(), grad_accum=2)
    for _ in range(2):
        over_staged.get_samples(1)
        over_staged.stage()
    over_staged.get_samples(1)
    with pytest.raises(RuntimeError, match="already staged its complete 2-row window"):
        over_staged.stage()
    assert over_staged._committed_rows == 0
    over_staged.abort_window()
    assert over_staged.state_dict()["committed_rows"] == 0

    partial_loader = _Loader()
    partial = WFMDataEnv(partial_loader, grad_accum=2)
    partial.get_samples(1)
    partial.stage()
    partial.abort_window()
    assert partial.state_dict()["committed_rows"] == 0
    assert partial_loader.start == 0
    assert partial.get_samples(1)[0].id == "wfm_source_00000000"


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_rejects_batch_or_restore_ambiguity() -> None:
    source = WFMDataEnv(_Loader())
    state = source.state_dict()
    with pytest.raises(ValueError, match="expects one conditioning"):
        source.get_samples(2)
    source.get_samples(1)
    with pytest.raises(RuntimeError, match="inside an accumulation window"):
        source.load_state_dict(state)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_restore_fails_before_mutating_the_durable_boundary() -> None:
    class _NonRestorableLoader:
        def __iter__(self) -> Iterator[dict[str, object]]:
            yield from iter(_Loader())

        def cursor_fingerprint(self) -> str:
            return "wfm-test-loader-v1"

    source = WFMDataEnv(_NonRestorableLoader())
    state = {
        "version": 1,
        "resume_kind": "optimizer_window_rows",
        "committed_rows": 2,
        "optimizer_step": 2,
        "grad_accum": 1,
        "source_fingerprint": "wfm-test-loader-v1",
    }
    with pytest.raises(ValueError, match="cannot restore a structural iteration"):
        source.load_state_dict(state)
    assert source.state_dict()["committed_rows"] == 0

    mismatched = dict(source.state_dict(), source_fingerprint="different-source")
    with pytest.raises(ValueError, match="fingerprint differs"):
        source.load_state_dict(mismatched)
    assert source.state_dict()["committed_rows"] == 0

    malformed = dict(source.state_dict(), committed_rows=True)
    with pytest.raises(ValueError, match="invalid WFM committed_rows"):
        source.load_state_dict(malformed)
    assert source.state_dict()["committed_rows"] == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_retries_same_acquired_row_after_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _Loader()
    source = WFMDataEnv(loader)
    original = wfm_env_module.make_t2v_conditioning_input
    attempts = 0

    def fail_once(
        source_sample: dict[str, Any],
        *,
        source_id: str,
        source_fingerprint: str,
    ) -> WFMConditioningInput:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("one-shot conditioning failure")
        return original(
            source_sample,
            source_id=source_id,
            source_fingerprint=source_fingerprint,
        )

    monkeypatch.setattr(wfm_env_module, "make_t2v_conditioning_input", fail_once)
    with pytest.raises(RuntimeError, match="one-shot conditioning failure"):
        source.get_samples(1)
    source.abort()
    assert loader.yield_count == 1
    with pytest.raises(RuntimeError, match="inside an accumulation window"):
        source.state_dict()
    initial_state = WFMDataEnv(_Loader()).state_dict()
    with pytest.raises(RuntimeError, match="inside an accumulation window"):
        source.load_state_dict(initial_state)

    first = _conditioning(source.get_samples(1)[0])
    assert loader.yield_count == 1
    assert first.id == "wfm_source_00000000"
    assert first.prompt == "prompt-0"

    source.commit()
    second = _conditioning(source.get_samples(1)[0])
    assert loader.yield_count == 2
    assert second.id == "wfm_source_00000001"
    assert second.prompt == "prompt-1"


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_malformed_row_fails_closed_without_advancing() -> None:
    loader = _MalformedThenValidLoader()
    source = WFMDataEnv(loader)

    for _ in range(2):
        with pytest.raises(ValueError, match="exactly one video source row"):
            source.get_samples(1)
        source.abort()

    assert loader.yield_count == 1
    with pytest.raises(RuntimeError, match="inside an accumulation window"):
        source.state_dict()
    with pytest.raises(RuntimeError, match="without a pending row"):
        source.commit()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "attribute",
    (
        "condition_frame_indexes_vision",
        "condition_frame_indexes_lidar",
        "condition_frame_indexes_action",
        "condition_frame_indexes_sound",
    ),
)
def test_t2v_source_rejects_conditioned_frames_before_zeroing_video(attribute: str) -> None:
    sample = next(iter(_Loader()))
    sample["sequence_plan"] = SimpleNamespace(**{attribute: [0]})

    with pytest.raises(NotImplementedError, match=attribute):
        make_t2v_conditioning_input(
            sample,
            source_id="conditioned",
            source_fingerprint=_SOURCE_FINGERPRINT,
        )


class _RankLoader(_Loader):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def __iter__(self) -> Iterator[dict[str, object]]:
        yield {
            "video": [torch.full((1, 2), float(self.rank))],  # one sample [C,T]
            "ai_caption": [f"rank-{self.rank}"],
            "x": torch.tensor([float(self.rank), 1.0]),  # [D]
        }


class _RankFingerprintLoader(_Loader):
    def __init__(self, rank: int) -> None:
        super().__init__()
        self.rank = rank

    def cursor_fingerprint(self) -> str:
        return f"wfm-test-loader-rank-{self.rank}"


class _FailingRankZeroLoader(_RankLoader):
    def __iter__(self) -> Iterator[dict[str, object]]:
        if self.rank == 0:
            raise ValueError("rank-zero source failure")
        yield from super().__iter__()


def _broadcast_source_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        conditioning = _conditioning(WFMDataEnv(_RankLoader(rank)).get_samples(1)[0])
        assert conditioning.id == "wfm_source_00000000"
        assert conditioning.prompt == "rank-0"
        assert torch.equal(conditioning.model_batch["x"], torch.tensor([0.0, 1.0]))
        assert torch.count_nonzero(conditioning.model_batch["video"][0]) == 0
    finally:
        dist.destroy_process_group()


def _broadcast_source_failure_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        with pytest.raises(RuntimeError, match="online WFM source fetch failed on one or more ranks"):
            WFMDataEnv(_FailingRankZeroLoader(rank)).get_samples(1)
    finally:
        dist.destroy_process_group()


def _broadcast_source_retry_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    original = wfm_env_module.make_t2v_conditioning_input
    attempts = 0

    def fail_once(
        source_sample: dict[str, Any],
        *,
        source_id: str,
        source_fingerprint: str,
    ) -> WFMConditioningInput:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("rank-zero one-shot conditioning failure")
        return original(
            source_sample,
            source_id=source_id,
            source_fingerprint=source_fingerprint,
        )

    wfm_env_module.make_t2v_conditioning_input = fail_once
    loader = _Loader()
    source = WFMDataEnv(loader)
    try:
        with pytest.raises(RuntimeError, match="online WFM source fetch failed on one or more ranks"):
            source.get_samples(1)
        source.abort()
        initial_state = WFMDataEnv(_Loader()).state_dict()
        with pytest.raises(RuntimeError, match="inside an accumulation window"):
            source.load_state_dict(initial_state)
        with pytest.raises(RuntimeError, match="without a pending row"):
            source.commit()

        conditioning = _conditioning(source.get_samples(1)[0])
        assert conditioning.id == "wfm_source_00000000"
        assert conditioning.prompt == "prompt-0"
        assert loader.yield_count == (1 if rank == 0 else 0)
    finally:
        wfm_env_module.make_t2v_conditioning_input = original
        dist.destroy_process_group()


def _asymmetric_source_cursor_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    try:
        identity_mismatch = WFMDataEnv(_RankFingerprintLoader(rank))
        with pytest.raises(RuntimeError, match="source transaction signature differs across ranks"):
            identity_mismatch.get_samples(1)
        assert identity_mismatch.dataloader.yield_count == 0

        cursor_mismatch = WFMDataEnv(_Loader(), start_iteration=rank)
        with pytest.raises(RuntimeError, match="source transaction signature differs across ranks"):
            cursor_mismatch.get_samples(1)
        assert cursor_mismatch.dataloader.yield_count == 0

        loader = _Loader()
        window_mismatch = WFMDataEnv(loader)
        window_mismatch.get_samples(1)
        if rank == 1:
            window_mismatch.stage()
        with pytest.raises(RuntimeError, match="source transaction signature differs across ranks"):
            window_mismatch.get_samples(1)
        assert loader.yield_count == (1 if rank == 0 else 0)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_broadcasts_one_rank0_conditioning_group(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "wfm-source-rendezvous")
    mp.spawn(_broadcast_source_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_rank0_failure_converges_before_broadcast(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "wfm-source-failure-rendezvous")
    mp.spawn(_broadcast_source_failure_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_distributed_retry_reuses_rank0_acquired_row(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "wfm-source-retry-rendezvous")
    mp.spawn(_broadcast_source_retry_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_source_asymmetric_cursor_state_converges_before_fetch(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "wfm-source-cursor-rendezvous")
    mp.spawn(_asymmetric_source_cursor_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)
