# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU contract tests for final-latent WFM rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.roles.rollout import Rollout
from cosmos_framework.posttrain.roles.rollout.wfm import WFMOnlineRollout
from cosmos_framework.posttrain.schema import (
    WFM_CONDITIONING_KEY,
    ObsBundle,
    Status,
    WFMConditioningInput,
)
from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel
from cosmos_framework.posttrain.utils.wfm import make_t2v_conditioning_input, repeat_conditioning_batch

_SOURCE_FINGERPRINT = "wfm-rollout-test-source-v1"


def _obs(conditioning: WFMConditioningInput) -> ObsBundle:
    return ObsBundle(
        representations={WFM_CONDITIONING_KEY: conditioning},
        modality_tags=("text", "video"),
        id=conditioning.id,
    )


class _DeclaredSampler:
    def __init__(self, base_shift: float = 1.0) -> None:
        self.base_shift = base_shift

    def validation_semantics_identity(self) -> dict[str, float | str]:
        return {"family": "test-unipc", "base_shift": self.base_shift}


@pytest.mark.L0
@pytest.mark.CPU
def test_t2v_conditioning_removes_target_pixels() -> None:
    source_video = torch.full((3, 4, 8, 8), 7.0)  # [C,T,H,W]
    source = {
        "ai_caption": "a robot moves a box",
        "video": [source_video],
        "x": torch.tensor([1.0, 2.0]),  # [D]
    }
    conditioning = make_t2v_conditioning_input(
        source,
        source_id="prompt-a",
        source_fingerprint=_SOURCE_FINGERPRINT,
    )
    template = conditioning.model_batch["video"][0]
    assert isinstance(template, torch.Tensor)
    assert tuple(template.shape) == tuple(source_video.shape)
    assert torch.count_nonzero(template) == 0
    assert torch.count_nonzero(source_video) > 0, "source must not be mutated"
    assert conditioning.prompt == "a robot moves a box"
    assert conditioning.template_identity.startswith("wfm-t2v-template-v1:")


@pytest.mark.L0
@pytest.mark.CPU
def test_t2v_conditioning_owns_a_recursive_source_snapshot() -> None:
    source_tensor = torch.tensor([1.0, 2.0], requires_grad=True)  # [D]
    source = {
        "ai_caption": "a robot moves a box",
        "video": [torch.ones(1, 2)],  # one sample [C,T]
        "metadata": {
            "tensor": source_tensor,
            "nested": [torch.tensor([3.0])],  # one sample [D]
        },
    }
    conditioning = make_t2v_conditioning_input(
        source,
        source_id="prompt-a",
        source_fingerprint=_SOURCE_FINGERPRINT,
    )

    with torch.no_grad():
        source["metadata"]["tensor"].zero_()
    source["metadata"]["nested"].append(torch.tensor([4.0]))  # one sample [D]

    metadata = conditioning.model_batch["metadata"]
    torch.testing.assert_close(metadata["tensor"], torch.tensor([1.0, 2.0]))
    assert not metadata["tensor"].requires_grad
    assert metadata["tensor"].grad_fn is None
    assert metadata["tensor"].data_ptr() != source_tensor.data_ptr()
    assert len(metadata["nested"]) == 1
    assert torch.count_nonzero(conditioning.model_batch["video"][0]) == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_t2v_conditioning_identity_is_resume_stable_and_source_scoped() -> None:
    source = {
        "ai_caption": "a robot moves a box",
        "video": [torch.ones(1, 2)],  # one sample [C,T]
    }

    first = make_t2v_conditioning_input(
        source,
        source_id="wfm_source_00000007",
        source_fingerprint=_SOURCE_FINGERPRINT,
    )
    resumed = make_t2v_conditioning_input(
        source,
        source_id="wfm_source_00000007",
        source_fingerprint=_SOURCE_FINGERPRINT,
    )
    next_row = make_t2v_conditioning_input(
        source,
        source_id="wfm_source_00000008",
        source_fingerprint=_SOURCE_FINGERPRINT,
    )
    other_source = make_t2v_conditioning_input(
        source,
        source_id="wfm_source_00000007",
        source_fingerprint="different-source-v1",
    )

    assert first.template_identity == resumed.template_identity
    assert first.template_identity != next_row.template_identity
    assert first.template_identity != other_source.template_identity


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_rollout_is_seeded_grouped_and_shares_conditioning() -> None:
    policy = ToyFlowModel(dim=2, seed=0)
    rollout = WFMOnlineRollout(policy, sampler=_DeclaredSampler(), num_steps=2, shift=5.0, guidance=3.0)
    conditioning = WFMConditioningInput(
        id="prompt-a",
        template_identity="rollout-test-template-a",
        model_batch={"x": torch.tensor([[1.0, 2.0]])},  # [1,D]
        prompt="a prompt",
    )
    obs = _obs(conditioning)
    assert isinstance(rollout, Rollout)
    identity = rollout.validation_semantics_identity()
    assert identity["backend"] == "cosmos3-native-wfm"
    assert identity["num_steps"] == 2
    assert identity["sampler"]["declared"]["base_shift"] == 1.0
    trajectories = rollout.generate([obs], group=3, seed=20, weight_version=0)
    assert len(trajectories) == 3
    assert {trajectory.group_id for trajectory in trajectories} == {"prompt-a"}
    assert [trajectory.transitions[0].policy_info["seed"] for trajectory in trajectories] == [20, 21, 22]
    assert all(trajectory.status is Status.COMPLETED for trajectory in trajectories)
    assert all(trajectory.transitions[0].policy_info["weight_version"] == 0 for trajectory in trajectories)
    carried = [trajectory.transitions[0].obs.representations[WFM_CONDITIONING_KEY] for trajectory in trajectories]
    assert all(item is conditioning for item in carried), "one group must reference one shared conditioning object"
    assert all(tuple(trajectory.transitions[0].action.shape) == (1, 2) for trajectory in trajectories)
    assert not torch.equal(trajectories[0].transitions[0].action, trajectories[1].transitions[0].action)

    repeated = rollout.generate([obs], group=3, seed=20, weight_version=0)
    for first, second in zip(trajectories, repeated):
        torch.testing.assert_close(first.transitions[0].action, second.transitions[0].action)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_rollout_owns_bound_group_size_and_rejects_callsite_drift() -> None:
    policy = ToyFlowModel(dim=2, seed=0)
    rollout = WFMOnlineRollout(
        policy,
        sampler=_DeclaredSampler(),
        num_steps=2,
        shift=5.0,
        guidance=3.0,
        group_size=3,
    )
    conditioning = WFMConditioningInput(
        id="prompt-a",
        template_identity="rollout-test-template-a",
        model_batch={"x": torch.tensor([[1.0, 2.0]])},  # [1,D]
        prompt="a prompt",
    )

    trajectories = rollout.generate([_obs(conditioning)], seed=20, weight_version=0)

    assert [trajectory.transitions[0].policy_info["seed"] for trajectory in trajectories] == [20, 21, 22]
    with pytest.raises(ValueError, match="differs from bound local group"):
        rollout.generate([_obs(conditioning)], group=2, seed=20, weight_version=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_rollout_validation_identity_binds_sampler_configuration() -> None:
    policy = ToyFlowModel(dim=2, seed=0)
    first = WFMOnlineRollout(policy, sampler=_DeclaredSampler(1.0), num_steps=2, shift=5.0, guidance=3.0)
    second = WFMOnlineRollout(policy, sampler=_DeclaredSampler(2.0), num_steps=2, shift=5.0, guidance=3.0)
    assert first.validation_semantics_identity() != second.validation_semantics_identity()

    opaque = WFMOnlineRollout(policy, sampler=object(), num_steps=2, shift=5.0, guidance=3.0)
    with pytest.raises(TypeError, match="must implement validation_semantics_identity"):
        opaque.validation_semantics_identity()


@pytest.mark.L0
@pytest.mark.CPU
def test_conditioning_rebuilds_real_batch_axes_without_nested_lists() -> None:
    conditioning = WFMConditioningInput(
        id="prompt-a",
        template_identity="rollout-test-template-a",
        model_batch={
            "video": [torch.zeros(1, 2)],  # one sample [C,T]
            "ai_caption": "prompt",
            "fps": torch.tensor([16.0]),  # [B]
        },
        prompt="prompt",
    )
    batch = repeat_conditioning_batch(conditioning, 3)
    assert len(batch["video"]) == 3
    assert all(isinstance(item, torch.Tensor) for item in batch["video"])
    assert batch["ai_caption"] == ["prompt", "prompt", "prompt"]
    assert tuple(batch["fps"].shape) == (3,)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_rollout_requires_explicit_seed_and_positive_local_group() -> None:
    policy = ToyFlowModel(dim=2, seed=0)
    rollout = WFMOnlineRollout(policy, sampler=object(), num_steps=2, shift=5.0, guidance=3.0)
    conditioning = WFMConditioningInput(
        id="p",
        template_identity="rollout-test-template-p",
        model_batch={"x": torch.ones(1, 2)},  # [1,D]
        prompt="p",
    )
    obs = _obs(conditioning)
    with pytest.raises(ValueError, match="explicit group"):
        rollout.generate([obs], seed=0, weight_version=0)
    with pytest.raises(ValueError, match="explicit integer seed"):
        rollout.generate([obs], group=2, weight_version=0)
    with pytest.raises(ValueError, match="explicit weight_version"):
        rollout.generate([obs], group=1, seed=0)
    assert len(rollout.generate([obs], group=1, seed=0, weight_version=0)) == 1
    with pytest.raises(ValueError, match="group must be an integer >= 1"):
        rollout.generate([obs], group=0, seed=0, weight_version=0)
    with pytest.raises(RuntimeError, match="rollout holds version 0"):
        rollout.generate([obs], group=1, seed=0, weight_version=1)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_rollout_rejects_coercive_or_nonfinite_sampler_controls() -> None:
    cases: list[tuple[str, object]] = [
        ("num_steps", True),
        ("num_steps", 1.5),
        ("shift", float("nan")),
        ("shift", float("inf")),
        ("guidance", float("nan")),
        ("guidance", -1.0),
    ]
    for field, value in cases:
        kwargs: dict[str, object] = {"num_steps": 2, "shift": 5.0, "guidance": 3.0}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            WFMOnlineRollout(ToyFlowModel(dim=2, seed=0), sampler=object(), **kwargs)


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_rollout_rejects_coercive_or_out_of_range_generation_controls() -> None:
    cases: list[tuple[str, object]] = [
        ("group", True),
        ("group", 1.5),
        ("seed", True),
        ("seed", -1),
        ("seed", 1 << 63),
        ("weight_version", True),
        ("weight_version", 0.5),
    ]
    rollout = WFMOnlineRollout(ToyFlowModel(dim=2, seed=0), sampler=object(), num_steps=2, shift=5.0, guidance=3.0)
    conditioning = WFMConditioningInput(
        id="p",
        template_identity="rollout-test-template-p",
        model_batch={"x": torch.ones(1, 2)},  # [1,D]
        prompt="p",
    )
    for field, value in cases:
        kwargs: dict[str, object] = {"group": 1, "seed": 0, "weight_version": 0}
        kwargs[field] = value
        with pytest.raises((ValueError, RuntimeError), match=field):
            rollout.generate([_obs(conditioning)], **kwargs)


@pytest.mark.L0
@pytest.mark.CPU
def test_t2v_conditioning_rejects_unsupported_pixel_template_leaves() -> None:
    source = {"ai_caption": "prompt", "video": ["opaque-video-handle"]}
    with pytest.raises(TypeError, match="video template must contain only tensors"):
        make_t2v_conditioning_input(
            source,
            source_id="prompt-a",
            source_fingerprint=_SOURCE_FINGERPRINT,
        )


class _FailingRankToyFlowModel(ToyFlowModel):
    def __init__(self, rank: int) -> None:
        super().__init__(dim=2, seed=0)
        self.rank = rank
        self.sample_calls = 0

    def sample_final_latents(
        self,
        model_batch: dict[str, Any],
        *,
        sampler: object,
        seeds: list[int],
        num_steps: int,
        shift: float,
        guidance: float,
    ) -> list[torch.Tensor]:
        self.sample_calls += 1
        outputs = super().sample_final_latents(
            model_batch,
            sampler=sampler,
            seeds=seeds,
            num_steps=num_steps,
            shift=shift,
            guidance=guidance,
        )
        return outputs[:-1] if self.rank == 1 and self.sample_calls == 1 else outputs


def _rollout_convergence_worker(rank: int, world_size: int, rendezvous_path: str) -> None:
    import torch.distributed as dist

    init_gloo_process_group(rank, world_size, rendezvous_path)
    policy = _FailingRankToyFlowModel(rank)
    rollout = WFMOnlineRollout(policy, sampler=object(), num_steps=2, shift=5.0, guidance=3.0)
    first = WFMConditioningInput(
        id="p0",
        template_identity="rollout-test-template-p0",
        model_batch={"x": torch.ones(1, 2)},  # [1,D]
        prompt="p0",
    )
    second = WFMConditioningInput(
        id="p1",
        template_identity="rollout-test-template-p1",
        model_batch={"x": torch.ones(1, 2)},  # [1,D]
        prompt="p1",
    )
    try:
        asymmetric_prompts = [_obs(first), _obs(second)] if rank == 1 else [_obs(first)]
        with pytest.raises(RuntimeError, match="online WFM rollout preparation failed"):
            rollout.generate(asymmetric_prompts, group=1, seed=0, weight_version=0)
        assert policy.sample_calls == 0

        malformed = ObsBundle(id="p0") if rank == 1 else _obs(first)
        with pytest.raises(RuntimeError, match="online WFM rollout preparation failed"):
            rollout.generate([malformed], group=1, seed=0, weight_version=0)
        assert policy.sample_calls == 0

        with pytest.raises(RuntimeError, match="online WFM rollout group 0 generation failed"):
            rollout.generate([_obs(first)], group=1, seed=0, weight_version=0)
        assert policy.sample_calls == 1

        completed = rollout.generate([_obs(first)], group=1, seed=0, weight_version=0)
        assert len(completed) == 1
        action = completed[0].transitions[0].action  # [1,D]
        local_identity = (
            completed[0].id,
            completed[0].group_id,
            completed[0].transitions[0].policy_info["seed"],
            tuple(float(value) for value in action.flatten().tolist()),
        )
        identities: list[tuple[str, str | None, int, tuple[float, ...]] | None] = [None] * world_size
        dist.all_gather_object(identities, local_identity)
        assert len({identity[0] for identity in identities if identity is not None}) == world_size
        assert {identity[1] for identity in identities if identity is not None} == {"p0"}
        assert {identity[2] for identity in identities if identity is not None} == set(range(world_size))
        assert len({identity[3] for identity in identities if identity is not None}) == world_size
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_wfm_rollout_rank_local_contract_failures_converge(tmp_path: Path) -> None:
    import torch.multiprocessing as mp

    world_size = 2
    rendezvous_path = str(tmp_path / "wfm-rollout-convergence")
    mp.spawn(_rollout_convergence_worker, args=(world_size, rendezvous_path), nprocs=world_size, join=True)
