# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Analytic two-sensor oracles for joint SF-DMD rollout, scoring and gradients."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from cosmos_framework.callbacks.sigma_loss_analysis import SigmaLossAnalysis
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.sequence import ModalityData, PackedSequence
from cosmos_framework.model.generator.distillation import self_forcing_dmd_joint as joint
from cosmos_framework.model.generator.distillation.self_forcing_dmd_joint import (
    JointRollout,
    JointSelfForcingDMDModel,
    active_mean_square,
)
from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import build_interactive_multiview_mask_items
from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean, GenerationDataNoised
from cosmos_framework.trainer.distillation import DistillationTrainer

pytestmark = [pytest.mark.gpus(0)]


@pytest.mark.level(0)
def test_joint_scorers_require_bidirectional_networks() -> None:
    model = AnalyticJointModel()
    assert model._scorer_video_temporal_causal() is False


@pytest.mark.level(0)
@pytest.mark.parametrize("net_type", ["teacher", "fake_score"])
def test_joint_scorers_use_noisy_full_clip_without_clean_replay(
    inputs: tuple, monkeypatch: pytest.MonkeyPatch, net_type: str
) -> None:
    data, plans = inputs
    model = AnalyticJointModel()
    with torch.no_grad():
        rollout = model._joint_rollout(data, plans, [[1]])
        noised, timestep, _ = model._joint_scoring_inputs(rollout, ["480"])
    original_pack = model._pack_input_sequence

    def pack(*args: Any, **kwargs: Any) -> Any:
        assert model.config.video_temporal_causal is False
        return original_pack(*args, **kwargs)

    def denoise(
        *,
        net: torch.nn.Module,
        data_batch_packed: Any,
        memory: Any = None,
        video_temporal_causal: bool | None = None,
    ) -> dict[str, Any]:
        assert net is model.denoiser_nets[net_type]
        assert memory is None and video_temporal_causal is False
        for modality in ("vision", "lidar"):
            actual = getattr(data_batch_packed, modality).tokens
            expected = getattr(noised, f"xt_tokens_{modality}")
            assert len(actual) == len(expected) == 2
            for actual_item, expected_item in zip(actual, expected, strict=True):
                torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)
        return {"bidirectional": True}

    monkeypatch.setattr(model, "_pack_input_sequence", pack)
    monkeypatch.setattr(model, "denoise", denoise)
    assert model._joint_score(rollout, noised, timestep, plans, [[1]], net_type) == {"bidirectional": True}
    assert model.config.video_temporal_causal is True
    assert model.scoring_memories == []


@pytest.mark.level(0)
def test_joint_bidirectional_build_bypasses_causal_replay_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    model = AnalyticJointModel()
    model.config.video_temporal_causal = False
    policy = SimpleNamespace(backend="maskless", window=None)
    model.config.multiview_attention = policy
    result = model.make_net(0.1, 0.2)

    def native_build(self: Any, dtype: torch.dtype, **kwargs: Any) -> torch.nn.Module:
        assert self is model and dtype is torch.bfloat16
        assert self.config.video_temporal_causal is False
        assert self.config.multiview_attention is policy
        return result

    def reject_replay(*args: Any, **kwargs: Any) -> torch.nn.Module:
        raise AssertionError("A bidirectional scorer must not inherit the causal replay builder")

    monkeypatch.setattr(OmniMoTModel, "build_net", native_build)
    monkeypatch.setattr(OmniMoTCausalModel, "build_net", reject_replay)
    assert model.build_net(torch.bfloat16) is result


@pytest.mark.level(0)
def test_failed_scorer_pack_restores_student_causality(inputs: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    data, plans = inputs
    model = AnalyticJointModel()
    with torch.no_grad():
        rollout = model._joint_rollout(data, plans, [[1]])
        noised, timestep, _ = model._joint_scoring_inputs(rollout, ["480"])

    def fail_pack(*args: Any, **kwargs: Any) -> Any:
        assert model.config.video_temporal_causal is False
        raise ValueError("Invalid scorer metadata")

    monkeypatch.setattr(model, "_pack_input_sequence", fail_pack)
    with pytest.raises(ValueError, match="Invalid scorer metadata"):
        model._joint_score(rollout, noised, timestep, plans, [[1]], "teacher")
    assert model.config.video_temporal_causal is True
    with pytest.raises(ValueError, match="teacher or fake-score"):
        model._joint_score(rollout, noised, timestep, plans, [[1]], "student")


class AnalyticJointModel(JointSelfForcingDMDModel):
    """Run the real joint algorithm with independently known velocity fields."""

    calls: list[dict[str, Any]]
    prefixes: list[GenerationDataClean]
    scoring_memories: list[tuple[torch.nn.Module, Any, bool]]

    def __init__(self, sample_type: str = "sde", guidance: float = 3.0) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            joint_sensor_modalities=("vision", "lidar"),
            strict_regular_checkpoint_loading=False,
            teacher_negative_prompt_per_view_metadata=False,
            compile=SimpleNamespace(enabled=False),
            sound_gen=False,
            action_gen=False,
            backward_grad_steps=1,
            sf_replayed_training=True,
            sf_student_use_persistent_kv=False,
            sf_student_backward_use_persistent_kv=True,
            sf_critic_use_persistent_kv=False,
            sf_rollout_step_mode="max",
            sf_simulation_steps_per_frame=[4],
            sf_backward_timestep_schedules=None,
            sf_require_equal_rollout_lengths=True,
            sf_rollout_num_pixel_frames=None,
            teacher_forcing_frames_per_chunk=3,
            fixed_step_sampler_config=SimpleNamespace(t_list=[1.0, 0.9375, 5 / 6, 0.625], sample_type=sample_type),
            rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
            rectified_flow_training_config=SimpleNamespace(loss_scale=10.0, lidar_loss_scale=20.0),
            teacher_guidance=guidance,
            loss_scale_fake_score=1.0,
            loss_scale_sid=1.0,
            video_temporal_causal=True,
            student_update_freq=6,
            warmup_student_steps=0,
            warmup_critic_steps=0,
        )
        self.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        self.tensor_kwargs_fp32 = self.tensor_kwargs
        self.parallel_dims = None
        self.net = self.make_net(0.25, 0.5)
        self._net_teacher_holder = [self.make_net(0.1, 0.2)]
        self._net_fake_score_holder = [self.make_net(0.4, 0.6)]
        self._phase_trainable_parameter_masks = {
            phase: {name: True for name, _ in network.named_parameters()}
            for phase, network in (("student", self.net), ("critic", self.net_fake_score))
        }
        self.net_teacher.requires_grad_(False)
        self.denoiser_nets = {"student": self.net, "teacher": self.net_teacher, "fake_score": self.net_fake_score}
        self.calls = []
        self.prefixes = []
        self.scoring_memories = []

    @staticmethod
    def make_net(vision: float, lidar: float) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict(
            {
                "vision": torch.nn.Parameter(torch.tensor(vision)),  # []
                "lidar": torch.nn.Parameter(torch.tensor(lidar)),  # []
            }
        )

    def _uses_multiview_replay_kv(self) -> bool:
        return True

    def _get_teacher_forcing_replay_policy(self) -> Any:
        return SimpleNamespace(control_visibility="causal")

    def _validate_teacher_forcing_pack(self, pack: Any) -> None:
        pass

    def _pack_input_sequence(
        self, plans: list, text: list, data: GenerationDataClean, timestep: torch.Tensor, **kwargs: Any
    ) -> Any:  # timestep: [1,1]
        streams: dict[str, Any] = {}
        for modality, views in (("vision", data.num_views_per_vision_item[0]), ("lidar", 1)):
            if modality not in self._joint_modalities(data):
                streams[modality] = None
                continue
            values = getattr(data, f"x0_tokens_{modality}")
            frames = values[1].shape[2]
            condition = torch.zeros(frames, 1, 1)  # [T,1,1]
            if modality == "vision":
                condition[:: frames // views] = 1
            streams[modality] = SimpleNamespace(
                tokens=values,
                condition_mask=[torch.ones_like(condition), condition],  # list[[T,1,1]]
                timesteps=torch.zeros(frames),  # [T]
            )
        return SimpleNamespace(
            **streams, to_cuda=lambda: None, negative=text == [[0]], radar=None, sound=None, action=None
        )

    def _cast_generated_tokens_to_precision(self, pack: Any) -> None:
        pass

    def _build_clean_tf_cache(
        self,
        *,
        net: torch.nn.Module,
        packed_sequence: Any,
        gen_data_clean: GenerationDataClean,
        memory_info: dict[str, Any],
        detach_clean_kv: bool,
    ) -> Any:
        assert self.config.video_temporal_causal
        memory = SimpleNamespace(net=net, negative=packed_sequence.negative, detach_clean_kv=detach_clean_kv)
        self.scoring_memories.append((net, memory, packed_sequence.negative))
        return memory

    def denoise(
        self,
        *,
        net: torch.nn.Module,
        data_batch_packed: Any,
        memory: Any = None,
        video_temporal_causal: bool | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for modality in self.config.joint_sensor_modalities:
            if getattr(data_batch_packed, modality) is None:
                # Mirror the native network's absent-sensor head probe.
                result[f"preds_{modality}"] = [net[modality].reshape(1)]  # list[[1]]
                continue
            tokens = getattr(data_batch_packed, modality).tokens  # list[[1,C,T,H,W]]
            value = net[modality] + (0.05 if data_batch_packed.negative else 0.0)  # []
            result[f"preds_{modality}"] = [torch.ones_like(token) * value for token in tokens]
            # list[[1,C,T,H,W]]
        self.calls.append(
            {
                "net": net,
                "causal": self.config.video_temporal_causal if video_temporal_causal is None else video_temporal_causal,
                "memory": memory,
                "negative": data_batch_packed.negative,
                "vision": data_batch_packed.vision.tokens[1].detach().clone(),  # [1,Cv,Tv,Hv,Wv]
                "lidar": None
                if data_batch_packed.lidar is None
                else data_batch_packed.lidar.tokens[1].detach().clone(),  # [1,Cl,Tl,Hl,Wl] or None
            }
        )
        return result

    def _get_train_noise_level_vision(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([[400.0]]), torch.tensor([[0.4]])  # [1,1], [1,1]

    def _add_noise_to_input(
        self, data: GenerationDataClean, packed: Any, sigmas: torch.Tensor, *, sigmas_lidar: torch.Tensor | None
    ) -> GenerationDataNoised:  # sigmas, sigmas_lidar: [2,1]
        if data.x0_tokens_lidar is not None:
            torch.testing.assert_close(sigmas, sigmas_lidar, rtol=0, atol=0)
        else:
            assert sigmas_lidar is None
        noised = GenerationDataNoised(batch_size=1, epsilon_vision=[], xt_tokens_vision=[], vt_target_vision=[])
        for modality in self._joint_modalities(data):
            values = getattr(data, f"x0_tokens_{modality}")
            masks = getattr(packed, modality).condition_mask
            sigma = [sigmas[i] * (1 - mask) for i, mask in enumerate(masks)]  # list[[T,1,1]]
            noise = [torch.full_like(value, 0.7) for value in values]  # list[[1,C,T,H,W]]
            setattr(noised, f"sigmas_{modality}", sigma)
            setattr(noised, f"xt_tokens_{modality}", [x * (1 - s) + n * s for x, n, s in zip(values, noise, sigma)])
            # list[[1,C,T,H,W]]
            setattr(noised, f"vt_target_{modality}", [n - x for n, x in zip(noise, values)])
            # list[[1,C,T,H,W]]
        return noised


@pytest.fixture
def inputs() -> tuple[GenerationDataClean, list[Any]]:
    vision = torch.full((1, 2, 14, 1, 2), 3.0)  # [1,2,2*7,1,2]
    lidar = torch.full((1, 3, 9, 1, 1), 4.0)  # [1,3,9,1,1]
    data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[vision.clone(), vision],
        x0_tokens_lidar=[lidar.clone(), lidar],
        num_vision_items_per_sample=[2],
        num_lidar_items_per_sample=[2],
        num_views_per_vision_item=[2, 2],
    )
    plans = [SequencePlan(has_text=True, has_vision=True, has_lidar=True, has_action=False, has_sound=False)]
    return data, plans


def _install_analytic_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    def build(
        host: AnalyticJointModel,
        plans: list,
        data: GenerationDataClean,
        text: list,
        *,
        persistent_cache: Any = None,
    ) -> tuple[Any, None]:
        host.prefixes.append(data)
        return host._pack_input_sequence(plans, text, data, torch.zeros(1, 1)), None  # timestep: [1,1]

    sensors = [
        [
            SimpleNamespace(is_control=True),
            SimpleNamespace(is_control=False, seconds_per_frame=4 / 30),
            SimpleNamespace(is_control=True),
            SimpleNamespace(is_control=False, seconds_per_frame=0.1),
        ]
    ]
    monkeypatch.setattr(joint, "_build_replay_branch", build)
    monkeypatch.setattr(
        joint,
        "build_interactive_multiview_mask_items",
        lambda pack: sensors if pack.lidar is not None else [sensors[0][:2]],
    )


@pytest.fixture(autouse=True)
def analytic_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_analytic_replay(monkeypatch)


@pytest.mark.level(0)
@pytest.mark.parametrize("sample_type", ["sde", "ode"])
def test_joint_rollout_excludes_ground_truth_and_replays_exact_endpoints(inputs: tuple, sample_type: str) -> None:
    data, plans = inputs
    model = AnalyticJointModel(sample_type)
    with torch.no_grad():
        torch.manual_seed(123)
        rollout = model._joint_rollout(data, plans, [[1]])
        poisoned = replace(
            data,
            x0_tokens_vision=[x.clone() for x in data.x0_tokens_vision],
            x0_tokens_lidar=[x.clone() for x in data.x0_tokens_lidar],
        )
        poisoned.x0_tokens_vision[1][:, :, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13]] = float("nan")
        poisoned.x0_tokens_lidar[1].fill_(float("nan"))
        torch.manual_seed(123)
        other = model._joint_rollout(poisoned, plans, [[1]])
    assert len(rollout.frames) == 3
    assert rollout.frames[0].output_indexes["vision"] == []
    assert rollout.frames[0].output_indexes["lidar"] == [0]
    for modality in ("vision", "lidar"):
        torch.testing.assert_close(
            getattr(rollout.data, f"x0_tokens_{modality}")[1],
            getattr(other.data, f"x0_tokens_{modality}")[1],
            rtol=0,
            atol=0,
        )
    for frame in rollout.frames:
        prefix = model._joint_prefix(rollout, frame)
        for modality in ("vision", "lidar"):
            assert not getattr(prefix, f"x0_tokens_{modality}")[1][:, :, frame.prefix_indexes[modality]].count_nonzero()
            predicted = frame.noisy[modality] - frame.sigma * model.net[modality]  # [1,C,T_chunk,H,W]
            actual = getattr(rollout.data, f"x0_tokens_{modality}")[1][:, :, frame.output_indexes[modality]]
            # [1,C,T_chunk,H,W]
            torch.testing.assert_close(predicted, actual)
    torch.testing.assert_close(rollout.data.x0_tokens_vision[1][:, :, [0, 7]], data.x0_tokens_vision[1][:, :, [0, 7]])


@pytest.mark.level(0)
@pytest.mark.parametrize("guidance", [1.0, 3.0])
def test_generator_gradient_matches_full_sensor_oracle(inputs: tuple, guidance: float) -> None:
    data, plans = inputs
    model = AnalyticJointModel(guidance=guidance)
    model._setup_grad_requirements(6)
    with torch.no_grad():
        rollout = model._joint_rollout(data, plans, [[1]])
        targets, _ = model._joint_pseudo_targets(rollout, plans, [[1]], [[0]], ["480"])
    sigma = rollout.frames[0].sigma
    expected: dict[str, torch.Tensor] = {}
    for modality in ("vision", "lidar"):
        generated = getattr(rollout.data, f"x0_tokens_{modality}")[1]  # [1,C,T,H,W]
        mask = (1 - getattr(rollout.packed, modality).condition_mask[1]).expand_as(generated)  # [1,C,T,H,W]
        expected[modality] = -sigma * ((generated - targets[modality]) * mask).sum() / mask.sum()
        # []
        expected[modality] = expected[modality] * (1 if modality == "vision" else 2)  # []
    for frame in rollout.frames:
        _, loss = model._joint_replay_loss(rollout, frame, plans, [[1]], targets)
        loss.backward()
    for modality in ("vision", "lidar"):
        assert model.net[modality].grad is not None
        torch.testing.assert_close(model.net[modality].grad, expected[modality], atol=2e-6, rtol=2e-6)
        assert expected[modality].abs() > 0
        assert model.net_fake_score[modality].grad is None
        assert model.net_teacher[modality].grad is None


@pytest.mark.level(0)
@pytest.mark.parametrize("iteration", [0, 1], ids=["student", "critic"])
@pytest.mark.parametrize("chunk_size", [1, 3])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
def test_joint_merged_outputs_preserve_scoring_sigma_for_callback(
    inputs: tuple, monkeypatch: pytest.MonkeyPatch, iteration: int, chunk_size: int, guidance: float
) -> None:
    data, plans = inputs
    model = AnalyticJointModel(guidance=guidance)
    model.config.teacher_forcing_frames_per_chunk = chunk_size
    model.input_image_key = "images"
    model.input_video_key = "video"
    monkeypatch.setattr(model, "_prepare_self_forcing_inputs", lambda *args: ([[1]], plans, data, ["480"], [[0]]))
    scoring_draws: list[torch.Tensor] = []

    def sample_scoring_sigma(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.tensor([[0.4]], requires_grad=True)  # [1,1]; distinct from replay's 0.625
        scoring_draws.append(sigma)
        return sigma * 1000, sigma  # [1,1], [1,1]

    monkeypatch.setattr(model, "_get_train_noise_level_vision", sample_scoring_sigma)
    closures = list(model.training_step_closures({}, iteration))
    outputs = [closure() for _, closure, _ in closures]
    merged: dict[str, torch.Tensor] = {}
    for metrics, _ in outputs:
        DistillationTrainer._merge_output_batches(merged, metrics)
    total = sum(loss.detach() for _, loss in outputs)  # []

    # Exercise the unchanged callback before inspecting its payload, reproducing
    # the native post-step KeyError when the student omits its scoring sigma.
    callback = SigmaLossAnalysis(every_n=100, every_n_viz=100)
    callback.on_training_step_end(model, {"lidar": data.x0_tokens_lidar[1]}, merged, total, iteration=1)
    assert len(scoring_draws) == 1
    assert [index for index, (metrics, _) in enumerate(outputs) if "sigma" in metrics] == [0]
    assert merged["sigma"].shape == (1, 1) and not merged["sigma"].requires_grad
    torch.testing.assert_close(merged["sigma"], scoring_draws[0], rtol=0, atol=0)
    torch.testing.assert_close(merged["dmd_loss"], total)
    for modality in ("vision", "lidar"):
        key = f"flow_matching_loss_{modality}_per_instance"
        expected = sum(metrics[key] for metrics, _ in outputs)  # [1]
        torch.testing.assert_close(merged[key], expected)
    cached_sigma, cached_loss = callback.video_cache.get_arrays()  # [1], [1]
    assert cached_sigma.numel() == cached_loss.numel() == 1
    torch.testing.assert_close(cached_sigma, scoring_draws[0].detach().flatten().to(torch.bfloat16))
    torch.testing.assert_close(cached_loss, merged["flow_matching_loss_vision_per_instance"].to(torch.bfloat16))
    if iteration == 0:
        assert len(closures) > 1
        assert [index for index, (metrics, _) in enumerate(outputs) if "teacher_guidance" in metrics] == [0]
        assert merged["teacher_guidance"].item() == guidance


@pytest.mark.level(0)
def test_critic_uses_noisy_lidar_and_updates_only_fake_score(inputs: tuple) -> None:
    data, plans = inputs
    model = AnalyticJointModel()
    model._setup_grad_requirements(1)
    with torch.no_grad():
        rollout = model._joint_rollout(data, plans, [[1]])
    metrics, loss = model._joint_critic(rollout, plans, [[1]], ["480"])
    loss.backward()
    torch.testing.assert_close(
        loss.detach(), metrics["joint_critic_loss_vision"] + 2 * metrics["joint_critic_loss_lidar"]
    )
    for modality in ("vision", "lidar"):
        generated = getattr(rollout.data, f"x0_tokens_{modality}")[1]  # [1,C,T,H,W]
        mask = 1 - getattr(rollout.packed, modality).condition_mask[1]  # [T,1,1]
        expected_noisy = generated * (1 - 0.4 * mask) + 0.7 * 0.4 * mask  # [1,C,T,H,W]
        torch.testing.assert_close(model.calls[-1][modality], expected_noisy)
        assert model.net_fake_score[modality].grad is not None
        assert model.net_fake_score[modality].grad.abs() > 0
        assert model.net[modality].grad is None
        assert model.net_teacher[modality].grad is None


@pytest.mark.level(0)
def test_active_loss_has_zero_condition_gradient() -> None:
    prediction = torch.tensor([1.0, 5.0, 9.0], requires_grad=True).reshape(1, 1, 3, 1, 1)  # [1,1,3,1,1]
    prediction.retain_grad()
    target = torch.zeros_like(prediction)  # [1,1,3,1,1]
    mask = torch.tensor([0.0, 1.0, 1.0]).reshape(3, 1, 1)  # [3,1,1]
    loss = active_mean_square(prediction, target, mask)  # []
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(53.0))
    torch.testing.assert_close(prediction.grad.flatten(), torch.tensor([0.0, 5.0, 9.0]))


@pytest.mark.level(0)
def test_missing_lidar_fails_before_rollout(inputs: tuple) -> None:
    data, plans = inputs
    data.num_lidar_items_per_sample = [0]
    with pytest.raises(ValueError, match="control and target"):
        AnalyticJointModel()._joint_rollout(data, plans, [[1]])


@pytest.mark.level(0)
def test_negative_captions_follow_cp_owner_metadata(inputs: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    data, plans = inputs
    plans[0].text_view_ids = [0, 1]
    model = AnalyticJointModel()
    model.vlm_config = SimpleNamespace(use_system_prompt=False)
    seen_negative: list[list[list[int]]] = []
    original = model._joint_pseudo_targets

    def targets(
        rollout: Any, plans: list, text: list, negative: list[list[int]], resolutions: list[str]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:  # modality -> [1,C,T,H,W], [1,1]
        seen_negative.append(negative)
        return original(rollout, plans, text, negative, resolutions)

    monkeypatch.setattr(model, "_joint_pseudo_targets", targets)
    monkeypatch.setattr(
        model,
        "_get_training_inputs",
        lambda *args: ([[1], [2]], plans, data, {"_sf_dmd_unconditional_text_indexes": [[0, 195]]}, ["480"], []),
    )
    plans[0].condition_frame_indexes_vision = [0]
    closures = list(model.training_step_closures({"system_prompt": "Rank-local prompt"}, iteration=6))
    assert seen_negative == [[[0, 195], [0, 195]]]
    assert [sync for _, _, sync in closures] == [False, False, True]


@pytest.mark.level(0)
def test_scorers_use_bidirectional_cfg_without_student_cache(inputs: tuple) -> None:
    data, plans = inputs
    model = AnalyticJointModel()
    with torch.no_grad():
        rollout = model._joint_rollout(data, plans, [[1]])
        model._joint_pseudo_targets(rollout, plans, [[1]], [[0]], ["480"])
    assert [call["net"] for call in model.calls[-3:]] == [
        model.net_fake_score,
        model.net_teacher,
        model.net_teacher,
    ]
    assert [call["negative"] for call in model.calls[-3:]] == [False, False, True]
    assert model.scoring_memories == []
    assert all(call["memory"] is None and call["causal"] is False for call in model.calls[-3:])
    assert all(call["causal"] is True for call in model.calls[:-3])


class HistoryDependentJointModel(AnalyticJointModel):
    """Make velocity depend on clean history so stale replay changes endpoints."""

    def _joint_chunk_velocity(
        self,
        prefix: GenerationDataClean,
        pack: Any,
        memory: Any,
        frame: joint.JointReplayFrame,
        noisy: dict[str, torch.Tensor],  # modality -> [1,C,T_chunk,H,W]
        sigma: float,
    ) -> dict[str, torch.Tensor]:
        velocity = super()._joint_chunk_velocity(prefix, pack, memory, frame, noisy, sigma)
        assert prefix.x0_tokens_vision is not None and prefix.x0_tokens_lidar is not None
        history = prefix.x0_tokens_vision[1].mean() + prefix.x0_tokens_lidar[1].mean()  # []
        return {key: value + 0.1 * history for key, value in velocity.items()}  # each [1,C,T_chunk,H,W]


@pytest.mark.level(0)
def test_replay_recomputes_history_without_current_or_future_targets(inputs: tuple) -> None:
    data, plans = inputs
    model = HistoryDependentJointModel()
    with torch.no_grad():
        torch.manual_seed(41)
        rollout = model._joint_rollout(data, plans, [[1]])
        for frame in rollout.frames:
            prefix = model._joint_prefix(rollout, frame)
            pack, memory = joint._build_replay_branch(model, plans, prefix, [[1]])
            velocity = model._joint_chunk_velocity(prefix, pack, memory, frame, frame.noisy, frame.sigma)
            for modality in ("vision", "lidar"):
                expected = frame.noisy[modality] - frame.sigma * velocity[modality]  # [1,C,T_chunk,H,W]
                actual = getattr(rollout.data, f"x0_tokens_{modality}")[1][:, :, frame.output_indexes[modality]]
                # [1,C,T_chunk,H,W]
                torch.testing.assert_close(expected, actual)

        first = rollout.frames[0]
        baseline = model._joint_prefix(rollout, first)
        for modality in ("vision", "lidar"):
            for later in rollout.frames[1:]:
                getattr(rollout.data, f"x0_tokens_{modality}")[1][:, :, later.output_indexes[modality]] = float("nan")
        replay = model._joint_prefix(rollout, first)
        for modality in ("vision", "lidar"):
            torch.testing.assert_close(
                getattr(baseline, f"x0_tokens_{modality}")[1], getattr(replay, f"x0_tokens_{modality}")[1]
            )


@pytest.mark.level(0)
def test_joint_training_sde_reinjects_fresh_noise_at_each_sigma(inputs: tuple) -> None:
    data, plans = inputs
    model = AnalyticJointModel()
    with torch.no_grad():
        torch.manual_seed(73)
        model._joint_rollout(data, plans, [[1]])
    states = [call["lidar"][:, :, :1] for call in model.calls[:4]]  # each [1,Cl,1,Hl,Wl]
    sigmas = [1.0, 0.9375, 5 / 6, 0.625]
    noises = [states[0]]
    for index in range(3):
        endpoint = states[index] - sigmas[index] * 0.5  # [1,Cl,1,Hl,Wl]
        noise = (states[index + 1] - (1 - sigmas[index + 1]) * endpoint) / sigmas[index + 1]
        # [1,Cl,1,Hl,Wl]
        assert all(not torch.allclose(noise, prior) for prior in noises)
        noises.append(noise)


@pytest.mark.level(0)
def test_joint_setup_builds_bidirectional_scorers_and_loads_regular_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    model = AnalyticJointModel()
    model.vlm_config = SimpleNamespace(pretrained_weights=SimpleNamespace(enabled=False))
    model.config.vlm_config = model.vlm_config
    model.config.vlm_config_teacher = SimpleNamespace(pretrained_weights=SimpleNamespace(enabled=False))
    model.config.vlm_config_fake_score = SimpleNamespace(pretrained_weights=SimpleNamespace(enabled=False))
    model.config.ema = SimpleNamespace(enabled=False)
    model.config.load_teacher_weights = True
    model.config.teacher_load_from = {"load_path": "bidirectional/model", "credentials": None}
    model.config.student_load_from = {"load_path": "stage2/model", "credentials": None}
    model.config.parallelism = SimpleNamespace(enable_inference_mode=False, fsdp_master_dtype="float32")
    model.precision = torch.float32
    built: list[bool] = []
    loaded: list[tuple[str, str]] = []

    def build(*args: Any, **kwargs: Any) -> torch.nn.Module:
        built.append(model.config.video_temporal_causal)
        return model.make_net(0.25, 0.5)

    def load(self: Any, net: Any, path: str, prefix: str = "net_ema", **kwargs: Any) -> None:
        loaded.append((path, prefix))

    monkeypatch.setattr(OmniMoTCausalModel, "set_up_model", lambda self: None)
    monkeypatch.setattr(model, "build_net", build)
    monkeypatch.setattr(joint.SelfForcingDMDModel, "_load_checkpoint_to_net", load)
    model.set_up_model()
    assert built == [False, False]
    assert loaded == [("bidirectional/model", "net"), ("bidirectional/model", "net"), ("stage2/model", "net")]
    assert model.config.video_temporal_causal
    assert not model.net_teacher.training and not any(p.requires_grad for p in model.net_teacher.parameters())


@pytest.mark.level(0)
def test_joint_layout_uses_actual_packed_sensor_metadata(inputs: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    data, plans = inputs
    mask = torch.zeros(14, 1, 1)  # [2*7,1,1]
    mask[[0, 7]] = 1
    packed = PackedSequence(
        text_ids=torch.tensor([11, 12]),  # [2]
        sample_lens=[46],
        vision=ModalityData(
            tokens=data.x0_tokens_vision,
            token_shapes=[(14, 1, 1), (14, 1, 1)],
            condition_mask=[torch.ones_like(mask), mask],  # list[[14,1,1]]
            seconds_per_frame=[4 / 30, 4 / 30],
        ),
        lidar=ModalityData(
            tokens=data.x0_tokens_lidar,
            token_shapes=[(9, 1, 1), (9, 1, 1)],
            condition_mask=[torch.ones(9, 1, 1), torch.zeros(9, 1, 1)],  # list[[9,1,1]]
            seconds_per_frame=[0.1, 0.1],
        ),
        num_vision_items_per_sample=[2],
        num_lidar_items_per_sample=[2],
        num_views_per_vision_item=[2, 2],
    )
    packed.text_caption_lens = [[1, 1]]
    packed.text_caption_view_ids = [[0, 1]]
    model = AnalyticJointModel()
    monkeypatch.setattr(joint, "build_interactive_multiview_mask_items", build_interactive_multiview_mask_items)
    monkeypatch.setattr(model, "_pack_input_sequence", lambda *args: packed)
    actual, views, chunks = model._joint_layout(data, plans, [[11], [12]])
    assert actual is packed and views == 2
    assert [(c.vision_frames, c.lidar_frames) for c in chunks] == [
        ((0,), (0,)),
        ((1, 2, 3), (1, 2, 3, 4)),
        ((4, 5, 6), (5, 6, 7, 8)),
    ]


class DistributedNoiseJointModel(AnalyticJointModel):
    """Exercise real sensor noising with deliberately rank-divergent RNG/time."""

    def _add_noise_to_input(
        self, data: GenerationDataClean, packed: Any, sigmas: torch.Tensor, *, sigmas_lidar: torch.Tensor | None
    ) -> GenerationDataNoised:  # sigmas, sigmas_lidar: [2,1]
        return OmniMoTModel._add_noise_to_input(self, data, packed, sigmas, sigmas_lidar=sigmas_lidar)

    def _get_train_noise_level_vision(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = torch.tensor([[0.2 * (dist.get_rank() + 1)]])  # [1,1]
        return sigma * 1000, sigma  # [1,1], [1,1]


def _distributed_joint_noise_worker(rank: int, rendezvous: str, data: GenerationDataClean, plans: list) -> None:
    # The model import initializes the loss package before RectifiedFlow; direct
    # module-level import would enter the legacy loss-package import cycle.
    from cosmos_framework.model.generator.diffusion.rectified_flow import RectifiedFlow

    patches = pytest.MonkeyPatch()
    _install_analytic_replay(patches)
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=rendezvous, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "cp"))
        model = DistributedNoiseJointModel()
        model.parallel_dims = SimpleNamespace(cp_enabled=True, cfgp_enabled=False, cp_mesh=mesh["cp"])
        model.rectified_flow_video = RectifiedFlow(velocity_field=model.net, device=torch.device("cpu"))
        torch.manual_seed(1000 + rank)
        with torch.no_grad():
            rollout = model._joint_rollout(data, plans, [[1]])
            noised, timestep, sigma = model._joint_scoring_inputs(rollout, ["480"])
        result = torch.cat(
            [
                rollout.data.x0_tokens_vision[1].flatten(),  # [Nv]
                rollout.data.x0_tokens_lidar[1].flatten(),  # [Nl]
                noised.xt_tokens_vision[1].flatten(),  # [Nv]
                noised.xt_tokens_lidar[1].flatten(),  # [Nl]
                timestep.flatten(),  # [1]
                sigma.flatten(),  # [1]
            ]
        )  # [2*Nv+2*Nl+2]
        gathered = [torch.empty_like(result) for _ in range(4)]  # list[[2*Nv+2*Nl+2]]
        dist.all_gather(gathered, result)
        for owner, peer in ((0, 1), (2, 3)):
            torch.testing.assert_close(gathered[owner], gathered[peer], rtol=0, atol=0)
        assert not torch.equal(gathered[0], gathered[2])
        torch.testing.assert_close(sigma, torch.tensor([[0.2 if rank < 2 else 0.6]]))  # [1,1]
    finally:
        dist.destroy_process_group()


@pytest.mark.level(1)
def test_joint_noise_is_synchronized_within_cp_and_independent_across_dp(inputs: tuple, tmp_path: Path) -> None:
    data, plans = inputs
    torch.multiprocessing.start_processes(
        _distributed_joint_noise_worker,
        args=(f"file://{tmp_path / 'gloo-store'}", data, plans),
        nprocs=4,
        start_method="spawn",
    )


def _distributed_joint_length_worker(
    rank: int, rendezvous: str, data: GenerationDataClean, plans: list, unequal_horizons: bool
) -> None:
    patches = pytest.MonkeyPatch()
    _install_analytic_replay(patches)
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=rendezvous, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "cp"))
        model = AnalyticJointModel()
        model.parallel_dims = SimpleNamespace(cp_enabled=True, cfgp_enabled=False, cp_mesh=mesh["cp"])
        if rank >= 2 and unequal_horizons:
            data = replace(data, x0_tokens_lidar=[value[:, :, :5] for value in data.x0_tokens_lidar])
            # list[[1,Cl,5,Hl,Wl]]; each CP group is internally consistent.
        elif rank >= 2:
            data = replace(
                data,
                x0_tokens_vision=[value[:, :, :7] for value in data.x0_tokens_vision],  # list[[1,Cv,7,Hv,Wv]]
                num_views_per_vision_item=[1, 1],
            )
        if unequal_horizons:
            with pytest.raises(ValueError, match="across HSDP"):
                model._joint_rollout(data, plans, [[1]])
            assert model.calls == [] and model.prefixes == []
        else:
            # Production mixes rigs across independent replicas; this changes
            # token counts while preserving the number/order of backward calls.
            with torch.no_grad():
                rollout = model._joint_rollout(data, plans, [[1]])
            assert len(rollout.frames) == 3
    finally:
        dist.destroy_process_group()


@pytest.mark.level(1)
@pytest.mark.parametrize("unequal_horizons", [False, True])
def test_joint_validates_replica_horizons_without_rejecting_different_rigs(
    inputs: tuple, tmp_path: Path, unequal_horizons: bool
) -> None:
    data, plans = inputs
    torch.multiprocessing.start_processes(
        _distributed_joint_length_worker,
        args=(f"file://{tmp_path / 'unequal-gloo-store'}", data, plans, unequal_horizons),
        nprocs=4,
        start_method="spawn",
    )


class FullClipGradientScorerModel(AnalyticJointModel):
    """Expose critic gradients through every noisy frame, including future ones."""

    def denoise(
        self,
        *,
        net: torch.nn.Module,
        data_batch_packed: Any,
        memory: Any = None,
        video_temporal_causal: bool | None = None,
    ) -> dict[str, Any]:
        output = super().denoise(
            net=net,
            data_batch_packed=data_batch_packed,
            memory=memory,
            video_temporal_causal=video_temporal_causal,
        )
        if video_temporal_causal is False:
            assert memory is None
            for key in ("vision", "lidar"):
                noisy = getattr(data_batch_packed, key).tokens[1]  # [1,C,T,H,W]
                full_clip = noisy.mean() * net[key]  # []
                output[f"preds_{key}"] = [value + full_clip for value in output[f"preds_{key}"]]
                # list[[1,C,T,H,W]]
        return output


@pytest.mark.level(0)
def test_joint_critic_keeps_full_noisy_clip_parameter_gradients(inputs: tuple) -> None:
    data, plans = inputs
    model = FullClipGradientScorerModel()
    model._setup_grad_requirements(1)
    with torch.no_grad():
        rollout = model._joint_rollout(data, plans, [[1]])
    _, loss = model._joint_critic(rollout, plans, [[1]], ["480"])
    loss.backward()
    for modality in ("vision", "lidar"):
        generated = getattr(rollout.data, f"x0_tokens_{modality}")[1]  # [1,C,T,H,W]
        target = 0.7 - generated  # [1,C,T,H,W]
        mask = (1 - getattr(rollout.packed, modality).condition_mask[1]).expand_as(generated)  # [1,C,T,H,W]
        weight = model.net_fake_score[modality].detach()  # []
        noisy = generated * (1 - 0.4 * mask) + 0.7 * 0.4 * mask  # [1,C,T,H,W]
        scale = 1 + noisy.mean()  # []
        # Prediction includes all noisy frames. Both parameter paths contribute
        # gradients, while the generated sample remains detached from the student.
        expected = (2 * scale * (scale * weight - target) * mask).sum() / mask.sum()  # []
        expected = expected * model._joint_sensor_weight(modality)  # []
        torch.testing.assert_close(model.net_fake_score[modality].grad, expected)
    assert model.scoring_memories == []
    assert all(parameter.grad is None for parameter in model.net.parameters())
    assert all(parameter.grad is None for parameter in model.net_teacher.parameters())


@pytest.mark.level(0)
@pytest.mark.parametrize("mode", ["max", "cycle"])
@pytest.mark.parametrize("iteration", [0, 1, 2, 3, 4, 9])
def test_joint_rollout_consumes_exit_policy(inputs: tuple, mode: str, iteration: int) -> None:
    data, plans = inputs
    model = AnalyticJointModel()
    model.config.sf_rollout_step_mode = mode
    with torch.no_grad():
        rollout = model._joint_rollout(data, plans, [[1]], effective_iteration=iteration)
    count = 4 if mode == "max" else iteration % 4 + 1
    assert len(model.calls) == len(rollout.frames) * count
    expected = model.config.fixed_step_sampler_config.t_list[count - 1]
    assert all(frame.sigma == expected for frame in rollout.frames)


@pytest.mark.level(0)
def test_rgb_replay_retains_history_without_lidar_or_future_ground_truth(inputs: tuple) -> None:
    data, plans = inputs
    model = AnalyticJointModel()
    model.config.joint_sensor_modalities = ("vision",)
    data = replace(data, x0_tokens_lidar=None, num_lidar_items_per_sample=None)
    plans = [replace(plans[0], has_lidar=False)]
    with torch.no_grad():
        torch.manual_seed(71)
        rollout = model._joint_rollout(data, plans, [[1]], effective_iteration=0)
        targets, _ = model._joint_pseudo_targets(rollout, plans, [[1]], [[0]], ["480"])
    assert rollout.data.x0_tokens_lidar is None
    assert set(targets) == {"vision"}
    assert all(prefix.x0_tokens_lidar is None for prefix in model.prefixes)
    assert all(call["lidar"] is None for call in model.calls)
    generated = rollout.data.x0_tokens_vision[1]  # [1,C,V*T,H,W]
    for frame in rollout.frames:
        metrics, loss = model._joint_replay_loss(rollout, frame, plans, [[1]], targets)
        assert "joint_generator_loss_lidar" not in metrics
        loss.backward()
    assert model.net["vision"].grad is not None
    assert torch.isfinite(model.net["vision"].grad).all()
    assert model.net["lidar"].grad is None
    assert torch.isfinite(generated).all()
    with torch.no_grad():
        poisoned = replace(data, x0_tokens_vision=[value.clone() for value in data.x0_tokens_vision])
        # Preserve only the two conditioned camera frames. Future targets may
        # contain arbitrary values and must never affect the generated rollout.
        poisoned.x0_tokens_vision[1][:, :, 1:7] = 99.0  # [1,C,6,H,W]
        poisoned.x0_tokens_vision[1][:, :, 8:] = -99.0  # [1,C,6,H,W]
        torch.manual_seed(71)
        other = model._joint_rollout(poisoned, plans, [[1]], effective_iteration=0)
    torch.testing.assert_close(generated, other.data.x0_tokens_vision[1], rtol=0, atol=0)


class SimAutoPrefixJointModel(AnalyticJointModel):
    """Exercise sim-auto input preparation with plan-aware masks and analytic networks."""

    rollouts: list[JointRollout]

    def __init__(self) -> None:
        super().__init__()
        self.config.strict_regular_checkpoint_loading = True
        self.config.teacher_negative_prompt_per_view_metadata = True
        self.rollouts = []

    def _pack_input_sequence(
        self, plans: list[SequencePlan], text: list, data: GenerationDataClean, timestep: torch.Tensor, **kwargs: Any
    ) -> Any:  # timestep: [1,1]
        packed = super()._pack_input_sequence(plans, text, data, timestep, **kwargs)
        for modality, views in (("vision", data.num_views_per_vision_item[0]), ("lidar", 1)):
            if getattr(packed, modality) is None:
                continue
            mask = getattr(packed, modality).condition_mask[1]  # [V*T,1,1]
            mask.zero_()  # [V*T,1,1]
            frames = mask.shape[0] // views
            for view in range(views):
                for frame in getattr(plans[0], f"condition_frame_indexes_{modality}"):
                    if frame < frames:
                        mask[view * frames + frame] = 1.0  # [1,1]
        return packed

    def _joint_rollout(
        self,
        data: GenerationDataClean,
        plans: list[SequencePlan],
        text: list[list[int]],
        effective_iteration: int | None = None,
        *,
        use_persistent_kv: bool = False,
        save_replay_frames: bool = False,
    ) -> JointRollout:
        rollout = super()._joint_rollout(
            data,
            plans,
            text,
            effective_iteration=effective_iteration,
            use_persistent_kv=use_persistent_kv,
            save_replay_frames=save_replay_frames,
        )
        self.rollouts.append(rollout)
        return rollout


def _distributed_sim_auto_prefix_worker(rank: int, rendezvous: str, dropout: str) -> None:
    patches = pytest.MonkeyPatch()
    _install_analytic_replay(patches)
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=rendezvous, timeout=timedelta(seconds=90))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "cp"))
        # Independent replicas may sample different rigs, prefixes and RNG states.
        # Their 397-frame temporal schedule must still agree across both axes.
        views = 2 if rank < 2 else 1
        has_lidar = dropout == "none" or (dropout == "rank_zero" and rank >= 2) or (dropout == "other" and rank < 2)
        owner_rank = 2 if dropout == "rank_zero" else 0
        vision = torch.full((1, 2, views * 100, 1, 2), 3.0)  # [1,2,V*100,1,2]
        lidar = torch.full((1, 3, 133, 1, 1), 4.0)  # [1,3,133,1,1]
        data = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[vision.clone(), vision],  # list[[1,2,V*100,1,2]]
            x0_tokens_lidar=[lidar.clone(), lidar] if has_lidar else None,  # list[[1,3,133,1,1]] or None
            num_vision_items_per_sample=[2],
            num_lidar_items_per_sample=[2] if has_lidar else None,
            num_views_per_vision_item=[views, views],
        )
        for owner_prefix in (0, 1, 2):
            local_prefix = (owner_prefix + rank // 2 - owner_rank // 2) % 3
            plans = [
                SequencePlan(
                    has_text=True,
                    has_vision=True,
                    has_lidar=has_lidar,
                    condition_frame_indexes_vision=list(range(local_prefix)),
                    condition_frame_indexes_lidar=list(range((0, 1, 3)[local_prefix])) if has_lidar else [],
                )
            ]
            model = SimAutoPrefixJointModel()
            model.config.sf_rollout_step_mode = "max"
            model.parallel_dims = SimpleNamespace(cp_enabled=True, cfgp_enabled=False, cp_mesh=mesh["cp"])
            model._get_training_inputs = MagicMock(
                return_value=([[1]], plans, data, {"_sf_dmd_unconditional_text_indexes": [[0]]}, ["480"], [])
            )
            for iteration in (0, 1):
                torch.manual_seed(1000 + rank + iteration)
                model.net.zero_grad(set_to_none=True)
                model.net_fake_score.zero_grad(set_to_none=True)
                closures = list(model.training_step_closures({}, iteration))
                expected_closures = (34 if owner_prefix == 0 else 33) if iteration == 0 else 1
                assert len(closures) == expected_closures
                assert [sync for _, _, sync in closures] == [False] * (expected_closures - 1) + [True]
                for _, closure, _ in closures:
                    _, loss = closure()  # []
                    assert torch.isfinite(loss).all()
                    loss.backward()
                network = model.net if iteration == 0 else model.net_fake_score
                inactive = model.net_fake_score if iteration == 0 else model.net
                for modality, parameter in network.items():
                    assert parameter.grad is not None
                    assert torch.isfinite(parameter.grad).all()
                    if modality == "lidar" and not has_lidar:
                        assert parameter.grad.item() == 0
                    else:
                        assert parameter.grad.abs().sum() > 0
                    # Real world collectives verify completion across the replica
                    # boundary. This CPU oracle does not emulate full GPU FSDP.
                    dist.all_reduce(parameter.grad)  # []
                assert all(parameter.grad is None for parameter in inactive.parameters())
                assert all(parameter.grad is None for parameter in model.net_teacher.parameters())

                rollout = model.rollouts[-1]
                exit_sigma = torch.tensor([rollout.frames[0].sigma])  # [1]
                exits = [torch.empty_like(exit_sigma) for _ in range(4)]  # list[[1]]
                dist.all_gather(exits, exit_sigma)
                assert all(torch.equal(exits[0], value) for value in exits)
                assert all(frame.sigma == rollout.frames[0].sigma for frame in rollout.frames)
                for modality, frames, prefix, cameras in (
                    ("vision", 100, owner_prefix, views),
                    ("lidar", 133, (0, 1, 3)[owner_prefix], 1),
                ):
                    if modality == "lidar" and not has_lidar:
                        assert rollout.data.x0_tokens_lidar is None
                        assert all("lidar" not in chunk.output_indexes for chunk in rollout.frames)
                        continue
                    expected = [view * frames + frame for view in range(cameras) for frame in range(prefix, frames)]
                    actual = sorted(index for chunk in rollout.frames for index in chunk.output_indexes[modality])
                    assert actual == expected
                    conditioned = [view * frames + frame for view in range(cameras) for frame in range(prefix)]
                    generated = getattr(rollout.data, f"x0_tokens_{modality}")[1]  # [1,C,V*T,H,W]
                    original = getattr(data, f"x0_tokens_{modality}")[1]  # [1,C,V*T,H,W]
                    torch.testing.assert_close(generated[:, :, conditioned], original[:, :, conditioned])
                # Synchronization owns a new plan; cached rank-local loader plans
                # must remain intact for future CP data-window rotations.
                assert plans[0].condition_frame_indexes_vision == list(range(local_prefix))
                assert plans[0].condition_frame_indexes_lidar == (
                    list(range((0, 1, 3)[local_prefix])) if has_lidar else []
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.level(1)
@pytest.mark.parametrize("dropout", ["none", "rank_zero", "other", "all"])
def test_mixed_sensor_prefixes_cover_397_frames_across_independent_replicas(tmp_path: Path, dropout: str) -> None:
    torch.multiprocessing.start_processes(
        _distributed_sim_auto_prefix_worker,
        args=(f"file://{tmp_path / 'driving-prefix-gloo-store'}", dropout),
        nprocs=4,
        start_method="spawn",
    )


class PersistentHistoryJointModel(SimAutoPrefixJointModel):
    """Use summed clean sensor history as an independently checkable cache value."""

    def __init__(self) -> None:
        super().__init__()
        self.config.sf_critic_use_persistent_kv = True
        self.config.kv_cache_inference_size = None
        self.config.attention_sink_size = 0
        self.config.rolling_kv_cache_chunks = None
        self.net.num_hidden_layers = 1

    def _get_teacher_forcing_kv_implementation(self) -> str:
        return "multiview_maskless_kv"

    def _joint_chunk_velocity(
        self,
        prefix: GenerationDataClean,
        pack: Any,
        memory: Any,
        frame: joint.JointReplayFrame,
        noisy: dict[str, torch.Tensor],  # modality -> [1,C,T_chunk,H,W]
        sigma: float,
    ) -> dict[str, torch.Tensor]:  # modality -> [1,C,T_chunk,H,W]
        velocity = super()._joint_chunk_velocity(prefix, pack, memory, frame, noisy, sigma)
        history = sum(getattr(prefix, f"x0_tokens_{key}")[1].sum() for key in self._joint_modalities(prefix))  # []
        if memory is not None:
            history = history + memory.history  # []
        return {key: value + 0.001 * history for key, value in velocity.items()}  # each [1,C,T_chunk,H,W]


@pytest.fixture
def persistent_history(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, dict[str, torch.Tensor]]]:
    """Replace attention only; retain the production rollout, chunk slicing and noising."""
    commits: list[tuple[int, dict[str, torch.Tensor]]] = []

    def cache(reference: Any, num_layers: int) -> Any:
        return SimpleNamespace(history=torch.zeros(()))  # history: []

    def branch(
        host: PersistentHistoryJointModel,
        plans: list[SequencePlan],
        data: GenerationDataClean,
        text: list[list[int]],
        cache: Any,
        chunk: Any,
    ) -> tuple[Any, Any]:
        return host._pack_input_sequence(plans, text, data, torch.zeros(1, 1)), cache  # timestep: [1,1]

    def commit(
        host: PersistentHistoryJointModel,
        plans: list[SequencePlan],
        data: GenerationDataClean,
        branches: list[tuple[list[list[int]], Any]],
        chunk: Any,
    ) -> None:
        assert not torch.is_grad_enabled()
        values = {key: getattr(data, f"x0_tokens_{key}")[1].clone() for key in host._joint_modalities(data)}
        # values: modality -> [1,C,T_chunk,H,W]
        assert all(not value.requires_grad for value in values.values())
        commits.append((chunk.step, values))
        for _, cache in branches:
            cache.history = cache.history + sum(value.sum() for value in values.values())  # []

    monkeypatch.setattr(joint, "JointARCache", cache)
    monkeypatch.setattr(joint, "_build_persistent_branch", branch)
    monkeypatch.setattr(joint, "_commit_persistent_joint_chunk", commit)
    return commits


@pytest.mark.level(0)
@pytest.mark.parametrize("sample_type", ["sde", "ode"])
@pytest.mark.parametrize("conditioned", [0, 1, 2])
@pytest.mark.parametrize("has_lidar", [False, True])
def test_persistent_critic_rollout_matches_history_dependent_replay(
    inputs: tuple,
    persistent_history: list[tuple[int, dict[str, torch.Tensor]]],
    sample_type: str,
    conditioned: int,
    has_lidar: bool,
) -> None:
    data, plans = inputs
    if not has_lidar:
        data = replace(data, x0_tokens_lidar=None, num_lidar_items_per_sample=None)
    plans = [
        replace(
            plans[0],
            has_lidar=has_lidar,
            condition_frame_indexes_vision=list(range(conditioned)),
            condition_frame_indexes_lidar=list(range((0, 1, 3)[conditioned])) if has_lidar else [],
        )
    ]
    model = PersistentHistoryJointModel()
    model.config.fixed_step_sampler_config.sample_type = sample_type
    with torch.no_grad():
        torch.manual_seed(379)
        replay = model._joint_rollout(data, plans, [[1]])
        replay_rng = torch.get_rng_state()  # [R]
        torch.manual_seed(379)
        persistent = model._joint_rollout(data, plans, [[1]], use_persistent_kv=True)
        persistent_rng = torch.get_rng_state()  # [R]
    assert persistent.frames == []
    torch.testing.assert_close(replay_rng, persistent_rng, rtol=0, atol=0)
    assert [step for step, _ in persistent_history] == [0, 1]
    _, views, chunks = model._joint_layout(data, plans, [[1]])
    frames_per_view = data.x0_tokens_vision[1].shape[2] // views
    for key in model._joint_modalities(data):
        actual = getattr(persistent.data, f"x0_tokens_{key}")[1]  # [1,C,T,H,W]
        expected = getattr(replay.data, f"x0_tokens_{key}")[1]  # [1,C,T,H,W]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert not actual.requires_grad
        for (_, values), frame in zip(persistent_history, chunks):
            indexes = (
                [view * frames_per_view + index for view in range(views) for index in frame.vision_frames]
                if key == "vision"
                else list(frame.lidar_frames)
            )
            torch.testing.assert_close(values[key], actual[:, :, indexes])
    model._setup_grad_requirements(1)
    _, loss = model._joint_critic(persistent, plans, [[1]], ["480"])
    loss.backward()
    assert all(parameter.grad is None for parameter in model.net.parameters())
    assert all(parameter.grad is None for parameter in model.net_teacher.parameters())
    for key, parameter in model.net_fake_score.items():
        assert parameter.grad is not None
        if key in model._joint_modalities(data):
            assert parameter.grad.abs() > 0
        else:
            assert parameter.grad.item() == 0
    assert model.calls[-1]["memory"] is None and model.calls[-1]["causal"] is False


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "sample_type,conditioned,has_lidar",
    [("sde", 0, True), ("ode", 1, True), ("sde", 2, True), ("ode", 2, False)],
)
def test_persistent_student_rollout_preserves_replay_frames_loss_and_gradients(
    inputs: tuple,
    persistent_history: list[tuple[int, dict[str, torch.Tensor]]],
    sample_type: str,
    conditioned: int,
    has_lidar: bool,
) -> None:
    data, plans = inputs
    if not has_lidar:
        data = replace(data, x0_tokens_lidar=None, num_lidar_items_per_sample=None)
    plans = [
        replace(
            plans[0],
            has_lidar=has_lidar,
            condition_frame_indexes_vision=list(range(conditioned)),
            condition_frame_indexes_lidar=list(range((0, 1, 3)[conditioned])) if has_lidar else [],
        )
    ]
    model = PersistentHistoryJointModel()
    model.config.fixed_step_sampler_config.sample_type = sample_type
    model._setup_grad_requirements(0)
    with torch.no_grad():
        torch.manual_seed(379)
        replay = model._joint_rollout(data, plans, [[1]])
        replay_rng = torch.get_rng_state()  # [R]
        torch.manual_seed(379)
        persistent = model._joint_rollout(data, plans, [[1]], use_persistent_kv=True, save_replay_frames=True)
        persistent_rng = torch.get_rng_state()  # [R]
    torch.testing.assert_close(replay_rng, persistent_rng, rtol=0, atol=0)
    assert [step for step, _ in persistent_history] == [0, 1]
    assert len(persistent.frames) == len(replay.frames)
    for key in model._joint_modalities(data):
        actual = getattr(persistent.data, f"x0_tokens_{key}")[1]  # [1,C,T,H,W]
        expected = getattr(replay.data, f"x0_tokens_{key}")[1]  # [1,C,T,H,W]
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert not actual.requires_grad
    for actual, expected in zip(persistent.frames, replay.frames, strict=True):
        assert actual.chunk == expected.chunk
        assert actual.prefix_indexes == expected.prefix_indexes
        assert actual.output_indexes == expected.output_indexes
        assert actual.sigma == expected.sigma
        for key, noisy in actual.noisy.items():  # noisy: [1,C,T_chunk,H,W]
            torch.testing.assert_close(noisy, expected.noisy[key], rtol=1e-6, atol=1e-6)
            assert not noisy.requires_grad
        prefix = model._joint_prefix(persistent, actual)
        pack, memory = joint._build_replay_branch(model, plans, prefix, [[1]])
        with torch.no_grad():
            velocity = model._joint_chunk_velocity(prefix, pack, memory, actual, actual.noisy, actual.sigma)
            for key, noisy in actual.noisy.items():  # noisy: [1,C,T_chunk,H,W]
                endpoint = noisy - actual.sigma * velocity[key]  # [1,C,T_chunk,H,W]
                generated = getattr(persistent.data, f"x0_tokens_{key}")[1]  # [1,C,T,H,W]
                torch.testing.assert_close(endpoint, generated[:, :, actual.output_indexes[key]], rtol=1e-6, atol=1e-6)

    losses: list[torch.Tensor] = []  # each []
    gradients: list[dict[str, torch.Tensor]] = []  # each modality -> []
    for rollout in (replay, persistent):
        model.net.zero_grad(set_to_none=True)
        with torch.no_grad():
            targets, _ = model._joint_pseudo_targets(rollout, plans, [[1]], [[0]], ["480"])
        total = torch.zeros(())  # []
        for frame in rollout.frames:
            _, loss = model._joint_replay_loss(rollout, frame, plans, [[1]], targets)  # []
            loss.backward()
            total = total + loss.detach()  # []
        losses.append(total)
        current_gradients: dict[str, torch.Tensor] = {}
        for key, parameter in model.net.items():
            assert parameter.grad is not None
            current_gradients[key] = parameter.grad.detach().clone()  # []
            if key in model._joint_modalities(data):
                assert parameter.grad.abs() > 0
            else:
                assert parameter.grad.item() == 0
        gradients.append(current_gradients)
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-6, atol=1e-6)
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key], gradients[1][key], rtol=1e-6, atol=1e-6)
    assert all(parameter.grad is None for parameter in model.net_fake_score.parameters())
    assert all(parameter.grad is None for parameter in model.net_teacher.parameters())


@pytest.mark.level(0)
@pytest.mark.parametrize("backward_enabled,student", [(True, True), (False, True), (True, False)])
def test_backward_cache_is_retained_and_forwarded_only_when_enabled(
    inputs: tuple,
    persistent_history: list[tuple[int, dict[str, torch.Tensor]]],
    monkeypatch: pytest.MonkeyPatch,
    backward_enabled: bool,
    student: bool,
) -> None:
    data, plans = inputs
    model = PersistentHistoryJointModel()
    model.config.sf_student_backward_use_persistent_kv = backward_enabled
    with torch.no_grad():
        rollout = model._joint_rollout(data, plans, [[1]], use_persistent_kv=True, save_replay_frames=student)
    assert (rollout.cache is not None) == (student and backward_enabled)
    assert len(persistent_history) == 2
    if not student:
        return
    replay = joint._build_replay_branch
    seen: list[Any] = []

    def record(*args: Any, persistent_cache: Any = None, **kwargs: Any) -> Any:
        assert not torch.is_grad_enabled()
        seen.append(persistent_cache)
        return replay(*args, **kwargs)

    monkeypatch.setattr(joint, "_build_replay_branch", record)
    targets = {key: getattr(rollout.data, f"x0_tokens_{key}")[1] for key in model._joint_modalities(data)}
    # targets: modality -> [1,C,T,H,W]
    _, loss = model._joint_replay_loss(rollout, rollout.frames[-1], plans, [[1]], targets)
    loss.backward()
    assert len(seen) == 1 and seen[0] is rollout.cache


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "iteration,enabled,supported,used,fallback",
    [(0, True, True, 0, 0), (1, True, True, 1, 0), (1, True, False, 0, 1), (1, False, True, 0, 0)],
    ids=["student", "persistent", "fallback", "disabled"],
)
def test_critic_backend_metrics_report_actual_rollout(
    inputs: tuple,
    persistent_history: list[tuple[int, dict[str, torch.Tensor]]],
    monkeypatch: pytest.MonkeyPatch,
    iteration: int,
    enabled: bool,
    supported: bool,
    used: int,
    fallback: int,
) -> None:
    data, plans = inputs
    model = PersistentHistoryJointModel()
    model.config.sf_critic_use_persistent_kv = enabled
    if not supported:
        model.config.attention_sink_size = 1
    monkeypatch.setattr(model, "_prepare_self_forcing_inputs", lambda *args: ([[1]], plans, data, ["480"], [[0]]))
    outputs = [closure()[0] for _, closure, _ in model.training_step_closures({}, iteration)]
    expected = {"sf_critic_persistent_kv_used": used, "sf_critic_persistent_kv_fallback": fallback}
    for metrics in outputs:
        if iteration == 0:
            assert expected.keys().isdisjoint(metrics)
        else:
            for key, value in expected.items():
                assert metrics[key].shape == () and not metrics[key].requires_grad
                assert metrics[key].item() == value
    assert bool(persistent_history) == bool(used)


@pytest.mark.level(0)
@pytest.mark.parametrize(
    "iteration,student_enabled,critic_enabled,expected",
    [
        (0, True, False, True),
        (0, False, True, False),
        (0, True, True, True),
        (1, True, False, False),
        (1, False, True, True),
        (1, True, True, True),
    ],
)
@pytest.mark.parametrize("has_lidar", [False, True])
def test_phase_selects_its_persistent_rollout_flag(
    inputs: tuple,
    monkeypatch: pytest.MonkeyPatch,
    iteration: int,
    student_enabled: bool,
    critic_enabled: bool,
    expected: bool,
    has_lidar: bool,
) -> None:
    data, plans = inputs
    if not has_lidar:
        data = replace(data, x0_tokens_lidar=None, num_lidar_items_per_sample=None)
        plans = [replace(plans[0], has_lidar=False)]
    model = AnalyticJointModel()
    model.config.sf_student_use_persistent_kv = student_enabled
    model.config.sf_critic_use_persistent_kv = critic_enabled
    monkeypatch.setattr(model, "_prepare_self_forcing_inputs", lambda *args: ([[1]], plans, data, ["480"], [[0]]))
    rollout = model._joint_rollout
    requests: list[tuple[bool, bool]] = []

    def record(
        *args: Any, use_persistent_kv: bool = False, save_replay_frames: bool = False, **kwargs: Any
    ) -> JointRollout:
        assert not torch.is_grad_enabled()
        requests.append((use_persistent_kv, save_replay_frames))
        return rollout(*args, save_replay_frames=save_replay_frames, **kwargs)

    monkeypatch.setattr(model, "_joint_rollout", record)
    closures = list(model.training_step_closures({}, iteration))
    assert requests == [(expected, iteration == 0)]
    assert len(closures) == ((3 if has_lidar else 2) if iteration == 0 else 1)


@pytest.mark.level(0)
@pytest.mark.parametrize("save_replay_frames", [False, True], ids=["critic", "student"])
@pytest.mark.parametrize("unsupported", ["condition_after_generated_chunk", "flex", "window", "sink"])
def test_unsupported_persistent_layout_falls_back_to_replay(
    inputs: tuple, monkeypatch: pytest.MonkeyPatch, unsupported: str, save_replay_frames: bool
) -> None:
    data, plans = inputs
    model = PersistentHistoryJointModel()
    if unsupported == "condition_after_generated_chunk":
        plans = [replace(plans[0], condition_frame_indexes_vision=[0, 1], condition_frame_indexes_lidar=[])]
    elif unsupported == "flex":
        monkeypatch.setattr(model, "_get_teacher_forcing_kv_implementation", lambda: "multiview_flex_kv")
    elif unsupported == "window":
        model.config.kv_cache_inference_size = 3
    else:
        model.config.attention_sink_size = 1

    def reject_cache(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Unsupported samples must retain clean-prefix replay")

    monkeypatch.setattr(joint, "JointARCache", reject_cache)
    with torch.no_grad():
        torch.manual_seed(47)
        expected = model._joint_rollout(data, plans, [[1]])
        torch.manual_seed(47)
        actual = model._joint_rollout(data, plans, [[1]], use_persistent_kv=True, save_replay_frames=save_replay_frames)
    assert actual.frames
    assert actual.cache is None
    for key in model._joint_modalities(data):
        torch.testing.assert_close(
            getattr(actual.data, f"x0_tokens_{key}")[1], getattr(expected.data, f"x0_tokens_{key}")[1], rtol=0, atol=0
        )


def _distributed_persistent_eligibility_worker(
    rank: int, rendezvous: str, data: GenerationDataClean, plans: list[SequencePlan]
) -> None:
    patches = pytest.MonkeyPatch()
    _install_analytic_replay(patches)
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=rendezvous, timeout=timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "cp"))
        has_lidar = rank < 2
        if not has_lidar:
            data = replace(data, x0_tokens_lidar=None, num_lidar_items_per_sample=None)
        for prefix in (0, 1, 2):
            local_plans = [
                replace(
                    plans[0],
                    has_lidar=has_lidar,
                    condition_frame_indexes_vision=list(range(prefix)),
                    condition_frame_indexes_lidar=list(range((0, 1, 3)[prefix])) if has_lidar else [],
                )
            ]
            model = PersistentHistoryJointModel()
            model.parallel_dims = SimpleNamespace(cp_enabled=True, cfgp_enabled=False, cp_mesh=mesh["cp"])
            pack, views, chunks = model._joint_layout(data, local_plans, [[1]])
            # All HSDP ranks must choose the same number of student forwards, even
            # when just one independent CP group dropped LiDAR from its sample.
            assert model._can_use_persistent_joint_kv(pack, views, chunks)
            if not has_lidar:
                model._get_teacher_forcing_kv_implementation = lambda: "multiview_flex_kv"
            assert not model._can_use_persistent_joint_kv(pack, views, chunks)
    finally:
        dist.destroy_process_group()


@pytest.mark.level(1)
def test_persistent_critic_eligibility_agrees_across_independent_replicas(inputs: tuple, tmp_path: Path) -> None:
    data, plans = inputs
    torch.multiprocessing.start_processes(
        _distributed_persistent_eligibility_worker,
        args=(f"file://{tmp_path / 'persistent-gloo-store'}", data, plans),
        nprocs=4,
        start_method="spawn",
    )
