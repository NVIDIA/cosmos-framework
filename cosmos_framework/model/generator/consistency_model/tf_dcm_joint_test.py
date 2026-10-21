# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Analytic two-sensor integration checks for the opt-in joint TF-dCM solver."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from cosmos_framework.data.generator.sequence_packing.sequence import ModalityData, PackedSequence
from cosmos_framework.model.generator.consistency_model import tf_dcm_joint
from cosmos_framework.model.generator.consistency_model.tf_dcm_joint import TeacherForcingJointDCMModel
from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import build_interactive_multiview_mask_items
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean, GenerationDataNoised
from cosmos_framework.model.generator.utils.kv_cache import TeacherForcingMemoryState


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_noisy_endpoint_pack_stages_timestep_metadata_on_cpu(device: str) -> None:
    """Exercise the native helper with CPU and simulated CUDA ladder metadata."""
    calls: list[torch.Tensor] = []

    def pack_input(
        plans: list, text: list, clean: GenerationDataClean, timesteps: torch.Tensor, **kwargs: Any
    ) -> SimpleNamespace:  # timesteps: [B,1]
        # Match the production packer's device contract. FakeTensorMode keeps
        # this regression executable on CPU without pretending CUDA is CPU.
        if timesteps.is_cuda:
            raise ValueError("input_timesteps must be on CPU, not CUDA")
        calls.append(timesteps)
        return SimpleNamespace(to_cuda=lambda: None)

    model = SimpleNamespace(
        _pack_input_sequence=pack_input,
        _replace_clean_with_noised=lambda packed, noised: None,
    )
    with FakeTensorMode():
        timesteps = torch.empty((2, 1), dtype=torch.float32, device=device)  # [B,1]
        TeacherForcingJointDCMModel._build_noisy_target_pack(
            model,
            [],
            [],
            GenerationDataClean(batch_size=2, is_image_batch=False),
            None,
            timesteps,
            {"skip_text": False, "initial_temporal_offset": 0},
        )
        assert len(calls) == 1 and calls[0].device.type == "cpu"
        assert calls[0].shape == timesteps.shape and calls[0].dtype == timesteps.dtype
        assert timesteps.device.type == device


class _AnalyticJointModel(TeacherForcingJointDCMModel):
    """Use constant velocity fields with unequal camera/sweep counts and explicit controls."""

    calls: list[dict[str, Any]]
    clean: GenerationDataClean
    masks: dict[str, list[torch.Tensor]]

    def __init__(self, lidar_mode: str, guidance: float, conditioned_prefix: int) -> None:
        torch.nn.Module.__init__(self)
        self.config = SimpleNamespace(
            sound_gen=False,
            action_gen=False,
            lidar_state_ch=128,
            teacher_forcing_dcm_total_steps=48,
            teacher_forcing_dcm_skipping_interval_steps=1,
            teacher_forcing_dcm_teacher_guidance=guidance,
            teacher_forcing_dcm_detach_target_endpoint=True,
            teacher_forcing_detach_clean_kv=False,
            teacher_forcing_dcm_enable_load_balancing_loss=False,
            teacher_forcing_dcm_sample_timestep_shift=1.0,
            rectified_flow_training_config=SimpleNamespace(
                loss_scale=100.0, lidar_loss_scale=200.0, image_loss_scale=None, exclude_fully_conditioned_items=True
            ),
        )
        self.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        self.tensor_kwargs_fp32 = self.tensor_kwargs
        self.parallel_dims = None
        self.net = torch.nn.ParameterDict(
            {
                "vision": torch.nn.Parameter(torch.tensor(0.2)),  # []
                "lidar": torch.nn.Parameter(torch.tensor(0.4)),  # []
            }
        )
        self._net_teacher = [
            torch.nn.ParameterDict(
                {
                    "vision": torch.nn.Parameter(torch.tensor(0.3), requires_grad=False),  # []
                    "lidar": torch.nn.Parameter(torch.tensor(0.5), requires_grad=False),  # []
                }
            )
        ]
        self.rectified_flow_video = SimpleNamespace(
            noise_scheduler=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        )
        self.rectified_flow_image = self.rectified_flow_video
        self.clean = GenerationDataClean(batch_size=1, is_image_batch=False)
        self.masks = {}
        self.calls = []
        for modality, frames in (("vision", 6), ("lidar", 8)):
            if modality == "lidar" and lidar_mode == "missing":
                continue
            values = [torch.full((1, 1, frames, 1, 1), 2.0) for _ in range(2)]  # list[[1,C,T,H,W]]
            setattr(self.clean, f"x0_tokens_{modality}", values)
            setattr(self.clean, f"num_{modality}_items_per_sample", [2])
            target_mask = (torch.arange(frames) < conditioned_prefix).float().view(frames, 1, 1)  # [T,1,1]
            if modality == "lidar" and lidar_mode == "conditioned":
                target_mask = torch.ones_like(target_mask)  # [T,1,1]
            self.masks[modality] = [torch.ones(frames, 1, 1), target_mask]  # list[[T,1,1]]

    def _get_training_inputs(self, batch: dict, iteration: int) -> tuple:
        return (
            [[1]],
            [SimpleNamespace(text_view_ids=None)],
            self.clean,
            {"skip_text": False, "initial_temporal_offset": 0, "_tf_dcm_unconditional_text_indexes": [[0]]},
            ["480"],
            [],
        )

    def _pack_input_sequence(
        self, plans: list, text: list, clean: GenerationDataClean, timesteps: torch.Tensor, **kwargs: Any
    ) -> Any:  # timesteps: [B,1]
        packed = SimpleNamespace(
            vision=None,
            lidar=None,
            action=None,
            sound=None,
            to_cuda=lambda: None,
            text_indexes=torch.zeros(1),  # [N_text]
            sequence_length=15,
            split_lens=[15],
            attn_modes=[],
            negative=text == [[0]],
            timesteps=timesteps,
        )
        for modality, masks in self.masks.items():
            setattr(
                packed,
                modality,
                SimpleNamespace(
                    condition_mask=masks,
                    sequence_indexes=list(range(7)),
                    tokens=getattr(clean, f"x0_tokens_{modality}"),
                ),
            )
        return packed

    def _validate_teacher_forcing_pack(self, packed: Any) -> None:
        pass

    def _add_noise_to_input(
        self,
        clean: GenerationDataClean,
        packed: Any,
        vision_sigmas: torch.Tensor,
        *,
        sigmas_lidar: torch.Tensor | None,
        iteration: int,
    ) -> GenerationDataNoised:
        state = GenerationDataNoised(batch_size=1, epsilon_vision=[], xt_tokens_vision=[], vt_target_vision=[])
        for modality in self.masks:
            sigmas = vision_sigmas if modality == "vision" else sigmas_lidar  # [B_items,1]
            assert sigmas is not None
            torch.testing.assert_close(sigmas, torch.full_like(sigmas, 0.8))
            values = getattr(clean, f"x0_tokens_{modality}")
            eps = [torch.ones_like(value) for value in values]  # list[[1,C,T,H,W]]
            mask_sigmas = self._vision_sigma_masks(sigmas, self.masks[modality])  # list[[T,1,1]]
            noised = [
                value * (1.0 - sigma) + epsilon * sigma for value, epsilon, sigma in zip(values, eps, mask_sigmas)
            ]  # list[[1,C,T,H,W]]
            setattr(state, f"epsilon_{modality}", eps)
            setattr(state, f"xt_tokens_{modality}", noised)
            setattr(state, f"sigmas_{modality}", mask_sigmas)
        return state

    def _build_clean_tf_cache(self, **kwargs: Any) -> Any:
        return SimpleNamespace(net=kwargs["net"])

    def _replace_clean_with_noised(self, packed: Any, noised: GenerationDataNoised) -> None:
        for modality in self.masks:
            getattr(packed, modality).tokens = getattr(noised, f"xt_tokens_{modality}")

    def _denoise_with_replayed_tf(self, net: Any, packed: Any, clean: GenerationDataClean, memory: Any) -> dict:
        assert memory.net is net
        predictions = {}
        states = {}
        for modality in ("vision", "lidar"):
            if modality not in self.masks:
                predictions[f"preds_{modality}"] = [net[modality].view(1, 1, 1, 1)]  # list[[C,T,H,W]]
                continue
            values = getattr(packed, modality).tokens
            assert values is not None
            states[modality] = [value.detach().clone() for value in values]  # list[[1,C,T,H,W]]
            predictions[f"preds_{modality}"] = [
                torch.ones_like(value) * (net[modality] - (0.1 if packed.negative else 0.0)) for value in values
            ]  # list[[1,C,T,H,W]]
        self.calls.append(
            {
                "teacher": net is self.net_teacher,
                "negative": packed.negative,
                "timestep": packed.timesteps.item(),
                "states": states,
            }
        )
        return predictions


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("lidar_mode", ["active", "conditioned", "missing"])
@pytest.mark.parametrize("guidance", [1.0, 3.0])
@pytest.mark.parametrize("conditioned_prefix", [0, 1])
def test_joint_solver_matches_closed_form_and_sensor_gradients(
    monkeypatch: pytest.MonkeyPatch, lidar_mode: str, guidance: float, conditioned_prefix: int
) -> None:
    model = _AnalyticJointModel(lidar_mode, guidance, conditioned_prefix)
    ladder = torch.tensor([[0.8, 0.6]])  # [B,N+1]
    monkeypatch.setattr(tf_dcm_joint, "sample_teacher_forcing_dcm_ladder", lambda *args: (ladder * 1000, ladder))
    output, loss = model._training_step_teacher_forcing_dcm({}, 0)  # dict, []
    loss.backward()
    difference = 0.2 * (0.3 + (guidance - 1) * 0.1 - 0.2)
    expected = 0.0
    for modality, frames, scale in (("vision", 6, 100), ("lidar", 8, 200)):
        active = modality == "vision" or lidar_mode == "active"
        expected_loss = difference**2 * (frames - conditioned_prefix) if active else 0.0
        assert output[f"tf_dcm_loss_{modality}"].item() == pytest.approx(expected_loss, abs=1e-6)
        expected += expected_loss * scale
        gradient = -2 * difference * 0.8 * (frames - conditioned_prefix) * scale if active else 0.0
        assert model.net[modality].grad is not None
        assert model.net[modality].grad.item() == pytest.approx(gradient, abs=3e-4)
        assert model.net_teacher[modality].grad is None
    assert loss.item() == pytest.approx(expected, abs=1e-4)
    assert len(model.calls) == (4 if guidance > 1 else 3)
    assert model.calls[-1]["timestep"] == pytest.approx(600)
    assert model.calls[-2]["timestep"] == pytest.approx(800)
    for call in model.calls:
        for values in call["states"].values():
            torch.testing.assert_close(values[0], torch.full_like(values[0], 2.0))
    for modality, values in model.calls[-1]["states"].items():
        teacher_velocity = (0.3 if modality == "vision" else 0.5) + (guidance - 1) * 0.1
        expected_value = 1.2 - 0.2 * teacher_velocity
        if modality == "vision" or lidar_mode == "active":
            torch.testing.assert_close(
                values[1][:, :, conditioned_prefix:],
                torch.full_like(values[1][:, :, conditioned_prefix:], expected_value),
            )


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("lidar_mode", ["active", "conditioned", "missing"])
def test_joint_endpoint_mean_retains_stage1_conditioned_item_denominator(
    monkeypatch: pytest.MonkeyPatch, lidar_mode: str
) -> None:
    ladder = torch.tensor([[0.8, 0.6]])  # [B,N+1]
    monkeypatch.setattr(tf_dcm_joint, "sample_teacher_forcing_dcm_ladder", lambda *args: (ladder * 1000, ladder))
    generated_only = _AnalyticJointModel(lidar_mode, guidance=3.0, conditioned_prefix=1)
    all_items = _AnalyticJointModel(lidar_mode, guidance=3.0, conditioned_prefix=1)
    all_items.config.rectified_flow_training_config.exclude_fully_conditioned_items = False
    _, expected_loss = generated_only._training_step_teacher_forcing_dcm({}, 0)  # dict, []
    output, actual_loss = all_items._training_step_teacher_forcing_dcm({}, 0)  # dict, []
    expected_loss.backward()
    actual_loss.backward()
    # Each present sensor has one fully conditioned control and one target.
    # Including the zero-loss control halves both the loss and target gradients.
    torch.testing.assert_close(actual_loss, expected_loss * 0.5)
    for modality in ("vision", "lidar"):
        actual_gradient = all_items.net[modality].grad  # [] or None
        expected_gradient = generated_only.net[modality].grad  # [] or None
        assert actual_gradient is not None and expected_gradient is not None
        torch.testing.assert_close(actual_gradient, expected_gradient * 0.5)
        assert all_items.net_teacher[modality].grad is None
    assert output["tf_dcm_loss_vision_per_instance"][0].item() == 0.0


@pytest.mark.level(0)
@pytest.mark.gpus(0)
def test_joint_state_preserves_conditioning_metadata_and_original_storage() -> None:
    model = _AnalyticJointModel("active", 3.0, 1)
    action_mask = [torch.ones(2)]  # list[[T]]
    original = GenerationDataNoised(1, [], [], [], action_valid_mask=action_mask)
    states = {name: getattr(model.clean, f"x0_tokens_{name}") for name in model.masks}
    sigmas = model._joint_sigma_masks(torch.tensor([[0.7]]), model.clean, model.masks)  # modality -> list[[T,1,1]]
    result = model._joint_noised_state(original, states, sigmas)
    assert result is not original
    assert result.action_valid_mask is action_mask
    assert original.xt_tokens_lidar is None
    assert result.xt_tokens_lidar is not None
    for values in sigmas.values():
        assert torch.count_nonzero(values[0]) == 0
        assert values[1][0].item() == 0


@pytest.mark.level(0)
@pytest.mark.gpus(0)
@pytest.mark.parametrize("has_lidar", [False, True])
@pytest.mark.parametrize("already_saved", [False, True])
def test_joint_noisy_replay_preserves_sensor_roles_per_sample(has_lidar: bool, already_saved: bool) -> None:
    """The native mask consumer must receive both sensors in sample order on every endpoint."""
    vision_masks = [
        torch.ones(6),  # [V*T]
        torch.tensor([1, 0, 0, 1, 0, 0]),  # [V*T]
        torch.ones(6),  # [V*T]
        torch.tensor([0, 1, 0, 0, 1, 0]),  # [V*T]
    ]
    lidar_masks = [
        torch.ones(4),  # [T_lidar]
        torch.tensor([1, 0, 0, 0]),  # [T_lidar]
        torch.ones(4),  # [T_lidar]
        torch.tensor([0, 1, 0, 0]),  # [T_lidar]
    ]
    packed = PackedSequence(
        sample_lens=[20, 20] if has_lidar else [12, 12],
        vision=ModalityData(
            tokens=[torch.zeros(1) for _ in range(4)],  # list[[1]]
            token_shapes=[(6, 1, 1)] * 4,
            condition_mask=vision_masks,
            seconds_per_frame=[4 / 30] * 4,
        ),
        num_vision_items_per_sample=[2, 2],
        num_views_per_vision_item=[2] * 4,
    )
    if has_lidar:
        packed.lidar = ModalityData(
            tokens=[torch.zeros(1) for _ in range(4)],  # list[[1]]
            token_shapes=[(4, 1, 1)] * 4,
            condition_mask=lidar_masks,
            seconds_per_frame=[0.1] * 4,
        )
        packed.num_lidar_items_per_sample = [2, 2]
    expected = vision_masks[:2] + (lidar_masks[:2] if has_lidar else [])
    expected += vision_masks[2:] + (lidar_masks[2:] if has_lidar else [])
    saved = [mask.clone() for mask in expected]  # list[[T_sensor]]
    if already_saved:
        packed.teacher_forcing_original_condition_masks_sensors = saved
        # Clean replay rewrites the live masks; the original roles must survive.
        for modality in (packed.vision, packed.lidar):
            if modality is not None:
                modality.condition_mask = [torch.ones_like(mask) for mask in modality.condition_mask]
    model = object.__new__(TeacherForcingJointDCMModel)
    torch.nn.Module.__init__(model)
    memory = object.__new__(TeacherForcingMemoryState)
    memory.selected_clean_gen_token_indexes = torch.arange(2)  # [S_clean_real]
    memory.selected_clean_gen_padded_capacity = 2
    net = torch.nn.Identity()
    with patch.object(model, "denoise", return_value={}) as denoise:
        model._denoise_with_replayed_tf(net, packed, GenerationDataClean(2, False), memory)
    denoise.assert_called_once_with(net=net, data_batch_packed=packed, memory=memory)
    restored = packed.teacher_forcing_original_condition_masks_sensors
    items = build_interactive_multiview_mask_items(packed, condition_masks=restored)
    assert [len(sample) for sample in items] == ([4, 4] if has_lidar else [2, 2])
    for actual, original in zip([item for sample in items for item in sample], expected, strict=True):
        torch.testing.assert_close(actual.condition_mask, original)
        assert actual.condition_mask.data_ptr() != original.data_ptr()
    if already_saved:
        assert restored is saved
    assert memory.pass_number == 2
    assert packed.teacher_forcing_pass == "noisy"
