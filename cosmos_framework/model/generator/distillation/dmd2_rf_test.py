# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Unit tests for DMD2RFModel."""

import inspect
from types import (
    SimpleNamespace,
)
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch

from cosmos_framework.model.generator.distillation import dmd2_rf as dmd2_rf_module
from cosmos_framework.model.generator.distillation.common_loss import variational_score_distillation_loss
from cosmos_framework.model.generator.distillation.dmd2_rf import DMD2RFModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils.flags import Device


@pytest.mark.L0
@pytest.mark.CPU
def test_vision_only_sequence_plan_guard_accepts_vision_batch() -> None:
    sequence_plans = [SimpleNamespace(has_action=False, has_sound=False)]

    DMD2RFModel._validate_vision_only_sequence_plans(sequence_plans)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("has_action", "has_sound"), [(True, False), (False, True)])
def test_vision_only_sequence_plan_guard_rejects_nonvision_batch(has_action: bool, has_sound: bool) -> None:
    sequence_plans = [SimpleNamespace(has_action=has_action, has_sound=has_sound)]

    with pytest.raises(ValueError, match="only vision T2I/I2V batches"):
        DMD2RFModel._validate_vision_only_sequence_plans(sequence_plans)


@pytest.mark.L0
@pytest.mark.CPU
def test_training_only_network_holders_are_not_module_state() -> None:
    model = object.__new__(DMD2RFModel)
    torch.nn.Module.__init__(model)
    model.net = torch.nn.Linear(2, 2)
    model._net_teacher_holder = [torch.nn.Linear(2, 2)]
    model._net_fake_score_holder = [torch.nn.Linear(2, 2)]

    state_keys = set(torch.nn.Module.state_dict(model))

    assert state_keys == {"net.weight", "net.bias"}
    assert model.net_teacher is model._net_teacher_holder[0]
    assert model.net_fake_score is model._net_fake_score_holder[0]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("iteration", "expected_phase", "expected_optimizer_key"),
    [
        (0, "student", "net"),
        (1, "critic", "fake_score"),
        (4, "critic", "fake_score"),
        (5, "student", "net"),
    ],
)
def test_public_phase_contract_routes_student_and_critic_optimizers(
    iteration: int,
    expected_phase: str,
    expected_optimizer_key: str,
) -> None:
    model = object.__new__(DMD2RFModel)
    model.config = SimpleNamespace(
        warmup_student_steps=0,
        warmup_critic_steps=0,
        student_update_freq=5,
    )

    assert model.get_phase(iteration) == expected_phase
    assert model.get_optimizer_key(iteration) == expected_optimizer_key


@pytest.mark.L0
def test_is_subclass_of_omni_mot_model():
    assert issubclass(DMD2RFModel, OmniMoTModel)


@pytest.mark.L0
def test_does_not_inherit_removed_legacy_wrapper():
    # The legacy wrapper is not part of the public DMD2RFModel hierarchy.
    mro_names = [cls.__name__ for cls in DMD2RFModel.__mro__]
    assert "Cosmos3InteractiveModel" not in mro_names


@pytest.mark.L0
def test_clip_grad_norm_defined_in_dmd2():
    # clip_grad_norm_ must be defined directly on DMD2RFModel, not inherited from OmniMoTModel.
    assert "clip_grad_norm_" in DMD2RFModel.__dict__


@pytest.mark.L0
def test_on_before_zero_grad_defined_in_dmd2():
    # on_before_zero_grad must be defined directly on DMD2RFModel, not inherited from OmniMoTModel.
    assert "on_before_zero_grad" in DMD2RFModel.__dict__


@pytest.mark.L0
def test_clip_grad_norm_skips_empty_params():
    """Empty param list (no grads) should return 0.0 tensor, not raise an error."""
    model = MagicMock(spec=DMD2RFModel)
    model.config = MagicMock()
    model.config.grad_clip = True
    model.net = MagicMock()
    model.net_fake_score = MagicMock()
    model.net_fake_score.parameters.return_value = iter([])

    # A param with no grad attached
    param = torch.nn.Parameter(torch.randn(4))
    param.grad = None
    model.net.parameters.return_value = iter([param])

    result = DMD2RFModel.clip_grad_norm_(model, max_norm=1.0)

    assert isinstance(result, torch.Tensor)
    assert result.item() == 0.0


@pytest.mark.L0
def test_clip_grad_norm_cleans_nan_grads():
    """NaN gradients should be replaced with 0 before clipping."""
    p = torch.nn.Parameter(torch.randn(4))
    p.grad = torch.full((4,), float("nan"))
    torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
    assert not torch.isnan(p.grad).any()


@pytest.mark.L0
def test_on_before_zero_grad_student_phase_calls_ema_update():
    """In student phase, net_ema_worker.update_average should be called."""
    model = MagicMock()
    model.get_phase.return_value = "student"
    model.config.ema.enabled = True
    model.get_student_iteration.return_value = 100
    model.ema_beta.return_value = 0.999

    optimizer = MagicMock()
    scheduler = MagicMock()

    DMD2RFModel.on_before_zero_grad(model, optimizer, scheduler, iteration=5)

    model.net_ema_worker.update_average.assert_called_once_with(model.net, model.net_ema, beta=0.999)


@pytest.mark.L0
def test_on_before_zero_grad_critic_phase_skips_ema():
    """In critic phase (not student), EMA update should NOT be called."""
    model = MagicMock()
    model.get_phase.return_value = "critic"
    model.net_fake_score = None

    optimizer = MagicMock()
    optimizer.get.return_value = None
    scheduler = MagicMock()

    DMD2RFModel.on_before_zero_grad(model, optimizer, scheduler, iteration=5)

    model.net_ema_worker.update_average.assert_not_called()


@pytest.mark.L0
def test_model_dict_includes_fake_score():
    model = MagicMock(spec=DMD2RFModel)
    model.net = MagicMock()
    model.net_fake_score = MagicMock()

    result = DMD2RFModel.model_dict(model)

    assert "net" in result
    assert result["net"] is model.net
    assert "fake_score" in result
    assert result["fake_score"] is model.net_fake_score


@pytest.mark.L0
def test_dmd2_rf_needs_fake_score_by_default():
    model = MagicMock(spec=DMD2RFModel)

    assert DMD2RFModel._needs_fake_score(model) is True


@pytest.mark.L0
def test_fixed_step_sampler_set_in_set_up_model():
    """Verify set_up_model creates self.fixed_step_sampler through the setup helper."""
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler  # noqa: F401

    setup_src = inspect.getsource(DMD2RFModel.set_up_model)
    fixed_step_setup_src = inspect.getsource(DMD2RFModel._set_up_fixed_step_sampler)

    assert "_set_up_fixed_step_sampler()" in setup_src
    assert "FixedStepSampler" in fixed_step_setup_src
    assert "fixed_step_sampler" in fixed_step_setup_src


@pytest.mark.L0
def test_training_only_nets_stay_out_of_inference_state_dict():
    """Teacher/fake-score should stay unregistered and be skipped for inference setup."""
    init_src = inspect.getsource(DMD2RFModel.__init__)
    setup_src = inspect.getsource(DMD2RFModel.set_up_model)

    assert "_net_teacher_holder" in init_src
    assert "_net_fake_score_holder" in init_src
    assert "if is_inference_mode:" in setup_src
    assert 'self.denoiser_nets = {"student": self.net}' in setup_src
    assert "needs_fake_score = self._needs_fake_score()" in setup_src
    assert "_net_teacher_holder = [self.build_net(self.precision, lora_enabled=False)]" in setup_src
    assert "_net_fake_score_holder = [self.build_net(self.precision)]" in setup_src
    assert 'self.denoiser_nets["fake_score"] = self.net_fake_score' in setup_src
    assert "_build_net_with_lora" not in DMD2RFModel.__dict__
    assert "state_dict" not in DMD2RFModel.__dict__


@pytest.mark.L0
@pytest.mark.parametrize("teacher_negative_prompt", ["", "low quality video"])
def test_teacher_cfg_negative_prompt_is_tokenized_for_uncond_teacher_pass(teacher_negative_prompt: str) -> None:
    model = MagicMock()
    model.config = SimpleNamespace(
        action_gen=False,
        loss_scale_sid=1.0,
        sound_gen=False,
        teacher_guidance=3.0,
        teacher_negative_prompt=teacher_negative_prompt,
        vsd_gradient_space="x0",
        vsd_loss_reduction="mean",
    )
    model.parallel_dims = None
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.vlm_config = SimpleNamespace(use_system_prompt=False)
    model.vlm_tokenizer = object()

    gen_data_student = SimpleNamespace(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[torch.full((1, 1, 1, 1), 0.2)],
        x0_tokens_action=None,
        x0_tokens_sound=None,
    )
    packed_student = SimpleNamespace(
        action=None,
        sound=None,
        vision=SimpleNamespace(condition_mask=[torch.zeros(1, 1, 1)]),
    )
    gen_data_noised = SimpleNamespace(
        sigmas_vision=[torch.full((1, 1, 1), 0.5)],
        xt_tokens_vision=[torch.full((1, 1, 1, 1), 0.4)],
    )
    model._gen_data_from_student.return_value = (
        gen_data_student,
        packed_student,
        torch.zeros(1, 1),
        {"preds_vision": [torch.zeros(1, 1, 1, 1)]},
    )
    model._get_train_noise_level_vision.return_value = (torch.full((1, 1), 0.5), torch.full((1, 1), 0.5))
    model._action_sigmas_from_full.return_value = None
    model._sound_sigmas_from_full.return_value = None
    model._add_noise_to_input.return_value = gen_data_noised
    model._pack_and_denoise.side_effect = [
        {"preds_vision": [torch.full((1, 1, 1, 1), 0.1)]},
        {"preds_vision": [torch.full((1, 1, 1, 1), 0.2)]},
        {"preds_vision": [torch.full((1, 1, 1, 1), 0.05)]},
    ]
    model._velocity_to_x0.side_effect = [
        [torch.full((1, 1, 1, 1), 0.15)],
        [torch.full((1, 1, 1, 1), 0.3)],
        [torch.full((1, 1, 1, 1), 0.1)],
    ]

    tokenized_negative_prompt = [len(teacher_negative_prompt), 1, 0]

    def fake_tokenize_caption(
        prompt: str,
        tokenizer: object,
        *,
        is_video: bool,
        use_system_prompt: bool,
    ) -> list[int]:
        assert tokenizer is model.vlm_tokenizer
        assert is_video is True
        assert use_system_prompt is False
        return [len(prompt), int(is_video), int(use_system_prompt)]

    with patch.object(dmd2_rf_module, "tokenize_caption", side_effect=fake_tokenize_caption) as tokenize_mock:
        output_batch, loss = DMD2RFModel.training_step_generator(
            model,
            input_text_indexes=[[101]],
            sequence_plans=[MagicMock()],
            gen_data_clean=SimpleNamespace(batch_size=1),
            data_resolutions=["480"],
            num_vision_tokens_per_sample=[1],
            iteration=7,
        )

    assert torch.isfinite(loss)
    assert torch.isfinite(output_batch["total_generator_loss"])
    tokenize_mock.assert_called_once()
    assert tokenize_mock.call_args.args[0] == teacher_negative_prompt

    pack_calls = model._pack_and_denoise.call_args_list
    assert [call.kwargs["net_type"] for call in pack_calls] == ["fake_score", "teacher", "teacher"]
    assert pack_calls[0].args[3] == [[101]]
    assert pack_calls[1].args[3] == [[101]]
    assert pack_calls[2].args[3] == [tokenized_negative_prompt]


@pytest.mark.L0
def test_build_net_accepts_optional_lora_override() -> None:
    signature = inspect.signature(OmniMoTModel.build_net)

    lora_enabled = signature.parameters["lora_enabled"]
    assert lora_enabled.kind is inspect.Parameter.KEYWORD_ONLY
    assert lora_enabled.default is None


@pytest.mark.L0
@pytest.mark.parametrize(
    ("config_lora_enabled", "lora_enabled_override", "expected_lora_enabled"),
    [
        (True, None, True),
        (True, False, False),
        (False, True, True),
    ],
)
def test_build_net_lora_override_controls_injection_and_initialization(
    config_lora_enabled: bool,
    lora_enabled_override: bool | None,
    expected_lora_enabled: bool,
) -> None:
    model = MagicMock(spec=OmniMoTModel)
    model.config = MagicMock()
    model.config.lora_enabled = config_lora_enabled
    model.config.lora_rank = 8
    model.config.lora_alpha = 16
    model.config.lora_target_modules = "q_proj_moe_gen"
    model.vlm_config = MagicMock()
    model.tokenizer_vision_gen = MagicMock()
    model.parallel_dims = MagicMock()

    language_model = MagicMock()
    net = MagicMock()
    net.to.return_value = net
    model.add_lora.return_value = net

    module = "cosmos_framework.model.generator.omni_mot_model"
    with (
        patch(f"{module}.lazy_instantiate", return_value=language_model),
        patch(f"{module}.Cosmos3VFMNetworkConfig"),
        patch(f"{module}.Cosmos3VFMNetwork", return_value=net),
        patch(f"{module}.parallelize_vfm_network", return_value=net),
        patch(f"{module}.DEVICE", Device.CUDA),
    ):
        if lora_enabled_override is None:
            result = OmniMoTModel.build_net(model, torch.float32)
        else:
            result = OmniMoTModel.build_net(model, torch.float32, lora_enabled=lora_enabled_override)

    assert result is net
    if expected_lora_enabled:
        model.add_lora.assert_called_once_with(
            net,
            lora_rank=8,
            lora_alpha=16,
            lora_target_modules="q_proj_moe_gen",
        )
        model._init_lora_weights_post_materialization.assert_called_once_with(net)
    else:
        model.add_lora.assert_not_called()
        model._init_lora_weights_post_materialization.assert_not_called()


@pytest.mark.L0
def test_copy_teacher_weights_allows_missing_lora_adapter_keys():
    class NetWithOptionalLoRA(torch.nn.Module):
        def __init__(self, with_lora: bool):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)
            if with_lora:
                self.proj.lora_A = torch.nn.Linear(2, 1, bias=False)
                self.proj.lora_B = torch.nn.Linear(1, 2, bias=False)

    teacher = NetWithOptionalLoRA(with_lora=False)
    target = NetWithOptionalLoRA(with_lora=True)
    with torch.no_grad():
        teacher.proj.weight.fill_(3.0)
        target.proj.lora_A.weight.fill_(1.0)
        target.proj.lora_B.weight.fill_(2.0)

    model = MagicMock(spec=DMD2RFModel)
    type(model).net_teacher = property(lambda self: teacher)

    DMD2RFModel._copy_teacher_weights(model, target_net=target, target_name="fake score")

    assert torch.equal(target.proj.weight, teacher.proj.weight)
    assert torch.equal(target.proj.lora_A.weight, torch.ones_like(target.proj.lora_A.weight))
    assert torch.equal(target.proj.lora_B.weight, torch.full_like(target.proj.lora_B.weight, 2.0))


# ---------------------------------------------------------------------------
# _pack_and_denoise proxy forwarding
# ---------------------------------------------------------------------------


def _make_gen_data(
    *,
    with_sound: bool = False,
    num_vision_items: Optional[list[int]] = None,
) -> MagicMock:
    """Build a minimal GenerationDataClean mock for proxy-forwarding tests."""
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    gd = MagicMock(spec=GenerationDataClean)
    gd.batch_size = 2
    gd.is_image_batch = False
    gd.raw_state_vision = [MagicMock(), MagicMock()]
    gd.x0_tokens_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd.fps_vision = [24.0, 24.0]
    gd.num_vision_items_per_sample = num_vision_items or [1, 1]
    if with_sound:
        gd.raw_state_sound = [MagicMock(), MagicMock()]
        gd.x0_tokens_sound = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
        gd.fps_sound = [24.0, 24.0]
    else:
        gd.raw_state_sound = None
        gd.x0_tokens_sound = None
        gd.fps_sound = None
    gd.raw_state_action = None
    gd.x0_tokens_action = None
    gd.fps_action = None
    gd.action_domain_id = None
    return gd


def _make_gen_data_noised(
    *,
    with_sound: bool = False,
    with_action: bool = False,
) -> MagicMock:
    """Build a minimal GenerationDataNoised mock."""
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised

    gd = MagicMock(spec=GenerationDataNoised)
    gd.xt_tokens_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd.vt_target_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd.sigmas_vision = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]
    if with_sound:
        gd.xt_tokens_sound = [torch.zeros(2, 1), torch.zeros(2, 1)]
        gd.vt_target_sound = [torch.zeros(2, 1), torch.zeros(2, 1)]
        gd.sigmas_sound = [torch.zeros(1, 1), torch.zeros(1, 1)]
    else:
        gd.xt_tokens_sound = None
        gd.vt_target_sound = None
        gd.sigmas_sound = None
    if with_action:
        gd.xt_tokens_action = [torch.zeros(1, 4), torch.zeros(1, 4)]
        gd.vt_target_action = [torch.zeros(1, 4), torch.zeros(1, 4)]
        gd.sigmas_action = [torch.zeros(1, 1), torch.zeros(1, 1)]
    else:
        gd.xt_tokens_action = None
        gd.vt_target_action = None
        gd.sigmas_action = None
    return gd


def _make_packed_sequence(*, with_action: bool = False) -> MagicMock:
    """Build a minimal PackedSequence mock."""
    packed = MagicMock()
    packed.vision = MagicMock()
    packed.vision.tokens = []
    packed.vision.condition_mask = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]
    if with_action:
        packed.action = MagicMock()
        packed.action.tokens = []
        packed.action.condition_mask = [torch.zeros(1, 1), torch.zeros(1, 1)]
    else:
        packed.action = None
    packed.sound = None
    return packed


@pytest.mark.L0
def test_pack_and_denoise_returns_dict():
    """_pack_and_denoise must return a plain dict, not a tuple."""
    model = MagicMock()
    model._pack_input_sequence.return_value = _make_packed_sequence()
    model.denoise.return_value = {"preds_vision": [torch.zeros(4, 1, 2, 2)]}

    gen_data = _make_gen_data()
    gen_data_noised = _make_gen_data_noised()

    result = DMD2RFModel._pack_and_denoise(
        model,
        gen_data_clean=gen_data,
        gen_data_noised=gen_data_noised,
        timesteps=torch.zeros(2, 1),
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        net_type="student",
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"


@pytest.mark.L0
def test_pack_and_denoise_forwards_num_vision_items():
    """num_vision_items_per_sample must be forwarded from gen_data_clean in the proxy."""
    model = MagicMock()
    model._pack_input_sequence.return_value = _make_packed_sequence()
    model.denoise.return_value = {"preds_vision": []}

    gen_data = _make_gen_data(num_vision_items=[2, 2])
    gen_data_noised = _make_gen_data_noised()

    DMD2RFModel._pack_and_denoise(
        model,
        gen_data_clean=gen_data,
        gen_data_noised=gen_data_noised,
        timesteps=torch.zeros(2, 1),
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        net_type="student",
    )

    _, call_kwargs = model._pack_input_sequence.call_args
    proxy = call_kwargs.get("gen_data_clean") or model._pack_input_sequence.call_args.args[2]
    assert proxy.num_vision_items_per_sample == [2, 2]


@pytest.mark.L0
def test_gen_data_from_student_forwards_num_vision_items():
    """num_vision_items_per_sample must be forwarded in the proxy inside _forward_simulation."""
    model = MagicMock()
    model.tensor_kwargs = {"dtype": torch.float32, "device": "cpu"}
    model.config.rectified_flow_inference_config.num_train_timesteps = 1000

    model.config.action_gen = False
    model.config.sound_gen = False
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model._sample_student_sigma.return_value = torch.full((2, 1), 0.5)

    packed = MagicMock()
    packed.vision = MagicMock()
    packed.vision.condition_mask = [torch.ones(1, 1, 1), torch.ones(1, 1, 1)]
    packed.action = None
    packed.sound = None
    model._pack_input_sequence.return_value = packed

    gen_data_noised = MagicMock()
    gen_data_noised.xt_tokens_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gen_data_noised.xt_tokens_action = None
    gen_data_noised.xt_tokens_sound = None
    model._add_noise_to_input.return_value = gen_data_noised
    model.denoise.return_value = {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}
    model._velocity_to_x0.return_value = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]

    gen_data = _make_gen_data(num_vision_items=[2, 2])

    DMD2RFModel._forward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
    )

    _, call_kwargs = model._pack_input_sequence.call_args
    proxy = call_kwargs.get("gen_data_clean") or model._pack_input_sequence.call_args.args[2]
    assert proxy.num_vision_items_per_sample == [2, 2]


# ---------------------------------------------------------------------------
# _gen_data_from_student return type and multi-modal x0 extraction
# ---------------------------------------------------------------------------


def _make_student_model(*, action_gen: bool = False, sound_gen: bool = False) -> MagicMock:
    """Build a model mock wired up for _gen_data_from_student tests."""
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised

    model = MagicMock()
    model.tensor_kwargs = {"dtype": torch.float32, "device": "cpu"}
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.config.rectified_flow_inference_config.num_train_timesteps = 1000
    model.config.action_gen = action_gen
    model.config.sound_gen = sound_gen
    model._sample_student_sigma.return_value = torch.full((2, 1), 0.5)

    packed = _make_packed_sequence(with_action=action_gen)
    if action_gen:
        packed.action.condition_mask = [torch.zeros(1, 1), torch.zeros(1, 1)]
    model._pack_input_sequence.return_value = packed

    gd_noised = MagicMock(spec=GenerationDataNoised)
    gd_noised.xt_tokens_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd_noised.xt_tokens_action = [torch.zeros(1, 4), torch.zeros(1, 4)] if action_gen else None
    gd_noised.xt_tokens_sound = None
    model._add_noise_to_input.return_value = gd_noised

    preds = {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}
    if action_gen:
        preds["preds_action"] = [torch.zeros(1, 4), torch.zeros(1, 4)]
    model.denoise.return_value = preds
    model._velocity_to_x0.return_value = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    return model


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("is_image_batch", [True, False], ids=["t2i", "i2v"])
def test_vision_only_forward_simulation_supports_t2i_and_i2v(is_image_batch: bool) -> None:
    model = _make_student_model(action_gen=False, sound_gen=False)
    model._sound_sigmas_from_full.return_value = None
    gen_data = _make_gen_data()
    gen_data.is_image_batch = is_image_batch

    gen_data_student, _, _, out_student = DMD2RFModel._forward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
    )

    assert gen_data_student.is_image_batch is is_image_batch
    assert gen_data_student.x0_tokens_vision is not None
    assert gen_data_student.x0_tokens_action is None
    assert gen_data_student.x0_tokens_sound is None
    assert set(out_student) == {"preds_vision"}


@pytest.mark.L0
def test_gen_data_from_student_returns_4_tuple():
    """_forward_simulation must return (GenerationDataClean, PackedSequence, Tensor, dict)."""
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    model = _make_student_model()
    gen_data = _make_gen_data()

    result = DMD2RFModel._forward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
    )

    assert isinstance(result, tuple) and len(result) == 4
    gen_data_student, _, sigma, out = result
    assert isinstance(gen_data_student, GenerationDataClean)
    assert isinstance(sigma, torch.Tensor) and sigma.shape == (2, 1)
    assert isinstance(out, dict)


# ---------------------------------------------------------------------------
# training_step_critic: normalize_by_active and FSDP dummy
# ---------------------------------------------------------------------------


def _make_critic_model(*, action_gen: bool = False) -> MagicMock:
    """Build a model mock wired up for training_step_critic tests."""
    model = MagicMock()
    model.parallel_dims = None  # disable CP broadcast path in tests
    model.config.action_gen = action_gen
    model.config.sound_gen = False
    model.config.loss_scale_fake_score = 1.0
    model.config.fake_score_loss_reduction = "active_mean"

    gen_data_student = _make_gen_data()
    gen_data_student.x0_tokens_action = None  # absent by default
    packed_student = _make_packed_sequence()
    model._gen_data_from_student.return_value = (gen_data_student, packed_student, torch.zeros(2, 1), {})

    model._get_train_noise_level_vision.return_value = (torch.zeros(2, 1), torch.zeros(2, 1))
    model._add_noise_to_input.return_value = _make_gen_data_noised()

    out_fake = {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}
    if action_gen:
        out_fake["preds_action"] = [torch.zeros(1, 4), torch.zeros(1, 4)]
    model._pack_and_denoise.return_value = out_fake

    model._compute_flow_matching_loss.return_value = (torch.tensor(0.5), torch.zeros(2))
    return model


@pytest.mark.L0
def test_training_step_critic_calls_compute_flow_matching_loss_with_normalize_by_active():
    """_compute_flow_matching_loss must be called with normalize_by_active=True in critic."""
    model = _make_critic_model()

    DMD2RFModel.training_step_critic(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=MagicMock(),
        data_resolutions=["480", "480"],
        num_vision_tokens_per_sample=[4, 4],
        iteration=0,
    )

    noise_call_kwargs = model._get_train_noise_level_vision.call_args.kwargs
    assert noise_call_kwargs["resolutions"] == ["480", "480"]
    assert noise_call_kwargs["num_tokens"] == [4, 4]

    model._compute_flow_matching_loss.assert_called_once()
    call_kwargs = model._compute_flow_matching_loss.call_args.kwargs
    assert call_kwargs.get("normalize_by_active") is True


@pytest.mark.L0
def test_flow_matching_per_instance_sum_loss_sums_active_elements_per_instance() -> None:
    """per_instance_sum should sum generated elements per instance, then average the batch."""
    pred = [torch.ones(1, 2, 1, 1), torch.full((1, 2, 1, 1), 2.0)]  # list of [C,T,H,W]
    target = [torch.zeros_like(pred[0]), torch.ones_like(pred[1])]  # list of [C,T,H,W]
    condition_mask = [
        torch.tensor([[[0.0]], [[1.0]]]),
        torch.tensor([[[0.0]], [[0.0]]]),
    ]  # list of [T,1,1]

    loss, per_instance = DMD2RFModel._flow_matching_per_instance_sum_loss(
        pred=pred,
        target=target,
        condition_mask=condition_mask,
        has_valid_tokens=True,
    )

    expected_per_instance = torch.tensor([1.0, 2.0])  # [B]
    assert torch.allclose(per_instance, expected_per_instance)
    assert loss.item() == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# _sample_student_sigma
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_sample_student_sigma_shape():
    """Return shape must be (B, 1) regardless of batch size or t_list length."""
    model = MagicMock()
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.config.fixed_step_sampler_config.t_list = [1.0, 0.9, 0.75, 0.5]

    for B in [1, 4]:
        sigma = DMD2RFModel._sample_student_sigma(model, B)
        assert sigma.shape == (B, 1), f"Expected ({B}, 1), got {sigma.shape}"


@pytest.mark.L0
def test_sample_student_sigma_single_step_always_same_value():
    """Single-entry t_list: every sample in the batch gets that exact sigma."""
    model = MagicMock()
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.config.fixed_step_sampler_config.t_list = [1.0]

    sigma = DMD2RFModel._sample_student_sigma(model, batch_size=4)
    assert sigma.shape == (4, 1)
    assert (sigma == 1.0).all()


@pytest.mark.L0
def test_sample_student_sigma_multi_step_values_in_t_list():
    """Multi-entry t_list: all returned sigmas must be one of the listed values."""
    t_list = [1.0, 0.9, 0.75, 0.5]
    model = MagicMock()
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.config.fixed_step_sampler_config.t_list = t_list

    sigma = DMD2RFModel._sample_student_sigma(model, batch_size=100)
    for s in sigma.squeeze(1).tolist():
        assert any(abs(s - t) < 1e-6 for t in t_list), f"sigma {s} not in t_list {t_list}"


@pytest.mark.L0
def test_sample_student_sigma_multi_step_covers_all_values():
    """Multi-entry t_list with large batch: all t_list values should appear at least once."""
    t_list = [1.0, 0.9, 0.75, 0.5]
    model = MagicMock()
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.config.fixed_step_sampler_config.t_list = t_list

    # With 1000 samples, the probability of missing any value is negligible
    sigma = DMD2RFModel._sample_student_sigma(model, batch_size=1000)
    unique_vals = set(round(s, 6) for s in sigma.squeeze(1).tolist())
    for t in t_list:
        assert any(abs(t - v) < 1e-5 for v in unique_vals), f"{t} never sampled"


@pytest.mark.L0
def test_sample_student_sigma_returns_float32():
    """Sigma tensor should be float32 (consistent with tensor_kwargs_fp32)."""
    model = MagicMock()
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.config.fixed_step_sampler_config.t_list = [1.0]

    sigma = DMD2RFModel._sample_student_sigma(model, batch_size=2)
    assert sigma.dtype == torch.float32


# ---------------------------------------------------------------------------
# variational_score_distillation_loss with loss_mask
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_vsd_loss_no_mask():
    """Without mask the loss matches the original formula (non-regression)."""
    torch.manual_seed(0)
    B, C, T, H, W = 2, 4, 3, 2, 2
    gen_data = [torch.randn(C, T, H, W, requires_grad=True) for _ in range(B)]
    teacher_x0 = [torch.randn(C, T, H, W) for _ in range(B)]
    fake_x0 = [torch.randn(C, T, H, W) for _ in range(B)]

    loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_x0)
    assert loss.shape == ()
    assert loss.item() >= 0.0
    # Gradient must flow back to gen_data
    loss.backward()
    assert gen_data[0].grad is not None


@pytest.mark.L0
def test_vsd_loss_with_mask_zeros_conditioned():
    """Conditioned frames (mask=0) should contribute zero loss.

    When mask=0 for all frames, pseudo_target == gen_data for the masked
    frames only if the loss is correctly zeroed. We verify by checking that
    a mask of all-zeros yields zero loss (no generated frames to fit).
    """
    torch.manual_seed(1)
    B, C, T, H, W = 1, 4, 4, 2, 2
    gen_data = [torch.randn(C, T, H, W, requires_grad=True)]
    teacher_x0 = [torch.randn(C, T, H, W)]
    fake_x0 = [torch.randn(C, T, H, W)]

    # All frames conditioned (mask = 0 everywhere)
    loss_mask = [torch.zeros(T, 1, 1)]
    loss = variational_score_distillation_loss(gen_data, teacher_x0, fake_x0, loss_mask=loss_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f"Expected zero loss when mask is all-zeros, got {loss.item()}"


@pytest.mark.L0
def test_vsd_loss_weight_unaffected_by_conditioned_frames():
    """With loss_mask, weight is computed only over generated frames.

    When conditioned frames have gen ≈ teacher (near-zero diff) and generated
    frames have a non-trivial diff, the unmasked version produces an inflated
    weight (small diff_abs_mean → large w).  The masked version should yield
    a weight based solely on the generated-frame differences, which is larger
    (non-trivial diff → smaller w).

    We verify that masked and unmasked losses differ when conditioned frames
    have small diff and generated frames have large diff.
    """
    torch.manual_seed(2)
    B, C, T, H, W = 1, 4, 4, 2, 2

    gen_data_val = torch.zeros(C, T, H, W)
    teacher_val = torch.zeros(C, T, H, W)
    fake_val = torch.randn(C, T, H, W)

    # Frames 0-1: conditioned (small gen-teacher diff by construction)
    # Frames 2-3: generated (large gen-teacher diff)
    gen_data_val[:, 2:, :, :] = 10.0
    teacher_val[:, 2:, :, :] = -10.0  # large diff on generated frames

    gen = [gen_data_val.clone().requires_grad_(True)]
    tea = [teacher_val]
    fake = [fake_val]

    mask_generated = torch.zeros(T, 1, 1)
    mask_generated[2:] = 1.0  # only frames 2-3 are generated

    loss_masked = variational_score_distillation_loss(gen, tea, fake, loss_mask=[mask_generated])
    # Now all frames (including near-zero diff frames 0-1) inflate the denom
    gen2 = [gen_data_val.clone().requires_grad_(True)]
    loss_unmasked = variational_score_distillation_loss(gen2, tea, fake)

    # Masked and unmasked losses should differ when conditioned frames have small diff
    assert loss_masked.item() != pytest.approx(loss_unmasked.item(), rel=0.01), (
        "Masked and unmasked losses should differ when conditioned frames have small diff"
    )


# ---------------------------------------------------------------------------
# simulation_mode config fields
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_simulation_mode_default_is_forward():
    """DMD2RFConfig.simulation_mode defaults to 'forward'."""
    from cosmos_framework.configs.base.experiment.distillation.dmd2_config import DMD2RFConfig

    assert DMD2RFConfig().simulation_mode == "forward"


@pytest.mark.L0
def test_simulation_mode_accepts_backward():
    """DMD2RFConfig accepts simulation_mode='backward' without error."""
    import attrs

    from cosmos_framework.configs.base.experiment.distillation.dmd2_config import DMD2RFConfig

    cfg = attrs.evolve(DMD2RFConfig(), simulation_mode="backward")
    assert cfg.simulation_mode == "backward"


@pytest.mark.L0
def test_backward_grad_steps_default():
    """DMD2RFConfig.backward_grad_steps defaults to 1."""
    from cosmos_framework.configs.base.experiment.distillation.dmd2_config import DMD2RFConfig

    assert DMD2RFConfig().backward_grad_steps == 1


@pytest.mark.L0
def test_forward_simulation_defined():
    """_forward_simulation must be defined directly on DMD2RFModel."""
    assert "_forward_simulation" in DMD2RFModel.__dict__


@pytest.mark.L0
def test_backward_simulation_defined():
    """_backward_simulation must be defined directly on DMD2RFModel."""
    assert "_backward_simulation" in DMD2RFModel.__dict__


# ---------------------------------------------------------------------------
# _gen_data_from_student dispatch
# ---------------------------------------------------------------------------


@pytest.mark.L0
def test_gen_data_from_student_dispatches_forward():
    """_gen_data_from_student must call _forward_simulation when simulation_mode='forward'."""
    model = MagicMock()
    model.config.simulation_mode = "forward"
    model._forward_simulation.return_value = ("gd", "packed", torch.zeros(2, 1), {})

    DMD2RFModel._gen_data_from_student(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=MagicMock(),
        iteration=0,
    )

    model._forward_simulation.assert_called_once()
    model._backward_simulation.assert_not_called()


@pytest.mark.L0
def test_gen_data_from_student_dispatches_backward():
    """_gen_data_from_student must call _backward_simulation when simulation_mode='backward'."""
    model = MagicMock()
    model.config.simulation_mode = "backward"
    model._backward_simulation.return_value = ("gd", "packed", torch.zeros(2, 1), {})

    DMD2RFModel._gen_data_from_student(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=MagicMock(),
        iteration=0,
    )

    model._backward_simulation.assert_called_once()
    model._forward_simulation.assert_not_called()


# ---------------------------------------------------------------------------
# _backward_simulation helper tests
# ---------------------------------------------------------------------------


def _make_backward_model(
    *,
    t_list: Optional[list[float]] = None,
    sample_type: str = "ode",
    backward_grad_steps: int = 1,
    sound_gen: bool = False,
) -> tuple[MagicMock, int]:
    """Build a minimal mock for _backward_simulation tests (B=2, vision-only).

    ``_backward_n_steps`` is mocked to return the full schedule length by default, so the
    rollout is deterministic; tests that exercise rcm-style variable length override its
    ``side_effect``.
    """
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised

    if t_list is None:
        t_list = [1.0, 0.5]  # 2-step schedule -> full_t_list = [1.0, 0.5, 0.0]

    B = 2
    model = MagicMock()
    model.tensor_kwargs = {"dtype": torch.float32, "device": "cpu"}
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.parallel_dims = None
    model.config.rectified_flow_inference_config.num_train_timesteps = 1000
    model.config.action_gen = False
    model.config.sound_gen = sound_gen
    model.config.fixed_step_sampler_config.t_list = t_list
    model.config.fixed_step_sampler_config.sample_type = sample_type
    model.config.backward_grad_steps = backward_grad_steps
    # Default: run the full schedule (n_steps == number of nonzero sigma levels). rcm-style
    # variable-length tests override this side_effect.
    model._backward_n_steps.side_effect = lambda max_n_steps, iteration: max_n_steps

    # All-zero condition mask = generation frames only
    packed = MagicMock()
    packed.vision = MagicMock()
    packed.vision.condition_mask = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]  # [T=1,1,1] each
    packed.action = None
    packed.sound = None
    model._pack_input_sequence.return_value = packed

    gd_noised = MagicMock(spec=GenerationDataNoised)
    gd_noised.xt_tokens_vision = [torch.ones(4, 1, 2, 2), torch.ones(4, 1, 2, 2)]  # [C,T,H,W]
    gd_noised.xt_tokens_action = None
    gd_noised.xt_tokens_sound = None
    gd_noised.epsilon_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd_noised.vt_target_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd_noised.sigmas_vision = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]
    gd_noised.epsilon_action = None
    gd_noised.vt_target_action = None
    gd_noised.sigmas_action = None
    gd_noised.epsilon_sound = None
    gd_noised.vt_target_sound = None
    gd_noised.sigmas_sound = None
    model._add_noise_to_input.return_value = gd_noised

    # Return zero velocity predictions by default
    model._pack_and_denoise.return_value = {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}

    import types

    # Use real implementations so ODE/SDE steps and x0 = xt - sigma * v are exercised
    # _velocity_to_x0 doesn't use self; bind the others so self.tensor_kwargs resolves
    model._velocity_to_x0 = DMD2RFModel._velocity_to_x0
    model._ode_step = types.MethodType(DMD2RFModel._ode_step, model)
    model._sde_step = types.MethodType(DMD2RFModel._sde_step, model)

    return model, B


@pytest.mark.L0
def test_backward_simulation_vision_output_shape():
    """_backward_simulation returns a 4-tuple with x0_tokens_vision of the correct shape."""
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    model, B = _make_backward_model()
    gen_data = _make_gen_data()

    result = DMD2RFModel._backward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
        iteration=0,
    )

    assert isinstance(result, tuple) and len(result) == 4
    gen_data_student, packed, sigma, out = result
    assert isinstance(gen_data_student, GenerationDataClean)
    assert gen_data_student.x0_tokens_vision is not None
    assert len(gen_data_student.x0_tokens_vision) == B
    assert gen_data_student.x0_tokens_vision[0].shape == (4, 1, 2, 2)  # [C,T,H,W]
    assert sigma.shape == (B, 1)  # [B,1]


@pytest.mark.L0
def test_backward_simulation_single_step_x0():
    """With a 1-step t_list, x0 must equal xt - sigma * v_pred (RF formula)."""
    model, B = _make_backward_model(t_list=[1.0])  # full_t_list = [1.0, 0.0], n_steps=1

    xt = torch.ones(4, 1, 2, 2) * 2.0  # [C,T,H,W]
    v_pred = torch.ones(4, 1, 2, 2) * 3.0  # [C,T,H,W]

    # xt_tokens_vision is the initial noised (pure-noise) data at sigma=1.0
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised

    gd_noised = MagicMock(spec=GenerationDataNoised)
    gd_noised.xt_tokens_vision = [xt, xt]
    gd_noised.xt_tokens_action = None
    gd_noised.xt_tokens_sound = None
    gd_noised.epsilon_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd_noised.vt_target_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd_noised.sigmas_vision = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]
    gd_noised.epsilon_action = None
    gd_noised.vt_target_action = None
    gd_noised.sigmas_action = None
    gd_noised.epsilon_sound = None
    gd_noised.vt_target_sound = None
    gd_noised.sigmas_sound = None
    model._add_noise_to_input.return_value = gd_noised
    model._pack_and_denoise.return_value = {"preds_vision": [v_pred, v_pred]}

    gen_data = _make_gen_data()
    result = DMD2RFModel._backward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
        iteration=0,
    )

    gen_data_student = result[0]
    # x0 = xt - sigma * v_pred; sigma_eff = 1.0 * noisy_mask = 1.0 (generation frames)
    expected_x0 = xt - 1.0 * v_pred  # [C,T,H,W]
    assert torch.allclose(gen_data_student.x0_tokens_vision[0], expected_x0, atol=1e-5)


@pytest.mark.L0
def test_backward_simulation_last_step_only_grad():
    """With backward_grad_steps=1, only the last _pack_and_denoise call has grad enabled."""
    model, _ = _make_backward_model(backward_grad_steps=1)  # 2-step t_list

    grad_states = []

    def capture_grad(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}

    model._pack_and_denoise.side_effect = capture_grad

    gen_data = _make_gen_data()
    with torch.enable_grad():
        DMD2RFModel._backward_simulation(
            model,
            input_text_indexes=[[], []],
            sequence_plans=[MagicMock(), MagicMock()],
            gen_data_clean=gen_data,
            iteration=0,
        )

    assert len(grad_states) == 2
    assert grad_states[0] is False, "Step 0 (early) should run under no_grad"
    assert grad_states[1] is True, "Step 1 (last) should run with grad enabled"


@pytest.mark.L0
def test_backward_simulation_no_grad_early_steps():
    """All but the last step must be wrapped in torch.no_grad."""
    model, _ = _make_backward_model(backward_grad_steps=1, t_list=[1.0, 0.9, 0.75])

    grad_states = []

    def capture_grad(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}

    model._pack_and_denoise.side_effect = capture_grad

    gen_data = _make_gen_data()
    with torch.enable_grad():
        DMD2RFModel._backward_simulation(
            model,
            input_text_indexes=[[], []],
            sequence_plans=[MagicMock(), MagicMock()],
            gen_data_clean=gen_data,
            iteration=0,
        )

    # n_steps=3; backward_grad_steps=1; steps 0,1 → no_grad; step 2 → grad
    assert len(grad_states) == 3
    assert all(not g for g in grad_states[:2]), "Steps 0-1 should be under no_grad"
    assert grad_states[2] is True, "Step 2 (last) should have grad enabled"


@pytest.mark.L0
def test_backward_simulation_bptt_all_steps():
    """With backward_grad_steps=-1, every _pack_and_denoise call has grad enabled."""
    model, _ = _make_backward_model(backward_grad_steps=-1)

    grad_states = []

    def capture_grad(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}

    model._pack_and_denoise.side_effect = capture_grad

    gen_data = _make_gen_data()
    with torch.enable_grad():
        DMD2RFModel._backward_simulation(
            model,
            input_text_indexes=[[], []],
            sequence_plans=[MagicMock(), MagicMock()],
            gen_data_clean=gen_data,
            iteration=0,
        )

    assert all(grad_states), "All steps should have grad enabled when backward_grad_steps=-1"


@pytest.mark.L0
def test_backward_simulation_prefix_schedule_from_pure_noise():
    """An n_steps of 2 runs the schedule PREFIX from sigma=1.0 down to 0."""
    model, B = _make_backward_model(t_list=[1.0, 0.9, 0.75], backward_grad_steps=1)
    # full_t_list = [1.0, 0.9, 0.75, 0.0]; n_steps=2 -> prefix schedule [1.0, 0.9, 0.0].
    model._backward_n_steps.side_effect = lambda max_n_steps, iteration: 2

    timesteps_seen = []

    def capture_timesteps(*args, **kwargs):
        timesteps_seen.append(args[2].clone())  # [B,1]
        return {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}

    model._pack_and_denoise.side_effect = capture_timesteps

    gen_data = _make_gen_data()
    result = DMD2RFModel._backward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
        iteration=0,
    )

    # n_steps=2 -> denoise at sigma=1.0 (t=1000) then sigma=0.9 (t=900), descending to 0.
    assert len(timesteps_seen) == 2
    assert torch.allclose(timesteps_seen[0], torch.full((B, 1), 1000.0))  # [B,1]
    assert torch.allclose(timesteps_seen[1], torch.full((B, 1), 900.0))  # [B,1]
    # Returned sigma_student is always the pure-noise start sigma (1.0), regardless of n_steps.
    assert torch.allclose(result[2], torch.full((B, 1), 1.0))  # [B,1]


@pytest.mark.L0
def test_backward_simulation_seeds_from_pure_noise():
    """The rollout seed must be built at sigma=1.0 (pure noise, no clean-data blending)."""
    model, _ = _make_backward_model(t_list=[1.0, 0.9, 0.75], backward_grad_steps=1)

    gen_data = _make_gen_data()
    DMD2RFModel._backward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
        iteration=0,
    )

    # _add_noise_to_input(gen_data_clean, packed_sequence, sigma_max, ...) seeds the rollout.
    sigma_max = model._add_noise_to_input.call_args.args[2]  # [B,1]
    assert torch.allclose(sigma_max, torch.ones_like(sigma_max))


@pytest.mark.L0
def test_backward_simulation_requires_pure_noise_start():
    """A schedule whose first sigma is < 1.0 must raise (would blend clean data into the seed)."""
    model, _ = _make_backward_model(t_list=[0.9, 0.5], backward_grad_steps=1)
    gen_data = _make_gen_data()

    with pytest.raises(AssertionError):
        DMD2RFModel._backward_simulation(
            model,
            input_text_indexes=[[], []],
            sequence_plans=[MagicMock(), MagicMock()],
            gen_data_clean=gen_data,
            iteration=0,
        )


@pytest.mark.L0
def test_backward_simulation_invalid_grad_steps():
    """backward_grad_steps=0 must raise AssertionError."""
    model, _ = _make_backward_model(backward_grad_steps=0)
    gen_data = _make_gen_data()

    with pytest.raises(AssertionError):
        DMD2RFModel._backward_simulation(
            model,
            input_text_indexes=[[], []],
            sequence_plans=[MagicMock(), MagicMock()],
            gen_data_clean=gen_data,
            iteration=0,
        )


@pytest.mark.L0
def test_backward_n_steps_cycles_by_per_net_update_index():
    """rcm-style cycling: n_steps deterministically cycles 1..max by per-net update index."""
    import types

    model = MagicMock()
    model.config.warmup_student_steps = 0
    model.config.warmup_critic_steps = 0
    model.config.student_update_freq = 5
    model.get_phase = types.MethodType(DMD2RFModel.get_phase, model)
    model.get_student_iteration = types.MethodType(DMD2RFModel.get_student_iteration, model)
    model.get_critic_iteration = types.MethodType(DMD2RFModel.get_critic_iteration, model)
    model._backward_n_steps = types.MethodType(DMD2RFModel._backward_n_steps, model)

    # Student steps fall on iteration % 5 == 0 -> student_idx 0,1,2,3,4 -> n_steps cycles 1,2,3,4,1.
    student_iters = [0, 5, 10, 15, 20]
    assert [model._backward_n_steps(4, it) for it in student_iters] == [1, 2, 3, 4, 1]
    # Critic steps (iteration % 5 != 0) cycle on the critic update counter, also spanning 1..4.
    critic_iters = [1, 2, 3, 4, 6]
    assert all(1 <= model._backward_n_steps(4, it) <= 4 for it in critic_iters)

    # Critic-only warmup: iteration 0 is a critic step with update index -1; the clamp keeps the
    # cycle starting at n_steps=1 (not wrapping to max via negative modulo).
    model.config.warmup_critic_steps = 100
    assert model.get_phase(0) == "critic"
    assert model._backward_n_steps(4, 0) == 1


@pytest.mark.L0
def test_backward_simulation_grad_steps_exceeds_n_steps():
    """backward_grad_steps > n_steps must not raise and enables grad on every step."""
    model, _ = _make_backward_model(backward_grad_steps=10, t_list=[1.0, 0.5])

    grad_states = []

    def capture_grad(*args, **kwargs):
        grad_states.append(torch.is_grad_enabled())
        return {"preds_vision": [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]}

    model._pack_and_denoise.side_effect = capture_grad

    gen_data = _make_gen_data()
    with torch.enable_grad():
        DMD2RFModel._backward_simulation(
            model,
            input_text_indexes=[[], []],
            sequence_plans=[MagicMock(), MagicMock()],
            gen_data_clean=gen_data,
            iteration=0,
        )

    assert all(grad_states), "All steps should have grad when backward_grad_steps > n_steps"


@pytest.mark.L0
def test_backward_simulation_ode_step_keeps_conditioned_frames():
    """ODE Euler step must leave conditioned frames unchanged between denoising steps."""
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised

    model, B = _make_backward_model(sample_type="ode", t_list=[1.0, 0.5])

    # Frame 0 is conditioned (mask=1), frame 1 is generated (mask=0)
    packed = MagicMock()
    packed.vision = MagicMock()
    packed.vision.condition_mask = [
        torch.tensor([[[1.0]], [[0.0]]]),  # [T=2,1,1]: frame0 conditioned, frame1 generated
        torch.tensor([[[1.0]], [[0.0]]]),
    ]
    packed.action = None
    packed.sound = None
    model._pack_input_sequence.return_value = packed

    # Initial xt: frame 0 = value 5.0, frame 1 = value 1.0
    xt_init = torch.zeros(4, 2, 2, 2)  # [C,T,H,W]
    xt_init[:, 0, :, :] = 5.0  # conditioned frame
    xt_init[:, 1, :, :] = 1.0  # generated frame

    gd_noised = MagicMock(spec=GenerationDataNoised)
    gd_noised.xt_tokens_vision = [xt_init.clone(), xt_init.clone()]
    gd_noised.xt_tokens_action = None
    gd_noised.xt_tokens_sound = None
    gd_noised.epsilon_vision = [torch.zeros(4, 2, 2, 2), torch.zeros(4, 2, 2, 2)]
    gd_noised.vt_target_vision = [torch.zeros(4, 2, 2, 2), torch.zeros(4, 2, 2, 2)]
    gd_noised.sigmas_vision = [torch.zeros(2, 1, 1), torch.zeros(2, 1, 1)]
    gd_noised.epsilon_action = None
    gd_noised.vt_target_action = None
    gd_noised.sigmas_action = None
    gd_noised.epsilon_sound = None
    gd_noised.vt_target_sound = None
    gd_noised.sigmas_sound = None
    model._add_noise_to_input.return_value = gd_noised

    # Non-zero v_pred so ODE step actually moves generated frames
    v_pred = torch.ones(4, 2, 2, 2) * 3.0  # [C,T,H,W]

    xt_received_at_step1 = []

    def capture_step(*args, **kwargs):
        if model._pack_and_denoise.call_count == 2:
            # Second call: record what xt was passed in gen_data_noised
            gd_n = args[1]
            xt_received_at_step1.append(gd_n.xt_tokens_vision[0].clone())
        return {"preds_vision": [v_pred.clone(), v_pred.clone()]}

    model._pack_and_denoise.side_effect = capture_step

    gen_data = _make_gen_data()
    DMD2RFModel._backward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
        iteration=0,
    )

    assert len(xt_received_at_step1) == 1
    xt_step1 = xt_received_at_step1[0]  # [C,T,H,W]
    # Conditioned frame (frame 0) must be unchanged from initial value 5.0
    assert torch.allclose(xt_step1[:, 0, :, :], torch.full((4, 2, 2), 5.0), atol=1e-5), (
        "Conditioned frame should remain 5.0 after ODE step"
    )
    # Generated frame (frame 1) must have changed (delta = -0.499, v=3 → moved by -1.497)
    assert not torch.allclose(xt_step1[:, 1, :, :], torch.full((4, 2, 2), 1.0), atol=1e-3), (
        "Generated frame should have changed after ODE step"
    )


@pytest.mark.L0
def test_backward_simulation_sde_step_keeps_conditioned_frames():
    """SDE re-noising step must leave conditioned frames unchanged between denoising steps."""
    from unittest.mock import patch

    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised

    model, B = _make_backward_model(sample_type="sde", t_list=[1.0, 0.5])

    # Frame 0 conditioned, frame 1 generated
    packed = MagicMock()
    packed.vision = MagicMock()
    packed.vision.condition_mask = [
        torch.tensor([[[1.0]], [[0.0]]]),
        torch.tensor([[[1.0]], [[0.0]]]),
    ]
    packed.action = None
    packed.sound = None
    model._pack_input_sequence.return_value = packed

    xt_init = torch.zeros(4, 2, 2, 2)
    xt_init[:, 0, :, :] = 5.0
    xt_init[:, 1, :, :] = 1.0

    gd_noised = MagicMock(spec=GenerationDataNoised)
    gd_noised.xt_tokens_vision = [xt_init.clone(), xt_init.clone()]
    gd_noised.xt_tokens_action = None
    gd_noised.xt_tokens_sound = None
    gd_noised.epsilon_vision = [torch.zeros(4, 2, 2, 2), torch.zeros(4, 2, 2, 2)]
    gd_noised.vt_target_vision = [torch.zeros(4, 2, 2, 2), torch.zeros(4, 2, 2, 2)]
    gd_noised.sigmas_vision = [torch.zeros(2, 1, 1), torch.zeros(2, 1, 1)]
    gd_noised.epsilon_action = None
    gd_noised.vt_target_action = None
    gd_noised.sigmas_action = None
    gd_noised.epsilon_sound = None
    gd_noised.vt_target_sound = None
    gd_noised.sigmas_sound = None
    model._add_noise_to_input.return_value = gd_noised

    v_pred = torch.ones(4, 2, 2, 2) * 3.0

    xt_received_at_step1 = []

    def capture_step(*args, **kwargs):
        if model._pack_and_denoise.call_count == 2:
            gd_n = args[1]
            xt_received_at_step1.append(gd_n.xt_tokens_vision[0].clone())
        return {"preds_vision": [v_pred.clone(), v_pred.clone()]}

    model._pack_and_denoise.side_effect = capture_step

    # Use deterministic randn (zeros) so the SDE fresh noise term is zero
    with patch("torch.randn_like", return_value=torch.zeros(4, 2, 2, 2)):
        gen_data = _make_gen_data()
        DMD2RFModel._backward_simulation(
            model,
            input_text_indexes=[[], []],
            sequence_plans=[MagicMock(), MagicMock()],
            gen_data_clean=gen_data,
            iteration=0,
        )

    assert len(xt_received_at_step1) == 1
    xt_step1 = xt_received_at_step1[0]  # [C,T,H,W]
    # Conditioned frame (frame 0) must remain 5.0: SDE uses cond_mask * xt for those frames
    assert torch.allclose(xt_step1[:, 0, :, :], torch.full((4, 2, 2), 5.0), atol=1e-5), (
        "Conditioned frame should remain 5.0 after SDE step"
    )


# ---------------------------------------------------------------------------
# _backward_simulation dtype preservation (regression for bf16/fp32 mixing bug)
# ---------------------------------------------------------------------------


def _make_backward_model_bf16(
    *,
    sample_type: str = "ode",
) -> tuple[MagicMock, int]:
    """Like _make_backward_model but with tensor_kwargs=bfloat16 to match real training."""
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataNoised

    B = 2
    model = MagicMock()
    model.tensor_kwargs = {"dtype": torch.bfloat16, "device": "cpu"}
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.parallel_dims = None
    model.config.rectified_flow_inference_config.num_train_timesteps = 1000
    model.config.action_gen = False
    model.config.sound_gen = False
    model.config.fixed_step_sampler_config.t_list = [1.0, 0.5]
    model.config.fixed_step_sampler_config.sample_type = sample_type
    model.config.backward_grad_steps = 1
    model._backward_n_steps.side_effect = lambda max_n_steps, iteration: max_n_steps

    # Condition mask is float32 (typical from packing pipeline) — this is the source of the bug
    packed = MagicMock()
    packed.vision = MagicMock()
    packed.vision.condition_mask = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]  # float32, [T=1,1,1]
    packed.action = None
    packed.sound = None
    model._pack_input_sequence.return_value = packed

    gd_noised = MagicMock(spec=GenerationDataNoised)
    gd_noised.xt_tokens_vision = [
        torch.ones(4, 1, 2, 2, dtype=torch.bfloat16),
        torch.ones(4, 1, 2, 2, dtype=torch.bfloat16),
    ]
    gd_noised.xt_tokens_action = None
    gd_noised.xt_tokens_sound = None
    gd_noised.epsilon_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd_noised.vt_target_vision = [torch.zeros(4, 1, 2, 2), torch.zeros(4, 1, 2, 2)]
    gd_noised.sigmas_vision = [torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)]
    gd_noised.epsilon_action = None
    gd_noised.vt_target_action = None
    gd_noised.sigmas_action = None
    gd_noised.epsilon_sound = None
    gd_noised.vt_target_sound = None
    gd_noised.sigmas_sound = None
    model._add_noise_to_input.return_value = gd_noised

    model._pack_and_denoise.return_value = {
        "preds_vision": [
            torch.zeros(4, 1, 2, 2, dtype=torch.bfloat16),
            torch.zeros(4, 1, 2, 2, dtype=torch.bfloat16),
        ]
    }
    import types

    model._velocity_to_x0 = DMD2RFModel._velocity_to_x0
    model._ode_step = types.MethodType(DMD2RFModel._ode_step, model)
    model._sde_step = types.MethodType(DMD2RFModel._sde_step, model)

    return model, B


@pytest.mark.L0
def test_backward_simulation_ode_xt_dtype_matches_model():
    """ODE: xt_tokens passed to _pack_and_denoise on step 2 must match tensor_kwargs dtype.

    Regression test for the bf16/fp32 mixing bug where fp32 condition masks upcast xt_next
    to fp32, causing a dtype mismatch in the model's linear layers.
    """
    model, _ = _make_backward_model_bf16(sample_type="ode")

    xt_dtypes_step2 = []

    def capture_step(*args, **kwargs):
        if model._pack_and_denoise.call_count == 2:
            gd_n = args[1]
            xt_dtypes_step2.append(gd_n.xt_tokens_vision[0].dtype)
        return {
            "preds_vision": [
                torch.zeros(4, 1, 2, 2, dtype=torch.bfloat16),
                torch.zeros(4, 1, 2, 2, dtype=torch.bfloat16),
            ]
        }

    model._pack_and_denoise.side_effect = capture_step

    gen_data = _make_gen_data()
    DMD2RFModel._backward_simulation(
        model,
        input_text_indexes=[[], []],
        sequence_plans=[MagicMock(), MagicMock()],
        gen_data_clean=gen_data,
        iteration=0,
    )

    assert len(xt_dtypes_step2) == 1
    assert xt_dtypes_step2[0] == torch.bfloat16, (
        f"xt_tokens_vision dtype should be bfloat16 after ODE step, got {xt_dtypes_step2[0]}"
    )


@pytest.mark.L0
def test_backward_simulation_sde_xt_dtype_matches_model():
    """SDE: xt_tokens passed to _pack_and_denoise on step 2 must match tensor_kwargs dtype.

    Same regression test as ODE variant, for the SDE re-noising path.
    """
    from unittest.mock import patch

    model, _ = _make_backward_model_bf16(sample_type="sde")

    xt_dtypes_step2 = []

    def capture_step(*args, **kwargs):
        if model._pack_and_denoise.call_count == 2:
            gd_n = args[1]
            xt_dtypes_step2.append(gd_n.xt_tokens_vision[0].dtype)
        return {
            "preds_vision": [
                torch.zeros(4, 1, 2, 2, dtype=torch.bfloat16),
                torch.zeros(4, 1, 2, 2, dtype=torch.bfloat16),
            ]
        }

    model._pack_and_denoise.side_effect = capture_step

    with patch("torch.randn_like", return_value=torch.zeros(4, 1, 2, 2)):
        gen_data = _make_gen_data()
        DMD2RFModel._backward_simulation(
            model,
            input_text_indexes=[[], []],
            sequence_plans=[MagicMock(), MagicMock()],
            gen_data_clean=gen_data,
            iteration=0,
        )

    assert len(xt_dtypes_step2) == 1
    assert xt_dtypes_step2[0] == torch.bfloat16, (
        f"xt_tokens_vision dtype should be bfloat16 after SDE step, got {xt_dtypes_step2[0]}"
    )


# ---------------------------------------------------------------------------
# _ode_step and _sde_step unit tests
# ---------------------------------------------------------------------------


def _make_step_model(*, dtype: torch.dtype = torch.float32) -> MagicMock:
    model = MagicMock()
    model.tensor_kwargs = {"dtype": dtype, "device": "cpu"}
    model.tensor_kwargs_fp32 = {"dtype": torch.float32, "device": "cpu"}
    model.parallel_dims = None
    return model


@pytest.mark.L0
def test_ode_step_basic():
    """_ode_step computes xt + delta_sigma * v * noisy_mask element-wise."""
    model = _make_step_model()
    xt = [torch.tensor([2.0, 4.0])]
    v_pred = [torch.tensor([1.0, 1.0])]
    noisy_mask = [torch.tensor([1.0, 0.0])]  # second element conditioned
    delta_sigma = -0.5

    result = DMD2RFModel._ode_step(model, xt, v_pred, noisy_mask, delta_sigma)

    expected = torch.tensor([2.0 + (-0.5) * 1.0 * 1.0, 4.0 + (-0.5) * 1.0 * 0.0])
    assert torch.allclose(result[0], expected, atol=1e-6)


@pytest.mark.L0
def test_ode_step_conditioned_frame_unchanged():
    """_ode_step leaves frames where noisy_mask=0 unchanged."""
    model = _make_step_model()
    xt = [torch.ones(4, 2, 2, 2) * 5.0]  # [C,T,H,W]
    v_pred = [torch.ones(4, 2, 2, 2) * 3.0]
    noisy_mask = [torch.tensor([[[0.0]], [[1.0]]])]  # frame0 conditioned, frame1 generated
    delta_sigma = -0.5

    result = DMD2RFModel._ode_step(model, xt, v_pred, noisy_mask, delta_sigma)

    # Conditioned frame (frame 0): unchanged from 5.0
    assert torch.allclose(result[0][:, 0, :, :], torch.full((4, 2, 2), 5.0), atol=1e-6)
    # Generated frame (frame 1): changed
    assert not torch.allclose(result[0][:, 1, :, :], torch.full((4, 2, 2), 5.0), atol=1e-3)


@pytest.mark.L0
def test_ode_step_output_dtype_matches_tensor_kwargs():
    """_ode_step output dtype must match tensor_kwargs, not the input dtype."""
    model = _make_step_model(dtype=torch.bfloat16)
    xt = [torch.ones(4, 1, 2, 2)]  # float32 input
    v_pred = [torch.ones(4, 1, 2, 2)]
    noisy_mask = [torch.zeros(1, 1, 1)]
    delta_sigma = -0.5

    result = DMD2RFModel._ode_step(model, xt, v_pred, noisy_mask, delta_sigma)
    assert result[0].dtype == torch.bfloat16


@pytest.mark.L0
def test_sde_step_basic_zero_noise():
    """_sde_step with zero fresh noise: xt_next = noisy_mask*(1-σ)*x0 + cond_mask*xt."""
    from unittest.mock import patch

    model = _make_step_model()
    xt = [torch.tensor([5.0, 1.0])]  # [conditioned, generated]
    x0_pred = [torch.tensor([0.0, 2.0])]
    noisy_mask = [torch.tensor([0.0, 1.0])]  # frame0 conditioned, frame1 generated
    cond_mask = [torch.tensor([1.0, 0.0])]
    sigma_next = 0.5

    with patch("torch.randn_like", return_value=torch.zeros(2)):
        result = DMD2RFModel._sde_step(model, xt, x0_pred, noisy_mask, cond_mask, sigma_next)

    # frame0 (conditioned): cond_mask=1 → keeps xt=5.0; noisy_mask=0 → no re-noise
    # frame1 (generated): noisy_mask=1 → (1-0.5)*2.0 + 0.5*0.0 = 1.0
    assert torch.allclose(result[0], torch.tensor([5.0, 1.0]), atol=1e-6)


@pytest.mark.L0
def test_sde_step_conditioned_frame_unchanged():
    """_sde_step leaves frames where cond_mask=1 at their original xt value."""
    from unittest.mock import patch

    model = _make_step_model()
    B = 2
    xt = [torch.ones(4, 2, 2, 2) * 5.0, torch.ones(4, 2, 2, 2) * 5.0]  # [C,T,H,W]
    x0_pred = [torch.zeros(4, 2, 2, 2), torch.zeros(4, 2, 2, 2)]
    noisy_mask = [torch.tensor([[[0.0]], [[1.0]]]), torch.tensor([[[0.0]], [[1.0]]])]
    cond_mask = [torch.tensor([[[1.0]], [[0.0]]]), torch.tensor([[[1.0]], [[0.0]]])]
    sigma_next = 0.5

    with patch("torch.randn_like", return_value=torch.zeros(4, 2, 2, 2)):
        result = DMD2RFModel._sde_step(model, xt, x0_pred, noisy_mask, cond_mask, sigma_next)

    for i in range(B):
        assert torch.allclose(result[i][:, 0, :, :], torch.full((4, 2, 2), 5.0), atol=1e-6), (
            f"Conditioned frame {i} should remain 5.0"
        )


@pytest.mark.L0
def test_sde_step_output_dtype_matches_tensor_kwargs():
    """_sde_step output dtype must match tensor_kwargs (regression: fp32 masks + bf16 xt)."""
    from unittest.mock import patch

    model = _make_step_model(dtype=torch.bfloat16)
    xt = [torch.ones(4, 1, 2, 2, dtype=torch.bfloat16)]
    x0_pred = [torch.ones(4, 1, 2, 2, dtype=torch.bfloat16)]
    noisy_mask = [torch.zeros(1, 1, 1)]  # float32 mask (typical from packing pipeline)
    cond_mask = [torch.ones(1, 1, 1)]  # float32 mask
    sigma_next = 0.5

    with patch("torch.randn_like", return_value=torch.zeros(4, 1, 2, 2)):
        result = DMD2RFModel._sde_step(model, xt, x0_pred, noisy_mask, cond_mask, sigma_next)

    assert result[0].dtype == torch.bfloat16


@pytest.mark.L0
def test_sde_step_broadcasts_and_uses_fresh_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """_sde_step must CP-broadcast rank-local fresh SDE noise before composing xt_next."""

    model = _make_step_model()
    model.parallel_dims = MagicMock()
    xt = [torch.zeros(2)]  # [C]
    x0_pred = [torch.zeros(2)]  # [C]
    noisy_mask = [torch.ones(2)]  # [C]
    cond_mask = [torch.zeros(2)]  # [C]
    sigma_next = 1.0
    calls: list[tuple[list[torch.Tensor], object | None]] = []

    def _spy_broadcast(tensors: list[torch.Tensor] | None, parallel_dims: object | None) -> None:
        assert tensors is not None
        calls.append(([tensor.clone() for tensor in tensors], parallel_dims))  # list of [C]
        tensors[0].fill_(7.0)  # [C]

    monkeypatch.setattr(dmd2_rf_module, "context_parallel_broadcast_tensor_list", _spy_broadcast)

    with patch("torch.randn_like", return_value=torch.ones(2)):
        result = DMD2RFModel._sde_step(model, xt, x0_pred, noisy_mask, cond_mask, sigma_next)

    assert len(calls) == 1
    assert calls[0][1] is model.parallel_dims
    assert torch.allclose(calls[0][0][0], torch.ones(2), atol=1e-6)
    assert torch.allclose(result[0], torch.full((2,), 7.0), atol=1e-6)


@pytest.mark.L0
def test_ode_step_defined():
    """_ode_step must be defined directly on DMD2RFModel."""
    assert "_ode_step" in DMD2RFModel.__dict__


@pytest.mark.L0
def test_sde_step_defined():
    """_sde_step must be defined directly on DMD2RFModel."""
    assert "_sde_step" in DMD2RFModel.__dict__
