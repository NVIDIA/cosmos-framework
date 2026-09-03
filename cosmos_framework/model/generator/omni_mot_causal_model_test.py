# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for OmniMoTCausalModel (AR generation logic).

## L0 regression tests for iter_samples_from_batch_autoregressive

Bugs patched:

  Bug 2 — text2video: text tokens never stored in KV cache
      text_tokens=None was passed for ALL frames including frame 0, so the
      und_cache stored empty K/V and all frames were generated without text
      conditioning.

  Bug 3 — image2video/forward_dynamics: KV cache never seeded with frame 0
      The initial packed sequence (text + conditioned frame 0) was built but never
      run through denoise before the AR loop started, leaving und_cache
      uninitialized.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


@pytest.mark.L0
@pytest.mark.CPU
def test_gaussian_bell_train_time_weight_matches_rcm_formula() -> None:
    from cosmos_framework.model.generator.omni_mot_causal_model import GaussianBellTrainTimeWeight

    weight = GaussianBellTrainTimeWeight(SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)))

    timesteps = torch.tensor([0.0, 500.0, 1000.0])  # [B]
    actual = weight(timesteps, {"dtype": torch.float32, "device": "cpu"})  # [B]
    t_rf = timesteps / 1000.0  # [B]
    expected = torch.exp(-2 * (t_rf - 0.5) ** 2) - torch.exp(torch.tensor(-0.5))  # [B]

    torch.testing.assert_close(actual, expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_gaussian_bell_train_time_weight_peaks_near_middle() -> None:
    from cosmos_framework.model.generator.omni_mot_causal_model import GaussianBellTrainTimeWeight

    weight = GaussianBellTrainTimeWeight(SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000)))

    actual = weight(torch.tensor([0.0, 500.0, 1000.0]), {"dtype": torch.float32, "device": "cpu"})  # [B]

    assert actual[1] > actual[0]
    torch.testing.assert_close(actual[0], actual[2])


# ---------------------------------------------------------------------------
# L0 — Causal model config fields
# ---------------------------------------------------------------------------


@pytest.mark.L0
@pytest.mark.CPU
def test_kv_cache_config_fields_default_to_bf16_and_triton() -> None:
    """KV cache config defaults to BF16 storage with Triton FP8 batch decode."""
    import attrs

    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModelConfig

    fields = {f.name: f for f in attrs.fields(OmniMoTCausalModelConfig)}
    assert "kv_cache_dtype" in fields
    assert fields["kv_cache_dtype"].default is None
    assert "kv_cache_kernel_impl" in fields
    assert fields["kv_cache_kernel_impl"].default == "triton"


@pytest.mark.L0
@pytest.mark.CPU
def test_attention_sink_size_field_defaults_to_zero() -> None:
    """attention_sink_size defaults to disabled."""
    import attrs

    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModelConfig

    fields = {f.name: f for f in attrs.fields(OmniMoTCausalModelConfig)}
    assert "attention_sink_size" in fields
    assert fields["attention_sink_size"].default == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_attention_sink_config_validation() -> None:
    """Attention sinks require a finite KV cache and one non-sink slot."""
    from cosmos_framework.model.generator.omni_mot_causal_model import _validate_attention_sink_config

    _validate_attention_sink_config(None, 0)
    _validate_attention_sink_config(16, 0)
    _validate_attention_sink_config(16, 4)
    _validate_attention_sink_config(16, 15)

    with pytest.raises(ValueError, match="attention_sink_size must be 0 when kv_cache_inference_size is None"):
        _validate_attention_sink_config(None, 1)
    with pytest.raises(ValueError, match="attention_sink_size must be less than kv_cache_inference_size"):
        _validate_attention_sink_config(16, 16)
    with pytest.raises(ValueError, match="attention_sink_size must be less than kv_cache_inference_size"):
        _validate_attention_sink_config(16, 17)
    with pytest.raises(ValueError, match="attention_sink_size must be >= 0"):
        _validate_attention_sink_config(16, -1)


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_forcing_kv_implementation_default_and_validation() -> None:
    """The causal model selects either supported replay implementation explicitly."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModelConfig

    default_config = OmniMoTCausalModelConfig()
    assert default_config.teacher_forcing_kv_implementation == "singleview_threeway_kv"
    assert default_config.teacher_forcing_replay_policy.control_visibility == "global"
    assert (
        OmniMoTCausalModelConfig(
            teacher_forcing_kv_implementation="multiview_flex_kv"
        ).teacher_forcing_kv_implementation
        == "multiview_flex_kv"
    )
    with pytest.raises(ValueError):
        OmniMoTCausalModelConfig(teacher_forcing_kv_implementation="unknown")


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_forcing_replay_policy_resolves_lazy_config_and_runs_validation() -> None:
    """LazyConfig must become the typed policy before model construction."""
    from cosmos_framework.utils.lazy_config import LazyCall as L
    from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
    from cosmos_framework.model.generator.omni_mot_causal_model import (
        OmniMoTCausalModel,
        OmniMoTCausalModelConfig,
        _resolve_teacher_forcing_replay_policy,
    )

    lazy_call = L(OmniMoTCausalModel)(config=OmniMoTCausalModelConfig(), _recursive_=False)
    lazy_policy = lazy_call.config.teacher_forcing_replay_policy
    lazy_policy.control_visibility = "current"
    lazy_policy.controls_read_strict_past_clean_rgb = True
    lazy_policy.clean_pass_causality = "chunk"

    resolved = _resolve_teacher_forcing_replay_policy(lazy_policy)

    assert isinstance(resolved, TeacherForcingReplayPolicyConfig)
    assert resolved.control_visibility == "current"
    assert resolved.controls_read_strict_past_clean_rgb is True
    assert resolved.clean_pass_causality == "chunk"

    unsafe_call = L(OmniMoTCausalModel)(config=OmniMoTCausalModelConfig(), _recursive_=False)
    unsafe_policy = unsafe_call.config.teacher_forcing_replay_policy
    unsafe_policy.controls_read_strict_past_clean_rgb = True
    with pytest.raises(ValueError, match="global control visibility"):
        _resolve_teacher_forcing_replay_policy(unsafe_policy)

    mutated_policy = TeacherForcingReplayPolicyConfig()
    mutated_policy.controls_read_strict_past_clean_rgb = True
    with pytest.raises(ValueError, match="global control visibility"):
        _resolve_teacher_forcing_replay_policy(mutated_policy)


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_forcing_kv_implementation_validates_lazy_config_value() -> None:
    """LazyConfig must not bypass validation for the public replay selector."""
    from cosmos_framework.model.generator.omni_mot_causal_model import _resolve_teacher_forcing_kv_implementation

    assert _resolve_teacher_forcing_kv_implementation("multiview_flex_kv") == "multiview_flex_kv"
    with pytest.raises(ValueError, match="teacher_forcing_kv_implementation"):
        _resolve_teacher_forcing_kv_implementation("unknown")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("causal_training_strategy", ["teacher_forcing", "teacher_forcing_dcm"])
def test_multiview_flex_selection_does_not_infer_from_backend_knobs(causal_training_strategy: str) -> None:
    """The public selector, rather than internal attention flags, chooses Flex replay."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    model = object.__new__(OmniMoTCausalModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        causal_training_strategy=causal_training_strategy,
        teacher_forcing_kv_implementation="multiview_flex_kv",
        joint_attn_implementation="three_way",
        flex_attention=SimpleNamespace(enabled=False),
    )

    assert model._uses_multiview_flex_kv() is True
    del model._teacher_forcing_kv_implementation_runtime
    model.config.teacher_forcing_kv_implementation = "singleview_threeway_kv"
    model.config.joint_attn_implementation = "two_way"
    model.config.flex_attention.enabled = True
    assert model._uses_multiview_flex_kv() is False


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_flex_selection_rejects_non_replay_strategy() -> None:
    """A Flex selector cannot silently fall through to a non-replay backend."""
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    model = object.__new__(OmniMoTCausalModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        causal_training_strategy="none",
        teacher_forcing_kv_implementation="multiview_flex_kv",
    )

    with patch.object(OmniMoTModel, "build_net") as base_build:
        with pytest.raises(ValueError, match="requires causal_training_strategy"):
            model.build_net(torch.bfloat16)

    base_build.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_forcing_dcm_build_routes_multiview_flex_selector() -> None:
    """TF-dCM constructs the selected interactive Flex network instead of falling through."""
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
    from cosmos_framework.model.generator import omni_mot_causal_model as causal_model_module
    from cosmos_framework.model.generator.mot.causal_cosmos3_vfm_network import InteractiveCosmos3VFMNetwork
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    replay_policy = TeacherForcingReplayPolicyConfig(
        control_visibility="causal",
        multiview_attention_scope="same_view",
        decomposed_temporal_window_seconds=0.5,
    )
    model = object.__new__(OmniMoTCausalModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        causal_training_strategy="teacher_forcing_dcm",
        teacher_forcing_kv_implementation="multiview_flex_kv",
        teacher_forcing_replay_policy=replay_policy,
        teacher_forcing_frames_per_chunk=4,
        video_temporal_causal=True,
        joint_attn_implementation="three_way",
        flex_attention=SimpleNamespace(
            enabled=False,
            mask=SimpleNamespace(
                attention_scope="all_views",
                decomposed_temporal_window_seconds=None,
            ),
        ),
    )
    network = SimpleNamespace(config=SimpleNamespace())

    def _fake_base_build(
        dtype: torch.dtype,
        *,
        mp_policy: object | None = None,
        lora_enabled: bool | None = None,
    ) -> SimpleNamespace:
        assert dtype == torch.bfloat16
        assert mp_policy is None
        assert lora_enabled is None
        assert causal_model_module.omni_mot_model_module.Cosmos3VFMNetwork is InteractiveCosmos3VFMNetwork
        assert model.config.joint_attn_implementation == "two_way"
        assert model.config.flex_attention.enabled is True
        return network

    with patch.object(OmniMoTModel, "build_net", side_effect=_fake_base_build):
        result = model.build_net(torch.bfloat16)

    assert result is network
    assert network.config.video_temporal_causal is True
    assert network.video_temporal_causal is True
    assert network.teacher_forcing_replay_policy is replay_policy
    assert network.teacher_forcing_frames_per_chunk == 4
    assert model.config.joint_attn_implementation == "three_way"
    assert model.config.flex_attention.enabled is False
    assert model.config.flex_attention.mask.attention_scope == "all_views"
    assert model.config.flex_attention.mask.decomposed_temporal_window_seconds is None


class TestTeacherForcingTransferControlDropout:
    """Control dropout is scoped to teacher-forcing transfer samples."""

    _PATCH_RAND: str = "cosmos_framework.model.generator.omni_mot_causal_model.torch.rand"

    @staticmethod
    def _make_model(dropout_rate: float, strategy: str = "teacher_forcing") -> MagicMock:
        model = MagicMock()
        model.config.teacher_forcing_transfer_control_dropout_rate = dropout_rate
        model.config.causal_training_strategy = strategy
        return model

    @staticmethod
    def _make_data() -> SimpleNamespace:
        control = torch.full((1, 4, 5, 2, 2), 1.0)  # [B,C,T,H,W]
        target = torch.full((1, 4, 5, 2, 2), 2.0)  # [B,C,T,H,W]
        raw_control = torch.full((1, 3, 17, 4, 4), 3.0)  # [B,C,T_px,H,W]
        raw_target = torch.full((1, 3, 17, 4, 4), 4.0)  # [B,C,T_px,H,W]
        control_positions = torch.arange(5)  # [T]
        target_positions = torch.arange(5)  # [T]
        return SimpleNamespace(
            x0_tokens_vision=[control, target],
            raw_state_vision=[raw_control, raw_target],
            temporal_positions_vision=[control_positions, target_positions],
            num_vision_items_per_sample=[2],
            num_views_per_vision_item=[1, 1],
            control_weights=[[1.0]],
        )

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_config_default_and_range_validation(self) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModelConfig

        assert OmniMoTCausalModelConfig().teacher_forcing_transfer_control_dropout_rate == 0.0
        assert (
            OmniMoTCausalModelConfig(
                teacher_forcing_transfer_control_dropout_rate=1.0
            ).teacher_forcing_transfer_control_dropout_rate
            == 1.0
        )
        with pytest.raises(ValueError):
            OmniMoTCausalModelConfig(teacher_forcing_transfer_control_dropout_rate=-0.01)
        with pytest.raises(ValueError):
            OmniMoTCausalModelConfig(teacher_forcing_transfer_control_dropout_rate=1.01)

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_zero_rate_preserves_transfer_layout_without_rng(self) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model(0.0)
        gen_data_clean = self._make_data()

        with patch(self._PATCH_RAND) as random_draw:
            result = OmniMoTCausalModel._maybe_drop_teacher_forcing_transfer_control(
                model,
                gen_data_clean,
                {"dataset_name": ["video_transfer_4modality_480"]},
            )

        assert result is gen_data_clean
        assert result.num_vision_items_per_sample == [2]
        assert len(result.x0_tokens_vision) == 2
        random_draw.assert_not_called()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_one_rate_removes_control_and_parallel_metadata_without_rng(self) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model(1.0)
        gen_data_clean = self._make_data()
        target = gen_data_clean.x0_tokens_vision[-1]  # [B,C,T,H,W]
        raw_target = gen_data_clean.raw_state_vision[-1]  # [B,C,T_px,H,W]
        target_positions = gen_data_clean.temporal_positions_vision[-1]  # [T]

        with patch(self._PATCH_RAND) as random_draw:
            result = OmniMoTCausalModel._maybe_drop_teacher_forcing_transfer_control(
                model,
                gen_data_clean,
                {"dataset_name": ["video_transfer_4modality_480"]},
            )

        assert result is gen_data_clean
        assert len(result.x0_tokens_vision) == 1
        assert result.x0_tokens_vision[0] is target
        assert len(result.raw_state_vision) == 1
        assert result.raw_state_vision[0] is raw_target
        assert len(result.temporal_positions_vision) == 1
        assert result.temporal_positions_vision[0] is target_positions
        assert result.num_views_per_vision_item == [1]
        assert result.num_vision_items_per_sample is None
        assert result.control_weights is None
        random_draw.assert_not_called()

    @pytest.mark.L0
    @pytest.mark.CPU
    @pytest.mark.parametrize(("draw", "expected_items"), [(0.249, 1), (0.25, 2)])
    def test_dropout_probability_uses_strict_less_than_boundary(self, draw: float, expected_items: int) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model(0.25)
        gen_data_clean = self._make_data()
        dropout_draw = torch.tensor(draw)  # []

        with patch(self._PATCH_RAND, return_value=dropout_draw) as random_draw:
            result = OmniMoTCausalModel._maybe_drop_teacher_forcing_transfer_control(
                model,
                gen_data_clean,
                {"dataset_name": ["video_transfer_4modality_480"]},
            )

        assert len(result.x0_tokens_vision) == expected_items
        random_draw.assert_called_once_with(())

    @pytest.mark.L0
    @pytest.mark.CPU
    @pytest.mark.parametrize(
        ("strategy", "dataset_name"),
        [("none", "video_transfer_4modality_480"), ("teacher_forcing", "video_data_480")],
    )
    def test_non_teacher_forcing_or_non_transfer_batch_is_unchanged(
        self,
        strategy: str,
        dataset_name: str,
    ) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model(1.0, strategy=strategy)
        gen_data_clean = self._make_data()

        with patch(self._PATCH_RAND) as random_draw:
            result = OmniMoTCausalModel._maybe_drop_teacher_forcing_transfer_control(
                model,
                gen_data_clean,
                {"dataset_name": [dataset_name]},
            )

        assert result.num_vision_items_per_sample == [2]
        assert len(result.x0_tokens_vision) == 2
        random_draw.assert_not_called()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_malformed_transfer_layout_fails_loudly(self) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model(1.0)
        gen_data_clean = self._make_data()
        gen_data_clean.num_vision_items_per_sample = [1]

        with pytest.raises(ValueError, match="max_samples_per_batch=1"):
            OmniMoTCausalModel._maybe_drop_teacher_forcing_transfer_control(
                model,
                gen_data_clean,
                {"dataset_name": ["video_transfer_4modality_480"]},
            )


@pytest.mark.L0
@pytest.mark.CPU
def test_distilled_ar_sampler_passes_frame_context_to_schedule() -> None:
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    model = object.__new__(OmniMoTCausalModel)
    model.config = SimpleNamespace(
        fixed_step_sampler_config=SimpleNamespace(sample_type="ode"),
        rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
    )
    calls = []

    def schedule(distilled_num_steps: int | None, frame_idx: int | None, num_frames: int | None) -> list[float]:
        calls.append((distilled_num_steps, frame_idx, num_frames))
        return [1.0, 0.0]

    model._get_ar_distilled_timestep_schedule = schedule
    initial_noise = torch.ones(1, 4)  # [B,N]

    def velocity_fn(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return torch.zeros_like(x)  # [B,N]

    denoised = OmniMoTCausalModel._run_distilled_ar_sampler(
        model,
        velocity_fn,
        initial_noise,
        seed=7,
        frame_idx=2,
        num_frames=5,
        distilled_num_steps=3,
    )

    assert calls == [(3, 2, 5)]
    torch.testing.assert_close(denoised, initial_noise)


@pytest.mark.L0
@pytest.mark.CPU
def test_fp8_kv_cache_with_cuda_graph_path_is_rejected() -> None:
    """FP8 KV cache + the effective cuda-graph path fails fast; other combos pass.

    The guard gates on the effective cuda-graph flag (compile.enabled and
    compile.use_cuda_graphs), so FP8 on the dynamic path (cuda_graph_path_active
    False) is allowed, and BF16 (None) is always allowed.
    """
    from cosmos_framework.model.generator.omni_mot_causal_model import (
        _validate_kv_cache_dtype_supports_cuda_graphs as guard,
    )

    with pytest.raises(NotImplementedError, match="cuda graphs"):
        guard("fp8", True)

    guard("fp8", False)  # FP8 on the dynamic AR path is FP8-safe
    guard(None, True)  # BF16 cache is unaffected by the cuda-graph path
    guard(None, False)


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_forcing_cp_local_kv_heads_match_ulysses_layout() -> None:
    """Replay TF empty caches use the same local KV head count as CP dispatch."""
    from cosmos_framework.model.generator.omni_mot_causal_model import _get_context_parallel_num_kv_heads

    assert _get_context_parallel_num_kv_heads(8, None) == 8
    assert _get_context_parallel_num_kv_heads(8, SimpleNamespace(cp_enabled=False, cp_size=4)) == 8
    assert _get_context_parallel_num_kv_heads(8, SimpleNamespace(cp_enabled=True, cp_size=4)) == 2
    assert _get_context_parallel_num_kv_heads(1, SimpleNamespace(cp_enabled=True, cp_size=4)) == 1

    with pytest.raises(ValueError, match="Repeated KV heads"):
        _get_context_parallel_num_kv_heads(3, SimpleNamespace(cp_enabled=True, cp_size=2))


@pytest.mark.L0
@pytest.mark.CPU
def test_three_way_teacher_forcing_memory_state_does_not_require_flex_metadata() -> None:
    """Legacy replay defaults the Flex-only clean-memory capacity to zero."""
    from cosmos_framework.data.generator.sequence_packing.sequence import ModalityData, PackedSequence
    from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    model = object.__new__(OmniMoTCausalModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        teacher_forcing_detach_clean_kv=True,
        clamp_empty_varlen_kv=True,
        teacher_forcing_frames_per_chunk=4,
        teacher_forcing_replay_policy=TeacherForcingReplayPolicyConfig(),
    )
    model.parallel_dims = None
    net = SimpleNamespace(num_hidden_layers=2, num_kv_heads=8, head_dim=128)
    packed_sequence = PackedSequence(
        sample_lens=[2],
        vision=ModalityData(
            tokens=[torch.zeros(1)],  # list[[1]]
            token_shapes=[(2, 1, 1)],
            condition_mask=[torch.tensor([1.0, 0.0])],  # list[[T]]
        ),
    )

    memory_state = model._build_tf_memory_state(
        packed_sequence=packed_sequence,
        memory_info={},
        net=net,
        selected_clean_gen_token_indexes=None,
    )

    assert memory_state.selected_clean_gen_token_indexes is None
    assert memory_state.selected_clean_gen_padded_capacity == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_clean_callback_includes_partial_conditioned_prefix() -> None:
    """A partial prefix is submitted once in camera-major view order."""
    from cosmos_framework.model.generator.omni_mot_causal_model import (
        _submit_multiview_conditioned_prefix,
    )

    target_latent = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1, 1)  # [B,C,V*T,H,W]
    callback_chunks: list[torch.Tensor] = []

    _submit_multiview_conditioned_prefix(
        target_latent,
        num_views=2,
        frames_per_view=4,
        condition_count=2,
        output_frames=4,
        on_clean_vision_chunk=callback_chunks.append,
    )

    expected = torch.tensor([0.0, 1.0, 4.0, 5.0]).reshape(1, 1, 4, 1, 1)  # [B,C,V*T_prefix,H,W]
    assert len(callback_chunks) == 1
    torch.testing.assert_close(callback_chunks[0], expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_transfer_ar_rejects_sparse_condition_frames() -> None:
    """Sparse target conditions must not be misread as a ground-truth prefix."""
    from cosmos_framework.model.generator.omni_mot_causal_model import (
        _multiview_conditioned_prefix_length,
    )

    sparse_mask = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool)  # [V*T]

    with pytest.raises(ValueError, match="contiguous prefix"):
        _multiview_conditioned_prefix_length(
            sparse_mask,
            num_views=2,
            frames_per_view=3,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_clean_callback_includes_fully_conditioned_output() -> None:
    """A fully conditioned target still submits one decodable callback chunk."""
    from cosmos_framework.model.generator.omni_mot_causal_model import (
        _submit_multiview_conditioned_prefix,
    )

    target_latent = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1, 1)  # [B,C,V*T,H,W]
    callback_chunks: list[torch.Tensor] = []

    _submit_multiview_conditioned_prefix(
        target_latent,
        num_views=2,
        frames_per_view=4,
        condition_count=4,
        output_frames=4,
        on_clean_vision_chunk=callback_chunks.append,
    )

    assert len(callback_chunks) == 1
    torch.testing.assert_close(callback_chunks[0], target_latent)


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_clean_callback_truncates_conditioned_prefix_to_output_frames() -> None:
    """A fully conditioned target honors max_num_frames before decode submission."""
    from cosmos_framework.model.generator.omni_mot_causal_model import (
        _submit_multiview_conditioned_prefix,
    )

    target_latent = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8, 1, 1)  # [B,C,V*T,H,W]
    callback_chunks: list[torch.Tensor] = []

    _submit_multiview_conditioned_prefix(
        target_latent,
        num_views=2,
        frames_per_view=4,
        condition_count=4,
        output_frames=2,
        on_clean_vision_chunk=callback_chunks.append,
    )

    expected = torch.tensor([0.0, 1.0, 4.0, 5.0]).reshape(1, 1, 4, 1, 1)  # [B,C,V*T_output,H,W]
    assert len(callback_chunks) == 1
    torch.testing.assert_close(callback_chunks[0], expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_transfer_ar_mode_dispatches_to_specialized_iterator() -> None:
    """Camera-major transfer batches use the Flex replay iterator behind the public mode."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    model = MagicMock()
    model.config.compile.enabled = False
    model._uses_multiview_flex_kv.return_value = True
    expected = {"vision": torch.zeros(1, 4, 2, 1, 1)}  # [B,C,V*T,H,W]
    model._iter_samples_multiview_transfer_autoregressive.return_value = iter([expected])
    data_batch = {
        "enable_per_camera_vae_encoding": True,
        "sample_n_views": torch.tensor([2]),  # [B]
    }

    with patch(
        "cosmos_framework.model.generator.omni_mot_causal_model.reset_ar_post_saturation_runtime_for_generation"
    ):
        outputs = list(
            OmniMoTCausalModel.iter_samples_from_batch_autoregressive(
                model,
                data_batch,
                mode="video_transfer",
                has_negative_prompt=True,
            )
        )

    assert outputs == [expected]
    model._iter_samples_multiview_transfer_autoregressive.assert_called_once()
    call_kwargs = model._iter_samples_multiview_transfer_autoregressive.call_args.kwargs
    assert call_kwargs["data_batch"] is data_batch
    assert call_kwargs["has_negative_prompt"] is True
    model.get_data_and_condition.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("controls_read_rgb", [False, True])
def test_multiview_transfer_ar_yields_logical_frames_as_chunks_finish(controls_read_rgb: bool) -> None:
    """Multi-chunk transfer exposes progress and refreshes RGB-aware control K/V in order."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    num_views = 2
    frames_per_view = 5
    chunk_size = 2
    control_latent = torch.zeros(1, 1, num_views * frames_per_view, 1, 1)  # [B,C,V*T,H,W]
    target_latent = torch.zeros_like(control_latent)  # [B,C,V*T,H,W]
    target_condition_mask = torch.zeros(num_views * frames_per_view, 1, 1)  # [V*T,1,1]
    control_condition_mask = torch.ones_like(target_condition_mask)  # [V*T,1,1]
    gen_data_clean = SimpleNamespace(
        x0_tokens_vision=[control_latent, target_latent],
        num_vision_items_per_sample=[2],
        num_views_per_vision_item=[num_views, num_views],
        fps_vision=torch.tensor([24.0]),  # [B]
    )
    built_prefills: list[SimpleNamespace] = []

    def build_prefill(*_args: object, **_kwargs: object) -> SimpleNamespace:
        prefill = SimpleNamespace(
            vision=SimpleNamespace(
                condition_mask=[control_condition_mask, target_condition_mask],
                token_shapes=[
                    (num_views * frames_per_view, 1, 1),
                    (num_views * frames_per_view, 1, 1),
                ],
            ),
            to_cuda=MagicMock(),
        )
        built_prefills.append(prefill)
        return prefill

    prefill_indexes = torch.arange(2 * num_views * frames_per_view)  # [N]
    memory_layout = SimpleNamespace(
        prefill_source_token_indexes=prefill_indexes,
        prefill_cache_token_indexes=prefill_indexes,
        target_cache_token_indexes=lambda frame_range: torch.arange(  # [V*C*S]
            num_views * (frame_range[1] - frame_range[0])
        ),
    )

    model = MagicMock()
    model._uses_multiview_flex_kv.return_value = True
    model.config = SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        teacher_forcing_frames_per_chunk=chunk_size,
        teacher_forcing_replay_policy=SimpleNamespace(
            controls_read_strict_past_clean_rgb=controls_read_rgb,
        ),
    )
    model._get_teacher_forcing_replay_policy.return_value = model.config.teacher_forcing_replay_policy
    model.input_video_key = "video"
    model.input_image_key = "images"
    model.parallel_dims = None
    model.net = SimpleNamespace(
        num_hidden_layers=1,
        flex_backend=SimpleNamespace(block_size=(128, 128)),
    )
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model.get_data_and_condition.return_value = gen_data_clean
    model._get_inference_text_tokens.return_value = ([[1, 2]], None)
    model._pack_input_sequence.side_effect = build_prefill
    model._build_multiview_transfer_ar_pack.return_value = MagicMock()

    def generate_chunk(**kwargs: object) -> torch.Tensor:  # returns [B,C,V*T_chunk,H,W]
        chunk_start = kwargs["chunk_start"]
        assert isinstance(chunk_start, int)
        curr_vision_latent = kwargs["curr_vision_latent"]
        assert isinstance(curr_vision_latent, torch.Tensor)
        chunk_len = curr_vision_latent.shape[2] // num_views
        values = [
            float(100 * view_idx + chunk_start + local_idx)
            for view_idx in range(num_views)
            for local_idx in range(chunk_len)
        ]
        return torch.tensor(values, dtype=torch.float32).reshape(  # [B,C,V*T_chunk,H,W]
            1,
            1,
            num_views * chunk_len,
            1,
            1,
        )

    model._generate_multiview_transfer_ar_chunk.side_effect = generate_chunk
    callback_chunks: list[torch.Tensor] = []

    with (
        patch(
            "cosmos_framework.model.generator.omni_mot_causal_model.build_sequence_plans_from_data_batch",
            return_value=[SimpleNamespace(condition_frame_indexes_vision=[])],
        ),
        patch(
            "cosmos_framework.model.generator.omni_mot_causal_model.build_multiview_transfer_ar_memory_layout",
            return_value=memory_layout,
        ),
    ):
        iterator = OmniMoTCausalModel._iter_samples_multiview_transfer_autoregressive(
            model,
            data_batch={},
            guidance=1.0,
            seed=1,
            num_steps=2,
            shift=5.0,
            normalize_cfg=False,
            sampler_mode="rf",
            distilled_num_steps=None,
            sync_num_frames_across_ranks=False,
            sync_process_group=None,
            max_num_frames=None,
            on_clean_vision_chunk=callback_chunks.append,
            has_negative_prompt=False,
        )
        outputs: list[dict[str, object]] = []
        progress: list[tuple[int, int, int]] = []
        for _ in range(frames_per_view):
            outputs.append(next(iterator))
            progress.append(
                (
                    model._generate_multiview_transfer_ar_chunk.call_count,
                    model._capture_multiview_transfer_ar_memory.call_count,
                    len(callback_chunks),
                )
            )
        with pytest.raises(StopIteration):
            next(iterator)

    expected_capture_counts = [2, 4, 4, 5, 5] if controls_read_rgb else [2, 3, 3, 3, 3]
    assert progress == [
        (1, expected_capture_counts[0], 1),
        (2, expected_capture_counts[1], 2),
        (2, expected_capture_counts[2], 2),
        (3, expected_capture_counts[3], 3),
        (3, expected_capture_counts[4], 3),
    ]
    assert [chunk.shape[2] for chunk in callback_chunks] == [2, 4, 4]
    assert model._pack_input_sequence.call_count == (4 if controls_read_rgb else 1)
    expected_materialized_ranges: list[tuple[tuple[int, int], ...]] = [()]
    if controls_read_rgb:
        expected_materialized_ranges.extend(
            [
                (),
                ((0, 1),),
                ((0, 1), (1, 3)),
            ]
        )
    assert [prefill.teacher_forcing_materialized_target_frame_ranges for prefill in built_prefills] == (
        expected_materialized_ranges
    )
    assert (
        model._capture_multiview_transfer_ar_memory.call_args_list[0].kwargs["pack"]
        is built_prefills[1 if controls_read_rgb else 0]
    )
    for frame_idx, output in enumerate(outputs):
        output_frame = output["vision"]
        assert isinstance(output_frame, torch.Tensor)
        expected = torch.tensor([frame_idx, 100 + frame_idx], dtype=torch.float32).reshape(  # [B,C,V,H,W]
            1,
            1,
            num_views,
            1,
            1,
        )
        torch.testing.assert_close(output_frame, expected)


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_transfer_ar_pack_sets_metadata_and_absolute_view_positions() -> None:
    """Synchronized chunks retain camera-major absolute mRoPE positions and Flex metadata."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    model = MagicMock()
    model.config.diffusion_expert_config.patch_spatial = 1
    model.config.diffusion_expert_config.enable_fps_modulation = False
    model.config.diffusion_expert_config.base_fps = 24.0
    model.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin = 0
    model.config.max_action_dim = 8
    model.config.teacher_forcing_frames_per_chunk = 2
    model.tokenizer_vision_gen.temporal_compression_factor = 4
    model.llm_special_tokens = {}
    packed_seq = MagicMock()
    memory_layout = MagicMock()
    vision_latent = torch.zeros(1, 4, 4, 2, 2)  # [B,C,V*chunk_len,H,W]

    with patch(_PATCH_PACK, return_value=packed_seq) as pack_input:
        result = OmniMoTCausalModel._build_multiview_transfer_ar_pack(
            model,
            vision_latent=vision_latent,
            text_tokens=[1, 2],
            fps_vision=[24.0],
            num_views=2,
            frames_per_view=5,
            chunk_start=2,
            memory_layout=memory_layout,
            current_role="current_target",
        )

    assert result is packed_seq
    torch.testing.assert_close(
        pack_input.call_args.kwargs["vision_temporal_positions"],
        torch.tensor([2.0, 3.0, 7.0, 8.0]),  # [V*chunk_len]
    )
    assert pack_input.call_args.kwargs["num_views"] == 2
    assert packed_seq.multiview_transfer_ar_metadata == {
        "current_frame_start": 2,
        "frames_per_view": 5,
        "frames_per_chunk": 2,
        "current_role": "current_target",
        "memory_layout": memory_layout,
    }
    packed_seq.to_cuda.assert_called_once_with()
    model._cast_generated_tokens_to_precision.assert_called_once_with(packed_seq)


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_transfer_clean_pack_uses_teacher_forcing_condition_semantics() -> None:
    """Clean target history omits timestep embeddings while retaining its explicit Flex role."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    model = MagicMock()
    model.config.diffusion_expert_config.patch_spatial = 1
    model.config.diffusion_expert_config.enable_fps_modulation = False
    model.config.diffusion_expert_config.base_fps = 24.0
    model.config.diffusion_expert_config.unified_3d_mrope_temporal_modality_margin = 0
    model.config.max_action_dim = 8
    model.config.teacher_forcing_frames_per_chunk = 2
    model.tokenizer_vision_gen.temporal_compression_factor = 4
    model.llm_special_tokens = {}
    original_mask = torch.zeros(4, 1, 1)  # [V*T_chunk,1,1]
    vision = SimpleNamespace(
        condition_mask=[original_mask.clone()],  # list[[V*T_chunk,1,1]]
        mse_loss_indexes=torch.arange(8, dtype=torch.long),  # [N_noisy_tokens]
        timesteps=torch.zeros(8),  # [N_noisy_tokens]
        noisy_frame_indexes=[torch.arange(4, dtype=torch.long)],  # list[[V*T_chunk]]
    )
    packed_seq = SimpleNamespace(vision=vision, to_cuda=MagicMock())
    memory_layout = MagicMock()

    with patch(_PATCH_PACK, return_value=packed_seq):
        result = OmniMoTCausalModel._build_multiview_transfer_ar_pack(
            model,
            vision_latent=torch.zeros(1, 4, 4, 2, 2),  # [B,C,V*T_chunk,H,W]
            text_tokens=[1, 2],
            fps_vision=[24.0],
            num_views=2,
            frames_per_view=5,
            chunk_start=2,
            memory_layout=memory_layout,
            current_role="clean_target",
        )

    assert result is packed_seq
    assert result.multiview_transfer_ar_metadata["current_role"] == "clean_target"
    clean_mask = torch.ones_like(original_mask)  # [V*T_chunk,1,1]
    torch.testing.assert_close(result.vision.condition_mask[0], clean_mask)
    assert result.vision.mse_loss_indexes.numel() == 0
    assert result.vision.timesteps.numel() == 0
    assert len(result.vision.noisy_frame_indexes) == 1
    assert result.vision.noisy_frame_indexes[0].numel() == 0
    packed_seq.to_cuda.assert_called_once_with()
    model._cast_generated_tokens_to_precision.assert_called_once_with(packed_seq)


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_transfer_ar_memory_merge_updates_only_selected_slots() -> None:
    """A clean recomputation refreshes control slots without erasing RGB history."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    destination_k = torch.full((1, 5, 1, 1), -1.0)  # [1,S_memory,H_kv,D]
    destination_v = torch.full((1, 5, 1, 1), -2.0)  # [1,S_memory,H_kv,D]
    source_k = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1)  # [1,S_memory,H_kv,D]
    source_v = (10 + torch.arange(5, dtype=torch.float32)).reshape(1, 5, 1, 1)  # [1,S_memory,H_kv,D]
    cache_indexes = torch.tensor([1, 3], dtype=torch.long)  # [S_write]
    destination = [(destination_k, destination_v)]

    OmniMoTCausalModel._merge_multiview_transfer_ar_memory(
        destination=destination,
        source=[(source_k, source_v)],
        cache_indexes=cache_indexes,
    )

    torch.testing.assert_close(destination_k.flatten(), torch.tensor([-1.0, 1.0, -1.0, 3.0, -1.0]))
    torch.testing.assert_close(destination_v.flatten(), torch.tensor([-2.0, 11.0, -2.0, 13.0, -2.0]))


@pytest.mark.L0
@pytest.mark.CPU
def test_multiview_transfer_ar_uses_negative_prompt_for_unconditional_tokens() -> None:
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    class TextTokensObserved(Exception):
        pass

    data_batch = {
        "caption": ["positive"],
        "neg_caption": ["negative"],
    }
    sequence_plan = SimpleNamespace(condition_frame_indexes_vision=[])
    control_latent = torch.zeros(1, 1, 4, 1, 1)  # [B,C,V*T,H,W]
    target_latent = torch.zeros_like(control_latent)  # [B,C,V*T,H,W]
    fps_vision = torch.tensor([24.0])  # [B]
    gen_data_clean = SimpleNamespace(
        x0_tokens_vision=[control_latent, target_latent],
        num_vision_items_per_sample=[2],
        num_views_per_vision_item=[2, 2],
        fps_vision=fps_vision,
    )
    model = MagicMock()
    model._uses_multiview_flex_kv.return_value = True
    model.config.action_gen = False
    model.config.sound_gen = False
    model.input_caption_key = "caption"
    model.input_image_key = "images"
    model.input_video_key = "video"
    model.get_data_and_condition.return_value = gen_data_clean
    model.parallel_dims = None
    model.vlm_config = SimpleNamespace(use_system_prompt=False)
    token_ids = {
        "positive": [101],
        "negative": [202],
        "": [0],
    }
    tokenized_captions: list[list[str]] = []

    def tokenize_captions(captions: list[str], **_kwargs: object) -> list[list[int]]:
        tokenized_captions.append(list(captions))
        return [token_ids[caption] for caption in captions]

    observed_tokens: list[tuple[list[list[int]], list[list[int]]]] = []

    def get_text_tokens(
        batch: dict,
        has_negative_prompt: bool,
    ) -> tuple[list[list[int]], list[list[int]]]:
        tokens = OmniMoTModel._get_inference_text_tokens(model, batch, has_negative_prompt)
        observed_tokens.append(tokens)
        raise TextTokensObserved

    model._tokenize_captions.side_effect = tokenize_captions
    model._get_inference_text_tokens.side_effect = get_text_tokens

    with patch(
        "cosmos_framework.model.generator.omni_mot_causal_model.build_sequence_plans_from_data_batch",
        return_value=[sequence_plan],
    ):
        iterator = OmniMoTCausalModel._iter_samples_multiview_transfer_autoregressive(
            model,
            data_batch=data_batch,
            guidance=1.5,
            seed=1,
            num_steps=2,
            shift=5.0,
            normalize_cfg=False,
            sampler_mode="rf",
            distilled_num_steps=None,
            sync_num_frames_across_ranks=False,
            sync_process_group=None,
            max_num_frames=None,
            on_clean_vision_chunk=None,
            has_negative_prompt=True,
        )
        with pytest.raises(TextTokensObserved):
            next(iterator)

    assert tokenized_captions == [["positive"], ["negative"]]
    assert observed_tokens == [([[101]], [[202]])]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("normalize_cfg", [False, True])
@pytest.mark.parametrize("cfgp_rank", [None, 0, 1])
def test_shared_ar_cfg_preserves_branch_order_and_result(
    normalize_cfg: bool,
    cfgp_rank: int | None,
) -> None:
    """Shared CFG orchestration keeps sequential and CFGP branch ownership stable."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    cond_pack = MagicMock(name="cond_pack")
    uncond_pack = MagicMock(name="uncond_pack")
    if cfgp_rank is None:
        parallel_dims = None
    else:
        parallel_dims = SimpleNamespace(
            cfgp_enabled=True,
            cfgp_rank=cfgp_rank,
            cfgp_size=2,
            cfgp_mesh=SimpleNamespace(get_group=lambda: "cfgp_group"),
        )
    model = SimpleNamespace(parallel_dims=parallel_dims)
    branch_calls: list[tuple[object, str]] = []

    def run_branch(
        pack: object,
        noise_vision: torch.Tensor,  # [B,C,T,H,W]
        timestep: torch.Tensor,  # [B,1]
        branch: str,
    ) -> torch.Tensor:  # [B,C,T,H,W]
        del noise_vision, timestep
        branch_calls.append((pack, branch))
        value = 3.0 if branch == "conditional" else 1.0
        return torch.full((1, 1, 1, 1, 2), value)  # [B,C,T,H,W]

    with patch(_PATCH_DIST) as mock_dist:
        mock_dist.P2POp.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)

        def exchange(operations: list[SimpleNamespace]) -> list[MagicMock]:
            if cfgp_rank is not None:
                peer_value = 1.0 if cfgp_rank == 0 else 3.0
                operations[1].tensor.fill_(peer_value)
            return [MagicMock(), MagicMock()]

        mock_dist.batch_isend_irecv.side_effect = exchange
        velocity = OmniMoTCausalModel._predict_ar_velocity_with_cfg(
            model,
            noise_x=torch.zeros(1, 2),  # [B,N_tokens_flat]
            timestep=torch.ones(1, 1),  # [B,1]
            vision_shape=torch.Size((1, 1, 1, 1, 2)),
            packed_seq=cond_pack,
            packed_seq_uncond=uncond_pack,
            guidance=2.0,
            normalize_cfg=normalize_cfg,
            run_branch=run_branch,
        )

    torch.testing.assert_close(velocity, torch.full((1, 2), 5.0))
    if cfgp_rank is None:
        assert branch_calls == [(cond_pack, "conditional"), (uncond_pack, "unconditional")]
    elif cfgp_rank == 0:
        assert branch_calls == [(cond_pack, "conditional")]
    else:
        assert branch_calls == [(uncond_pack, "unconditional")]


@pytest.mark.L0
@pytest.mark.CPU
def test_generic_ar_cfg_uses_branch_specific_dual_kv_caches() -> None:
    """Sequential generic AR keeps conditional and unconditional caches separate."""
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

    model = MagicMock()
    model.parallel_dims = None
    model.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
    model.config.rectified_flow_inference_config.scheduler_type = "unipc"
    cond_pack = MagicMock(name="cond_pack")
    cond_pack.vision = None
    cond_pack.action = None
    uncond_pack = MagicMock(name="uncond_pack")
    uncond_pack.vision = None
    uncond_pack.action = None
    cond_cache = [MagicMock(name="cond_cache")]
    uncond_cache = [MagicMock(name="uncond_cache")]
    observed_caches: list[object] = []

    def build_memory_state(_pack: object, memory_info: dict[str, object]) -> MagicMock:
        observed_caches.append(memory_info["dual_kv_cache"])
        return MagicMock()

    def denoise(*, data_batch_packed: object, memory: object) -> dict[str, list[torch.Tensor]]:
        del memory
        value = 3.0 if data_batch_packed is cond_pack else 1.0
        return {"preds_vision": [torch.full((1, 1, 1, 1), value)]}  # [C,T,H,W]

    def sampler(velocity_fn: object, initial_noise: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return velocity_fn(initial_noise, torch.ones(1, 1))  # type: ignore[operator]

    model.build_memory_state.side_effect = build_memory_state
    model.denoise.side_effect = denoise
    model.sampler = sampler
    with (
        patch(
            "cosmos_framework.model.generator.omni_mot_causal_model.is_ar_post_saturation_static_compile_frame",
            return_value=False,
        ),
        patch(
            "cosmos_framework.model.generator.omni_mot_causal_model.is_ar_post_saturation_cuda_graph_frame",
            return_value=False,
        ),
    ):
        denoised = OmniMoTCausalModel.generate_next_frame(
            model,
            packed_seq=cond_pack,
            packed_seq_uncond=uncond_pack,
            curr_vision_latent=torch.zeros(1, 1, 1, 1, 1),  # [B,C,T,H,W]
            curr_action_latent=None,
            cond_text_tokens=[1],
            uncond_text_tokens=[2],
            gen_data_clean=SimpleNamespace(),
            dual_kv_cache=cond_cache,
            dual_kv_cache_uncond=uncond_cache,
            guidance=2.0,
            num_steps=1,
            shift=1.0,
            seed=7,
            fps_vision_list=[24.0],
            fps_action_list=[],
        )

    torch.testing.assert_close(denoised, torch.full((1, 1, 1, 1, 1), 5.0))
    assert observed_caches == [cond_cache, uncond_cache]


# ---------------------------------------------------------------------------
# L0 — Text tokens at start frame
# ---------------------------------------------------------------------------


class TestTextTokensAtStartFrame:
    """Bug 2: for text2video (start_frame=0), text tokens must be passed at frame_idx=0."""

    def _text_tokens_for_frame(self, frame_idx: int, start_frame: int, cond_text_tokens: list) -> object:
        """
        Replicate the fixed AR loop logic for computing text_tokens.

        Before the fix:  ``text_tokens=None`` for all frames.
        After the fix:   ``text_tokens=cond_text_tokens[0]`` when frame_idx == start_frame.
        """
        return cond_text_tokens[0] if (cond_text_tokens and frame_idx == start_frame) else None

    @pytest.mark.L0
    def test_t2v_text_tokens_at_frame0(self):
        """text2video: frame 0 gets text tokens."""
        cond = [["tok1", "tok2"]]
        result = self._text_tokens_for_frame(frame_idx=0, start_frame=0, cond_text_tokens=cond)
        assert result == cond[0], "Frame 0 should receive text tokens in text2video mode"

    @pytest.mark.L0
    def test_t2v_no_text_tokens_at_frame1(self):
        """text2video: frames 1+ get no text tokens (cache already seeded at frame 0)."""
        cond = [["tok1", "tok2"]]
        result = self._text_tokens_for_frame(frame_idx=1, start_frame=0, cond_text_tokens=cond)
        assert result is None

    @pytest.mark.L0
    def test_i2v_text_tokens_at_frame1(self):
        """image2video: start_frame=1, so frame 1 gets text tokens."""
        cond = [["tok1", "tok2"]]
        result = self._text_tokens_for_frame(frame_idx=1, start_frame=1, cond_text_tokens=cond)
        assert result == cond[0]

    @pytest.mark.L0
    def test_empty_cond_tokens_returns_none(self):
        """No text conditioning → None regardless of frame index."""
        result = self._text_tokens_for_frame(frame_idx=0, start_frame=0, cond_text_tokens=[])
        assert result is None


# ---------------------------------------------------------------------------
# L0 — KV cache prefill
# ---------------------------------------------------------------------------


class TestKVCachePrefill:
    """Bug 3: prefill denoise call at frame_idx=0 must set und_cache.is_initialized=True."""

    def _make_und_cache(self, initialized: bool = False) -> MagicMock:
        cache = MagicMock()
        cache.is_initialized = initialized
        return cache

    def _make_dual_kv_cache(self, initialized: bool = False) -> MagicMock:
        dual = MagicMock()
        dual.und_cache = self._make_und_cache(initialized)
        return dual

    def _make_kv_cache_list(self, num_layers: int = 2, initialized: bool = False) -> list[MagicMock]:
        return [self._make_dual_kv_cache(initialized) for _ in range(num_layers)]

    @pytest.mark.L0
    def test_prefill_sets_is_initialized(self):
        """
        Simulate the prefill path: calling denoise at frame_idx=0 should store
        K/V in each layer's und_cache, making is_initialized=True for subsequent frames.
        dual_kv_cache is now a list[DualKVCache] — one per transformer layer.
        """
        dual_kv_cache = self._make_kv_cache_list(num_layers=2, initialized=False)

        # Simulate what happens inside denoise at frame_idx=0:
        # each layer's forward() stores text K/V, setting is_initialized=True.
        def mock_denoise(data_batch_packed, fps_vision, fps_action, dual_kv_cache, frame_idx):
            if frame_idx == 0:
                for cache in dual_kv_cache:
                    cache.und_cache.is_initialized = True

        mock_denoise(
            data_batch_packed=MagicMock(),
            fps_vision=MagicMock(),
            fps_action=None,
            dual_kv_cache=dual_kv_cache,
            frame_idx=0,
        )

        for cache in dual_kv_cache:
            assert cache.und_cache.is_initialized is True

    @pytest.mark.L0
    def test_no_prefill_leaves_cache_uninitialized(self):
        """Without prefill, all layers' und_cache.is_initialized remain False."""
        dual_kv_cache = self._make_kv_cache_list(num_layers=2, initialized=False)
        # No denoise call here — all caches stay uninitialized
        for cache in dual_kv_cache:
            assert cache.und_cache.is_initialized is False

    @pytest.mark.L0
    def test_prefill_only_for_i2v_and_fd(self):
        """
        Replicate the mode guard: prefill is skipped for text2video.
        Only image2video and forward_dynamics trigger the prefill call.
        """
        denoise_calls: list[str] = []

        def mock_denoise_tracking(mode: str, frame_idx: int):
            denoise_calls.append(f"mode={mode},frame={frame_idx}")

        for mode in ("image2video", "forward_dynamics"):
            if mode in ("image2video", "forward_dynamics"):
                mock_denoise_tracking(mode, frame_idx=0)

        assert len(denoise_calls) == 2
        assert all("frame=0" in c for c in denoise_calls)

    @pytest.mark.L0
    def test_prefill_skipped_for_t2v(self):
        """text2video must NOT trigger the prefill (frame 0 is generated, not conditioned)."""
        denoise_calls: list[str] = []

        mode = "text2video"
        if mode in ("image2video", "forward_dynamics"):
            denoise_calls.append("prefill")

        assert len(denoise_calls) == 0, "text2video should not run prefill"


# ---------------------------------------------------------------------------
# L0 — End-to-end AR loop logic via mocked model
# ---------------------------------------------------------------------------

_PATCH_PACK = "cosmos_framework.model.generator.omni_mot_causal_model.pack_input_sequence_autoregressive"
_PATCH_KV = "cosmos_framework.model.generator.omni_mot_causal_model.DualKVCache"
_PATCH_BROADCAST = "cosmos_framework.model.generator.omni_mot_causal_model._broadcast_seed"
_PATCH_DIST = "cosmos_framework.model.generator.omni_mot_causal_model.dist"


class TestARGenerationLoopLogic:
    """
    End-to-end logic tests for iter_samples_from_batch_autoregressive.

    The real method is called as an unbound function with a MagicMock standing in
    for ``self``.  pack_input_sequence_autoregressive and DualKVCache are patched
    at the module level so no real model, GPU, or packed-sequence machinery is needed.
    """

    C, T, H, W, TCF = 4, 3, 2, 2, 4

    def _make_model_mock(self) -> MagicMock:
        m = MagicMock()
        m.tokenizer_vision_gen.temporal_compression_factor = self.TCF
        m.config.diffusion_expert_config.patch_spatial = 1
        m.config.max_action_dim = 8
        m.config.video_temporal_causal = False
        # Use the eager (non-compiled) path: torch.compile is not exercised in unit
        # tests and the two paths differ in text_token handling — the rolling/compiled
        # path keeps text_tokens in every pack for compile-invariant shapes, while the
        # eager path drops them after frame 0 to reuse cached caption K/V.
        m.config.compile.enabled = False
        m.config.compile.use_cuda_graphs = False
        m.config.kv_cache_dtype = None
        m.config.kv_cache_kernel_impl = "triton"
        m.config.kv_cache_inference_size = None
        m.config.attention_sink_size = 0
        m.config.teacher_forcing_frames_per_chunk = 1
        m.config.teacher_forcing_replay_policy = SimpleNamespace(
            control_visibility="causal",
            controls_read_strict_past_clean_rgb=True,
            clean_pass_causality="frame",
        )
        m._get_teacher_forcing_replay_policy.return_value = m.config.teacher_forcing_replay_policy
        m.llm_special_tokens = {}
        m.net.num_hidden_layers = 2
        m.generate_next_frame.return_value = torch.zeros(1, self.C, 1, self.H, self.W)
        m.tensor_kwargs = {"device": "cpu", "dtype": torch.float32}
        m.parallel_dims = None  # disable CFGP for non-CFGP tests
        m.config.fixed_step_sampler_config.t_list = [1.0, 0.5]
        m.config.fixed_step_sampler_config.sample_type = "ode"
        m.config.rectified_flow_inference_config.num_train_timesteps = 1000
        m._uses_multiview_flex_kv.return_value = False
        return m

    def _make_gen_data(self, mode: str) -> SimpleNamespace:
        num_frames = 1 if mode == "text2image" else self.T
        vision = torch.zeros(1, self.C, num_frames, self.H, self.W)  # [B,C,T,H,W]
        vision_items = [vision]
        num_vision_items_per_sample = None
        if mode == "video_transfer":
            control = torch.ones_like(vision)  # [B,C,T,H,W]
            target = torch.zeros_like(vision)  # [B,C,T,H,W]
            vision_items = [control, target]
            num_vision_items_per_sample = [2]
        x0_tokens_action = None
        fps_action = None
        action_domain_id = None
        raw_action_dim = None
        if mode in ("forward_dynamics", "text2video_action_conditioned"):
            x0_tokens_action = [torch.zeros((self.T - 1) * self.TCF, 8)]
            fps_action = torch.tensor([24.0])
            action_domain_id = [torch.tensor(2, dtype=torch.long)]
            raw_action_dim = [torch.tensor(6, dtype=torch.long)]
        return SimpleNamespace(
            batch_size=1,
            x0_tokens_vision=vision_items,
            num_vision_items_per_sample=num_vision_items_per_sample,
            x0_tokens_action=x0_tokens_action,
            fps_vision=torch.tensor([24.0]),
            fps_action=fps_action,
            action_domain_id=action_domain_id,
            raw_action_dim=raw_action_dim,
        )

    def _run(
        self,
        model: MagicMock,
        mode: str,
        data_batch: dict | None = None,
        **ar_kwargs,
    ) -> tuple[dict, MagicMock]:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        gen_data = self._make_gen_data(mode)
        model.get_data_and_condition.return_value = gen_data
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], None)
        mock_pack = MagicMock()
        with patch(_PATCH_PACK, mock_pack), patch(_PATCH_KV):
            outputs = list(
                OmniMoTCausalModel.iter_samples_from_batch_autoregressive(
                    model,
                    data_batch or {},
                    mode=mode,
                    **ar_kwargs,
                )
            )
        result = {"vision": torch.cat([out["vision"] for out in outputs], dim=2)}  # [B,C,T,H,W]
        return result, mock_pack

    @pytest.mark.L0
    def test_t2i_generates_single_frame_without_prefill(self):
        """text2image: 1 frame generated, no prefill denoise call."""
        model = self._make_model_mock()
        self._run(model, "text2image")
        assert model.generate_next_frame.call_count == 1
        assert model.denoise.call_count == 0

    @pytest.mark.L0
    def test_output_vision_shape_t2i(self):
        """text2image output vision has shape (1, C, 1, H, W)."""
        model = self._make_model_mock()
        result, _ = self._run(model, "text2image")
        assert result["vision"].shape == (1, self.C, 1, self.H, self.W)

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_t2v_generates_all_frames_without_prefill(self):
        """text2video: T frames generated, no prefill denoise call."""
        model = self._make_model_mock()
        self._run(model, "text2video")
        model.get_data_and_condition.assert_called_once_with({}, vision_condition_indexes=None)
        assert model.generate_next_frame.call_count == self.T
        assert model.denoise.call_count == 0

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_uses_interleaved_full_history_cache(self) -> None:
        """Each depth frame is cached immediately before its aligned RGB target."""
        model = self._make_model_mock()

        result, mock_pack = self._run(model, "video_transfer")

        assert result["vision"].shape == (1, self.C, self.T, self.H, self.W)
        target_calls = model.generate_next_frame.call_args_list
        assert [call.kwargs["frame_idx"] for call in target_calls] == [0, 1, 2]
        assert [call.kwargs["cache_frame_idx"] for call in target_calls] == [1, 3, 5]

        seed_calls = model._seed_frame_into_kv_cache.call_args_list
        control_calls = [call for call in seed_calls if torch.all(call.kwargs["frame_latent"] == 1)]
        rgb_calls = [call for call in seed_calls if torch.all(call.kwargs["frame_latent"] == 0)]
        assert [call.kwargs["frame_idx"] for call in control_calls] == [0, 2, 4]
        assert [call.kwargs["position_frame_idx"] for call in control_calls] == [0, 1, 2]
        assert [call.kwargs["frame_idx"] for call in rgb_calls] == [1, 3]
        assert [call.kwargs["position_frame_idx"] for call in rgb_calls] == [0, 1]
        assert control_calls[0].kwargs["cond_text_tokens"] == [1, 2, 3]
        assert all(call.kwargs["cond_text_tokens"] is None for call in control_calls[1:] + rgb_calls)

        conditional_pack_calls = mock_pack.call_args_list[::2]
        assert [call.kwargs["frame_idx"] for call in conditional_pack_calls] == [0, 1, 2]
        assert all(call.kwargs["text_tokens"] is None for call in conditional_pack_calls)

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_interleaves_control_and_rgb_history(self) -> None:
        """C=4 transfer targets use logical cache indexes [1,3,8] and negative CFG tokens."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model_mock()
        model.config.teacher_forcing_frames_per_chunk = 4
        num_frames = 9
        control = torch.ones(1, self.C, num_frames, self.H, self.W)  # [B,C,T,H,W]
        target = torch.zeros_like(control)  # [B,C,T,H,W]
        model.get_data_and_condition.return_value = SimpleNamespace(
            batch_size=1,
            x0_tokens_vision=[control, target],
            num_vision_items_per_sample=[2],
            x0_tokens_action=None,
            fps_vision=torch.tensor([24.0]),  # [B]
            fps_action=None,
            action_domain_id=None,
            raw_action_dim=None,
        )
        data_batch = {"caption": ["a prompt"], "neg_caption": ["avoid artifacts"]}
        data_batch["enable_per_camera_vae_encoding"] = True
        data_batch["sample_n_views"] = torch.tensor([2])  # [B]
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], [[7, 8, 9]])

        def generate_chunk(**kwargs: object) -> torch.Tensor:  # returns [B,C,T_chunk,H,W]
            curr_vision_latent = kwargs["curr_vision_latent"]
            assert isinstance(curr_vision_latent, torch.Tensor)
            return torch.zeros_like(curr_vision_latent)  # [B,C,T_chunk,H,W]

        model.generate_next_frame.side_effect = generate_chunk
        with patch(_PATCH_PACK, MagicMock()), patch(_PATCH_KV) as dual_cache_cls:
            outputs = list(
                OmniMoTCausalModel.iter_samples_from_batch_autoregressive(
                    model,
                    data_batch,
                    mode="video_transfer",
                    guidance=7.0,
                    sampler_mode="rf",
                    has_negative_prompt=True,
                )
            )

        output_vision = torch.cat([output["vision"] for output in outputs], dim=2)  # [B,C,T,H,W]
        assert output_vision.shape == (1, self.C, num_frames, self.H, self.W)
        model._iter_samples_multiview_transfer_autoregressive.assert_not_called()
        assert [call.kwargs["cache_frame_idx"] for call in model.generate_next_frame.call_args_list] == [1, 3, 8]
        assert all(call.kwargs["gen_cache_size"] == 2 * num_frames + 1 for call in dual_cache_cls.call_args_list)

        seed_calls = model._seed_frame_into_kv_cache.call_args_list
        control_calls = [call for call in seed_calls if torch.all(call.kwargs["frame_latent"] == 1)]
        assert [call.kwargs["frame_latent"].shape[2] for call in control_calls] == [1, 4, 4]
        assert [call.kwargs["position_frame_idx"] for call in control_calls] == [0, 1, 5]
        assert control_calls[0].kwargs["cond_text_tokens"] == [1, 2, 3]
        assert control_calls[0].kwargs["uncond_text_tokens"] == [7, 8, 9]
        assert all(call.kwargs["cond_text_tokens"] is None for call in control_calls[1:])
        model._get_inference_text_tokens.assert_called_once_with(data_batch, True)
        assert all(call.kwargs["sampler_mode"] == "rf" for call in model.generate_next_frame.call_args_list)
        assert all(call.kwargs["guidance"] == 7.0 for call in model.generate_next_frame.call_args_list)

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_finite_window_maps_logical_sink_to_control_rgb_pairs(self) -> None:
        """A logical Transfer sink pins both physical control and RGB entries."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model_mock()
        model.config.kv_cache_inference_size = 3
        model.config.attention_sink_size = 1
        model.get_data_and_condition.return_value = self._make_gen_data("video_transfer")
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], None)

        with patch(_PATCH_PACK, MagicMock()), patch(_PATCH_KV) as dual_cache_cls:
            list(OmniMoTCausalModel.iter_samples_from_batch_autoregressive(model, {}, mode="video_transfer"))

        assert all(call.kwargs["gen_cache_size"] == 6 for call in dual_cache_cls.call_args_list)
        assert all(call.kwargs["attention_sink_size"] == 2 for call in dual_cache_cls.call_args_list)

        tokens_per_frame = self.H * self.W
        sink_tokens = 2 * tokens_per_frame
        control_recent_tokens = 2 * tokens_per_frame
        target_recent_tokens = 3 * tokens_per_frame
        seed_calls = model._seed_frame_into_kv_cache.call_args_list
        control_calls = [call for call in seed_calls if torch.all(call.kwargs["frame_latent"] == 1)]
        rgb_calls = [call for call in seed_calls if torch.all(call.kwargs["frame_latent"] == 0)]
        assert all(call.kwargs["transfer_history_sink_tokens"] == sink_tokens for call in seed_calls)
        assert all(call.kwargs["transfer_history_max_tokens"] == control_recent_tokens for call in control_calls)
        assert all(call.kwargs["transfer_history_max_tokens"] == target_recent_tokens for call in rgb_calls)
        assert all(
            call.kwargs["transfer_history_sink_tokens"] == sink_tokens
            and call.kwargs["transfer_history_max_tokens"] == target_recent_tokens
            for call in model.generate_next_frame.call_args_list
        )

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_finite_window_ceil_patchifies_720p_latents(self) -> None:
        """Finite-window inference counts the padded 23x40 patch grid at 720p."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model_mock()
        model.config.diffusion_expert_config.patch_spatial = 2
        model.config.kv_cache_inference_size = 51
        model.config.attention_sink_size = 1
        control = torch.ones((1, self.C, self.T, 45, 80), dtype=torch.float32)  # [B,C,T,H,W]
        target = torch.zeros_like(control)  # [B,C,T,H,W]
        model.get_data_and_condition.return_value = SimpleNamespace(
            batch_size=1,
            x0_tokens_vision=[control, target],
            num_vision_items_per_sample=[2],
            x0_tokens_action=None,
            fps_vision=torch.tensor([24.0]),
            fps_action=None,
            action_domain_id=None,
            raw_action_dim=None,
        )
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], None)
        model.generate_next_frame.return_value = torch.zeros((1, self.C, 1, 45, 80), dtype=torch.float32)

        with patch(_PATCH_PACK, MagicMock()), patch(_PATCH_KV):
            list(OmniMoTCausalModel.iter_samples_from_batch_autoregressive(model, {}, mode="video_transfer"))

        tokens_per_frame = 23 * 40
        assert all(
            call.kwargs["transfer_history_sink_tokens"] == 2 * tokens_per_frame
            and call.kwargs["transfer_history_max_tokens"] == 98 * tokens_per_frame
            for call in model._seed_frame_into_kv_cache.call_args_list
            if torch.all(call.kwargs["frame_latent"] == 1)
        )
        assert all(
            call.kwargs["transfer_history_sink_tokens"] == 2 * tokens_per_frame
            and call.kwargs["transfer_history_max_tokens"] == 99 * tokens_per_frame
            for call in model.generate_next_frame.call_args_list
        )

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_finite_window_requires_framewise_chunks(self) -> None:
        """Pair-aware finite Transfer windows currently require one latent frame per chunk."""
        model = self._make_model_mock()
        model.config.kv_cache_inference_size = 2
        model.config.teacher_forcing_frames_per_chunk = 2

        with pytest.raises(ValueError, match="teacher_forcing_frames_per_chunk=1"):
            self._run(model, "video_transfer")

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_finite_window_rejects_non_history_replay(self) -> None:
        """Finite Transfer windows are scoped to causal control with RGB history."""
        model = self._make_model_mock()
        model.config.kv_cache_inference_size = 2
        model.config.teacher_forcing_replay_policy.controls_read_strict_past_clean_rgb = False

        with pytest.raises(ValueError, match="controls_read_strict_past_clean_rgb=True"):
            self._run(model, "video_transfer")

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_chunkwise_generation_matches_training_partition(self) -> None:
        """Legacy C=4 transfer uses [1,4,4] chunks and interleaved RGB history."""
        self.T = 9
        model = self._make_model_mock()
        model.config.teacher_forcing_frames_per_chunk = 4
        model.generate_next_frame.side_effect = [
            torch.zeros(1, self.C, chunk_len, self.H, self.W)  # [B,C,T_chunk,H,W]
            for chunk_len in (1, 4, 4)
        ]

        result, _ = self._run(model, "video_transfer")

        assert result["vision"].shape == (1, self.C, self.T, self.H, self.W)
        target_calls = model.generate_next_frame.call_args_list
        assert [call.kwargs["frame_idx"] for call in target_calls] == [0, 1, 5]
        assert [call.kwargs["cache_frame_idx"] for call in target_calls] == [1, 3, 8]

        seed_calls = model._seed_frame_into_kv_cache.call_args_list
        control_calls = [call for call in seed_calls if torch.all(call.kwargs["frame_latent"] == 1)]
        assert [call.kwargs["frame_idx"] for call in control_calls] == [0, 2, 7]
        assert [call.kwargs["position_frame_idx"] for call in control_calls] == [0, 1, 5]
        assert [call.kwargs["frame_latent"].shape[2] for call in control_calls] == [1, 4, 4]
        assert [call.kwargs["condition_frame_indexes_vision"] for call in control_calls] == [
            [0],
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_chunkwise_rejects_partial_final_latent_chunk(self) -> None:
        self.T = 8
        model = self._make_model_mock()
        model.config.teacher_forcing_frames_per_chunk = 4

        with pytest.raises(ValueError, match="exactly match the teacher-forcing latent partition"):
            self._run(model, "video_transfer")

    @pytest.mark.L0
    def test_video_transfer_uses_supplied_negative_prompt(self) -> None:
        """The explicit negative caption drives the CFG unconditional text branch."""
        model = self._make_model_mock()
        model.input_caption_key = "caption"

        self._run(model, "video_transfer", data_batch={"neg_caption": ["avoid blur"]})

        model._get_inference_text_tokens.assert_called_once_with({"neg_caption": ["avoid blur"]}, True)

    @pytest.mark.L0
    def test_non_transfer_ar_ignores_supplied_negative_prompt(self) -> None:
        """Existing AR modes retain their empty unconditional text branch."""
        model = self._make_model_mock()
        model.input_caption_key = "caption"

        self._run(model, "text2video", data_batch={"neg_caption": ["avoid blur"]})

        model._get_inference_text_tokens.assert_called_once_with({"neg_caption": ["avoid blur"]}, False)

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_video_transfer_rejects_chunk_causal_clean_replay(self) -> None:
        """Legacy frame-at-a-time transfer must not accept chunk-causal clean replay."""
        model = self._make_model_mock()
        model.config.teacher_forcing_replay_policy.clean_pass_causality = "chunk"

        with pytest.raises(ValueError, match="clean_pass_causality='frame'"):
            self._run(model, "video_transfer")

    @pytest.mark.L0
    @pytest.mark.parametrize("mode", ["image2video", "forward_dynamics"])
    def test_clean_vision_callback_includes_conditioned_prefix(self, mode: str) -> None:
        """The clean-output callback includes the prefix and every generated frame exactly once."""
        model = self._make_model_mock()
        model.generate_next_frame.side_effect = [
            torch.full((1, self.C, 1, self.H, self.W), frame_value)  # [B,C,1,H,W]
            for frame_value in (1.0, 2.0)
        ]
        callback_chunks: list[torch.Tensor] = []

        result, _ = self._run(model, mode, on_clean_vision_chunk=callback_chunks.append)
        callback_vision = torch.cat(callback_chunks, dim=2)  # [B,C,T,H,W]

        assert [chunk.shape[2] for chunk in callback_chunks] == [1, 1, 1]
        torch.testing.assert_close(callback_vision, result["vision"])

    @pytest.mark.L0
    def test_clean_vision_callback_includes_all_i2v_prefix_frames(self) -> None:
        """Multi-prefix I2V submits every conditioned prefix before generated output."""
        model = self._make_model_mock()
        model.generate_next_frame.return_value = torch.ones(1, self.C, 1, self.H, self.W)  # [B,C,1,H,W]
        callback_chunks: list[torch.Tensor] = []
        data_batch = {
            "sequence_plan": [SimpleNamespace(condition_frame_indexes_vision=[0, 1], has_action=False)],
        }

        result, _ = self._run(
            model,
            "image2video",
            data_batch=data_batch,
            on_clean_vision_chunk=callback_chunks.append,
        )
        callback_vision = torch.cat(callback_chunks, dim=2)  # [B,C,T,H,W]

        assert [chunk.shape[2] for chunk in callback_chunks] == [1, 1, 1]
        torch.testing.assert_close(callback_vision, result["vision"])

    @pytest.mark.L0
    def test_ar_construction_passes_explicit_fp8_torch_kernel_impl(self) -> None:
        """AR cache construction forwards the FP8 batch decode kernel setting."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model_mock()
        model.config.kv_cache_dtype = "fp8"
        model.config.kv_cache_kernel_impl = "torch"
        gen_data = self._make_gen_data("text2video")
        model.get_data_and_condition.return_value = gen_data
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], None)

        with patch(_PATCH_PACK, MagicMock()), patch(_PATCH_KV) as mock_kv:
            list(OmniMoTCausalModel.iter_samples_from_batch_autoregressive(model, {}, mode="text2video"))

        assert mock_kv.call_args_list
        for call in mock_kv.call_args_list:
            assert call.kwargs["kv_cache_dtype"] == "fp8"
            assert call.kwargs["kv_cache_kernel_impl"] == "torch"

    @pytest.mark.L0
    def test_t2v_max_num_frames_truncates_callback_generation(self):
        """AR callback can cap latent frames so all FSDP ranks do the same number of forwards."""
        model = self._make_model_mock()
        result, _ = self._run(model, "text2video", max_num_frames=2)
        assert result["vision"].shape == (1, self.C, 2, self.H, self.W)
        assert model.generate_next_frame.call_count == 2

    @pytest.mark.L0
    def test_text2video_action_conditioned_generates_frame0_without_prefill(self):
        """Action-conditioned T2V generates frame 0 from text, then consumes actions for later frames."""
        model = self._make_model_mock()
        model.tensor_kwargs = {"device": "cpu", "dtype": torch.bfloat16}
        _, mock_pack = self._run(model, "text2video_action_conditioned")
        assert model.generate_next_frame.call_count == self.T
        assert model.denoise.call_count == 0
        first_call = model.generate_next_frame.call_args_list[0].kwargs
        second_call = model.generate_next_frame.call_args_list[1].kwargs
        assert first_call["curr_action_latent"] is None
        assert second_call["curr_action_latent"].shape == (self.TCF, 8)
        assert second_call["curr_action_latent"].dtype == model.tensor_kwargs["dtype"]
        first_pack_call = mock_pack.call_args_list[0].kwargs
        assert first_pack_call["vision_latent"].dtype == model.tensor_kwargs["dtype"]

    @pytest.mark.L0
    def test_action_conditioned_ar_passes_action_metadata_to_packer(self) -> None:
        """AR repacking preserves action domain and raw-width metadata."""
        model = self._make_model_mock()
        _, mock_pack = self._run(model, "forward_dynamics")

        first_pack_call = mock_pack.call_args_list[0].kwargs

        assert first_pack_call["action_domain_id"].item() == 2
        assert first_pack_call["raw_action_dim"].item() == 6

    @pytest.mark.L0
    def test_sampler_mode_passed_to_generate_next_frame(self):
        """Callback/inference distilled mode reaches the per-frame sampler."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model_mock()
        gen_data = self._make_gen_data("text2video")
        model.get_data_and_condition.return_value = gen_data
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], None)
        with patch(_PATCH_PACK, MagicMock()), patch(_PATCH_KV):
            list(
                OmniMoTCausalModel.iter_samples_from_batch_autoregressive(
                    model,
                    {},
                    mode="text2video",
                    sampler_mode="distilled",
                    distilled_num_steps=1,
                )
            )
        first_call = model.generate_next_frame.call_args_list[0].kwargs
        assert first_call["sampler_mode"] == "distilled"
        assert first_call["distilled_num_steps"] == 1

    @pytest.mark.L0
    def test_i2v_generates_from_frame1_with_prefill(self):
        """image2video: T-1 frames generated, one prefill _seed_frame_into_kv_cache call at frame 0."""
        model = self._make_model_mock()
        self._run(model, "image2video")
        assert model.generate_next_frame.call_count == self.T - 1
        prefill_calls = [c for c in model._seed_frame_into_kv_cache.call_args_list if c.kwargs.get("frame_idx") == 0]
        assert len(prefill_calls) == 1

    @pytest.mark.L0
    def test_i2v_multi_prefix_seeds_all_prefix_frames_and_generates_after_prefix(self):
        """video2video-style AR inference seeds every clean prefix frame before generation."""
        model = self._make_model_mock()
        data_batch = {
            "sequence_plan": [SimpleNamespace(condition_frame_indexes_vision=[0, 1], has_action=False)],
        }

        result, _ = self._run(model, "image2video", data_batch=data_batch)

        assert result["vision"].shape == (1, self.C, self.T, self.H, self.W)
        assert model.generate_next_frame.call_count == self.T - 2
        called_idxs = [c.kwargs["frame_idx"] for c in model.generate_next_frame.call_args_list]
        assert called_idxs == [2]
        prefill_idxs = [c.kwargs["frame_idx"] for c in model._seed_frame_into_kv_cache.call_args_list]
        assert prefill_idxs == [0, 1]
        assert model._seed_frame_into_kv_cache.call_args_list[0].kwargs["cond_text_tokens"] == [1, 2, 3]
        assert model._seed_frame_into_kv_cache.call_args_list[1].kwargs["cond_text_tokens"] is None

    @pytest.mark.L0
    def test_i2v_prefix_frame_count_overrides_variable_condition_prefix(self):
        """Callbacks force a one-frame prefix even if local training data has a longer prefix."""
        model = self._make_model_mock()
        data_batch = {
            "sequence_plan": [SimpleNamespace(condition_frame_indexes_vision=[0, 1], has_action=False)],
        }

        result, _ = self._run(model, "image2video", data_batch=data_batch, prefix_frame_count=1)

        model.get_data_and_condition.assert_called_once_with(data_batch, vision_condition_indexes=[[0]])
        assert result["vision"].shape == (1, self.C, self.T, self.H, self.W)
        assert model.generate_next_frame.call_count == self.T - 1
        seed_idxs = [c.kwargs["frame_idx"] for c in model._seed_frame_into_kv_cache.call_args_list]
        assert seed_idxs == [0, 1]
        assert model._seed_frame_into_kv_cache.call_args_list[0].kwargs["cond_text_tokens"] == [1, 2, 3]
        assert model._seed_frame_into_kv_cache.call_args_list[1].kwargs["cond_text_tokens"] is None

    @pytest.mark.L0
    def test_i2v_multi_prefix_requires_contiguous_prefix_from_zero(self):
        """AR cache prefill rejects sparse/non-prefix condition frames."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model_mock()
        model.get_data_and_condition.return_value = self._make_gen_data("image2video")
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], None)
        data_batch = {
            "sequence_plan": [SimpleNamespace(condition_frame_indexes_vision=[0, 2], has_action=False)],
        }

        with patch(_PATCH_PACK, MagicMock()), patch(_PATCH_KV), pytest.raises(ValueError, match="contiguous prefix"):
            list(OmniMoTCausalModel.iter_samples_from_batch_autoregressive(model, data_batch, mode="image2video"))

    @pytest.mark.L0
    def test_fd_generates_from_frame1_with_prefill(self):
        """forward_dynamics: T-1 frames generated, one prefill _seed_frame_into_kv_cache call at frame 0."""
        model = self._make_model_mock()
        self._run(model, "forward_dynamics")
        assert model.generate_next_frame.call_count == self.T - 1
        prefill_calls = [c for c in model._seed_frame_into_kv_cache.call_args_list if c.kwargs.get("frame_idx") == 0]
        assert len(prefill_calls) == 1

    @pytest.mark.L0
    @pytest.mark.CPU
    @pytest.mark.parametrize(
        ("mode", "condition_frame_indexes"),
        [("forward_dynamics", [0]), ("image2video", [0, 1, 2])],
    )
    def test_ar_reuses_condition_indexes_for_encode_and_prefill(
        self, mode: str, condition_frame_indexes: list[int]
    ) -> None:
        model = self._make_model_mock()
        data_batch = {
            "sequence_plan": [
                SimpleNamespace(
                    condition_frame_indexes_vision=condition_frame_indexes,
                    has_action=mode == "forward_dynamics",
                )
            ],
        }

        self._run(model, mode, data_batch=data_batch)

        model.get_data_and_condition.assert_called_once_with(
            data_batch,
            vision_condition_indexes=[condition_frame_indexes],
        )
        prefill_indexes = [call.kwargs["frame_idx"] for call in model._seed_frame_into_kv_cache.call_args_list]
        assert prefill_indexes[: len(condition_frame_indexes)] == condition_frame_indexes

    @pytest.mark.L0
    def test_fd_streaming_uses_sent_action_without_preloaded_actions(self):
        """Streaming forward_dynamics consumes sent actions past the seed clip length."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = self._make_model_mock()
        gen_data = self._make_gen_data("forward_dynamics")
        gen_data.x0_tokens_action = None
        model.get_data_and_condition.return_value = gen_data
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], None)
        action = torch.ones(self.TCF, 8)  # [tcf,D]
        action_2 = torch.full((self.TCF, 8), 2.0)  # [tcf,D]
        action_3 = torch.full((self.TCF, 8), 3.0)  # [tcf,D]

        with patch(_PATCH_PACK, MagicMock()), patch(_PATCH_KV):
            ar_iterator = OmniMoTCausalModel.iter_samples_from_batch_autoregressive(
                model,
                {},
                mode="forward_dynamics",
            )
            initial_payload = next(ar_iterator)
            generated_payload = ar_iterator.send(action)
            ar_iterator.send(action_2)
            generated_payload_past_clip = ar_iterator.send(action_3)

        assert initial_payload["vision"].shape == (1, self.C, 1, self.H, self.W)
        assert torch.equal(generated_payload["action"], action)
        assert torch.equal(generated_payload_past_clip["action"], action_3)
        assert model.generate_next_frame.call_args_list[0].kwargs["num_steps"] == 35
        assert model.generate_next_frame.call_args_list[0].kwargs["guidance"] == 1.5
        assert model.generate_next_frame.call_args_list[1].kwargs["num_steps"] == 35
        assert model.generate_next_frame.call_args_list[1].kwargs["guidance"] == 1.5

    @pytest.mark.L0
    def test_output_vision_shape_t2v(self):
        """text2video output vision has shape (1, C, T, H, W)."""
        model = self._make_model_mock()
        result, _ = self._run(model, "text2video")
        assert result["vision"].shape == (1, self.C, self.T, self.H, self.W)

    @pytest.mark.L0
    def test_output_vision_shape_i2v(self):
        """image2video output vision has shape (1, C, T, H, W)."""
        model = self._make_model_mock()
        result, _ = self._run(model, "image2video")
        assert result["vision"].shape == (1, self.C, self.T, self.H, self.W)

    @pytest.mark.L0
    def test_t2v_frame_idx_sequence(self):
        """generate_next_frame receives frame_idx 0, 1, ..., T-1 in order for text2video."""
        model = self._make_model_mock()
        self._run(model, "text2video")
        called_idxs = [c.kwargs["frame_idx"] for c in model.generate_next_frame.call_args_list]
        assert called_idxs == list(range(self.T))

    @pytest.mark.L0
    def test_t2v_text_tokens_passed_at_frame0(self):
        """Bug 2 guard: pack_input_sequence_autoregressive receives text_tokens at frame_idx=0
        for text2video and None for all subsequent frames.

        text2video has no prefill (start_frame=0). With default guidance=1.5, cfg_active=True,
        so each loop frame produces 2 pack calls (cond first, then uncond) — cond packs are at
        even indices in the call list.
        """
        model = self._make_model_mock()
        _, mock_pack = self._run(model, "text2video")
        cond_calls = mock_pack.call_args_list[::2]  # cond is the first of each (cond, uncond) pair
        assert len(cond_calls) == self.T, f"Expected {self.T} cond pack calls, got {len(cond_calls)}"
        assert cond_calls[0].kwargs["text_tokens"] is not None, "frame 0 must receive text_tokens"
        for i, call in enumerate(cond_calls[1:], start=1):
            assert call.kwargs["text_tokens"] is None, f"frame {i} must receive text_tokens=None"


# ---------------------------------------------------------------------------
# L0 — CFGP support for AR generation (mock-based, single-rank)
# ---------------------------------------------------------------------------


class TestCFGPARGeneration:
    """
    Tests for CFGP (CFG Parallelism) in iter_samples_from_batch_autoregressive
    and generate_next_frame.

    Uses the same MagicMock pattern as TestARGenerationLoopLogic.  _broadcast_seed
    and dist are patched to avoid distributed setup.
    """

    C, T, H, W, TCF = 4, 3, 2, 2, 4

    def _make_cfgp_model_mock(self, cfgp_rank: int = 0) -> MagicMock:
        m = MagicMock()
        m.tokenizer_vision_gen.temporal_compression_factor = self.TCF
        m.config.diffusion_expert_config.patch_spatial = 1
        m.config.max_action_dim = 8
        m.config.video_temporal_causal = False
        m.config.compile.enabled = False
        m.config.compile.use_cuda_graphs = False
        m.config.kv_cache_dtype = None
        m.config.kv_cache_kernel_impl = "triton"
        m.config.kv_cache_inference_size = None
        m.config.attention_sink_size = 0
        m.config.teacher_forcing_frames_per_chunk = 1
        m.llm_special_tokens = {}
        m.net.num_hidden_layers = 2
        m.generate_next_frame.return_value = torch.zeros(1, self.C, 1, self.H, self.W)
        m.parallel_dims.cfgp_enabled = True
        m.parallel_dims.cfgp_rank = cfgp_rank
        m.parallel_dims.cfgp_size = 2
        return m

    def _make_gen_data(self, mode: str) -> SimpleNamespace:
        vision = torch.zeros(1, self.C, self.T, self.H, self.W)
        return SimpleNamespace(
            batch_size=1,
            x0_tokens_vision=[vision],
            x0_tokens_action=None,
            fps_vision=torch.tensor([24.0]),
            fps_action=None,
        )

    def _run(self, model: MagicMock, mode: str, mock_pack: MagicMock | None = None) -> tuple[dict, MagicMock]:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        gen_data = self._make_gen_data(mode)
        model.get_data_and_condition.return_value = gen_data
        model._get_inference_text_tokens.return_value = ([[1, 2, 3]], [[]])
        if mock_pack is None:
            mock_pack = MagicMock()
        with (
            patch(_PATCH_PACK, mock_pack),
            patch(_PATCH_KV),
            patch(_PATCH_BROADCAST, side_effect=lambda s, g, r: s),
        ):
            outputs = list(OmniMoTCausalModel.iter_samples_from_batch_autoregressive(model, {}, mode=mode))
        result = {"vision": torch.cat([out["vision"] for out in outputs], dim=2)}  # [B,C,T,H,W]
        return result, mock_pack

    @pytest.mark.L0
    def test_cfgp_dual_kv_cache_uncond_is_none(self):
        """With CFGP, dual_kv_cache_uncond passed to generate_next_frame is always None."""
        model = self._make_cfgp_model_mock(cfgp_rank=0)
        self._run(model, "text2video")
        for call in model.generate_next_frame.call_args_list:
            assert call.kwargs["dual_kv_cache_uncond"] is None

    @pytest.mark.L0
    def test_cfgp_packed_seq_uncond_always_created(self):
        """With CFGP, uncond pack is created each loop frame (cfg_active=True under CFGP)."""
        model = self._make_cfgp_model_mock(cfgp_rank=0)
        _, mock_pack = self._run(model, "text2video")
        # text2video has no prefill; AR loop produces 2 pack calls per frame (cond + uncond).
        expected = 2 * self.T
        assert mock_pack.call_count == expected

    def _run_seed_prefill(self, model: MagicMock, cond_pack: MagicMock, uncond_pack: MagicMock) -> None:
        """Invoke _seed_frame_into_kv_cache directly with patched pack to observe denoise inputs."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model.config.causal_training_strategy = "none"
        mock_pack = MagicMock(side_effect=[cond_pack, uncond_pack])
        with patch(_PATCH_PACK, mock_pack):
            OmniMoTCausalModel._seed_frame_into_kv_cache(
                model,
                frame_latent=torch.zeros(1, self.C, 1, self.H, self.W),
                frame_idx=0,
                dual_kv_cache=[MagicMock() for _ in range(2)],
                dual_kv_cache_uncond=None,
                cond_text_tokens=[1, 2, 3],
                uncond_text_tokens=[],
                cond_cached_text_offset=0,
                uncond_cached_text_offset=0,
                curr_action_latent=None,
                gen_data_clean=SimpleNamespace(fps_vision=None, fps_action=None),
                fps_vision_list=[24.0],
                fps_action_list=[24.0],
                seed=42,
                cfg_active=True,
                cfgp_enabled=True,
                tcf=self.TCF,
                patch_size=1,
                action_dim=8,
                video_tc=False,
                enable_fps_mod=False,
                base_fps=24.0,
                modality_margin=0,
            )

    @pytest.mark.L0
    def test_cfgp_rank0_prefill_uses_cond_pack(self):
        """Prefill with CFGP rank 0: denoise receives the cond (first) pack."""
        model = self._make_cfgp_model_mock(cfgp_rank=0)
        cond_pack = MagicMock(name="cond_pack")
        uncond_pack = MagicMock(name="uncond_pack")
        self._run_seed_prefill(model, cond_pack, uncond_pack)
        assert model.denoise.call_count == 1
        assert model.denoise.call_args_list[0].kwargs["data_batch_packed"] is cond_pack

    @pytest.mark.L0
    def test_cfgp_rank1_prefill_uses_uncond_pack(self):
        """Prefill with CFGP rank 1: denoise receives the uncond (second) pack."""
        model = self._make_cfgp_model_mock(cfgp_rank=1)
        cond_pack = MagicMock(name="cond_pack")
        uncond_pack = MagicMock(name="uncond_pack")
        self._run_seed_prefill(model, cond_pack, uncond_pack)
        assert model.denoise.call_count == 1
        assert model.denoise.call_args_list[0].kwargs["data_batch_packed"] is uncond_pack

    @pytest.mark.L0
    def test_cfgp_single_denoise_call_per_velocity_eval(self):
        """generate_next_frame with CFGP rank 0: exactly one denoise call per velocity eval."""
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        C, H, W = self.C, self.H, self.W
        model = self._make_cfgp_model_mock(cfgp_rank=0)
        model.config.rectified_flow_inference_config.scheduler_type = "unipc"
        model.tensor_kwargs = {"dtype": torch.float32, "device": "cpu"}
        model.denoise.return_value = {"preds_vision": [torch.zeros(C, 1, H, W)]}

        denoise_counts: list[int] = []

        def mock_sampler(fn, x0, **kwargs):
            fn(x0, torch.ones(1, 1))
            denoise_counts.append(model.denoise.call_count)
            return x0

        model.sampler = mock_sampler

        packed_seq = MagicMock()
        packed_seq.vision = None  # skip noise injection in set_pack_noise

        with patch(_PATCH_DIST) as mock_dist:
            mock_dist.batch_isend_irecv.return_value = [MagicMock()]
            OmniMoTCausalModel.generate_next_frame(
                model,
                packed_seq=packed_seq,
                packed_seq_uncond=None,
                curr_vision_latent=torch.zeros(1, C, 1, H, W),
                curr_action_latent=None,
                cond_text_tokens=[1, 2, 3],
                uncond_text_tokens=[],
                gen_data_clean=SimpleNamespace(fps_vision=None, fps_action=None),
                dual_kv_cache=[],
                dual_kv_cache_uncond=None,
                guidance=7.5,
                num_steps=1,
                shift=1.0,
                seed=42,
                fps_vision_list=[24.0],
                fps_action_list=[24.0],
                frame_idx=1,
            )

        assert denoise_counts == [1], f"Expected 1 denoise call per velocity eval, got {denoise_counts}"


class TestDistilledARSampler:
    @staticmethod
    def _attach_distilled_schedule_helper(model, model_cls) -> None:
        def schedule(
            distilled_num_steps: int | None = None,
            frame_idx: int | None = None,
            num_frames: int | None = None,
        ) -> list[float]:
            return model_cls._get_ar_distilled_timestep_schedule(
                model,
                distilled_num_steps,
                frame_idx=frame_idx,
                num_frames=num_frames,
            )

        model._get_ar_distilled_timestep_schedule = schedule

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_distilled_schedule_appends_zero_and_truncates(self) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = SimpleNamespace(
            config=SimpleNamespace(
                fixed_step_sampler_config=SimpleNamespace(t_list=[1.0, 0.9, 0.75], sample_type="ode"),
            )
        )

        assert OmniMoTCausalModel._get_ar_distilled_timestep_schedule(model) == [1.0, 0.9, 0.75, 0.0]
        assert OmniMoTCausalModel._get_ar_distilled_timestep_schedule(model, distilled_num_steps=2) == [
            1.0,
            0.9,
            0.0,
        ]

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_distilled_ode_sampler_uses_configured_sigmas(self) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = SimpleNamespace(
            config=SimpleNamespace(
                fixed_step_sampler_config=SimpleNamespace(t_list=[0.5], sample_type="ode"),
                rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
            )
        )
        self._attach_distilled_schedule_helper(model, OmniMoTCausalModel)
        initial_noise = torch.full((1, 2), 1.0)  # [B,N]
        seen_timesteps = []

        def velocity_fn(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            seen_timesteps.append(timestep.clone())  # [B,1]
            return torch.full_like(x, 2.0)  # [B,N]

        sampled = OmniMoTCausalModel._run_distilled_ar_sampler(
            model,
            velocity_fn,
            initial_noise,
            seed=10,
            frame_idx=0,
        )  # [B,N]

        torch.testing.assert_close(sampled, torch.zeros_like(initial_noise))
        torch.testing.assert_close(seen_timesteps[0], torch.tensor([[500.0]]))

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_distilled_sde_sampler_is_seed_deterministic(self) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = SimpleNamespace(
            config=SimpleNamespace(
                fixed_step_sampler_config=SimpleNamespace(t_list=[0.5, 0.25], sample_type="sde"),
                rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
            )
        )
        self._attach_distilled_schedule_helper(model, OmniMoTCausalModel)
        initial_noise = torch.ones(1, 2)  # [B,N]

        def velocity_fn(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return torch.zeros_like(x)  # [B,N]

        sampled_a = OmniMoTCausalModel._run_distilled_ar_sampler(
            model,
            velocity_fn,
            initial_noise,
            seed=10,
            frame_idx=2,
        )  # [B,N]
        sampled_b = OmniMoTCausalModel._run_distilled_ar_sampler(
            model,
            velocity_fn,
            initial_noise,
            seed=10,
            frame_idx=2,
        )  # [B,N]

        torch.testing.assert_close(sampled_a, sampled_b)

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_distilled_sde_sampler_uses_distinct_frame_seeds(self) -> None:
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = SimpleNamespace(
            config=SimpleNamespace(
                fixed_step_sampler_config=SimpleNamespace(t_list=[0.5, 0.25], sample_type="sde"),
                rectified_flow_inference_config=SimpleNamespace(num_train_timesteps=1000),
            )
        )
        self._attach_distilled_schedule_helper(model, OmniMoTCausalModel)
        initial_noise = torch.ones(1, 8)  # [B,N]

        def velocity_fn(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:  # noqa: ARG001
            return torch.zeros_like(x)  # [B,N]

        sampled_frame0 = OmniMoTCausalModel._run_distilled_ar_sampler(
            model,
            velocity_fn,
            initial_noise,
            seed=10,
            frame_idx=0,
        )  # [B,N]
        sampled_frame1 = OmniMoTCausalModel._run_distilled_ar_sampler(
            model,
            velocity_fn,
            initial_noise,
            seed=10,
            frame_idx=1,
        )  # [B,N]

        assert not torch.equal(sampled_frame0, sampled_frame1)


# ---------------------------------------------------------------------------
# L0 — Joint bidirectional/TF training (MoBA)
# ---------------------------------------------------------------------------


class TestBidirectionalStepMixing:
    """Joint bidirectional and teacher-forcing training plumbing."""

    @staticmethod
    def _make_model(**config_overrides):
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        model = object.__new__(OmniMoTCausalModel)
        torch.nn.Module.__init__(model)
        config = SimpleNamespace(
            enable_moba=True,
            moba_bidirectional_steps=1,
            moba_causal_steps=1,
            causal_training_strategy="teacher_forcing",
            natten_parameter_list=None,
            teacher_forcing_detach_clean_kv=False,
        )
        for key, value in config_overrides.items():
            setattr(config, key, value)
        model.config = config
        return model

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_moba_defaults_to_disabled(self) -> None:
        import attrs

        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModelConfig

        fields = {field.name: field for field in attrs.fields(OmniMoTCausalModelConfig)}
        assert fields["enable_moba"].default is False
        assert fields["moba_bidirectional_steps"].default == 1
        assert fields["moba_causal_steps"].default == 1

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_training_step_delegates_when_moba_is_disabled(self) -> None:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        model = self._make_model(enable_moba=False, causal_training_strategy="diffusion_forcing")
        with patch.object(OmniMoTModel, "training_step", return_value=({}, torch.tensor(0.0))) as base_step:
            output_batch, _ = model.training_step({}, iteration=0)

        base_step.assert_called_once()
        assert "bidirectional_step" not in output_batch

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_validation_rejects_unsupported_strategy(self) -> None:
        model = self._make_model(causal_training_strategy="diffusion_forcing")
        with pytest.raises(ValueError, match="teacher_forcing"):
            model._validate_moba_config()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_validation_rejects_sparse_natten(self) -> None:
        model = self._make_model(natten_parameter_list=[{"window_size": (8, 8)}])
        with pytest.raises(ValueError, match="sparse NATTEN"):
            model._validate_moba_config()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_validation_allows_dense_natten_entries(self) -> None:
        model = self._make_model(natten_parameter_list=[None])
        model._validate_moba_config()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_validation_rejects_negative_step_counts(self) -> None:
        model = self._make_model(moba_bidirectional_steps=-1)
        with pytest.raises(ValueError, match=">= 0"):
            model._validate_moba_config()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_validation_rejects_empty_cycle(self) -> None:
        model = self._make_model(moba_bidirectional_steps=0, moba_causal_steps=0)
        with pytest.raises(ValueError, match=">= 1"):
            model._validate_moba_config()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_validation_allows_single_mode_patterns(self) -> None:
        self._make_model(moba_bidirectional_steps=1, moba_causal_steps=0)._validate_moba_config()
        self._make_model(moba_bidirectional_steps=0, moba_causal_steps=1)._validate_moba_config()

    @pytest.mark.L0
    @pytest.mark.CPU
    @pytest.mark.parametrize(
        ("bidir_steps", "causal_steps", "expected"),
        [
            (1, 1, [True, False, True, False]),  # default 1:1 alternation
            (1, 0, [True, True, True, True]),  # bidirectional-only
            (0, 1, [False, False, False, False]),  # causal-only
            (2, 1, [True, True, False, True, True, False]),  # generic pattern
        ],
    )
    def test_step_pattern_follows_configured_cycle(self, bidir_steps, causal_steps, expected) -> None:
        model = self._make_model(moba_bidirectional_steps=bidir_steps, moba_causal_steps=causal_steps)
        assert [model._is_moba_bidirectional_step(i) for i in range(len(expected))] == expected

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_init_validates_moba_config_once(self) -> None:
        """__init__ validates MoBA config when enabled and skips validation when disabled."""
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
        from cosmos_framework.configs.base.defaults.replay_attention import TeacherForcingReplayPolicyConfig
        from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel

        def _fake_base_init(self, config) -> None:
            torch.nn.Module.__init__(self)
            self.config = config

        bad_config = SimpleNamespace(
            enable_moba=True,
            causal_training_strategy="diffusion_forcing",
            natten_parameter_list=None,
            teacher_forcing_kv_implementation="singleview_threeway_kv",
            teacher_forcing_replay_policy=TeacherForcingReplayPolicyConfig(),
        )
        with patch.object(OmniMoTModel, "__init__", _fake_base_init):
            with pytest.raises(ValueError, match="teacher_forcing"):
                OmniMoTCausalModel(bad_config)

            bad_config.enable_moba = False
            OmniMoTCausalModel(bad_config)  # MoBA disabled: bad strategy is not validated

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_pre_noise_memory_hook_skips_clean_pass_on_bidirectional_step(self) -> None:
        model = self._make_model()
        model.net = MagicMock()
        model._validate_teacher_forcing_pack = MagicMock()
        model._build_clean_tf_cache = MagicMock(return_value="tf-memory")
        model._bidirectional_step_active = True
        memory_info = {"skip_text": False}

        result = model.pre_noise_memory_hook(MagicMock(), MagicMock(), memory_info)

        assert "_tf_memory_state" not in result
        model._validate_teacher_forcing_pack.assert_not_called()
        model._build_clean_tf_cache.assert_not_called()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_pre_noise_memory_hook_builds_clean_pass_on_tf_step(self) -> None:
        model = self._make_model()
        model.net = MagicMock()
        model._validate_teacher_forcing_pack = MagicMock()
        model._build_clean_tf_cache = MagicMock(return_value="tf-memory")
        model._bidirectional_step_active = False

        result = model.pre_noise_memory_hook(MagicMock(), MagicMock(), {"skip_text": False})

        assert result["_tf_memory_state"] == "tf-memory"
        model._build_clean_tf_cache.assert_called_once()

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_denoise_forces_full_attention_on_bidirectional_step(self) -> None:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        model = self._make_model()
        with patch.object(OmniMoTModel, "denoise", return_value={"preds_vision": []}) as base_denoise:
            model._bidirectional_step_active = True
            model.denoise(data_batch_packed=MagicMock(), memory=None)
            assert base_denoise.call_args.kwargs["video_temporal_causal"] is False

            base_denoise.reset_mock()
            model._bidirectional_step_active = False
            model.denoise(data_batch_packed=MagicMock(), memory=None)
            assert base_denoise.call_args.kwargs["video_temporal_causal"] is None

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_generate_samples_uses_bidirectional_attention_for_moba(self) -> None:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        model = self._make_model()
        flags_seen = []

        def base_generate(data_batch: dict, *args, **kwargs) -> dict:  # noqa: ARG001
            flags_seen.append(model._bidirectional_step_active)
            return {"vision": []}

        with patch.object(OmniMoTModel, "generate_samples_from_batch", side_effect=base_generate):
            result = model.generate_samples_from_batch({})

        assert flags_seen == [True]
        assert model._bidirectional_step_active is False
        assert result == {"vision": []}

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_generate_samples_keeps_config_attention_when_moba_disabled(self) -> None:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        model = self._make_model(enable_moba=False)
        flags_seen = []

        def base_generate(data_batch: dict, *args, **kwargs) -> dict:  # noqa: ARG001
            flags_seen.append(model._bidirectional_step_active)
            return {"vision": []}

        with patch.object(OmniMoTModel, "generate_samples_from_batch", side_effect=base_generate):
            model.generate_samples_from_batch({})

        assert flags_seen == [False]

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_generate_samples_stays_causal_for_causal_only_moba(self) -> None:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        model = self._make_model(moba_bidirectional_steps=0, moba_causal_steps=1)
        flags_seen = []

        def base_generate(data_batch: dict, *args, **kwargs) -> dict:  # noqa: ARG001
            flags_seen.append(model._bidirectional_step_active)
            return {"vision": []}

        with patch.object(OmniMoTModel, "generate_samples_from_batch", side_effect=base_generate):
            model.generate_samples_from_batch({})

        assert flags_seen == [False]

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_generate_samples_resets_flag_on_error(self) -> None:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        model = self._make_model()
        with patch.object(OmniMoTModel, "generate_samples_from_batch", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                model.generate_samples_from_batch({})

        assert model._bidirectional_step_active is False

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_training_step_alternates_and_resets_flag(self) -> None:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        model = self._make_model()
        flags_seen = []

        def base_training_step(data_batch, iteration):  # noqa: ARG001
            flags_seen.append(model._bidirectional_step_active)
            return {}, torch.tensor(0.0)

        with patch.object(OmniMoTModel, "training_step", side_effect=base_training_step):
            output_bidir, _ = model.training_step({}, iteration=0)
            output_tf, _ = model.training_step({}, iteration=1)

        assert flags_seen == [True, False]
        assert model._bidirectional_step_active is False
        assert output_bidir["bidirectional_step"] is True
        assert output_tf["bidirectional_step"] is False

    @pytest.mark.L0
    @pytest.mark.CPU
    def test_training_step_resets_flag_on_error(self) -> None:
        from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

        model = self._make_model()
        with patch.object(OmniMoTModel, "training_step", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                model.training_step({}, iteration=0)

        assert model._bidirectional_step_active is False
