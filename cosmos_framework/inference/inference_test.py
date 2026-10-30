# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


def _diffusion_cache_setup_args(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = dict(
        diffusion_cache=True,
        diffusion_cache_thresh=None,
        diffusion_cache_residual_order=None,
        diffusion_cache_max_consecutive_cached=None,
        quantization_method=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("explicit_config", [False, True])
@pytest.mark.parametrize(
    "window,sink,expected_window,expected_sink",
    [(None, None, 51, 1), (30, 3, 30, 3), (0, 0, None, 0), (30, None, 30, 1), (None, 3, 51, 3)],
)
def test_hf_model_construction_applies_public_streaming_cache_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_config: bool,
    window: int | None,
    sink: int | None,
    expected_window: int | None,
    expected_sink: int,
) -> None:
    monkeypatch.setenv("NVTE_CUDA_INCLUDE_DIR", "")
    import torch

    from cosmos_framework.inference import model as model_module
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.args import ConfigFileType
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "cosmos3_omni",
                "model": {
                    "config": {
                        "kv_cache_inference_size": 51,
                        "attention_sink_size": 1,
                        "vlm_config": {"tokenizer": {}},
                    }
                },
                "per_view_captions": True,
                "per_view_caption_task": "av_joint_camera_lidar_transfer",
            }
        )
    )
    (checkpoint / "model.safetensors").touch()
    # An explicit module config preserves the MODULE/HF route; the default local
    # HF resolver otherwise switches to JSON before model construction.
    module_config = tmp_path / "explicit_config.py"
    module_config.touch()
    saved_config = config_path.read_text()
    setup = OmniSetupOverrides(
        checkpoint_path=str(checkpoint),
        output_dir=tmp_path / "output",
        kv_cache_inference_size=window,
        attention_sink_size=sink,
        diffusion_cache=False,
        use_torch_compile=False,
        use_cuda_graphs=False,
        dp_replicate_size=1,
        dp_shard_size=1,
        cp_size=1,
        cfgp_size=1,
        config_file=str(config_path if explicit_config else module_config),
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=192 * 1024**3)
    assert setup.config_file_type == (ConfigFileType.JSON if explicit_config else ConfigFileType.MODULE)

    constructed: list[tuple[int | None, int]] = []

    def record_construction(self: Cosmos3OmniModel, config: Cosmos3OmniConfig) -> None:
        # Keep the real HF config loading and from_pretrained_dcp construction path;
        # replace only GPU network allocation and checkpoint tensor loading.
        torch.nn.Module.__init__(self)
        # The real DCP loader validates that the model has nonempty state.
        self.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))  # [1]
        self.config = config
        model_config = config.model["config"]
        constructed.append((model_config["kv_cache_inference_size"], model_config["attention_sink_size"]))
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                **model_config,
                rectified_flow_inference_config=SimpleNamespace(scheduler_type=setup.sampler),
            ),
            tokenizer_vision_gen=object(),
        )

    load_weights = Mock()
    monkeypatch.setattr(Cosmos3OmniModel, "__init__", record_construction)
    monkeypatch.setattr(model_module.dcp, "load", load_weights)

    pipe = OmniInference._create(setup, guardrails=None, _timer=None)

    load_weights.assert_called_once()
    assert constructed == [(expected_window, expected_sink)]
    assert pipe.model_config.kv_cache_inference_size == expected_window
    assert pipe.model_config.attention_sink_size == expected_sink
    assert pipe.per_view_captions is True
    assert pipe.model_config._auto_camera_fps == 30
    assert config_path.read_text() == saved_config


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("explicit_config", [False, True])
@pytest.mark.parametrize(
    ("checkpoint_layout", "requested_layout", "expected_layout"),
    [
        (None, "sequence_sharded", None),
        (None, "replicated", "replicated"),
        ("replicated", "sequence_sharded", "sequence_sharded"),
    ],
)
def test_hf_model_construction_applies_attention_io_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit_config: bool,
    checkpoint_layout: str | None,
    requested_layout: str,
    expected_layout: str | None,
) -> None:
    """``--attention-io-layout`` overrides the checkpoint; the default leaves an unset checkpoint unset."""
    monkeypatch.setenv("NVTE_CUDA_INCLUDE_DIR", "")
    import torch

    from cosmos_framework.inference import model as model_module
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    model_config: dict[str, Any] = {"vlm_config": {"tokenizer": {}}}
    if checkpoint_layout is not None:
        model_config["attention_io_layout"] = checkpoint_layout
    config_path.write_text(json.dumps({"model_type": "cosmos3_omni", "model": {"config": model_config}}))
    (checkpoint / "model.safetensors").touch()
    module_config = tmp_path / "explicit_config.py"
    module_config.touch()
    setup = OmniSetupOverrides(
        checkpoint_path=str(checkpoint),
        output_dir=tmp_path / "output",
        attention_io_layout=requested_layout,
        diffusion_cache=False,
        use_torch_compile=False,
        use_cuda_graphs=False,
        dp_replicate_size=1,
        dp_shard_size=1,
        cp_size=1,
        cfgp_size=1,
        config_file=str(config_path if explicit_config else module_config),
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=192 * 1024**3)

    constructed: list[str | None] = []

    def record_construction(self: Cosmos3OmniModel, config: Cosmos3OmniConfig) -> None:
        torch.nn.Module.__init__(self)
        self.register_parameter("weight", torch.nn.Parameter(torch.zeros(1)))  # [1]
        self.config = config
        constructed.append(config.model["config"].get("attention_io_layout"))
        self.model = SimpleNamespace(
            config=SimpleNamespace(rectified_flow_inference_config=SimpleNamespace(scheduler_type=setup.sampler)),
            tokenizer_vision_gen=object(),
        )

    monkeypatch.setattr(Cosmos3OmniModel, "__init__", record_construction)
    monkeypatch.setattr(model_module.dcp, "load", Mock())

    OmniInference._create(setup, guardrails=None, _timer=None)

    assert constructed == [expected_layout]


@pytest.mark.L0
@pytest.mark.CPU
def test_unbatch_transfer_debug_tensor_returns_single_sample() -> None:
    torch = pytest.importorskip("torch")
    from cosmos_framework.inference import inference

    tensor = torch.arange(24).reshape(1, 3, 2, 2, 2)  # [B,C,T,H,W]

    result = inference._unbatch_transfer_debug_tensor(tensor, name="vision")  # [C,T,H,W]

    assert result.shape == (3, 2, 2, 2)
    expected = tensor[0]  # [C,T,H,W]
    assert torch.equal(result, expected)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("vision", (3, 2, 2, 2)),
        ("control_depth", (2, 3, 2, 2, 2)),
        ("control_seg", (1, 1, 3, 2, 2, 2)),
    ],
)
def test_unbatch_transfer_debug_tensor_rejects_invalid_shape(
    name: str,
    shape: tuple[int, ...],
) -> None:
    torch = pytest.importorskip("torch")
    from cosmos_framework.inference import inference

    tensor = torch.zeros(shape)  # [*shape]

    with pytest.raises(ValueError) as error:
        inference._unbatch_transfer_debug_tensor(tensor, name=name)

    assert str(error.value) == f"Transfer debug tensor {name!r} must have shape [1,C,T,H,W]; got {shape}"


def _bf16_model() -> Any:
    torch = pytest.importorskip("torch")
    return torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))


def _modelopt_fp8_model() -> Any:
    torch = pytest.importorskip("torch")
    from cosmos_framework.utils.generator.quantization import _ModelOptFloat8Linear

    # The loader swaps ModelOpt FP8 targets to this module type; its presence is
    # what marks a loaded model as W8A8-quantized.
    return torch.nn.Sequential(torch.nn.Linear(4, 4), _ModelOptFloat8Linear(4, 4, bias=False))


def _patch_cache_install(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    from cosmos_framework.inference import inference
    from cosmos_framework.model.generator.mot import diffusion_cache

    install = Mock()
    warning = Mock()
    monkeypatch.setattr(diffusion_cache, "install_diffusion_cache", install)
    monkeypatch.setattr(inference.log, "warning", warning)
    return install, warning


def test_diffusion_cache_max_consecutive_cached_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference import inference

    install, _warning = _patch_cache_install(monkeypatch)
    pipe = SimpleNamespace(model=_bf16_model())
    setup_args = _diffusion_cache_setup_args(diffusion_cache_max_consecutive_cached=3)

    inference.OmniInference._maybe_install_diffusion_cache(pipe, setup_args)

    install.assert_called_once_with(
        pipe=pipe,
        enabled=True,
        sample_args_list=[],
        config_overrides={"max_consecutive_cached": 3},
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_diffusion_cache_respects_setting_for_modelopt_fp8_model(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    from cosmos_framework.inference import inference

    install, warning = _patch_cache_install(monkeypatch)
    pipe = SimpleNamespace(model=_modelopt_fp8_model())

    inference.OmniInference._maybe_install_diffusion_cache(pipe, _diffusion_cache_setup_args(diffusion_cache=enabled))

    if enabled:
        install.assert_called_once_with(pipe=pipe, enabled=True, sample_args_list=[], config_overrides=None)
    else:
        install.assert_not_called()
    warning.assert_not_called()


@pytest.mark.parametrize("quantization_method", ["mxfp8", "nvfp4"])
@pytest.mark.parametrize("enabled", [False, True])
def test_diffusion_cache_respects_setting_for_runtime_quantization(
    monkeypatch: pytest.MonkeyPatch, quantization_method: str, enabled: bool
) -> None:
    from cosmos_framework.inference import inference

    install, warning = _patch_cache_install(monkeypatch)
    pipe = SimpleNamespace(model=_bf16_model())

    inference.OmniInference._maybe_install_diffusion_cache(
        pipe,
        _diffusion_cache_setup_args(quantization_method=quantization_method, diffusion_cache=enabled),
    )

    if enabled:
        install.assert_called_once_with(pipe=pipe, enabled=True, sample_args_list=[], config_overrides=None)
    else:
        install.assert_not_called()
    warning.assert_not_called()


def test_diffusion_cache_unchanged_for_unquantized_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference import inference

    install, warning = _patch_cache_install(monkeypatch)
    pipe = SimpleNamespace(model=_bf16_model())

    inference.OmniInference._maybe_install_diffusion_cache(pipe, _diffusion_cache_setup_args())

    install.assert_called_once_with(pipe=pipe, enabled=True, sample_args_list=[], config_overrides=None)
    warning.assert_not_called()


def test_finalize_data_batch_does_not_mutate_reusable_video_list() -> None:
    torch = pytest.importorskip("torch")

    from cosmos_framework.inference.inference import _finalize_data_batch

    model = SimpleNamespace(input_video_key="video", input_image_key="images", input_caption_key="caption")
    original_video = torch.zeros(3, 2, 4, 4, dtype=torch.uint8)
    source_batch = {"video": [original_video], "caption": ["prompt"]}

    first_pass = _finalize_data_batch(source_batch, batch_size=1, model=model)
    first_pass["video"][0] = first_pass["video"][0].float() / 127.5 - 1.0
    first_pass["is_preprocessed"] = True
    second_pass = _finalize_data_batch(source_batch, batch_size=1, model=model)

    assert first_pass["video"] is not source_batch["video"]
    assert second_pass["video"] is not source_batch["video"]
    assert source_batch["video"][0] is original_video
    assert source_batch["video"][0].dtype == torch.uint8
    assert "is_preprocessed" not in source_batch


def test_compute_num_tokens_uses_config_when_vision_tokenizer_is_not_loaded() -> None:
    from cosmos_framework.inference.inference import _compute_num_tokens_for_sample

    model = SimpleNamespace(
        tokenizer_vision_gen=None,
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(
                spatial_compression_factor=16,
                temporal_compression_factor=4,
            ),
            diffusion_expert_config=SimpleNamespace(patch_spatial=2),
        ),
    )
    sample_args = SimpleNamespace(vision_size=(256, 128), num_frames=9, multiview=None)

    assert _compute_num_tokens_for_sample(sample_args, model) == 96


def test_separate_vision_decode_gpu_is_ignored_without_vision_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import DEFAULT_CHECKPOINT_NAME, OmniSetupOverrides
    from cosmos_framework.inference.common.args import ConfigFileType

    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
        guardrails=False,
        use_separate_pipeline_vision_decode_gpu=True,
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=80 * 1024**3)
    setup_args.config_file_type = ConfigFileType.MODULE

    model = SimpleNamespace(
        config=SimpleNamespace(
            rectified_flow_inference_config=SimpleNamespace(scheduler_type=setup_args.sampler),
        ),
        tokenizer_vision_gen=None,
    )
    load_model = Mock(return_value=SimpleNamespace(model=model))
    device_count = Mock(side_effect=AssertionError("separate VAE setup should not inspect CUDA devices"))
    monkeypatch.setattr(inference, "_download_on_rank0", lambda _download: tmp_path)
    monkeypatch.setattr(inference.Cosmos3OmniModel, "from_pretrained_dcp", load_model)
    monkeypatch.setattr(inference.torch.cuda, "device_count", device_count)

    pipe = inference.OmniInference._create(setup_args, guardrails=None, _timer=None)

    assert pipe.model is model
    assert pipe.vae_decode_stream is None
    load_model.assert_called_once()
    device_count.assert_not_called()


def _make_v2v_sample_args(**overrides: Any) -> SimpleNamespace:
    """v2v ``OmniSampleArgs`` stand-in for ``get_sample_data`` tests."""
    from cosmos_framework.inference.args import ModelMode, NegativeMetadataMode

    defaults = dict(
        action_path=None,
        aspect_ratio="16,9",
        autoregressive=False,
        camera_trajectory=None,
        condition_frame_indexes_vision=[0, 1],
        condition_video_keep=None,
        condition_vision_mode="video",
        duration_template=None,
        enable_sound=False,
        fps=24,
        inverse_duration_template=None,
        inverse_resolution_template=None,
        # get_sample_data dispatches joint camera+LiDAR before the multiview branch, so the
        # stand-in needs the flag the real OmniSampleArgs exposes.
        is_joint_cam_lidar_transfer=False,
        model_mode=ModelMode.VIDEO2VIDEO,
        multiview=None,
        native_prompt_upsampling=False,
        negative_metadata_mode=NegativeMetadataMode.NONE,
        negative_prompt=None,
        num_frames=125,
        num_outputs=1,
        prompt="prompt",
        resolution_template=None,
        transfer_hints={},
        vision_path="conditioning.mp4",
        vision_size=(32, 16),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    ("condition_video_keep", "expected_loader_keep"),
    [
        ("last", "last"),
        ("first", "first"),
        (None, "first"),  # default
    ],
)
def test_video_conditioning_plumbs_keep_and_pixel_frame_count(
    monkeypatch: pytest.MonkeyPatch,
    condition_video_keep: str | None,
    expected_loader_keep: str,
) -> None:
    """v2v: tokenizer derives pixel-frame count from latent count, ``keep`` passes through to the loader."""
    torch = pytest.importorskip("torch")

    from cosmos_framework.inference import inference

    class Tokenizer:
        calls: list[int]

        def __init__(self) -> None:
            self.calls = []

        def get_pixel_num_frames(self, num_latent_frames: int) -> int:
            self.calls.append(num_latent_frames)
            return 5

    tokenizer = Tokenizer()
    model = SimpleNamespace(
        input_image_key="image",
        input_video_key="video",
        input_caption_key="caption",
        tokenizer_vision_gen=tokenizer,
    )
    sample_args = _make_v2v_sample_args(condition_video_keep=condition_video_keep)
    conditioning_frames = torch.zeros(3, 5, 16, 32)
    sequence_plan = ["sequence-plan"]
    load_conditioning_video_mock = Mock(return_value=conditioning_frames)
    build_conditioned_video_batch_mock = Mock(
        return_value={
            "video": [torch.zeros(1, 3, 125, 16, 32)],
            "sequence_plan": sequence_plan,
        }
    )
    monkeypatch.setattr(inference, "load_conditioning_video", load_conditioning_video_mock)
    monkeypatch.setattr(inference, "build_conditioned_video_batch", build_conditioned_video_batch_mock)

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert tokenizer.calls == [2]  # max([0, 1]) + 1
    load_conditioning_video_mock.assert_called_once_with(
        Path("conditioning.mp4"),
        target_h=16,
        target_w=32,
        max_frames=5,
        keep=expected_loader_keep,
    )
    build_conditioned_video_batch_mock.assert_called_once()
    build_args, build_kwargs = build_conditioned_video_batch_mock.call_args
    assert build_args == (conditioning_frames,)
    assert build_kwargs == {
        "condition_frames_vision": [0, 1],
        "w": 32,
        "h": 16,
        "num_frames": 125,
        "fps": 24,
        "batch_size": 1,
    }
    assert out["sequence_plan"] is sequence_plan


def test_json_prompt_metadata_for_single_frame_omits_temporal_fields() -> None:
    from cosmos_framework.inference.inference import _format_json_prompt_with_template

    prompt = _format_json_prompt_with_template(
        {"subjects": [], "duration": "8s", "fps": 24.0},
        fps=24,
        num_frames=1,
        aspect_ratio="1,1",
        h=1024,
        w=1024,
        include_temporal_metadata=False,
    )

    assert prompt == '{"subjects": [], "resolution": {"H": 1024, "W": 1024}, "aspect_ratio": "1,1"}'
    parsed = json.loads(prompt)
    assert parsed["resolution"] == {"H": 1024, "W": 1024}
    assert parsed["aspect_ratio"] == "1,1"
    assert "duration" not in parsed
    assert "fps" not in parsed


def test_json_prompt_metadata_for_video_keeps_temporal_fields() -> None:
    from cosmos_framework.inference.inference import _format_json_prompt_with_template

    prompt = _format_json_prompt_with_template(
        {"subjects": []},
        fps=24,
        num_frames=189,
        aspect_ratio="16,9",
        h=720,
        w=1280,
        include_temporal_metadata=True,
    )

    assert prompt == (
        '{"subjects": [], "duration": "7s", "fps": 24.0, "resolution": {"H": 720, "W": 1280}, "aspect_ratio": "16,9"}'
    )
    assert json.loads(prompt) == {
        "subjects": [],
        "duration": "7s",
        "fps": 24.0,
        "resolution": {"H": 720, "W": 1280},
        "aspect_ratio": "16,9",
    }


def _make_reasoner_sample_args(**overrides: Any) -> SimpleNamespace:
    """Reasoner ``OmniSampleArgs`` stand-in for ``get_sample_data`` tests."""
    from cosmos_framework.inference.args import ModelMode

    defaults = dict(
        model_mode=ModelMode.REASONER,
        prompt="Describe a robotic arm.",
        vision_path=None,
        video_fps=None,
        max_new_tokens=8,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        num_outputs=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.L0
def test_get_sample_data_reasoner_text_only() -> None:
    from cosmos_framework.inference import inference

    model = SimpleNamespace(input_caption_key="caption")
    sample_args = _make_reasoner_sample_args()

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert out == {
        "caption": ["Describe a robotic arm."],
        "reasoner_images": [None],
        "reasoner_videos": [None],
    }


@pytest.mark.L0
def test_get_sample_data_reasoner_with_image(tmp_path: Path) -> None:
    PIL = pytest.importorskip("PIL.Image")
    from cosmos_framework.inference import inference

    img_path = tmp_path / "arm.png"
    PIL.new("RGB", (8, 8), color="red").save(img_path)

    model = SimpleNamespace(input_caption_key="caption")
    sample_args = _make_reasoner_sample_args(vision_path=str(img_path))

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert list(out) == ["caption", "reasoner_images", "reasoner_videos"]
    assert out["caption"] == ["Describe a robotic arm."]
    assert out["reasoner_videos"] == [None]
    assert len(out["reasoner_images"]) == 1
    assert out["reasoner_images"][0].size == (8, 8)
    assert out["reasoner_images"][0].mode == "RGB"


@pytest.mark.L0
def test_get_sample_data_reasoner_with_video(monkeypatch: pytest.MonkeyPatch) -> None:
    """A video ``vision_path`` routes through ``_decode_reasoner_video`` into ``reasoner_videos``.

    The decoder is monkeypatched (real decode needs torchvision + an actual clip);
    this asserts the routing/contract, not the decode itself."""
    from cosmos_framework.inference import inference

    decoded = {"frames": ["F0", "F1"], "fps": 2.0}
    monkeypatch.setattr(inference, "_decode_reasoner_video", lambda path, fps: decoded)
    model = SimpleNamespace(input_caption_key="caption")
    sample_args = _make_reasoner_sample_args(vision_path="/tmp/clip.mp4", video_fps=2.0)

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert out["caption"] == ["Describe a robotic arm."]
    assert out["reasoner_videos"] == [decoded]
    assert out["reasoner_images"] == [None]


@pytest.mark.L0
def test_reasoner_defaults_json_round_trip() -> None:
    import json as _json

    from cosmos_framework.inference.args import PACKAGE_DIR, _load_modality_defaults

    defaults = _load_modality_defaults("reasoner")
    assert defaults["model_mode"] == "reasoner"
    assert defaults["max_new_tokens"] == 1024
    on_disk = _json.loads((PACKAGE_DIR / "defaults/reasoner/sample_args.json").read_text())
    assert defaults == on_disk


@pytest.mark.L0
def test_reasoner_overrides_round_trip() -> None:
    import pydantic

    from cosmos_framework.inference.args import ModelMode, ReasonerDataOverrides

    overrides = ReasonerDataOverrides(max_new_tokens=128, temperature=0.7, top_p=0.9)
    assert overrides.max_new_tokens == 128
    assert overrides.temperature == 0.7
    assert overrides.top_p == 0.9
    with pytest.raises(pydantic.ValidationError):
        ReasonerDataOverrides(top_p=1.5)
    with pytest.raises(pydantic.ValidationError):
        ReasonerDataOverrides(temperature=0)
    assert ModelMode.REASONER.is_reasoner
    assert not ModelMode.TEXT2VIDEO.is_reasoner
    assert not ModelMode.REASONER.is_action


@pytest.mark.L0
def test_generate_reasoner_batch_writes_outputs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import ModelMode

    out_dir = tmp_path / "hello"

    class _SA(SimpleNamespace):
        def model_dump(self, **_):
            return {"name": self.name, "model_mode": self.model_mode}

        def model_dump_json(self, **_):
            import json as _json

            return _json.dumps(self.model_dump())

    sample_args = _SA(
        name="hello",
        model_mode=ModelMode.REASONER,
        output_dir=out_dir,
        prompt="Describe a robotic arm.",
        max_new_tokens=8,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        seed=None,
    )

    def _fake_generate_reasoner_text(prompts, *, images=None, **kwargs):
        assert prompts == ["Describe a robotic arm."]
        assert images is None
        return ["A six-axis arm with a parallel-jaw gripper."]

    model = SimpleNamespace(
        input_caption_key="caption",
        generate_reasoner_text=_fake_generate_reasoner_text,
    )

    pipe = inference.OmniInference.__new__(inference.OmniInference)
    pipe.model = model
    pipe.should_process_sample = lambda sa: True  # type: ignore[attr-defined]
    pipe._run_text_guardrail = lambda *_a, **_kw: None  # type: ignore[attr-defined]
    pipe._handle_sample_exception = lambda sa, e: (_ for _ in ()).throw(e)  # type: ignore[attr-defined]

    from contextlib import nullcontext

    pipe._get_timer = lambda *_a, **_kw: nullcontext()  # type: ignore[attr-defined]

    data_batch = {"caption": ["Describe a robotic arm."], "reasoner_images": [None]}
    results = pipe._generate_reasoner_batch([sample_args], data_batch)

    assert len(results) == 1
    so = results[0]
    assert so.outputs[0].content == {"reasoner_text": "A six-axis arm with a parallel-jaw gripper."}
    txt_file = out_dir / "reasoner_text.txt"
    assert txt_file.read_text() == "A six-axis arm with a parallel-jaw gripper."
    assert (out_dir / "sample_args.json").is_file()
    assert (out_dir / "sample_outputs.json").is_file()


@pytest.mark.L0
def test_generate_reasoner_batch_rejects_mixed_image_text_only(tmp_path: Path) -> None:
    PIL = pytest.importorskip("PIL.Image")
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import ModelMode

    pipe = inference.OmniInference.__new__(inference.OmniInference)
    pipe.model = SimpleNamespace(input_caption_key="caption")
    pipe.should_process_sample = lambda sa: False  # type: ignore[attr-defined]

    sa1 = SimpleNamespace(model_mode=ModelMode.REASONER, output_dir=tmp_path / "a")
    sa2 = SimpleNamespace(model_mode=ModelMode.REASONER, output_dir=tmp_path / "b")

    data_batch = {
        "caption": ["p1", "p2"],
        "reasoner_images": [PIL.new("RGB", (8, 8)), None],
    }
    with pytest.raises(ValueError, match="mixes image-conditioned and text-only"):
        pipe._generate_reasoner_batch([sa1, sa2], data_batch)


@pytest.mark.L0
def test_reasoner_build_rejects_empty_prompt() -> None:
    from cosmos_framework.inference.args import ModelMode, OmniSampleOverrides, SampleMeta, VisionMode

    overrides = OmniSampleOverrides(prompt="   ")
    meta = SampleMeta(model_mode=ModelMode.REASONER, vision_mode=VisionMode.IMAGE, condition_vision_mode=None)
    with pytest.raises(ValueError, match="non-empty 'prompt'"):
        overrides._build_reasoner_data(model_config=None, sample_meta=meta)


@pytest.mark.L0
def test_reasoner_defaults_validate_against_overrides() -> None:
    """The defaults JSON must validate against ``OmniSampleOverrides`` so
    ``build_sample`` cannot silently drop a field after an upstream rename."""
    from cosmos_framework.inference.args import OmniSampleOverrides, _load_modality_defaults

    defaults = _load_modality_defaults("reasoner")
    filtered = {k: v for k, v in defaults.items() if k in OmniSampleOverrides.model_fields}
    assert set(defaults) - set(filtered) == set(), f"defaults has unknown fields: {set(defaults) - set(filtered)}"
    OmniSampleOverrides.model_validate(filtered)


def _prompt_sample_args(**overrides: Any) -> SimpleNamespace:
    """Only the fields _format_sample_prompt reads."""
    fields: dict[str, Any] = dict(
        fps=30,
        num_frames=93,
        aspect_ratio="16,9",
        native_prompt_upsampling=False,
        resolution_template=None,
        num_outputs=1,
        negative_prompt=None,
        prompt="a merged prompt.",
        duration_template=None,
        inverse_duration_template=None,
        negative_metadata_mode=None,
        autoregressive=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.L0
@pytest.mark.CPU
def test_native_upsampled_prompt_is_not_stripped() -> None:
    """The upstream codebase returns this branch's prompt verbatim; stripping it here diverged the token ids.

    Latent at the defaults -- native_prompt_upsampling is False -- and only visible for a non-JSON
    prompt carrying leading or trailing whitespace. Where it is not latent, the two runtimes
    tokenize different text from the same input, which nothing downstream would reveal.
    """
    from cosmos_framework.inference.inference import _format_sample_prompt

    raw = "  a car drives along a tree-lined road.  "
    formatted = _format_sample_prompt(
        raw,
        _prompt_sample_args(native_prompt_upsampling=True),
        h=480,
        w=832,
        duration_template=None,
    )

    assert formatted == raw


@pytest.mark.L0
@pytest.mark.CPU
def test_a_json_prompt_still_takes_the_json_branch() -> None:
    """Extraction must not change which branch a prompt takes; JSON wins over upsampling."""
    from cosmos_framework.inference.inference import _format_sample_prompt

    raw = json.dumps({"caption": "a car drives"})
    formatted = _format_sample_prompt(
        raw,
        _prompt_sample_args(native_prompt_upsampling=True),
        h=480,
        w=832,
        duration_template=None,
    )

    assert formatted != raw
    assert json.loads(formatted)["caption"] == "a car drives"


@pytest.mark.L0
@pytest.mark.CPU
def test_a_plain_prompt_is_decorated_with_the_template() -> None:
    from cosmos_framework.inference.inference import _format_prompt_with_template, _format_sample_prompt

    sample_args = _prompt_sample_args()
    raw = "a car drives along a tree-lined road."

    assert _format_sample_prompt(raw, sample_args, h=480, w=832, duration_template="{duration}s.") == (
        _format_prompt_with_template(
            raw,
            fps=sample_args.fps,
            num_frames=sample_args.num_frames,
            duration_template="{duration}s.",
            resolution_template=sample_args.resolution_template,
            h=480,
            w=832,
        )
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_every_caption_decorates_independently() -> None:
    """What the per-view path needs: N captions decorated once each, not one decorated N times."""
    from cosmos_framework.inference.inference import _format_sample_prompt

    sample_args = _prompt_sample_args()
    captions = ["the road ahead.", "a parked van."]
    formatted = [
        _format_sample_prompt(caption, sample_args, h=480, w=832, duration_template="{duration}s.")
        for caption in captions
    ]

    assert len(set(formatted)) == 2
    assert all(caption.rstrip(".") in out for caption, out in zip(captions, formatted))


def _caption_model(key: str = "ai_caption") -> SimpleNamespace:
    return SimpleNamespace(input_caption_key=key)


@pytest.mark.L0
@pytest.mark.CPU
def test_per_view_captions_use_whole_second_durations() -> None:
    """Multiview training wrote whole seconds; a fractional duration is text the weights never saw.

    297 frames at 30 fps is 9.9 seconds. The JSON branch already truncates
    (``int(num_frames / fps)`` in _format_json_prompt_with_template), so before this the two
    branches disagreed with each other as well as with the upstream codebase: a JSON caption read "9s" and a
    plain-text one "9.9s" for the same clip.

    Measured rather than theoretical: this one character moved every decoded pixel of an 11-camera
    run -- 0/11 views identical, mean 5.05/255 against the upstream codebase -- while leaving a video that
    looks entirely correct.
    """
    from cosmos_framework.inference.inference import _get_per_view_prompt_sample_data

    out = _get_per_view_prompt_sample_data(
        _prompt_sample_args(fps=30, num_frames=297),
        _caption_model(),
        ["the road ahead.", "a parked van."],
        h=480,
        w=832,
        duration_template="{duration} seconds.",
    )

    captions = out["ai_caption"][0]
    assert all("9 seconds." in caption for caption in captions), captions
    assert not any("9.9" in caption for caption in captions), captions


@pytest.mark.L0
@pytest.mark.CPU
def test_per_view_captions_are_nested_one_level_deeper() -> None:
    """The model reads a list of captions per sample here, not one prompt per sample.

    Shape is (num_outputs, num_cameras); flattening it would hand the rig's captions to the
    tokenizer as separate samples.
    """
    from cosmos_framework.inference.inference import _get_per_view_prompt_sample_data

    out = _get_per_view_prompt_sample_data(
        _prompt_sample_args(num_outputs=2, negative_prompt=None),
        _caption_model(),
        ["the road ahead.", "a parked van.", "a cyclist."],
        h=480,
        w=832,
        duration_template=None,
    )

    captions = out["ai_caption"]
    assert len(captions) == 2
    assert all(len(per_sample) == 3 for per_sample in captions)
    assert all(isinstance(caption, str) for per_sample in captions for caption in per_sample)


@pytest.mark.L0
@pytest.mark.CPU
def test_no_negative_caption_key_is_emitted() -> None:
    """The omission is the feature.

    With no neg_<caption_key> the model falls to one empty caption per (sample, camera), which is
    the unconditional branch training's caption dropout produced. Emitting the sample's real
    negative would instead fan it out across the cameras via _expand_negative_captions,
    conditioning the unconditional pass on text these weights were never trained against --
    invisibly, since the output is still a well-formed video.
    """
    from cosmos_framework.inference.inference import _get_per_view_prompt_sample_data

    out = _get_per_view_prompt_sample_data(
        _prompt_sample_args(num_outputs=1, negative_prompt="blurry, low quality"),
        _caption_model(),
        ["the road ahead.", "a parked van."],
        h=480,
        w=832,
        duration_template=None,
    )

    assert set(out) == {"ai_caption"}
    assert "neg_ai_caption" not in out


@pytest.mark.L0
@pytest.mark.CPU
def test_each_camera_caption_is_decorated_on_its_own() -> None:
    """The merged path's decoration, once per caption rather than once per sample.

    With one deliberate difference: the per-view path truncates the duration to whole seconds,
    because that is what multiview training wrote. Passing ``truncate_duration=True`` here keeps
    this test about *per-caption* decoration; the truncation itself is pinned by
    test_per_view_captions_use_whole_second_durations.
    """
    from cosmos_framework.inference.inference import _format_sample_prompt, _get_per_view_prompt_sample_data

    sample_args = _prompt_sample_args(num_outputs=1, negative_prompt=None)
    view_prompts = ["the road ahead.", "a parked van."]

    out = _get_per_view_prompt_sample_data(
        sample_args, _caption_model(), view_prompts, h=480, w=832, duration_template="{duration}s."
    )

    assert out["ai_caption"][0] == [
        _format_sample_prompt(p, sample_args, h=480, w=832, duration_template="{duration}s.", truncate_duration=True)
        for p in view_prompts
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_a_sample_with_no_camera_captions_raises() -> None:
    """Per-view captions were requested; silently generating from nothing is the failure to avoid."""
    from cosmos_framework.inference.inference import _get_per_view_prompt_sample_data

    with pytest.raises(ValueError, match="no camera captions"):
        _get_per_view_prompt_sample_data(
            _prompt_sample_args(num_outputs=1, negative_prompt=None),
            _caption_model(),
            [],
            h=480,
            w=832,
            duration_template=None,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_the_caption_key_follows_the_model() -> None:
    from cosmos_framework.inference.inference import _get_per_view_prompt_sample_data

    out = _get_per_view_prompt_sample_data(
        _prompt_sample_args(num_outputs=1, negative_prompt=None),
        _caption_model("t5_text_embeddings"),
        ["a."],
        h=480,
        w=832,
        duration_template=None,
    )

    assert set(out) == {"t5_text_embeddings"}


@pytest.mark.L0
@pytest.mark.CPU
def test_view_prompts_route_to_the_per_view_builder() -> None:
    """The dispatch: given view_prompts, the merged path must not run at all."""
    from cosmos_framework.inference.inference import _get_per_view_prompt_sample_data, _get_prompt_sample_data

    sample_args = _prompt_sample_args(prompt="merged text.")
    view_prompts = ["the road ahead.", "a parked van."]

    routed = _get_prompt_sample_data(
        sample_args, _caption_model(), h=480, w=832, device="cpu", view_prompts=view_prompts
    )

    assert routed == _get_per_view_prompt_sample_data(
        sample_args, _caption_model(), view_prompts, h=480, w=832, duration_template=None
    )
    assert "neg_ai_caption" not in routed
    assert "merged text." not in json.dumps(routed)


def test_compile_config_carries_cuda_graph_scope() -> None:
    """The CLI's cuda_graph_scope reaches CompileConfig unchanged."""
    from cosmos_framework.inference import inference

    setup_args = SimpleNamespace(
        use_torch_compile=True,
        use_cuda_graphs=True,
        dp_shard_size=1,
        cp_size=1,
        cfgp_size=1,
        compiled_region="language",
        compile_dynamic=True,
        cuda_graph_scope="forward",
        ar_post_saturation_mode="static-compile",
    )
    compile_config = inference.OmniInference._get_compile_config(setup_args)
    assert compile_config.enabled is True
    assert compile_config.use_cuda_graphs is True
    assert compile_config.cuda_graph_scope == "forward"
    assert compile_config.ar_post_saturation_mode == "static-compile"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("granularity", ["per_row", "per_tensor"])
def test_quantization_config_carries_fp8_granularity(tmp_path: Path, granularity: str) -> None:
    """``--quantization-fp8-granularity`` reaches the model's QuantizationConfig unchanged."""
    from cosmos_framework.inference.args import DEFAULT_CHECKPOINT_NAME, OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference

    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
        dp_shard_size=1,
        quantization_method="fp8",
        quantization_fp8_granularity=granularity,
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=80 * 1024**3)

    quantization_config = OmniInference._get_quantization_config(setup_args)

    assert quantization_config.method == "fp8"
    assert quantization_config.fp8_granularity == granularity


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_defaults_keep_existing_runtime_behaviour(tmp_path: Path) -> None:
    """The newly exposed setup flags default to what the model ran with before they existed."""
    from cosmos_framework.inference.args import DEFAULT_CHECKPOINT_NAME, OmniSetupOverrides
    from cosmos_framework.inference.inference import OmniInference

    setup_args = OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=80 * 1024**3)

    assert setup_args.use_batched_cfg is False
    assert setup_args.attention_io_layout == "sequence_sharded"
    assert OmniInference._get_compile_config(setup_args).ar_post_saturation_mode == "default"
    quantization_config = OmniInference._get_quantization_config(setup_args)
    assert quantization_config.method is None
    assert quantization_config.fp8_granularity == "per_row"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("save_preview", [False, True])
def test_save_transfer_vision_writes_preview_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, save_preview: bool
) -> None:
    torch = pytest.importorskip("torch")
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import ModelMode

    calls: list[dict[str, Any]] = []

    def record(tensor: Any, path: str, **kwargs: Any) -> None:
        calls.append({"path": path, **kwargs})

    monkeypatch.setattr(inference, "save_img_or_video", record)
    sample_args = SimpleNamespace(
        model_mode=ModelMode.TEXT2VIDEO,
        video_save_quality=8,
        image_save_quality=95,
        save_preview=save_preview,
        preview_crf=30,
        preview_preset="fast",
    )

    written = inference._save_transfer_vision(
        torch.zeros(3, 5, 8, 8),  # [C,T,H,W]
        tmp_path / "vision.mp4",
        fps=24,
        sample_args=sample_args,
    )

    if save_preview:
        assert written == [tmp_path / "vision.mp4", tmp_path / "vision_preview.mp4"]
        assert calls[-1] == {
            "path": str(tmp_path / "vision_preview"),
            "fps": 24,
            "quality": None,
            "crf": 30,
            "preset": "fast",
            "ffmpeg_params": None,
        }
    else:
        assert written == [tmp_path / "vision.mp4"]
        assert len(calls) == 1


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("cp_size", "cfgp_size", "separate_vae_gpu", "expected_group"),
    [(4, 1, False, 4), (2, 2, False, 4), (1, 1, False, 1), (4, 1, True, 1)],
)
def test_parallelism_config_sizes_the_vae_sharing_group_to_one_replica(
    cp_size: int, cfgp_size: int, separate_vae_gpu: bool, expected_group: int
) -> None:
    """The replica's cp x cfgp ranks share VAE work, unless only one of them runs the VAE."""
    from cosmos_framework.inference.inference import OmniInference

    setup_args = SimpleNamespace(
        dp_shard_size=1,
        cp_size=cp_size,
        cfgp_size=cfgp_size,
        use_separate_pipeline_vision_decode_gpu=separate_vae_gpu,
    )

    config = OmniInference._get_parallelism_config(setup_args)

    assert config.enable_inference_mode is True
    assert config.data_parallel_shard_degree == 1
    assert config.context_parallel_shard_degree == cp_size
    assert config.cfg_parallel_shard_degree == cfgp_size
    assert config.vae_load_balance_group_size == expected_group
