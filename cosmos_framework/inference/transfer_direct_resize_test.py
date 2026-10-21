# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Direct-resize integration regressions ported from the i4 Transfer workflow."""

import json
from collections.abc import Generator, Iterator
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

import cosmos_framework.inference.args as args_module
from cosmos_framework.inference import transfer
from cosmos_framework.inference.args import ModelMode, OmniSampleOverrides, TransferHintKey
from cosmos_framework.inference.transfer_preprocessing import TransferMedia

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _fake_vfm_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        action_gen=False,
        sound_gen=False,
        resolution="480",
        tokenizer=SimpleNamespace(temporal_compression_factor=4),
        video_temporal_causal=False,
        vlm_config=SimpleNamespace(model_name="Qwen/Qwen3-VL-8B-Instruct"),
    )


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("resize_mode", ["center_crop", "direct_resize"])
def test_transfer_resize_policy_is_opt_in_and_autodetects_direct_aspect(
    tmp_path: Path, resize_mode: Literal["center_crop", "direct_resize"]
) -> None:
    control_path = tmp_path / "depth.mp4"
    control_path.touch()
    args = OmniSampleOverrides(
        name="direct_geometry",
        output_dir=tmp_path / "output",
        prompt="A controlled scene.",
        model_mode=ModelMode.VIDEO2VIDEO,
        depth={"control_path": str(control_path)},
        transfer_resize_mode=resize_mode,
    ).build_sample(model_config=_fake_vfm_model_config())
    assert args.transfer_resize_mode == resize_mode
    assert args.transfer_output_size == "model"
    assert args.aspect_ratio == (None if resize_mode == "direct_resize" else "16,9")


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_transfer_preserves_explicit_aspect_and_original_output(tmp_path: Path) -> None:
    control_path = tmp_path / "depth.mp4"
    control_path.touch()
    args = OmniSampleOverrides(
        name="direct_geometry",
        output_dir=tmp_path / "output",
        prompt="A controlled scene.",
        model_mode=ModelMode.VIDEO2VIDEO,
        depth={"control_path": str(control_path)},
        aspect_ratio="4,3",
        transfer_resize_mode="direct_resize",
        transfer_output_size="original",
    ).build_sample(model_config=_fake_vfm_model_config())
    assert args.aspect_ratio == "4,3"
    assert args.transfer_output_size == "original"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("resolution", "shift"), [("480", 5.0), ("720", 10.0)])
@pytest.mark.parametrize("shift_source", ["builtin", "sample", "defaults_file"])
@pytest.mark.parametrize("model_name", ["Qwen/Qwen3-VL-8B-Instruct", "nvidia/Cosmos3-Nano-Reasoner"])
def test_direct_stage1_transfer_enables_framewise_ar(
    tmp_path: Path, resolution: str, shift: float, shift_source: str, model_name: str
) -> None:
    config = _fake_vfm_model_config()
    config.vlm_config.model_name = model_name
    config.video_temporal_causal = True
    config.causal_training_strategy = "teacher_forcing"
    config.teacher_forcing_frames_per_chunk = 1
    control_path = tmp_path / "depth.mp4"
    control_path.touch()
    overrides: dict[str, Any] = {}
    if shift_source == "sample":
        overrides["shift"] = 7.0
    elif shift_source == "defaults_file":
        defaults = args_module._load_modality_defaults("video2video").copy()
        defaults["shift"] = 8.0
        defaults_file = tmp_path / "defaults.json"
        defaults_file.write_text(json.dumps(defaults))
        overrides["defaults_file"] = defaults_file
    args = OmniSampleOverrides(
        name="stage1_direct",
        output_dir=tmp_path / "output",
        prompt="A controlled scene.",
        model_mode=ModelMode.VIDEO2VIDEO,
        resolution=resolution,
        depth={"control_path": str(control_path)},
        transfer_resize_mode="direct_resize",
        transfer_output_size="original",
        control_guidance=1.0,
        **overrides,
    ).build_sample(model_config=config)
    assert args.autoregressive is True
    assert args.aspect_ratio == ("16,9" if shift_source == "defaults_file" else None)
    assert args.shift == {"builtin": shift, "sample": 7.0, "defaults_file": 8.0}[shift_source]
    assert args.num_first_chunk_conditional_frames == 0
    assert args.share_vision_temporal_positions is True
    assert args.transfer_output_size == "original"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("config_override", "sample_override", "message"),
    [
        ({"causal_training_strategy": "teacher_forcing_dcm"}, {}, "Stage 1 teacher_forcing"),
        ({"simulation_mode": "self_forcing"}, {}, "Stage 3 SF-DMD"),
        (
            {
                "causal_training_strategy": "teacher_forcing_dcm",
                "fixed_step_sampler_config": SimpleNamespace(t_list=[1.0], sample_type="sde"),
                "simulation_mode": "self_forcing",
            },
            {"autoregressive": False},
            "Stage 3 SF-DMD",
        ),
        (
            {"fixed_step_sampler_config": SimpleNamespace(t_list=[1.0], sample_type="sde")},
            {},
            "Stage 1 teacher_forcing",
        ),
        ({"teacher_forcing_frames_per_chunk": 2}, {}, "one latent frame"),
        ({}, {"autoregressive": False}, "Stage 1 teacher_forcing"),
        ({}, {"num_first_chunk_conditional_frames": 1}, "exactly one control hint"),
        ({}, {"share_vision_temporal_positions": False}, "share_vision_temporal_positions"),
        ({}, {"control_guidance": 2.0}, "control_guidance=1.0"),
        ({}, {"edge": {}}, "exactly one control hint"),
    ],
)
def test_direct_causal_transfer_rejects_unsupported_contract(
    tmp_path: Path, config_override: dict[str, Any], sample_override: dict[str, Any], message: str
) -> None:
    config = _fake_vfm_model_config()
    config.video_temporal_causal = True
    config.causal_training_strategy = "teacher_forcing"
    config.teacher_forcing_frames_per_chunk = 1
    for key, value in config_override.items():
        setattr(config, key, value)
    control_path = tmp_path / "depth.mp4"
    control_path.touch()
    values: dict[str, Any] = {
        "name": "invalid_stage1_direct",
        "output_dir": tmp_path / "output",
        "model_mode": ModelMode.VIDEO2VIDEO,
        "depth": {"control_path": str(control_path)},
        "transfer_resize_mode": "direct_resize",
        "control_guidance": 1.0,
        **sample_override,
    }
    with pytest.raises(ValueError, match=message):
        OmniSampleOverrides(**values).build_sample(model_config=config)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("has_rgb", [False, True])
@pytest.mark.parametrize("conditional_frames", [None, 0])
def test_direct_transfer_loads_one_geometry_and_decoded_fps(
    monkeypatch: pytest.MonkeyPatch, has_rgb: bool, conditional_frames: int | None
) -> None:
    pixels = torch.zeros((3, 3, 480, 832), dtype=torch.uint8)  # [3,3,H,W]
    media = TransferMedia(pixels, 30.0, (1080, 1920), (0.0, 1 / 30, 2 / 30), True)
    read_media = Mock(return_value=media)
    monkeypatch.setattr(transfer, "read_direct_transfer_media", read_media)
    monkeypatch.setattr(transfer, "probe_media_hw", Mock(return_value=(1080, 1920)))
    legacy_read = Mock(side_effect=AssertionError("direct mode must not invoke center cropping"))
    monkeypatch.setattr(transfer, "read_and_resize_media", legacy_read)
    sample = SimpleNamespace(
        transfer_hints={
            TransferHintKey.DEPTH: SimpleNamespace(control_path=Path("depth.mp4")),
            TransferHintKey.SEG: SimpleNamespace(control_path=Path("seg.mp4")),
        },
        vision_path=Path("rgb.mp4") if has_rgb else None,
        num_first_chunk_conditional_frames=conditional_frames,
        resolution="480",
        aspect_ratio=None,
        max_frames=3,
    )
    prepared = transfer._load_direct_transfer_inputs(sample)
    rgb, controls, fps, geometry = prepared.input_frames, prepared.control_frames, prepared.fps, prepared.geometry
    assert (rgb is pixels) if has_rgb else (rgb is None)
    assert set(controls) == {TransferHintKey.DEPTH, TransferHintKey.SEG}
    assert fps == 30.0
    assert geometry.original_hw == (1080, 1920)
    assert geometry.target_hw == (480, 832)
    calls = read_media.call_args_list
    assert [call.kwargs["modality"] for call in calls] == (["rgb"] if has_rgb else []) + ["depth", "seg"]
    assert all(call.kwargs["geometry"] is geometry for call in calls)
    legacy_read.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_transfer_computes_edge_from_resized_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    pixels = torch.zeros((3, 3, 480, 832), dtype=torch.uint8)  # [3,3,H,W]
    media = TransferMedia(pixels, 24.0, (1080, 1920), (0.0, 1 / 24, 2 / 24), True)
    monkeypatch.setattr(transfer, "read_direct_transfer_media", Mock(return_value=media))
    monkeypatch.setattr(transfer, "probe_media_hw", Mock(return_value=(1080, 1920)))
    compute_control = Mock(return_value=pixels)
    monkeypatch.setattr(transfer, "load_transfer_control_frames", compute_control)
    sample = SimpleNamespace(
        transfer_hints={TransferHintKey.EDGE: SimpleNamespace(control_path=None)},
        vision_path=Path("rgb.mp4"),
        num_first_chunk_conditional_frames=0,
        resolution="480",
        aspect_ratio=None,
        max_frames=3,
    )
    transfer._load_direct_transfer_inputs(sample)
    assert compute_control.call_args.kwargs["input_frames"] is pixels


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("strategy", "resolution", "target_hw"),
    [
        ("teacher_forcing", "480", (480, 832)),
        ("teacher_forcing", "720", (720, 1280)),
        ("teacher_forcing_dcm", "480", (480, 832)),
        ("none", "480", (480, 832)),
    ],
)
def test_direct_causal_transfer_uses_shared_geometry_and_framewise_ar(
    monkeypatch: pytest.MonkeyPatch, strategy: str, resolution: str, target_hw: tuple[int, int]
) -> None:
    """Run direct preparation through the real AR wrapper and retain fractional depth."""
    pixels = torch.arange(6, dtype=torch.float32).view(1, 6, 1, 1).expand(3, 6, *target_hw) + 128.25  # [C,T,H,W]
    media = TransferMedia(pixels, 29.97, (1080, 1920), tuple(index / 29.97 for index in range(6)), True)
    read_media = Mock(return_value=media)
    monkeypatch.setattr(transfer, "read_direct_transfer_media", read_media)
    monkeypatch.setattr(transfer, "probe_media_hw", Mock(return_value=(1080, 1920)))
    legacy_read = Mock(side_effect=AssertionError("Direct causal resizing must not center crop"))
    monkeypatch.setattr(transfer, "read_and_resize_media", legacy_read)
    prompt_data = Mock(return_value={"caption": ["prompt"]})
    monkeypatch.setattr(transfer, "_get_prompt_sample_data", prompt_data)
    build_batch = Mock(return_value={"prepared": True})
    monkeypatch.setattr(transfer, "build_transfer_batch", build_batch)
    ar_calls: list[dict[str, Any]] = []

    def iter_ar(**kwargs: Any) -> Iterator[dict[str, torch.Tensor]]:
        ar_calls.append(kwargs)
        for frame_count in (1, 4, 4):
            decoded = torch.zeros(1, 3, frame_count, *target_hw)  # [B,C,T,H,W]
            kwargs["on_clean_vision_chunk"](decoded)
            yield {"vision": decoded}

    def decode(latent: torch.Tensor) -> torch.Tensor:  # [B,C,T,H,W] -> [B,C,T,H,W]
        return latent

    model = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer=SimpleNamespace(temporal_compression_factor=4),
            teacher_forcing_frames_per_chunk=1,
            causal_training_strategy=strategy,
            simulation_mode="self_forcing" if strategy == "none" else None,
            video_temporal_causal=True,
            fixed_step_sampler_config={"t_list": [1.0, 15 / 16, 5 / 6, 5 / 8], "sample_type": "sde"}
            if strategy != "teacher_forcing"
            else None,
        ),
        fixed_step_sampler=Mock(),
        input_caption_key="caption",
        eval=Mock(),
        tokenizer_vision_gen=SimpleNamespace(use_cached_decoder=nullcontext),
        iter_samples_from_batch_autoregressive=iter_ar,
        decode=decode,
    )
    sample = SimpleNamespace(
        transfer_hints={TransferHintKey.DEPTH: SimpleNamespace(control_path=Path("depth.mp4"))},
        transfer_resize_mode="direct_resize",
        resolution=resolution,
        vision_path=None,
        aspect_ratio=None,
        max_frames=6,
        num_video_frames_per_chunk=5,
        num_conditional_frames=1,
        num_first_chunk_conditional_frames=0,
        fps=24,
        autoregressive=True,
        emphasize_control_in_prompt=False,
        seed=7,
        guidance=4.0,
        control_guidance=1.0,
        share_vision_temporal_positions=True,
        num_steps=35,
        shift=5.0 if resolution == "480" else 10.0,
        normalize_cfg=False,
        show_control_condition=False,
        show_input=False,
        model_copy=Mock(return_value=SimpleNamespace()),
    )
    prepared = transfer._load_direct_transfer_inputs(cast(Any, sample))
    assert prepared.geometry.target_hw == target_hw
    output = transfer.generate_transfer_sample(cast(Any, sample), cast(Any, model), prepared_direct_inputs=prepared)

    read_media.assert_called_once()
    legacy_read.assert_not_called()
    assert len(ar_calls) == 1
    assert ar_calls[0]["mode"] == "video_transfer"
    distilled = strategy != "teacher_forcing"
    assert ar_calls[0]["sampler_mode"] == ("distilled" if distilled else "rf")
    assert ar_calls[0]["guidance"] == (1.0 if distilled else sample.guidance)
    assert ar_calls[0]["shift"] == sample.shift
    assert output.output_video.shape == (1, 3, 6, *target_hw)
    assert output.spatial_geometry is prepared.geometry
    assert output.original_hw == (1080, 1920)
    assert output.fps == 29.97
    normalized = build_batch.call_args.kwargs["control_videos"][0]  # [C,T,H,W]
    assert normalized.dtype == torch.float32
    assert normalized.shape == (3, 9, *target_hw)
    assert build_batch.call_args.kwargs["num_frames"] == 9
    torch.testing.assert_close(normalized[:, :6], pixels / 127.5 - 1.0)
    # Reflection includes the final source frame, then walks backward through time.
    torch.testing.assert_close(normalized[:, 6:], (pixels[:, [5, 4, 3]] / 127.5 - 1.0))
    torch.testing.assert_close(output.control_videos[TransferHintKey.DEPTH], normalized[:, :6].unsqueeze(0))
    assert build_batch.call_args.kwargs["share_vision_temporal_positions"] is True
    assert sample.model_copy.call_args.kwargs["update"]["num_frames"] == 6


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("conditional_frames", [1, 5])
def test_direct_transfer_rejects_missing_rgb_before_probing_or_decoding(
    monkeypatch: pytest.MonkeyPatch, conditional_frames: int
) -> None:
    probe = Mock()
    decode = Mock()
    monkeypatch.setattr(transfer, "probe_media_hw", probe)
    monkeypatch.setattr(transfer, "read_direct_transfer_media", decode)
    sample = SimpleNamespace(vision_path=None, num_first_chunk_conditional_frames=conditional_frames)

    with pytest.raises(ValueError, match="requires 'vision_path'"):
        transfer._load_direct_transfer_inputs(cast(Any, sample))

    probe.assert_not_called()
    decode.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("output_size", ["model", "original"])
def test_save_transfer_rgb_retains_native_output_and_records_geometry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output_size: str
) -> None:
    torch = pytest.importorskip("torch")
    from cosmos_framework.inference import inference

    source = torch.rand(3, 3, 4, 8)  # [C,T,H_model,W_model]
    saved: dict[str, Any] = {}

    def save_vision(video: Any, path: Path, *, fps: float, sample_args: Any) -> list[Path]:  # video: [C,T,H,W]
        assert fps == 29.97
        saved[path.name] = video.clone()  # [C,T,H,W]
        path.touch()
        return [path]

    monkeypatch.setattr(inference, "_save_transfer_vision", save_vision)
    sample_args = SimpleNamespace(
        output_dir=tmp_path,
        transfer_output_size=output_size,
        transfer_resize_mode="direct_resize",
        vision_extension=".mp4",
    )
    content: dict[str, Any] = {}
    files = inference._save_transfer_rgb(
        source,
        original_hw=(6, 12),
        model_hw=(4, 8),
        fps=29.97,
        sample_args=cast(Any, sample_args),
        content=content,
    )
    assert all(path.is_file() for path in files)
    assert content["transfer_geometry"]["source_hw"] == [6, 12]
    assert content["transfer_geometry"]["model_hw"] == [4, 8]
    if output_size == "original":
        assert saved["vision.mp4"].shape == (3, 3, 6, 12)
        assert torch.equal(saved["vision_model.mp4"], source)
        assert content["native_vision_file"] == str(tmp_path / "vision_model.mp4")
    else:
        assert list(saved) == ["vision.mp4"]
        assert torch.equal(saved["vision.mp4"], source)
        assert "native_vision_file" not in content


@pytest.mark.L0
@pytest.mark.CPU
def test_transfer_image_edit_native_file_metadata_uses_actual_png(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import ModelMode

    video = torch.rand(3, 1, 8, 8)  # [C,1,H_model,W_model]
    sample = SimpleNamespace(
        output_dir=tmp_path,
        transfer_output_size="original",
        transfer_resize_mode="direct_resize",
        vision_extension=".jpg",
        model_mode=ModelMode.IMAGE2IMAGE,
    )
    content: dict[str, Any] = {}
    files = inference._save_transfer_rgb(
        video,
        original_hw=(9, 9),
        model_hw=(8, 8),
        fps=24,
        sample_args=cast(Any, sample),
        content=content,
    )

    native = Path(content["native_vision_file"])
    assert native == tmp_path / "vision_model.png"
    assert native in files and native.is_file()
    assert not (tmp_path / "vision_model.jpg").exists()
    with Image.open(native) as image:
        assert image.size == (8, 8)
    with Image.open(tmp_path / "vision.png") as image:
        assert image.size == (9, 9)
    assert content["transfer_geometry"]["resized_hw"] == [9, 9]
    assert content["transfer_geometry"]["output_hw"] == [9, 9]
    assert content["transfer_geometry"]["encoder_padding_hw"] == [0, 0]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("source_hw", [(65, 65), (65, 64), (64, 65)])
def test_transfer_output_metadata_matches_real_odd_size_mp4_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_hw: tuple[int, int]
) -> None:
    """Probe real encoded pixels instead of assuming the writer preserves tensor dimensions."""
    av = pytest.importorskip("av")
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    torch = pytest.importorskip("torch")
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import ModelMode

    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    # Use the real saver and encoder, with a bounded CPU thread budget for CI.
    monkeypatch.setattr(inference, "_save_transfer_vision", partial(inference._save_transfer_vision, encoder_threads=1))
    video = torch.rand(3, 3, 64, 64)  # [C,T,H_model,W_model]
    sample = SimpleNamespace(
        output_dir=tmp_path,
        transfer_output_size="original",
        transfer_resize_mode="direct_resize",
        vision_extension=".mp4",
        model_mode=ModelMode.VIDEO2VIDEO,
        video_save_quality=8,
        image_save_quality=95,
        save_preview=False,
    )
    content: dict[str, Any] = {}
    files = inference._save_transfer_rgb(
        video,
        original_hw=source_hw,
        model_hw=(64, 64),
        fps=24,
        sample_args=cast(Any, sample),
        content=content,
    )

    with av.open(str(tmp_path / "vision.mp4")) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        encoded_hw = [stream.height, stream.width]
        assert sum(1 for _ in container.decode(stream)) == 3
    geometry = content["transfer_geometry"]
    assert geometry["source_hw"] == list(source_hw)
    assert geometry["resized_hw"] == list(source_hw)
    assert geometry["output_hw"] == encoded_hw
    assert geometry["encoder_padding_hw"] == [
        encoded - resized for encoded, resized in zip(encoded_hw, source_hw, strict=True)
    ]
    assert all(size % 2 == 0 for size in encoded_hw)
    assert all(path.is_file() for path in files)
    with av.open(content["native_vision_file"]) as container:
        stream = container.streams.video[0]
        assert (stream.height, stream.width) == (64, 64)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("process_sample", (False, True))
@pytest.mark.parametrize("autoregressive", (False, True))
def test_direct_transfer_geometry_failure_is_synchronized_and_next_sample_runs(
    monkeypatch: pytest.MonkeyPatch,
    process_sample: bool,
    autoregressive: bool,
) -> None:
    """Reject unsupported geometry before model collectives, then continue the batch."""
    monkeypatch.setenv("NVTE_CUDA_INCLUDE_DIR", "")
    from cosmos_framework.inference import inference, transfer

    pipe = inference.OmniInference.__new__(inference.OmniInference)
    pipe.model = Mock()
    pipe.setup_args = SimpleNamespace(condition_stream_encode=False)
    pipe.transfer_output_video_chunk_callback = None
    pipe.should_process_sample = Mock(return_value=process_sample)
    pipe._handle_sample_exception = Mock(return_value="handled")
    sample_args = SimpleNamespace(transfer_resize_mode="direct_resize", autoregressive=autoregressive)
    prepared = object()
    loader = Mock(side_effect=[ValueError("bad geometry"), prepared])
    generate = Mock(return_value=SimpleNamespace())
    active_boundary = False
    synchronized_errors: list[str] = []

    @contextmanager
    def sync_boundary(*_args: Any, **_kwargs: Any) -> Generator[None, None, None]:
        nonlocal active_boundary
        active_boundary = True
        try:
            yield
        except ValueError as error:
            synchronized_errors.append(str(error))
            raise
        finally:
            active_boundary = False

    def load_inside_boundary(args: Any) -> Any:
        assert active_boundary, "preprocessing must complete before model collectives"
        return loader(args)

    monkeypatch.setattr(inference, "sync_distributed_errors", sync_boundary)
    monkeypatch.setattr(transfer, "_load_direct_transfer_inputs", load_inside_boundary)
    monkeypatch.setattr(transfer, "generate_transfer_sample", generate)

    assert pipe._generate_transfer_batch(sample_args, save_outputs=False) == (["handled"] if process_sample else [])
    generate.assert_not_called()
    assert synchronized_errors == ["bad geometry"]
    assert pipe._generate_transfer_batch(sample_args, save_outputs=False) == []
    assert loader.call_count == 2
    generate.assert_called_once()
    assert generate.call_args.kwargs["prepared_direct_inputs"] is prepared
    assert pipe._handle_sample_exception.call_count == int(process_sample)


def test_direct_transfer_rejects_condition_streaming_before_legacy_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.inference import transfer_streaming

    read_media = Mock(side_effect=AssertionError("must fail before center cropping"))
    monkeypatch.setattr(transfer_streaming, "read_and_resize_media", read_media)
    sample = SimpleNamespace(transfer_resize_mode="direct_resize")
    with pytest.raises(ValueError, match="condition_stream_encode=false"):
        transfer_streaming._load_streaming_transfer_media(cast(Any, sample))
    read_media.assert_not_called()
