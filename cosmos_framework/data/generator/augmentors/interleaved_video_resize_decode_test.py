# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Bounded transfer decoding preserves the direct modality resize contract."""

import weakref
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torchvision.transforms.functional as transforms_F
from torchvision.transforms.v2 import Resize

import cosmos_framework.data.generator.augmentors.interleaved_video_parsing as video_parsing

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


_STATIC_CONTROL_PARSERS = [
    video_parsing.VideoTransferAlignedFullFramesParsing,
    video_parsing.VideoTransferAlignedLegacyChunkParsing,
    video_parsing.VideoTransferAlignedChunkedFramesParsing,
]


def _make_parser(
    direct_resize: bool = True,
    *,
    parser_class: type[video_parsing.VideoTransferAlignedFullFramesParsing] = (
        video_parsing.VideoTransferAlignedFullFramesParsing
    ),
    control_keys: tuple[str, ...] = (),
) -> video_parsing.VideoTransferAlignedFullFramesParsing:
    args: dict[str, object] = {"max_stride": 1, "min_stride": 1}
    if direct_resize:
        args["direct_resize"] = {}
    if issubclass(parser_class, video_parsing.VideoTransferAlignedLegacyChunkParsing):
        args.update(
            key_for_caption="t2w_windows",
            min_duration=0,
            num_video_frames=5,
            use_native_fps=False,
            use_original_fps=False,
        )
    return parser_class(input_keys=["metas", "video", *control_keys], args=args)


@pytest.mark.parametrize(
    "control_key,modality",
    [
        ("depth_custom_v3", "depth"),
        ("depth", "depth"),
        ("segmentation_custom_v3", "seg"),
        ("segmentation", "seg"),
        ("segmentation_depth", "depth"),
        ("persisted_control", "depth"),
        ("persisted_control", "seg"),
    ],
)
def test_control_resize_accepts_merger_keys_and_preserves_modality_policy(
    monkeypatch: pytest.MonkeyPatch, control_key: str, modality: str
) -> None:
    resize = Mock(wraps=transforms_F.resize)
    monkeypatch.setattr(video_parsing.transforms_F, "resize", resize)
    parser = _make_parser(control_keys=(control_key,))
    sample = {"aspect_ratio": "1,1", "_res_size_map": {"1,1": (3, 2)}, "_selected_control_modality": modality}
    source = torch.arange(2 * 3 * 7 * 11).reshape(2, 3, 7, 11).remainder(251).to(torch.uint8)  # [2,3,7,11]

    transforms = parser._build_control_decode_transform(sample, control_key)
    assert transforms is not None
    actual = transforms[0](source)  # [2,3,2,3]

    interpolation = (
        transforms_F.InterpolationMode.BILINEAR if modality == "depth" else transforms_F.InterpolationMode.NEAREST
    )
    assert resize.call_args.kwargs == {
        "size": (2, 3),
        "interpolation": interpolation,
        "antialias": modality == "depth",
    }
    expected = transforms_F.resize(
        source, (2, 3), interpolation=interpolation, antialias=modality == "depth"
    )  # [2,3,2,3]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if modality == "seg":
        assert torch.isin(actual, source).all()


@pytest.mark.parametrize("parser_class", _STATIC_CONTROL_PARSERS)
def test_direct_resize_rejects_unknown_static_control_configuration_at_construction(
    parser_class: type[video_parsing.VideoTransferAlignedFullFramesParsing],
) -> None:
    with pytest.raises(video_parsing.ControlResizeConfigurationError, match="no policy for control key 'optical_flow'"):
        _make_parser(parser_class=parser_class, control_keys=("optical_flow",))

    legacy_parser = _make_parser(direct_resize=False, parser_class=parser_class, control_keys=("optical_flow",))
    assert legacy_parser._build_control_decode_transform({}, "optical_flow") is None


@pytest.mark.parametrize("parser_class", _STATIC_CONTROL_PARSERS)
@pytest.mark.parametrize("modality", [None, "edge", "unknown"])
def test_direct_resize_rejects_invalid_persisted_modality_before_probing_or_decoding(
    monkeypatch: pytest.MonkeyPatch,
    parser_class: type[video_parsing.VideoTransferAlignedFullFramesParsing],
    modality: str | None,
) -> None:
    parser = _make_parser(parser_class=parser_class, control_keys=("persisted_control",))
    decoder = Mock(side_effect=AssertionError("Invalid control configuration must fail before video probing/decoding."))
    warning = Mock()
    monkeypatch.setattr(video_parsing, "VideoDecoder", decoder)
    monkeypatch.setattr(parser, "_validate_and_probe", decoder)
    monkeypatch.setattr(video_parsing.log, "warning", warning)

    with pytest.raises(video_parsing.ControlResizeConfigurationError, match="requires _selected_control_modality"):
        parser({"_selected_control_modality": modality})

    decoder.assert_not_called()
    warning.assert_not_called()


@pytest.mark.parametrize("modality", [None, "unknown", []])
def test_selected_parser_rejects_unknown_resize_modality_without_decode_warning(
    monkeypatch: pytest.MonkeyPatch, modality: object
) -> None:
    parser = _make_parser(parser_class=video_parsing.VideoTransferAlignedSelectedControlParsing)
    decoder = Mock(side_effect=AssertionError("Invalid selected modality must fail before video probing/decoding."))
    warning = Mock()
    monkeypatch.setattr(video_parsing, "VideoDecoder", decoder)
    monkeypatch.setattr(parser, "_validate_and_probe", decoder)
    monkeypatch.setattr(video_parsing.log, "warning", warning)

    with pytest.raises(video_parsing.ControlResizeConfigurationError, match="requires _selected_control_modality"):
        parser({"_selected_control_modality": modality})

    decoder.assert_not_called()
    warning.assert_not_called()

    legacy_parser = _make_parser(
        direct_resize=False, parser_class=video_parsing.VideoTransferAlignedSelectedControlParsing
    )
    assert legacy_parser({"_selected_control_modality": modality}) is None
    warning.assert_called_once()


def _install_decoder(
    monkeypatch: pytest.MonkeyPatch,
    source: torch.Tensor,  # [T,C,H,W]
    *,
    native_output_dtype: bool = True,
    native_transforms: bool = True,
) -> tuple[list[dict[str, object]], list[list[int]]]:
    initializations: list[dict[str, object]] = []
    requests: list[list[int]] = []
    previous_frames: weakref.ReferenceType[torch.Tensor] | None = None

    class FakeDecoder:
        output_dtype: torch.dtype
        transforms: object

        def __init__(self, _video: bytes, **kwargs: object) -> None:
            if not native_output_dtype and "output_dtype" in kwargs:
                raise TypeError("VideoDecoder.__init__() got an unexpected keyword argument 'output_dtype'")
            if not native_transforms and "transforms" in kwargs:
                raise TypeError("VideoDecoder.__init__() got an unexpected keyword argument 'transforms'")
            output_dtype = kwargs.get("output_dtype", torch.uint8)
            assert isinstance(output_dtype, torch.dtype)
            self.output_dtype = output_dtype
            self.transforms = kwargs.get("transforms")
            initializations.append(kwargs)

        def get_frames_at(self, indices: list[int]) -> SimpleNamespace:
            nonlocal previous_frames
            # The implementation must release the prior native allocation before
            # fetching another batch, even when resize returns its input unchanged.
            assert previous_frames is None or previous_frames() is None
            requests.append(list(indices))
            frames = source[indices]  # [Tc,C,H,W]
            if self.output_dtype == torch.float32:
                frames = frames.float().div_(255.0)  # [Tc,C,H,W]
            if self.transforms is not None:
                assert isinstance(self.transforms, list)
                for transform in self.transforms:
                    assert callable(transform)
                    frames = transform(frames)  # [Tc,C,Htarget,Wtarget]
            previous_frames = weakref.ref(frames)
            return SimpleNamespace(data=frames)

    monkeypatch.setattr(video_parsing, "VideoDecoder", FakeDecoder)
    monkeypatch.setattr(video_parsing, "_SUPPORTS_VIDEO_DECODER_TRANSFORMS", None)
    monkeypatch.setattr(video_parsing, "_SUPPORTS_VIDEO_DECODER_OUTPUT_DTYPE", None)
    monkeypatch.setattr(video_parsing, "_WARNED_POST_DECODE_TRANSFORMS", False)
    monkeypatch.setattr(video_parsing, "_WARNED_POST_DECODE_OUTPUT_DTYPE", False)
    return initializations, requests


@pytest.mark.parametrize(
    "interpolation",
    [
        transforms_F.InterpolationMode.BICUBIC,
        transforms_F.InterpolationMode.BILINEAR,
        transforms_F.InterpolationMode.NEAREST,
    ],
)
def test_direct_decode_bounds_batches_and_matches_modality_resize(
    monkeypatch: pytest.MonkeyPatch, interpolation: transforms_F.InterpolationMode
) -> None:
    source = torch.arange(23 * 3 * 13 * 19).reshape(23, 3, 13, 19).remainder(251).to(torch.uint8)  # [23,3,13,19]
    initializations, requests = _install_decoder(monkeypatch, source)
    indices = [18, 0, 7, 7, 2, 19, 4, 8, 1, 6, 20, 3, 22, 11, 12, 15, 21, 10, 14]
    resize = partial(transforms_F.resize, size=(5, 9), interpolation=interpolation, antialias=True)

    frames = _make_parser()._decode_frames_at(b"video", indices, [resize])  # [3,T,5,9]

    expected = transforms_F.resize(source[indices], (5, 9), interpolation=interpolation, antialias=True).permute(
        1, 0, 2, 3
    )  # [3,T,5,9]
    torch.testing.assert_close(frames, expected, rtol=0, atol=0)
    assert frames.dtype == torch.uint8
    assert len(initializations) == 1
    assert "transforms" not in initializations[0]
    assert [index for request in requests for index in request] == indices
    assert [len(request) for request in requests] == [1, 8, 8, 2]


@pytest.mark.parametrize("height,width,expected_cap", [(1024, 1024, 2), (1536, 2048, 1)])
def test_direct_decode_caps_native_float_working_size(
    monkeypatch: pytest.MonkeyPatch, height: int, width: int, expected_cap: int
) -> None:
    source = torch.arange(9, dtype=torch.uint8).view(9, 1, 1, 1).expand(9, 3, height, width)  # [9,3,H,W]
    _, requests = _install_decoder(monkeypatch, source)
    resize = Resize((2, 2), interpolation=transforms_F.InterpolationMode.NEAREST)

    frames = _make_parser()._decode_frames_at(b"video", list(range(9)), [resize])  # [3,9,2,2]

    assert requests[0] == [0]
    assert max(map(len, requests)) == expected_cap
    assert [index for request in requests for index in request] == list(range(9))
    torch.testing.assert_close(frames[0, :, 0, 0], torch.arange(9, dtype=torch.uint8))  # [9]


@pytest.mark.parametrize("native_output_dtype", [False, True])
def test_direct_depth_decode_converts_before_bilinear_resize(
    monkeypatch: pytest.MonkeyPatch, native_output_dtype: bool
) -> None:
    source = torch.tensor([0, 255, 17, 199], dtype=torch.uint8).view(1, 1, 2, 2).expand(19, 3, 2, 2)  # [19,3,2,2]
    initializations, requests = _install_decoder(monkeypatch, source, native_output_dtype=native_output_dtype)
    seen_dtypes: list[torch.dtype] = []

    def resize_float_frames(frames: torch.Tensor) -> torch.Tensor:  # frames: [Tc,3,2,2], returns [Tc,3,1,1]
        seen_dtypes.append(frames.dtype)
        assert frames.dtype == torch.float32
        return transforms_F.resize(
            frames, (1, 1), interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
        )  # [Tc,3,1,1]

    frames = _make_parser()._decode_frames_at(
        b"depth", list(range(19)), [resize_float_frames], output_dtype=torch.float32
    )  # [3,19,1,1]

    expected = transforms_F.resize(
        source.float().div(255.0), (1, 1), interpolation=transforms_F.InterpolationMode.BILINEAR, antialias=True
    ).permute(1, 0, 2, 3)  # [3,19,1,1]
    torch.testing.assert_close(frames, expected)
    assert seen_dtypes == [torch.float32] * 4
    assert len(initializations) == 1
    assert "transforms" not in initializations[0]
    assert [len(request) for request in requests] == [1, 8, 8, 2]


def test_legacy_transform_fallback_also_decodes_bounded_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.arange(19 * 3 * 5 * 7).reshape(19, 3, 5, 7).remainder(251).to(torch.uint8)  # [19,3,5,7]
    initializations, requests = _install_decoder(monkeypatch, source, native_transforms=False)
    resize = Resize((2, 3), antialias=True)

    frames = _make_parser(direct_resize=False)._decode_frames_at(b"video", list(range(19)), [resize])  # [3,19,2,3]

    expected = resize(source).permute(1, 0, 2, 3)  # [3,19,2,3]
    torch.testing.assert_close(frames, expected)
    assert len(initializations) == 1
    assert [len(request) for request in requests] == [1, 8, 8, 2]


@pytest.mark.parametrize("use_resize", [False, True])
def test_non_direct_decode_keeps_native_transform_api(monkeypatch: pytest.MonkeyPatch, use_resize: bool) -> None:
    source = torch.zeros(19, 3, 5, 7, dtype=torch.uint8)  # [19,3,5,7]
    initializations, requests = _install_decoder(monkeypatch, source)
    transforms = [Resize((2, 3))] if use_resize else None

    frames = _make_parser(direct_resize=False)._decode_frames_at(b"video", list(range(19)), transforms)  # [3,19,H,W]

    assert requests == [list(range(19))]
    assert initializations[0].get("transforms") is transforms
    expected_shape = (3, 19, 2, 3) if use_resize else (3, 19, 5, 7)
    assert frames.shape == expected_shape


def test_direct_same_size_transform_releases_native_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.zeros(19, 3, 5, 7, dtype=torch.uint8)  # [19,3,5,7]
    _, requests = _install_decoder(monkeypatch, source)

    frames = _make_parser()._decode_frames_at(b"video", list(range(19)), [Resize((5, 7))])  # [3,19,5,7]

    torch.testing.assert_close(frames, source.permute(1, 0, 2, 3))  # [3,19,5,7]
    assert [len(request) for request in requests] == [1, 8, 8, 2]
