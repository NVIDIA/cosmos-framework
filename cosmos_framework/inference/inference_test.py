# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


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
        model_mode=ModelMode.VIDEO2VIDEO,
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

    assert out == {"caption": ["Describe a robotic arm."], "reasoner_images": [None]}


@pytest.mark.L0
def test_get_sample_data_reasoner_with_image(tmp_path: Path) -> None:
    PIL = pytest.importorskip("PIL.Image")
    from cosmos_framework.inference import inference

    img_path = tmp_path / "arm.png"
    PIL.new("RGB", (8, 8), color="red").save(img_path)

    model = SimpleNamespace(input_caption_key="caption")
    sample_args = _make_reasoner_sample_args(vision_path=str(img_path))

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert list(out) == ["caption", "reasoner_images"]
    assert out["caption"] == ["Describe a robotic arm."]
    assert len(out["reasoner_images"]) == 1
    assert out["reasoner_images"][0].size == (8, 8)
    assert out["reasoner_images"][0].mode == "RGB"


@pytest.mark.L0
def test_reasoner_defaults_json_round_trip() -> None:
    import json as _json

    from cosmos_framework.inference.args import PACKAGE_DIR, _load_modality_defaults

    defaults = _load_modality_defaults("reasoner")
    assert defaults["model_mode"] == "reasoner"
    assert defaults["max_new_tokens"] == 64
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
    results = pipe._generate_reasoner_batch([sample_args], data_batch, warmup=False)

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
        pipe._generate_reasoner_batch([sa1, sa2], data_batch, warmup=False)


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
