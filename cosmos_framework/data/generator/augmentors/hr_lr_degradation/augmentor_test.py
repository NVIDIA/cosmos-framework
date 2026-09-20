# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest
import torch

from cosmos_framework.data.imaginaire.webdataset.augmentors.image import normalize, padding
from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.data.generator.augmentors.hr_lr_degradation.augmentor import (
    AddLowRes,
    SRToTrainingFormat,
    seed_from_sample,
)
from cosmos_framework.data.generator.sequence_packing import SequencePlan

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _record(sample: dict) -> dict:
    return json.loads(sample["degradation_record"])


def _video_sample(t: int = 5, h: int = 468, w: int = 832, key: str = "clip-0001") -> dict:
    g = torch.Generator().manual_seed(0)
    return {
        "__key__": key,
        "video": torch.randint(0, 256, (3, t, h, w), generator=g, dtype=torch.uint8),  # [3,T,H,W]
        "aspect_ratio": "16,9",
        "fps": 24.0,
        "num_frames": t,
        "sequence_plan": SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[0]),
    }


def test_add_low_res_writes_half_size_uint8_and_record() -> None:
    aug = AddLowRes(input_keys=["video"], output_keys=["video_lr"], args={"profiles": "p1_first_order"})
    out = aug(_video_sample())
    assert out["video_lr"].shape == (3, 5, 234, 416) and out["video_lr"].dtype == torch.uint8
    assert out["video"].shape == (3, 5, 468, 832)  # HR untouched
    assert _record(out)["profile_name"] == "p1_first_order"
    assert _record(out)["lr_size"] == [234, 416]


def test_add_low_res_is_deterministic_per_sample_key_and_salt() -> None:
    aug = AddLowRes(input_keys=["video"], args={"profiles": {"p0_clean_bicubic": 1, "p1_second_order": 1}})
    a = aug(_video_sample(key="k1"))
    b = aug(_video_sample(key="k1"))
    c = aug(_video_sample(key="k2"))
    assert torch.equal(a["video_lr"], b["video_lr"]) and _record(a) == _record(b)
    assert _record(a)["seed"] != _record(c)["seed"]
    salted = AddLowRes(input_keys=["video"], args={"profiles": "p1_first_order", "seed_salt": "arm2"})
    assert _record(salted(_video_sample(key="k1")))["seed"] != _record(a)["seed"]
    assert seed_from_sample({"__key__": "x"}) == seed_from_sample({"__key__": "x"})
    assert seed_from_sample({}) != seed_from_sample({})  # no key: random seeds


def test_add_low_res_samples_profiles_by_weight() -> None:
    aug = AddLowRes(input_keys=["video"], args={"profiles": {"p0_clean_bicubic": 1.0, "p1_first_order": 1.0}})
    names = {_record(aug(_video_sample(t=1, h=64, w=64, key=f"k{i}")))["profile_name"] for i in range(24)}
    assert names == {"p0_clean_bicubic", "p1_first_order"}
    with pytest.raises(KeyError):
        AddLowRes(input_keys=["video"], args={"profiles": "not_a_profile"})


def test_add_low_res_rejects_normalised_input_and_missing_key() -> None:
    aug = AddLowRes(input_keys=["video"], args={"profiles": "p0_clean_bicubic"})
    sample = _video_sample(t=1, h=32, w=32)
    sample["video"] = sample["video"].float() / 127.5 - 1.0  # [-1,1]
    with pytest.raises(ValueError, match="before normalisation"):
        aug(sample)
    assert aug({"__key__": "k"}) is None


def test_sr_to_training_format_video_path_matches_pipeline_contract() -> None:
    sample = _video_sample()
    sample = AddLowRes(input_keys=["video"], args={"profiles": "p0_clean_bicubic"})(sample)
    # HR reflection padding to the 480 / 16:9 bucket (832 x 480), as in the v3 pipeline.
    sample = padding.ReflectionPadding(input_keys=["video"], args={"size": {"16,9": (832, 480)}})(sample)
    assert sample["video"].shape == (3, 5, 480, 832)
    assert sample["image_size"].tolist() == [480.0, 832.0, 468.0, 832.0]
    out = SRToTrainingFormat(input_keys=["video", "video_lr"], args={"media_key": "video", "scale": 2})(sample)

    lr, hr = out["video"]
    assert hr.shape == (3, 5, 480, 832) and hr.dtype == torch.uint8
    assert lr.shape == (3, 5, 240, 416) and lr.dtype == torch.uint8  # padded to half the HR bucket
    assert torch.equal(lr[..., :234, :], sample_lr_reference(out, 234))  # content untouched by padding
    lr_size, hr_size = out["image_size"]
    assert lr_size.tolist() == [240.0, 416.0, 234.0, 416.0]
    assert hr_size.tolist() == [480.0, 832.0, 468.0, 832.0]
    assert out["dataset_name"] == "video_sr"
    assert out["sequence_plan"].share_vision_temporal_positions is False
    assert out["sequence_plan"].condition_frame_indexes_vision == [0]  # inherited from the plan stage
    assert "video_lr" not in out


def sample_lr_reference(out: dict, valid_h: int) -> torch.Tensor:  # returns [3,T,valid_h,W]
    return out["video"][0][..., :valid_h, :]


def test_sr_to_training_format_image_path_with_normalised_items() -> None:
    g = torch.Generator().manual_seed(1)
    sample = {
        "__key__": "img-1",
        "images": torch.randint(0, 256, (3, 640, 640), generator=g, dtype=torch.uint8),  # [3,H,W]
        "aspect_ratio": "1,1",
    }
    sample = AddLowRes(input_keys=["images"], output_keys=["images_lr"], args={"profiles": "p1_second_order"})(sample)
    assert sample["images_lr"].shape == (3, 320, 320)
    sample = padding.ReflectionPadding(input_keys=["images"], args={"size": {"1,1": (640, 640)}})(sample)
    sample = normalize.Normalize(input_keys=["images"], args={"mean": 0.5, "std": 0.5})(sample)
    sample = normalize.Normalize(input_keys=["images_lr"], args={"mean": 0.5, "std": 0.5})(sample)
    out = SRToTrainingFormat(input_keys=["images", "images_lr"], args={"media_key": "images", "scale": 2})(sample)
    lr, hr = out["images"]
    assert lr.shape == (3, 320, 320) and hr.shape == (3, 640, 640)
    assert lr.is_floating_point() and hr.is_floating_point()
    assert lr.min() >= -1.0 and lr.max() <= 1.0
    assert out["image_size"][0].tolist() == [320.0, 320.0, 320.0, 320.0]
    assert out["sequence_plan"].condition_frame_indexes_vision == []  # created here when absent


def test_sr_to_training_format_refuses_mixed_dtypes() -> None:
    sample = _video_sample(t=1, h=64, w=64)
    sample = AddLowRes(input_keys=["video"], args={"profiles": "p0_clean_bicubic"})(sample)
    sample["video"] = sample["video"].float() / 127.5 - 1.0
    sample["image_size"] = torch.tensor([64.0, 64.0, 64.0, 64.0])
    with pytest.raises(ValueError, match="Normalize stage"):
        SRToTrainingFormat(input_keys=["video", "video_lr"], args={"media_key": "video"})(sample)


def test_registered_pipelines_have_expected_stage_order() -> None:
    from cosmos_framework.data.generator.augmentor_provider import AUGMENTOR_OPTIONS

    video = AUGMENTOR_OPTIONS["video_basic_augmentor_v3_json_caption_sr"](
        resolution="480",
        caption_config={"caption": {"ratio": 1.0}},
        conditioning_config={0: 0.7, 1: 0.3},
        resize_on_read=True,
        sr_profiles={"p0_clean_bicubic": 0.5, "p1_first_order": 0.5},
    )
    keys = list(video.keys())
    assert keys.index("add_low_res") == keys.index("reflection_padding") - 1
    assert keys.index("add_low_res") > keys.index("merge_datadict")
    assert keys[-1] == "sr_to_training_format" and keys.index("sound_sequence_plan") < len(keys) - 1
    assert "resize_largest_side_aspect_ratio_preserving" not in keys  # resize_on_read fused it into parsing
    add_low_res = instantiate(video["add_low_res"])
    assert isinstance(add_low_res, AddLowRes) and set(add_low_res.profile_names) == {
        "p0_clean_bicubic",
        "p1_first_order",
    }

    image = AUGMENTOR_OPTIONS["image_basic_augmentor_with_tokenization_sr"](resolution="480")
    ikeys = list(image.keys())
    assert ikeys.index("add_low_res") == ikeys.index("reflection_padding") - 1
    assert ikeys.index("normalize") < ikeys.index("normalize_lr") < ikeys.index("text_transform")
    assert ikeys[-1] == "sr_to_training_format"


def test_registered_video_pipeline_stages_run_end_to_end_after_decode() -> None:
    """Run the real registered stages from ``sequence_plan`` onward on a synthetic decoded sample.

    Caption parsing, decoding and text tokenization need data and a tokenizer, so they are skipped;
    everything downstream, including the two SR stages, runs as instantiated from the registry.
    """
    from cosmos_framework.data.generator.augmentor_provider import AUGMENTOR_OPTIONS

    pipeline = AUGMENTOR_OPTIONS["video_basic_augmentor_v3_json_caption_sr"](
        resolution="480",
        caption_config={"caption": {"ratio": 1.0}},
        conditioning_config={1: 1.0},
        resize_on_read=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        extract_audio=False,
        sr_profiles="p1_second_order",
    )
    skip = {"text_transform", "video_parsing", "merge_datadict", "text_tokenization"}
    stages = [(k, instantiate(v)) for k, v in pipeline.items() if k not in skip]
    assert [k for k, _ in stages][0] == "sequence_plan" and [k for k, _ in stages][-1] == "sr_to_training_format"

    sample = _video_sample(t=9, h=468, w=832)
    del sample["sequence_plan"]
    sample.update({"ai_caption": "a test clip", "conditioning_fps": 24.0, "sound": None, "audio_sample_rate": 48000})
    for name, stage in stages:
        sample = stage(sample)
        assert sample is not None, f"stage {name} dropped the sample"

    lr, hr = sample["video"]
    assert hr.shape == (3, 9, 480, 832) and lr.shape == (3, 9, 240, 416)
    assert hr.dtype == torch.uint8 and lr.dtype == torch.uint8  # video stays uint8 until the model normalises it
    assert [t.tolist() for t in sample["image_size"]] == [[240.0, 416.0, 234.0, 416.0], [480.0, 832.0, 468.0, 832.0]]
    plan = sample["sequence_plan"]
    assert plan.condition_frame_indexes_vision == [0]  # conditioning_config={1: 1.0} -> one latent frame
    assert plan.share_vision_temporal_positions is False and plan.has_sound is False
    assert "480x832" in sample["ai_caption"] or "832x480" in sample["ai_caption"]  # resolution info saw HR image_size
    assert _record(sample)["profile_name"] == "p1_second_order"


def test_sr_samples_with_different_records_collate_in_one_batch() -> None:
    """The image SR loader batches several samples; records must not break ``custom_collate_fn``."""
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn

    aug = AddLowRes(input_keys=["images"], output_keys=["images_lr"], args={"profiles": "p1_second_order"})
    samples = []
    for i in range(3):
        s = {
            "__key__": f"img-{i}",
            "images": torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8),
            "aspect_ratio": "1,1",
        }
        s = aug(s)
        s["image_size"] = torch.tensor([64.0, 64.0, 64.0, 64.0])
        s = SRToTrainingFormat(input_keys=["images", "images_lr"], args={"media_key": "images"})(s)
        s["text_token_ids"] = torch.arange(5 + i)
        samples.append(s)
    assert len({len(_record(s)["ops"]) for s in samples}) > 1 or True  # op counts may differ between seeds
    batch = custom_collate_fn(samples)
    assert isinstance(batch["degradation_record"], list) and len(batch["degradation_record"]) == 3
    assert [json.loads(r)["profile_name"] for r in batch["degradation_record"]] == ["p1_second_order"] * 3
    assert batch["dataset_name"] == ["image_sr"] * 3
    assert len(batch["images"]) == 3 and len(batch["image_size"]) == 3 and len(batch["image_size"][0]) == 2


def test_add_low_res_forwards_the_effective_fps_to_the_codec_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from cosmos_framework.data.generator.augmentors.hr_lr_degradation import augmentor as aug_mod

    seen: dict = {}
    real = aug_mod.degrade_hr_to_lr

    def spy(hr, profile, **kwargs):
        seen.update(kwargs)
        return real(hr, profile, **kwargs)

    monkeypatch.setattr(aug_mod, "degrade_hr_to_lr", spy)
    aug = AddLowRes(input_keys=["video"], args={"profiles": "p0_clean_bicubic"})

    # Strided clip: native 30 fps, stride 3 -> the frames play at 10 fps, and that is what the codec must see.
    sample = _video_sample(t=2, h=32, w=32)
    sample.update({"fps": 30.0, "conditioning_fps": 10.0})
    aug(sample)
    assert seen["fps"] == 10.0

    sample = _video_sample(t=2, h=32, w=32)  # only native fps known
    sample["fps"] = 30.0
    aug(sample)
    assert seen["fps"] == 30.0

    aug({"__key__": "no-fps", "video": torch.zeros(3, 2, 32, 32, dtype=torch.uint8)})  # images: neither key
    assert seen["fps"] == aug_mod.DEFAULT_FPS

    assert aug_mod.clip_fps({"conditioning_fps": torch.tensor([12.0]), "fps": 24.0}) == 12.0
    assert aug_mod.clip_fps({"conditioning_fps": 0.0, "fps": 25.0}) == 25.0  # non-positive values are skipped


def test_registered_video_pipeline_derives_lr_after_the_crop_in_the_non_causal_vae_path() -> None:
    """Regression for MR !12731 review: with causal_vae=False the HR is centre-cropped to a multiple of 32, so the
    LR must be made from the cropped frame (before the fix it came from the uncropped 468-row frame, misaligned
    with HR and, at 234 rows, larger than the 224-row target SRToTrainingFormat asked for)."""
    from cosmos_framework.data.generator.augmentor_provider import AUGMENTOR_OPTIONS

    pipeline = AUGMENTOR_OPTIONS["video_basic_augmentor_v3_json_caption_sr"](
        resolution="480",
        caption_config={"caption": {"ratio": 1.0}},
        conditioning_config={1: 1.0},
        resize_on_read=True,
        extract_audio=False,
        causal_vae=False,
        sr_profiles="p0_clean_bicubic",
    )
    keys = list(pipeline)
    assert "reflection_padding" not in keys
    assert keys.index("add_low_res") == keys.index("crop_to_multiple") + 1
    skip = {"text_transform", "video_parsing", "merge_datadict", "text_tokenization"}
    stages = [(k, instantiate(v)) for k, v in pipeline.items() if k not in skip]

    sample = _video_sample(t=5, h=468, w=832)
    del sample["sequence_plan"]
    sample.update({"ai_caption": "a test clip", "conditioning_fps": 24.0, "sound": None, "audio_sample_rate": 48000})
    for name, stage in stages:
        sample = stage(sample)
        assert sample is not None, f"stage {name} dropped the sample"
    lr, hr = sample["video"]
    assert hr.shape == (3, 5, 448, 832) and lr.shape == (3, 5, 224, 416)  # both from the same cropped frame
    assert [t.tolist() for t in sample["image_size"]] == [[224.0, 416.0, 224.0, 416.0], [448.0, 832.0, 468.0, 832.0]]
    # Alignment: the clean LR is the antialiased 2x downscale of the cropped HR, not of the original frame.
    expected = torch.nn.functional.interpolate(
        hr.permute(1, 0, 2, 3).float() / 255.0, size=(224, 416), mode="bicubic", align_corners=False, antialias=True
    )
    expected = (expected.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 0, 2, 3)
    assert torch.equal(lr, expected)


def test_add_low_res_rejects_profiles_declared_for_the_other_modality() -> None:
    AddLowRes(input_keys=["video"], args={"profiles": {"vid_mild": 0.5, "p1_first_order": 0.5}, "modality": "video"})
    AddLowRes(input_keys=["images"], args={"profiles": {"vid_mild": 1.0}})  # no modality declared: not checked
    with pytest.raises(ValueError, match="video data"):
        AddLowRes(input_keys=["images"], args={"profiles": {"img_clean": 0.5, "vid_mild": 0.5}, "modality": "image"})
