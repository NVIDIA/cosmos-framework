# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: Qwen3.5 image-only teacher request shaping."""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.data.packer.qwen3_vl import Qwen3VLDataPacker
from cosmos_framework.posttrain.schema import ObsBundle


class _Tokenizer:
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int | None = None
    unk_token_id: int | None = None
    additional_special_tokens_ids: list[int] = [90, 91, 92, 93]
    all_special_ids: list[int] = [0, 2, 90, 91, 92, 93]
    chat_template: str = "fixture-qwen3.5-template"
    name_or_path: str = "fixture-qwen3.5"

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {
            "<pad>": 0,
            "<eos>": 2,
            "<vision_start>": 90,
            "<image>": 91,
            "<video>": 92,
            "<vision_end>": 93,
        }


class _ProcessorComponent:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": {"longest_edge": 4096}}


class _Processor:
    def __init__(
        self,
        grid_rows: tuple[tuple[int, ...], ...] | None = None,
        marker_grid_rows: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        self.tokenizer = _Tokenizer()
        self.image_processor = _ProcessorComponent("image")
        self.video_processor = _ProcessorComponent("video")
        self.grid_rows = grid_rows
        self.marker_grid_rows = marker_grid_rows
        self.chat_template: str | None = None
        self.last_chat_template: str | None = None
        self.last_preview_images: list[Image.Image] = []
        self.last_preview_images_kwargs: dict[str, Any] = {}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str,
        enable_thinking: bool = False,
        return_dict: bool = False,
        return_tensors: str | None = None,
        images_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        assert add_generation_prompt is True and enable_thinking is False
        self.last_chat_template = chat_template
        if tokenize:
            assert return_dict is True and return_tensors == "pt"
            images = [
                block["image"] for message in messages for block in message["content"] if block.get("type") == "image"
            ]
            assert all(isinstance(image, Image.Image) for image in images)
            self.last_preview_images = images
            self.last_preview_images_kwargs = {} if images_kwargs is None else dict(images_kwargs)
            grid_rows = self.grid_rows or ((1, 18, 18),) * len(images)
            if len(grid_rows) != len(images):
                raise ValueError("fixture grid/image count mismatch")
            marker_grid_rows = self.marker_grid_rows or grid_rows
            if len(marker_grid_rows) != len(images):
                raise ValueError("fixture marker/image count mismatch")
            prompt_ids = [5]
            for grid_row in marker_grid_rows:
                marker_width = math.prod(grid_row) // 4 if len(grid_row) == 3 else 1
                prompt_ids.extend([90, *([91] * marker_width), 93])
            prompt_ids.append(7)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long)  # [1,L]
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)  # [1,L]
            patch_rows = sum(math.prod(grid_row) if len(grid_row) == 3 else 1 for grid_row in grid_rows)
            pixel_values = torch.zeros((patch_rows, 3), dtype=torch.float32)  # [N_patch,3]
            image_grid_thw = torch.tensor(grid_rows, dtype=torch.long)  # [N_image,D_grid]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        rendered: list[str] = []
        for message in messages:
            blocks = message["content"] if isinstance(message["content"], list) else []
            content = "".join(
                "<image>" if block.get("type") in {"image", "image_url"} else block.get("text", "") for block in blocks
            )
            rendered.append(f"<{message['role']}>{content}</{message['role']}>")
        rendered.append("<assistant><think>\n\n</think>\n\n")
        return "".join(rendered)


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _packer(
    *,
    grid_rows: tuple[tuple[int, ...], ...] | None = None,
    marker_grid_rows: tuple[tuple[int, ...], ...] | None = None,
) -> Qwen3VLDataPacker:
    config = SimpleNamespace(
        model_type="qwen3_5_moe",
        image_token_id=91,
        video_token_id=92,
        vision_start_token_id=90,
        vision_end_token_id=93,
        vision_config=SimpleNamespace(
            spatial_merge_size=2,
            patch_size=16,
            temporal_patch_size=2,
        ),
    )
    packer = Qwen3VLDataPacker(
        processor=_Processor(grid_rows, marker_grid_rows),
        model_config=config,
    )
    packer.setup("fixture-qwen3.5")
    return packer


def _image_obs(images: list[Any]) -> ObsBundle:
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        content.extend(
            [
                {"type": "image", "image": image},
                {"type": "text", "text": f" image-{index}"},
            ]
        )
    if not content:
        content.append({"type": "text", "text": "text-only row"})
    return ObsBundle(
        representations={"messages": [{"role": "user", "content": content}]},
        modality_tags=("text", "image") if images else ("text",),
        id="sample",
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_token_teacher_request_preserves_image_order_ids_and_uuids(tmp_path: Path) -> None:
    packer = _packer()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first_bytes = _png_bytes((255, 0, 0))
    second_bytes = _png_bytes((0, 255, 0))
    first.write_bytes(first_bytes)
    second.write_bytes(second_bytes)
    image_paths = [str(first), str(second)]
    obs = _image_obs(image_paths)
    obs.representations["messages"][0]["content"][0]["max_pixels"] = 4096
    obs.representations["messages"][0]["content"][0]["min_pixels"] = 16
    token_ids = [5, 90, 91, 93, 6, 90, 91, 93, 7]

    rendered = packer.render_cross_token_multimodal_prompt(obs, enable_thinking=False)
    prepared = packer.get_cross_token_multimodal_teacher_input(obs, token_ids)

    assert rendered.endswith("<assistant><think>\n\n</think>\n\n")
    assert prepared.request["prompt_token_ids"] == token_ids
    image_payloads = prepared.request["multi_modal_data"]["image"]
    assert all(isinstance(image, Image.Image) for image in image_payloads)
    assert [image.getpixel((0, 0)) for image in image_payloads] == [(255, 0, 0), (0, 255, 0)]
    assert prepared.image_count == 2
    assert prepared.expected_image_marker_widths == (81, 81)
    assert prepared.vision_end_token_id == 93
    expected_uuid = hashlib.sha256(
        f"qwen3-vl-image-v1\0{0}\0{hashlib.sha256(first_bytes).hexdigest()}".encode("utf-8")
    ).hexdigest()
    assert prepared.request["multi_modal_uuids"]["image"][0] == expected_uuid
    assert prepared.image_uuids[0] == expected_uuid
    assert prepared.request["mm_processor_kwargs"] == {"max_pixels": 4096, "min_pixels": 16}
    assert packer.processor.last_chat_template == _Tokenizer.chat_template

    first.write_bytes(_png_bytes((0, 0, 255)))
    assert image_payloads[0].getpixel((0, 0)) == (255, 0, 0)

    reversed_prepared = packer.get_cross_token_multimodal_teacher_input(
        _image_obs(list(reversed(image_paths))),
        token_ids,
    )
    assert reversed_prepared.image_uuids != prepared.image_uuids


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_token_multimodal_binds_logical_markers_to_ordered_grid_widths() -> None:
    grid_rows = ((1, 18, 18), (1, 20, 20))
    obs = _image_obs([_png_bytes((1, 2, 3)), _png_bytes((4, 5, 6))])
    token_ids = [5, 90, 91, 93, 6, 90, 91, 93, 7]

    prepared = _packer(grid_rows=grid_rows).get_cross_token_multimodal_teacher_input(obs, token_ids)

    assert prepared.expected_image_marker_widths == (81, 100)
    with pytest.raises(ValueError, match="marker runs do not match ordered image grids"):
        _packer(
            grid_rows=grid_rows,
            marker_grid_rows=tuple(reversed(grid_rows)),
        ).get_cross_token_multimodal_teacher_input(obs, token_ids)

    image = _image_obs([_png_bytes((1, 2, 3))])
    with pytest.raises(ValueError, match="not followed by vision-end"):
        _packer().get_cross_token_multimodal_teacher_input(image, [5, 90, 91, 7])
    with pytest.raises(ValueError, match="exactly one logical image marker"):
        _packer().get_cross_token_multimodal_teacher_input(image, [5, 90, 91, 91, 93, 7])
    with pytest.raises(ValueError, match="stray vision-end"):
        _packer().get_cross_token_multimodal_teacher_input(_image_obs([]), [5, 93, 7])
    with pytest.raises(ValueError, match="not divisible by spatial merge area"):
        _packer(grid_rows=((1, 3, 3),)).get_cross_token_multimodal_teacher_input(
            image,
            [5, 90, 91, 93, 7],
        )
    with pytest.raises(ValueError, match=r"grid must be \[N,3\]"):
        _packer(grid_rows=((1, 18),)).get_cross_token_multimodal_teacher_input(
            image,
            [5, 90, 91, 93, 7],
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_token_multimodal_remote_uri_fails_closed() -> None:
    with pytest.raises(ValueError, match="no verifiable raw-media content hash"):
        _packer().get_cross_token_multimodal_teacher_input(
            _image_obs(["https://example.invalid/image.png"]),
            [5, 90, 91, 93, 7],
        )
