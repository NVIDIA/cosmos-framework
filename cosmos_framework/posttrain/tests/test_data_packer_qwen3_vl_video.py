# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Qwen3-VL video packer: the policy ids of a video row are the vLLM engine's spliced stream, the request
names its frames, and every guard refuses identically on both sides.

The fixture tokenizer models exactly one BPE merge, ``".<"``, because that merge IS the defect: the HF
processor writes ``<t seconds>`` into the text and tokenizes once (``"result."`` + ``"<1.3 seconds>"`` ->
``".<"``), the engine encodes every timestamp alone and splices ids (``"."``, ``"<"``). The fixture
processor emulates both transformers text forms of a video row -- 4.x replaces the whole placeholder
triple ("triple"), 5.x replaces only ``<|video_pad|>`` and keeps the outer marker pair ("outer") -- and
can corrupt its result to prove the cross-check stays closed. The engine stream the tests compare
against is derived here independently of the packer (render, tokenize, splice with the test's own
timestamp formula). An optional real-model test (``QWEN3_VL_VIDEO_TEST_MODEL``) pins the known answer
recorded from vLLM's own processor on a 16-frame row.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.data import packer as pk
from cosmos_framework.posttrain.data.packer.qwen3_vl import Qwen3VLDataPacker
from cosmos_framework.posttrain.data.packer.qwen3_vl_video import (
    HF_TEXT_FORMS,
    VIDEO_POLICY_IDS_SEMANTICS,
    VIDEO_UUID_SEED,
    Qwen3VLVideoDataPacker,
    make_packer,
)
from cosmos_framework.posttrain.roles.vllm_request import vllm_request_media_identity
from cosmos_framework.posttrain.schema import ObsBundle
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256

VS, VE, IP, VP = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"
_SPECIAL_TOKENS: dict[str, int] = {"<|im_start|>": 80, "<|im_end|>": 81, VS: 90, IP: 91, VP: 92, VE: 93}
_MERGE, _MERGE_ID = ".<", 200  # the one BPE merge the fixture models
_CHAR_BASE = 1000
_SPECIAL_RE = re.compile("|".join(re.escape(token) for token in _SPECIAL_TOKENS))
_PLACEHOLDER_RE = re.compile(re.escape(VS) + "(" + re.escape(IP) + "|" + re.escape(VP) + ")" + re.escape(VE))
TEMPORAL, MERGE, GRID_H, GRID_W = 2, 2, 2, 4  # tokens per frame group = GRID_H * GRID_W / MERGE**2 = 2
IMAGE_GRID = [1, 2, 4]  # one image = 2 pads
FPS = 0.4
TEXT = "Return only the JSON result."  # ends in '.', the merge class


class _TokenizerResult(dict):
    def __init__(self, input_ids: list[int]) -> None:
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids


class _Tokenizer:
    pad_token_id: int = 0
    eos_token_id: int = 2
    chat_template: str = "fixture-template"
    name_or_path: str = "fixture"

    def encode(self, text: str, add_special_tokens: bool = False, **_kwargs: Any) -> list[int]:
        ids = [1] if add_special_tokens else []
        position = 0
        for match in _SPECIAL_RE.finditer(text):
            ids.extend(self._encode_plain(text[position : match.start()]))
            ids.append(_SPECIAL_TOKENS[match.group(0)])
            position = match.end()
        ids.extend(self._encode_plain(text[position:]))
        return ids

    @staticmethod
    def _encode_plain(piece: str) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(piece):
            if piece[index : index + 2] == _MERGE:
                ids.append(_MERGE_ID)
                index += 2
            else:
                ids.append(_CHAR_BASE + ord(piece[index]))
                index += 1
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False, **_kwargs: Any) -> _TokenizerResult:
        return _TokenizerResult(self.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        reverse = {token_id: token for token, token_id in _SPECIAL_TOKENS.items()}
        pieces: list[str] = []
        for token_id in ids.tolist() if hasattr(ids, "tolist") else ids:
            token_id = int(token_id)
            if token_id in reverse:
                if not skip_special_tokens:
                    pieces.append(reverse[token_id])
            elif token_id == _MERGE_ID:
                pieces.append(_MERGE)
            elif token_id >= _CHAR_BASE:
                pieces.append(chr(token_id - _CHAR_BASE))
            else:
                pieces.append(f"<{token_id}>")
        return "".join(pieces)

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, _MERGE: _MERGE_ID, **_SPECIAL_TOKENS}


class _ProcessorComponent:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        for key, value in config.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.config)


def _hf_timestamps(frames_indices: list[int], fps: float) -> list[float]:
    """The HF/engine timestamp rule written independently: pad to a multiple of TEMPORAL, mean of each group."""
    indices = list(frames_indices)
    if len(indices) % TEMPORAL:
        indices += [indices[-1]] * (TEMPORAL - len(indices) % TEMPORAL)
    seconds = [index / fps for index in indices]
    return [(seconds[i] + seconds[i + TEMPORAL - 1]) / 2 for i in range(0, len(seconds), TEMPORAL)]


def _render(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"<|im_start|>{message['role']}\n")
        for block in message["content"]:
            kind = block.get("type")
            if kind == "text":
                parts.append(block["text"])
            elif kind in {"image", "image_url"}:
                parts.append(VS + IP + VE)
            elif kind in {"video", "video_frames"}:
                parts.append(VS + VP + VE)
        parts.append("<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class _Processor:
    """Fixture Qwen3-VL processor: renders placeholders, and on ``tokenize=True`` expands them IN THE TEXT the way
    transformers does (``hf_text_form``), tokenizes once and returns grid-consistent media tensors."""

    def __init__(self, *, hf_text_form: str = "triple", corrupt: str | None = None) -> None:
        self.tokenizer = _Tokenizer()
        self.image_processor = _ProcessorComponent({"image_processor_type": "fixture-image", "patch_size": 16})
        self.video_processor = _ProcessorComponent(
            {"video_processor_type": "fixture-video", "temporal_patch_size": TEMPORAL, "merge_size": MERGE}
        )
        self.chat_template = "fixture-template"
        self.hf_text_form = hf_text_form
        self.corrupt = corrupt

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_tensors: str | None = None,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> Any:
        assert add_generation_prompt is True
        rendered = _render(messages)
        if not tokenize:
            return rendered
        assert return_dict is True and return_tensors == "pt"
        video_blocks = [
            block
            for message in messages
            for block in message["content"]
            if block.get("type") in {"video", "video_frames"}
        ]
        raw_metadata = kwargs.get("videos_kwargs", {}).get("video_metadata")
        metadata = [] if raw_metadata is None else (raw_metadata if isinstance(raw_metadata, list) else [raw_metadata])
        assert len(metadata) == len(video_blocks)
        image_count = 0
        video_grid: list[list[int]] = []

        def expand(match: re.Match[str]) -> str:
            nonlocal image_count
            if match.group(1) == IP:
                image_count += 1
                return VS + IP * (IMAGE_GRID[1] * IMAGE_GRID[2] // MERGE**2) + VE
            row = metadata[len(video_grid)]
            block = video_blocks[len(video_grid)]
            n_frames = len(block["video"])
            groups = ""
            for group_index, timestamp in enumerate(_hf_timestamps(row["frames_indices"], row["fps"])):
                label = f"<{timestamp:.1f} seconds>"
                if self.corrupt == "timestamp" and not video_grid and group_index == 0:
                    label = f"<{timestamp + 0.1:.1f} seconds>"
                n_pads = GRID_H * GRID_W // MERGE**2
                if self.corrupt == "pad" and not video_grid and group_index == 0:
                    n_pads -= 1
                groups += label + VS + VP * n_pads + VE
            video_grid.append([(n_frames + TEMPORAL - 1) // TEMPORAL, GRID_H, GRID_W])
            if self.hf_text_form == "triple":
                return groups
            if self.hf_text_form == "outer":
                return VS + groups + VE
            raise ValueError(self.hf_text_form)

        text = _PLACEHOLDER_RE.sub(expand, rendered)
        if self.corrupt == "stray":
            text = text.replace("<|im_start|>user\n", "<|im_start|>user\n" + VS, 1)
        prompt_ids = self.tokenizer.encode(text)
        result: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long),  # [1,P]
            "attention_mask": torch.ones(1, len(prompt_ids), dtype=torch.bool),  # [1,P]
        }
        if image_count:
            patches = IMAGE_GRID[0] * IMAGE_GRID[1] * IMAGE_GRID[2] * image_count
            result["pixel_values"] = torch.arange(patches * 3, dtype=torch.float32).reshape(patches, 3)  # [N_patch,D]
            result["image_grid_thw"] = torch.tensor([IMAGE_GRID] * image_count, dtype=torch.long)  # [N_img,3]
        if video_grid:
            patches = sum(t * h * w for t, h, w in video_grid)
            result["pixel_values_videos"] = torch.arange(patches * 3, dtype=torch.float32).reshape(patches, 3)
            result["video_grid_thw"] = torch.tensor(video_grid, dtype=torch.long)  # [N_vid,3]
        return result


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        image_token_id=_SPECIAL_TOKENS[IP],
        video_token_id=_SPECIAL_TOKENS[VP],
        vision_start_token_id=_SPECIAL_TOKENS[VS],
        vision_end_token_id=_SPECIAL_TOKENS[VE],
        vision_config=SimpleNamespace(spatial_merge_size=MERGE),
    )


def _video_packer(processor: _Processor | None = None) -> Qwen3VLVideoDataPacker:
    packer = Qwen3VLVideoDataPacker(processor=processor or _Processor(), model_config=_config())
    packer.setup("stub")
    return packer


def _base_packer(processor: _Processor | None = None) -> Qwen3VLDataPacker:
    packer = Qwen3VLDataPacker(processor=processor or _Processor(), model_config=_config())
    packer.setup("stub")
    return packer


def _frames(n: int, *, size: tuple[int, int] = (4, 2), shade: int = 0) -> list[Image.Image]:
    return [Image.new("RGB", size, color=(index + shade, 2 * index, 7)) for index in range(n)]


def _patterned_frames(n: int, *, size: tuple[int, int] = (4, 2)) -> list[Image.Image]:
    """Frames whose pixels are not symmetric under rotation, so an applied EXIF orientation shows in the array."""
    width, height = size
    return [
        Image.fromarray(
            (np.arange(height * width * 3, dtype=np.uint16).reshape(height, width, 3) * 7 + 3 * index).astype(np.uint8)
        )
        for index in range(n)
    ]


def _video_block(frames: list[Any], *, fps: float = FPS, indices: bool = True, **overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "video", "video": frames, "fps": fps, "total_num_frames": len(frames)}
    if indices:
        block["frames_indices"] = list(range(len(frames)))
    block.update(overrides)
    return block


def _messages(*blocks: dict[str, Any], text: str = TEXT) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "text", "text": text}, *blocks]}]


def _obs(messages: list[dict[str, Any]], oid: str = "row") -> ObsBundle:
    return ObsBundle(representations={"prompt": messages}, modality_tags=("text", "video"), id=oid)


def _engine_stream(packer: Qwen3VLDataPacker, messages: list[dict[str, Any]], grid: list[list[int]]) -> list[int]:
    """Independent derivation of the engine's prompt ids: render, tokenize with the placeholders, splice per group."""
    tokenizer = packer.tokenizer
    ids = tokenizer.encode(packer.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    video_blocks = [
        block for message in messages for block in message["content"] if block.get("type") in {"video", "video_frames"}
    ]
    vs, ve, ip, vp = (_SPECIAL_TOKENS[token] for token in (VS, VE, IP, VP))
    out: list[int] = []
    index = 0
    video_index = 0
    while index < len(ids):
        if ids[index : index + 3] == [vs, vp, ve]:
            block = video_blocks[video_index]
            t, h, w = grid[video_index]
            indices = block.get("frames_indices", list(range(len(block["video"]))))
            for timestamp in _hf_timestamps(indices, block["fps"]):
                out += tokenizer.encode(f"<{timestamp:.1f} seconds>") + [vs] + [vp] * (h * w // MERGE**2) + [ve]
            video_index += 1
            index += 3
            continue
        if ids[index : index + 3] == [vs, ip, ve]:
            out += [vs] + [ip] * (IMAGE_GRID[1] * IMAGE_GRID[2] // MERGE**2) + [ve]
            index += 3
            continue
        out.append(ids[index])
        index += 1
    return out


def _refusal(fn: Any, *args: Any, **kwargs: Any) -> str:
    try:
        fn(*args, **kwargs)
    except ValueError as error:
        return f"ValueError: {error}"
    return "ACCEPTED"


# ------------------------------------------------------------------------------------------------ the defect


@pytest.mark.L0
@pytest.mark.CPU
def test_video_policy_ids_are_the_engine_stream_the_base_packer_refuses() -> None:
    """The base packer's HF path BPE-merges the text into the first timestamp ('.<'), one id fewer than the engine's
    stream, so its stored-vs-processor check refuses every stored rollout of such a row. The video packer's policy
    ids ARE the engine stream; the media tensors stay the HF processor's."""
    messages = _messages(_video_block(_frames(4)))
    base, video = _base_packer(), _video_packer()
    completion = video.tokenizer.encode('{"a": 1}')

    hf_ids, _hf_attention, hf_inputs = base._processor_inputs(_obs(messages))
    grid = hf_inputs["video_grid_thw"].tolist()
    engine = _engine_stream(video, messages, grid)
    assert grid == [[2, GRID_H, GRID_W]]
    assert len(hf_ids) == len(engine) - 1 and _MERGE_ID in hf_ids and _MERGE_ID not in engine
    merge_at = hf_ids.index(_MERGE_ID)
    assert engine[merge_at : merge_at + 2] == [_CHAR_BASE + ord("."), _CHAR_BASE + ord("<")]
    with pytest.raises(ValueError, match=f"stored rollout prompt_ids differ .* index {merge_at}"):
        base.get_policy_input(_obs(messages), completion, prompt_ids=engine)

    ids, attention, model_inputs = video._processor_inputs(_obs(messages))
    assert ids == engine
    assert attention == [True] * len(ids)
    assert video.last_hf_text_form == "triple"
    assert torch.equal(model_inputs["pixel_values_videos"], hf_inputs["pixel_values_videos"])
    assert model_inputs["video_grid_thw"].tolist() == grid
    assert set(model_inputs) == set(hf_inputs)
    vs, ve, vp = _SPECIAL_TOKENS[VS], _SPECIAL_TOKENS[VE], _SPECIAL_TOKENS[VP]
    first_group = video.tokenizer.encode(f"<{_hf_timestamps([0, 1, 2, 3], FPS)[0]:.1f} seconds>")
    position = ids.index(vs)
    assert ids[position - len(first_group) : position] == first_group
    assert ids[position + 1 : position + 3] == [vp, vp] and ids[position + 3] == ve
    assert ids.count(vs) == ids.count(ve) == 2 and ids.count(vp) == 4 == hf_ids.count(vp)

    out = video.get_policy_input(_obs(messages), list(completion), prompt_ids=list(engine))
    prompt_len, completion_len = len(engine), len(completion)
    assert out.input_ids == engine + completion
    assert out.logprob_masks == [0] * (prompt_len - 1) + [1] * completion_len + [0]
    assert out.attention_mask == [True] * (prompt_len + completion_len)
    assert torch.equal(out.model_inputs["pixel_values_videos"], hf_inputs["pixel_values_videos"])
    shifted = video.get_policy_input(
        _obs(messages), list(completion), n_ignore_prefix_tokens=2, prompt_ids=list(engine)
    )
    assert shifted.logprob_masks == [0] * (prompt_len + 1) + [1] * (completion_len - 2) + [0]
    assert video.get_policy_input(_obs(messages), list(completion)).input_ids == engine + completion


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("form", HF_TEXT_FORMS)
def test_hf_cross_check_accepts_both_transformers_text_forms(form: str) -> None:
    """4.x replaces the whole placeholder triple (N vision_start), 5.x only <|video_pad|> (N + 1): the same
    engine ids come out either way and the form is recorded."""
    messages = _messages(_video_block(_frames(4)))
    video = _video_packer(_Processor(hf_text_form=form))
    hf_ids, _attention, hf_inputs = _base_packer(_Processor(hf_text_form=form))._processor_inputs(_obs(messages))
    vs = _SPECIAL_TOKENS[VS]
    assert hf_ids.count(vs) == 2 + (1 if form == "outer" else 0)

    ids, _attention, model_inputs = video._processor_inputs(_obs(messages))

    assert ids == _engine_stream(video, messages, hf_inputs["video_grid_thw"].tolist())
    assert video.last_hf_text_form == form
    assert torch.equal(model_inputs["pixel_values_videos"], hf_inputs["pixel_values_videos"])


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("form", HF_TEXT_FORMS)
@pytest.mark.parametrize(
    ("corrupt", "needle"),
    [
        ("pad", "spliced prompt has 4 video pads, the HF path 3"),
        ("timestamp", "matches neither text form"),
        ("stray", "matches neither text form"),
    ],
)
def test_hf_cross_check_refuses_a_corrupted_hf_result(
    form: str, corrupt: str, needle: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HF ids come back through one seam (``_hf_inputs``); hand it ids with one pad fewer, one changed timestamp
    digit or one stray marker (tensors untouched) and the cross-check names what disagrees. A processor that
    corrupts its own tensors/ids pair is refused earlier, by the base media validation."""
    messages = _messages(_video_block(_frames(4)))
    video = _video_packer(_Processor(hf_text_form=form))
    corrupted = _Processor(hf_text_form=form, corrupt=corrupt)

    def hf_inputs(msgs: list[dict[str, Any]], images: list[Any], videos: list[Any]) -> Any:
        _ids, _attention, model_inputs = Qwen3VLDataPacker._processor_inputs_from_messages(video, msgs, images, videos)
        bad = corrupted.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **video._processor_kwargs(msgs),
        )["input_ids"][0].tolist()
        return bad, [True] * len(bad), model_inputs

    monkeypatch.setattr(video, "_hf_inputs", hf_inputs)
    with pytest.raises(ValueError, match=re.escape(needle)):
        video._processor_inputs(_obs(messages))
    with pytest.raises(ValueError):
        _video_packer(corrupted)._processor_inputs(_obs(messages))


# ------------------------------------------------------------------------------------------------ the request


@pytest.mark.L0
@pytest.mark.CPU
def test_rollout_request_materialises_frames_and_names_them_the_base_request_is_refused_at_tp_gt_1() -> None:
    frames = _frames(4)
    messages = _messages(_video_block(frames))
    base, video = _base_packer(), _video_packer()

    base_request = base.get_rollout_input(_obs(messages))
    assert base_request.get("multi_modal_uuids") is None
    assert base_request["multi_modal_data"]["video"][0][0] is frames
    with pytest.raises(ValueError, match="video media requires ordered content UUIDs"):
        vllm_request_media_identity(base_request, context="vLLM training request", require_all_content_uuids=True)

    request = video.get_rollout_input(_obs(messages))
    assert request["prompt"] == _render(messages) == base_request["prompt"]
    array, metadata = request["multi_modal_data"]["video"][0]
    assert array.dtype.name == "uint8" and array.shape == (4, 2, 4, 3)
    assert array[1, 0, 0].tolist() == [1, 2, 7]
    assert metadata == {"fps": FPS, "total_num_frames": 4, "frames_indices": [0, 1, 2, 3]}
    assert request["mm_processor_kwargs"] == {"do_sample_frames": False}
    assert list(request["multi_modal_uuids"]) == ["video"]
    (uuid,) = request["multi_modal_uuids"]["video"]
    digests = [video._materialize_vllm_image(frame, location="f")[1] for frame in frames]
    content = canonical_json_sha256({"frames": digests, "metadata": metadata}, context="t")
    assert uuid == hashlib.sha256(f"{VIDEO_UUID_SEED}\0{0}\0{content}".encode("utf-8")).hexdigest()
    identity = vllm_request_media_identity(request, context="vLLM training request", require_all_content_uuids=True)
    assert identity["cardinality_only_modalities"] == []
    assert video.get_rollout_input(_obs(messages))["multi_modal_uuids"]["video"] == [uuid]
    assert video.get_rollout_input(_obs(_messages(_video_block(frames, fps=2.0))))["multi_modal_uuids"]["video"] != [
        uuid
    ]
    reordered = _messages(_video_block(list(reversed(frames))))
    assert video.get_rollout_input(_obs(reordered))["multi_modal_uuids"]["video"] != [uuid]
    recoloured = _messages(_video_block([*_frames(1, shade=9), *frames[1:]]))
    assert video.get_rollout_input(_obs(recoloured))["multi_modal_uuids"]["video"] != [uuid]
    assert video._unexpanded_prompt_ids(request["prompt"]) == video._server_prompt_ids(messages)
    # The ObsBundle's own block is never mutated by either side.
    assert "video" in messages[0]["content"][1] and messages[0]["content"][1]["video"] is frames


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_side_hands_hf_the_rollout_frames_with_the_exif_orientation_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollout request carries every frame decoded, EXIF-transposed and RGB; the policy side must hand the HF
    processor those same frames, not the row's originals (on transformers 4.57.6 the two paths gave
    different pixel tensors under identical grids). Orientation 3 (rotate 180) keeps the size, so only the pixels tell."""
    frames = _patterned_frames(4)
    for frame in frames:
        frame.getexif()[0x0112] = 3
    originals = np.stack([np.asarray(frame) for frame in frames])
    messages = _messages(_video_block(frames))
    video = _video_packer()
    request_array, _metadata = video.get_rollout_input(_obs(messages, "request"))["multi_modal_data"]["video"][0]
    assert request_array.shape == originals.shape and not np.array_equal(request_array, originals)
    assert np.array_equal(request_array, originals[:, ::-1, ::-1, :]), "orientation 3 = rotate 180"

    seen: list[tuple[list[dict[str, Any]], list[Any]]] = []
    hf_inputs = video._hf_inputs

    def spy(hf_messages: list[dict[str, Any]], images: list[Any], videos: list[Any]) -> Any:
        seen.append((hf_messages, videos))
        return hf_inputs(hf_messages, images, videos)

    monkeypatch.setattr(video, "_hf_inputs", spy)
    ids, _attention, inputs = video._processor_inputs(_obs(messages, "policy"))
    ((hf_messages, hf_videos),) = seen
    hf_array = np.stack([np.asarray(frame.convert("RGB")) for frame in hf_videos[0]])
    assert np.array_equal(hf_array, request_array), "the HF path processes the rollout's pixels"
    assert hf_messages[0]["content"][1]["video"] is hf_videos[0], "the block hands HF the bound frames"
    assert inputs["video_grid_thw"].tolist() == [[2, GRID_H, GRID_W]] and ids.count(_SPECIAL_TOKENS[VP]) == 4
    assert messages[0]["content"][1]["video"] is frames and frames[0].getexif()[0x0112] == 3, "ObsBundle untouched"


@pytest.mark.L0
@pytest.mark.CPU
def test_video_frames_alias_rows_shape_like_video_rows_and_never_rewrite_the_bundle() -> None:
    """A ``video_frames`` block is collected as a video by the base extraction but the HF metadata reader knows only
    ``video``; the packer's copy of the messages carries the alias as ``video`` on both sides (before the fix, the
    alias row was refused with "derived no video metadata")."""
    frames = _frames(4)
    plain = _messages(_video_block(frames))
    alias = _messages(_video_block(frames, type="video_frames"))
    video = _video_packer()
    ids_plain, _a, inputs_plain = video._processor_inputs(_obs(plain, "plain"))
    ids_alias, _a, inputs_alias = video._processor_inputs(_obs(alias, "alias"))
    assert ids_alias == ids_plain and torch.equal(inputs_alias["video_grid_thw"], inputs_plain["video_grid_thw"])
    request_plain = video.get_rollout_input(_obs(plain, "plain-request"))
    request_alias = video.get_rollout_input(_obs(alias, "alias-request"))
    assert request_alias["prompt"] == request_plain["prompt"]
    assert request_alias["multi_modal_uuids"] == request_plain["multi_modal_uuids"]
    assert request_alias["multi_modal_data"]["video"][0][1] == request_plain["multi_modal_data"]["video"][0][1]
    assert alias[0]["content"][1]["type"] == "video_frames" and alias[0]["content"][1]["video"] is frames


# ------------------------------------------------------------------------------------------------ the guards


@pytest.mark.L0
@pytest.mark.CPU
def test_video_guards_refuse_identically_on_the_request_side_and_the_policy_side() -> None:
    video = _video_packer()
    frames = _frames(4)

    def both(messages: list[dict[str, Any]]) -> tuple[str, str]:
        return (
            _refusal(video.get_rollout_input, _obs(messages, "request")),
            _refusal(video._processor_inputs, _obs(messages, "policy")),
        )

    cases = {
        "odd frame count": (
            _messages(_video_block(frames[:3])),
            "not a multiple of the video processor's temporal_patch_size 2",
        ),
        "odd frame count, no indices": (_messages(_video_block(frames[:3], indices=False)), "not a multiple of"),
        "literal placeholder": (
            _messages(_video_block(frames), text=TEXT + VS + VP + VE),
            "2 video / 0 image placeholder triple(s) for 1 video / 0 image",
        ),
        "stray video pad": (
            _messages(_video_block(frames), text=TEXT + VP),
            "outside a placeholder triple: 1 <|video_pad|>",
        ),
        "stray image pad": (
            _messages(_video_block(frames), text=TEXT + IP),
            "outside a placeholder triple: 1 <|image_pad|>",
        ),
        "stray vision start": (
            _messages(_video_block(frames), text=TEXT + VS),
            "outside a placeholder triple: 1 <|vision_start|>",
        ),
        "stray vision end": (
            _messages(_video_block(frames), text=TEXT + VE),
            "outside a placeholder triple: 1 <|vision_end|>",
        ),
        "missing fps": (_messages({"type": "video", "video": frames, "total_num_frames": 4}), "positive numeric fps"),
        "str video handle": (
            _messages({"type": "video", "video": "/clip.mp4", "fps": 1.0}),
            "non-empty list of frames",
        ),
    }
    for name, (messages, needle) in cases.items():
        request_side, policy_side = both(messages)
        assert request_side.startswith("ValueError: ") and needle in request_side, (name, request_side)
        assert request_side == policy_side, (name, request_side, policy_side)

    unequal = _messages(_video_block([*_frames(1, size=(2, 2)), *frames[1:]]))
    assert "share one HxWxC shape" in _refusal(video.get_rollout_input, _obs(unequal))
    with pytest.raises(ValueError, match="refuses a pretokenized request with video"):
        video.get_pretokenized_rollout_input(_obs(_messages(_video_block(frames))), [10, 11])
    good = _messages(_video_block(frames))
    assert both(good)[0] == "ACCEPTED" and both(good)[1] == "ACCEPTED"


# ------------------------------------------------------------------------------------------------ delegation


@pytest.mark.L0
@pytest.mark.CPU
def test_rows_without_video_are_the_base_packer_byte_for_byte() -> None:
    base, video = _base_packer(), _video_packer()
    text_row = _obs([{"role": "user", "content": [{"type": "text", "text": TEXT}]}], "text")
    assert video._processor_inputs(text_row)[0] == base._processor_inputs(text_row)[0]
    assert video.get_rollout_input(text_row) == base.get_rollout_input(text_row)
    image = Image.new("RGB", (2, 2), color=(11, 22, 33))
    image_row = _obs(_messages({"type": "image", "image": image}), "image")
    ids_base, _a, inputs_base = base._processor_inputs(image_row)
    ids_video, _a, inputs_video = video._processor_inputs(image_row)
    assert ids_base == ids_video and torch.equal(inputs_base["pixel_values"], inputs_video["pixel_values"])
    request_base, request_video = base.get_rollout_input(image_row), video.get_rollout_input(image_row)
    assert request_base["prompt"] == request_video["prompt"]
    assert request_base["multi_modal_uuids"] == request_video["multi_modal_uuids"]
    assert video.get_pretokenized_rollout_input(image_row, ids_video) == base.get_pretokenized_rollout_input(
        image_row, ids_base
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_video_and_image_plus_video_rows_splice_in_order() -> None:
    base, video = _base_packer(), _video_packer()
    frames = _frames(6)
    two = _messages(
        _video_block(frames[:4]),
        {"type": "text", "text": "Second clip."},
        _video_block(frames[4:], fps=2.0),
        {"type": "text", "text": "Compare."},
    )
    ids, _a, inputs = video._processor_inputs(_obs(two, "two"))
    grid = inputs["video_grid_thw"].tolist()
    assert grid == [[2, GRID_H, GRID_W], [1, GRID_H, GRID_W]]
    assert ids == _engine_stream(video, two, grid)
    assert ids.count(_SPECIAL_TOKENS[VP]) == 6
    hf_ids = base._processor_inputs(_obs(two, "two-base"))[0]
    assert video.tokenizer.decode(ids) == video.tokenizer.decode(hf_ids) and len(ids) == len(hf_ids) + 2
    request = video.get_rollout_input(_obs(two, "two-request"))
    assert len(request["multi_modal_data"]["video"]) == 2 and len(request["multi_modal_uuids"]["video"]) == 2
    assert request["multi_modal_data"]["video"][1][1]["fps"] == 2.0

    mixed = _messages(
        {"type": "image", "image": Image.new("RGB", (2, 2))},
        {"type": "text", "text": "Then the clip."},
        _video_block(frames[:4]),
    )
    ids, _a, inputs = video._processor_inputs(_obs(mixed, "mixed"))
    assert ids == _engine_stream(video, mixed, inputs["video_grid_thw"].tolist())
    assert ids.count(_SPECIAL_TOKENS[IP]) == 2 and ids.count(_SPECIAL_TOKENS[VP]) == 4
    assert "pixel_values" in inputs and "pixel_values_videos" in inputs
    request = video.get_rollout_input(_obs(mixed, "mixed-request"))
    assert sorted(request["multi_modal_uuids"]) == ["image", "video"]
    assert isinstance(request["multi_modal_data"]["image"][0], Image.Image)


# ------------------------------------------------------------------------------------------------ identity / wiring


@pytest.mark.L0
@pytest.mark.CPU
def test_identity_declares_the_video_policy_ids_and_the_factory_is_loadable() -> None:
    from cosmos_framework.posttrain.config.model import DataPackerConfig
    from cosmos_framework.posttrain.data.packer.base import PackerProcessorIdentity, PackerSemanticsIdentity

    base, video = _base_packer(), _video_packer()
    assert isinstance(video, PackerSemanticsIdentity) and isinstance(video, PackerProcessorIdentity)
    declared = video.validation_semantics_identity()
    assert declared.pop("video_policy_ids") == VIDEO_POLICY_IDS_SEMANTICS
    assert declared == base.validation_semantics_identity()
    assert video.processor_semantics_identity() == base.processor_semantics_identity()
    assert base.validation_semantics_identity() != video.validation_semantics_identity()

    factory = "cosmos_framework.posttrain.data.packer.qwen3_vl_video.make_packer"
    pk.validate_data_packer_factories(DataPackerConfig(factory=factory))
    assert type(make_packer()) is Qwen3VLVideoDataPacker
    assert pk.Qwen3VLVideoDataPacker is Qwen3VLVideoDataPacker
    # The registry default is unchanged: the image packer, not the video packer.
    assert type(pk.get_default_data_packer("qwen3_vl")) is Qwen3VLDataPacker


# ------------------------------------------------------------------------------------------------ real model (optional)

#: Recorded from vLLM's own ``Qwen3VLMultiModalProcessor`` (0.23 and 0.24 agree) on the row below with the Qwen3-VL
#: tokenizer: 16 frames of 576x320, fps 0.3976801988400994, the two texts. Any Qwen3-VL processor directory reproduces it.
KNOWN_ANSWER: dict[str, Any] = {
    "fps": 0.3976801988400994,
    "n_frames": 16,
    "frame_size": (576, 320),
    "text": "You are an autonomous-driving scene-recognition annotation expert. Return only the JSON result.",
    "analyze": "Analyze this video and return only the JSON result.",
    "n_ids": 1545,
    "sha256_csv": "1f2656de59ae16657b4bb3751318cd977cec9bb9845e7d821ee8f586099ef148",
    "grid": [[8, 20, 36]],
    "video_pads": 1440,
    "first_group_ids": [27, 16, 13, 18, 6486, 29],
    "timestamps": [f"<{value} seconds>" for value in ("1.3", "6.3", "11.3", "16.3", "21.4", "26.4", "31.4", "36.5")],
    "vision_start_positions": [36, 224, 413, 602, 791, 980, 1169, 1358],
    "n_hf_ids": {"triple": 1544, "outer": 1547},
}


def _real_engine_stream(
    packer: Qwen3VLVideoDataPacker, messages: list[dict[str, Any]], grid: list[list[int]]
) -> list[int]:
    """The independent derivation on a real tokenizer: render, tokenize, splice per temporal group."""
    tokenizer = packer.tokenizer
    rendered = packer.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template=packer._resolved_chat_template()
    )
    ids = [int(t) for t in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
    vs, ve, vp = packer.vision_start_token_id, packer.vision_end_token_id, packer.video_token_id
    blocks = [block for message in messages for block in message["content"] if block.get("type") == "video"]
    temporal = packer.processor.video_processor.temporal_patch_size
    merge = packer.processor.video_processor.merge_size
    out: list[int] = []
    index = video_index = 0
    while index < len(ids):
        if ids[index : index + 3] == [vs, vp, ve]:
            block = blocks[video_index]
            t, h, w = grid[video_index]
            indices = list(block.get("frames_indices", range(len(block["video"]))))
            if len(indices) % temporal:
                indices += [indices[-1]] * (temporal - len(indices) % temporal)
            seconds = [i / block["fps"] for i in indices]
            for g in range(0, len(seconds), temporal):
                timestamp = (seconds[g] + seconds[g + temporal - 1]) / 2
                out += tokenizer.encode(f"<{timestamp:.1f} seconds>", add_special_tokens=False)
                out += [vs] + [vp] * (h * w // merge**2) + [ve]
            video_index += 1
            index += 3
            continue
        out.append(ids[index])
        index += 1
    return out


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.skipif(
    not os.environ.get("QWEN3_VL_VIDEO_TEST_MODEL"), reason="QWEN3_VL_VIDEO_TEST_MODEL names no Qwen3-VL processor dir"
)
def test_real_processor_known_answer_and_optional_row(tmp_path: Path) -> None:
    """With a real Qwen3-VL processor: the known answer recorded from vLLM's processor, the HF text form of the
    installed transformers, and (``QWEN3_VL_VIDEO_TEST_ROW`` = a JSON file with ``messages``) one real row."""
    import json

    import numpy as np
    import transformers

    model = os.environ["QWEN3_VL_VIDEO_TEST_MODEL"]
    expected_form = "triple" if int(transformers.__version__.split(".")[0]) < 5 else "outer"
    frames: list[str] = []
    for i in range(KNOWN_ANSWER["n_frames"]):
        y, x = np.mgrid[0 : KNOWN_ANSWER["frame_size"][1], 0 : KNOWN_ANSWER["frame_size"][0]]
        array = np.stack([(x + 7 * i) % 256, (y + 11 * i) % 256, (x + y + 13 * i) % 256], axis=-1).astype(np.uint8)
        path = tmp_path / f"f{i:02d}.jpg"
        Image.fromarray(array).save(path, quality=90)
        frames.append(str(path))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": KNOWN_ANSWER["text"]},
                {"type": "text", "text": KNOWN_ANSWER["analyze"]},
                _video_block(frames, fps=KNOWN_ANSWER["fps"]),
            ],
        }
    ]
    base = Qwen3VLDataPacker()
    base.setup(model)
    video = Qwen3VLVideoDataPacker(processor=base.processor, model_config=base.model_config)
    video.setup(model)
    vs, ve, vp = base.vision_start_token_id, base.vision_end_token_id, base.video_token_id
    completion = [int(t) for t in base.tokenizer('{"Weather": "Clear"}', add_special_tokens=False)["input_ids"]]

    hf_ids, _a, hf_inputs = base._processor_inputs(_obs(messages, "hf"))
    grid = hf_inputs["video_grid_thw"].tolist()
    assert grid == KNOWN_ANSWER["grid"]
    assert len(hf_ids) == KNOWN_ANSWER["n_hf_ids"][expected_form]
    with pytest.raises(ValueError, match="stored rollout prompt_ids differ"):
        base.get_policy_input(_obs(messages, "hf"), completion, prompt_ids=_real_engine_stream(video, messages, grid))

    ids, attention, model_inputs = video._processor_inputs(_obs(messages, "video"))
    assert video.last_hf_text_form == expected_form
    assert ids == _real_engine_stream(video, messages, grid)
    assert len(ids) == KNOWN_ANSWER["n_ids"]
    assert hashlib.sha256(",".join(map(str, ids)).encode("utf-8")).hexdigest() == KNOWN_ANSWER["sha256_csv"]
    assert ids.count(vp) == KNOWN_ANSWER["video_pads"] == hf_ids.count(vp)
    assert ids.count(vs) == ids.count(ve) == 8
    assert [i for i, t in enumerate(ids) if t == vs] == KNOWN_ANSWER["vision_start_positions"]
    position = ids.index(vs)
    assert ids[position - len(KNOWN_ANSWER["first_group_ids"]) : position] == KNOWN_ANSWER["first_group_ids"]
    assert base.tokenizer.decode(KNOWN_ANSWER["first_group_ids"]) == KNOWN_ANSWER["timestamps"][0]
    assert torch.equal(model_inputs["pixel_values_videos"], hf_inputs["pixel_values_videos"])
    assert attention == [True] * len(ids)
    out = video.get_policy_input(_obs(messages, "video"), list(completion), prompt_ids=list(ids))
    assert out.input_ids == ids + completion
    assert out.logprob_masks == [0] * (len(ids) - 1) + [1] * len(completion) + [0]
    request = video.get_rollout_input(_obs(messages, "request"))
    array, metadata = request["multi_modal_data"]["video"][0]
    assert array.shape == (16, 320, 576, 3) and array.dtype == np.uint8
    assert metadata["fps"] == KNOWN_ANSWER["fps"] and metadata["frames_indices"] == list(range(16))
    assert len(request["multi_modal_uuids"]["video"]) == 1
    vllm_request_media_identity(request, context="vLLM training request", require_all_content_uuids=True)

    row_path = os.environ.get("QWEN3_VL_VIDEO_TEST_ROW")
    if row_path:
        real = json.loads(Path(row_path).read_text())["messages"]
        real_hf_ids, _a, real_inputs = base._processor_inputs(_obs(real, "real-hf"))
        real_grid = real_inputs["video_grid_thw"].tolist()
        real_engine = _real_engine_stream(video, real, real_grid)
        with pytest.raises(ValueError, match="stored rollout prompt_ids differ"):
            base.get_policy_input(_obs(real, "real-hf"), completion, prompt_ids=real_engine)
        real_ids, _a, real_model_inputs = video._processor_inputs(_obs(real, "real"))
        assert real_ids == real_engine and video.last_hf_text_form == expected_form
        assert torch.equal(real_model_inputs["pixel_values_videos"], real_inputs["pixel_values_videos"])
        real_out = video.get_policy_input(_obs(real, "real"), list(completion), prompt_ids=list(real_ids))
        assert real_out.logprob_masks == [0] * (len(real_ids) - 1) + [1] * len(completion) + [0]
        print(
            f"REAL_ROW n_engine_ids={len(real_ids)} n_hf_ids={len(real_hf_ids)} grid={real_grid} "
            f"video_pads={real_ids.count(vp)} form={video.last_hf_text_form} loss_tokens={sum(real_out.logprob_masks)}"
        )
