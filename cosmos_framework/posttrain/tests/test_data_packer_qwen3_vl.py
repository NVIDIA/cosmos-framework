# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.data.packer import BaseDataPacker
from cosmos_framework.posttrain.data.packer.decoder_only_llm import shape_policy_input
from cosmos_framework.posttrain.data.packer.qwen3_vl import Qwen3VLDataPacker, Qwen3VLPolicyInput
from cosmos_framework.posttrain.schema import ObsBundle, Transition


class _TokenizerResult:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class _Tokenizer:
    pad_token_id: int = 0
    eos_token_id: int = 2
    chat_template: str = "fixture-template"

    def __call__(self, text: str, add_special_tokens: bool = False, **_kwargs: Any) -> _TokenizerResult:
        prefix = [1] if add_special_tokens else []
        return _TokenizerResult(prefix + [20 + index for index, _character in enumerate(text)])

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {"<pad>": 0, "<eos>": 2, "<vision_start>": 90, "<image>": 91, "<video>": 92}


class _ProcessorComponent:
    """Fixture stand-in for a Hugging Face image/video processor: identity binds ``to_dict()``."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def to_dict(self) -> dict[str, Any]:
        return dict(self.config)


_IMAGE_PROCESSOR_CONFIG: dict[str, Any] = {"image_processor_type": "fixture-image", "patch_size": 16}
_VIDEO_PROCESSOR_CONFIG: dict[str, Any] = {
    "video_processor_type": "fixture-video",
    "fps": 2,
    "max_frames": 16,
    "size": {"longest_edge": 4096},
}


class _Processor:
    def __init__(self, *, bad_image_patch_count: bool = False, video_t: int = 1) -> None:
        self.tokenizer = _Tokenizer()
        self.image_processor = _ProcessorComponent(_IMAGE_PROCESSOR_CONFIG)
        self.video_processor = _ProcessorComponent(_VIDEO_PROCESSOR_CONFIG)
        self.bad_image_patch_count = bad_image_patch_count
        self.video_t = video_t
        self.last_messages: list[dict[str, Any]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
        return_tensors: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        assert add_generation_prompt is True
        self.last_messages = messages
        if not tokenize:
            return "<rendered>"
        assert return_dict is True and return_tensors == "pt"
        image_count = 0
        video_count = 0
        for message in messages:
            for block in message["content"]:
                image_count += int(block.get("type") in {"image", "image_url"})
                video_count += int(block.get("type") in {"video", "video_frames"})
        prompt_ids = [10]
        for _index in range(image_count):
            prompt_ids.extend([90, 91, 91])
        for _index in range(video_count):
            for _frame in range(self.video_t):
                prompt_ids.extend([90, 92, 92])
        prompt_ids.append(11)
        result: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor([prompt_ids], dtype=torch.long),  # [1,P]
            "attention_mask": torch.ones(1, len(prompt_ids), dtype=torch.bool),  # [1,P]
        }
        if image_count:
            image_patches = 7 if self.bad_image_patch_count else 8 * image_count
            result["pixel_values"] = torch.arange(image_patches * 3, dtype=torch.float32).reshape(
                image_patches, 3
            )  # [N_patch,D]
            result["image_grid_thw"] = torch.tensor([[1, 2, 4]] * image_count, dtype=torch.long)  # [N_img,3]
        if video_count:
            result["pixel_values_videos"] = torch.arange(
                8 * self.video_t * video_count * 3,
                dtype=torch.float32,
            ).reshape(
                8 * self.video_t * video_count,
                3,
            )  # [N_patch,D]
            result["video_grid_thw"] = torch.tensor(  # [N_vid,3]
                [[self.video_t, 2, 4]] * video_count,
                dtype=torch.long,
            )
            result["second_per_grid_ts"] = torch.ones(video_count, dtype=torch.float32)  # [N_vid]
        return result


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        image_token_id=91,
        video_token_id=92,
        vision_start_token_id=90,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )


def _packer(*, processor: _Processor | None = None) -> Qwen3VLDataPacker:
    packer = Qwen3VLDataPacker(processor=processor or _Processor(), model_config=_config())
    packer.setup("stub")
    return packer


def _image_obs(handle: Any | None = None) -> ObsBundle:
    image = object() if handle is None else handle
    return ObsBundle(
        representations={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            "images": [image],
        },
        modality_tags=("text", "image"),
        id="sample-image",
    )


def _video_obs(handle: Any | None = None) -> ObsBundle:
    video = object() if handle is None else handle
    return ObsBundle(
        representations={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "fps": 12.0},
                        {"type": "text", "text": "describe"},
                    ],
                }
            ],
            "videos": [video],
        },
        modality_tags=("text", "video"),
        id="sample-video",
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_qwen_packer_is_registered_for_dense_and_moe() -> None:
    for model_type in ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"):
        assert isinstance(BaseDataPacker.get_default_data_packer(model_type), Qwen3VLDataPacker)


@pytest.mark.L0
@pytest.mark.CPU
def test_rollout_and_pretokenized_requests_snapshot_images_and_preserve_obs(tmp_path: Path) -> None:
    image_path = tmp_path / "student-request.png"
    Image.new("RGB", (2, 2), color=(11, 22, 33)).save(image_path, format="PNG")
    image_bytes = image_path.read_bytes()
    obs = _image_obs(str(image_path))
    original_messages = obs.representations["messages"]
    packer = _packer()

    rollout_request = packer.get_rollout_input(obs)
    exact_ids = [10, 90, 91, 91, 11]
    pretokenized = packer.get_pretokenized_rollout_input(obs, exact_ids)

    assert rollout_request["prompt"] == "<rendered>"
    rollout_image = rollout_request["multi_modal_data"]["image"][0]
    assert isinstance(rollout_image, Image.Image)
    assert rollout_image.getpixel((0, 0)) == (11, 22, 33)
    # The teacher request carries the image run collapsed to one marker; vLLM re-expands it.
    assert pretokenized["prompt_token_ids"] == [10, 90, 91, 11]
    pretokenized_image = pretokenized["multi_modal_data"]["image"][0]
    assert isinstance(pretokenized_image, Image.Image)
    assert pretokenized_image.getpixel((0, 0)) == (11, 22, 33)
    raw_sha256 = hashlib.sha256(image_bytes).hexdigest()
    expected_uuid = hashlib.sha256(f"qwen3-vl-image-v1\0{0}\0{raw_sha256}".encode("utf-8")).hexdigest()
    assert rollout_request["multi_modal_uuids"] == {"image": [expected_uuid]}
    assert pretokenized["multi_modal_uuids"] == {"image": [expected_uuid]}
    assert obs.representations["messages"] is original_messages
    assert "image" not in obs.representations["messages"][0]["content"][0]
    assert isinstance(packer.processor.last_messages[0]["content"][0]["image"], Image.Image)


@pytest.mark.L0
@pytest.mark.CPU
def test_pretokenized_request_collapses_each_image_run_and_keeps_the_completion_verbatim() -> None:
    """vLLM re-expands the i-th `<|image_pad|>` match into the i-th image's run on token prompts, so
    the same-token teacher request carries one marker per image. The completion tail (here with its
    own repeated token) is never touched, and the exact ids are still validated against the media."""
    obs = ObsBundle(
        representations={
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "compare"}],
                }
            ],
            "images": [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))],
        },
        modality_tags=("text", "image"),
        id="sample-two-images",
    )
    processed_prompt = [10, 90, 91, 91, 90, 91, 91, 11]
    completion = [7, 7, 8]

    request = _packer().get_pretokenized_rollout_input(obs, [*processed_prompt, *completion])

    assert request["prompt_token_ids"] == [10, 90, 91, 90, 91, 11, 7, 7, 8]
    assert len(request["multi_modal_data"]["image"]) == 2
    assert len(request["multi_modal_uuids"]["image"]) == 2
    with pytest.raises(ValueError, match="placeholder count 3 != grid-derived visual token count 4"):
        _packer().get_pretokenized_rollout_input(obs, [10, 90, 91, 90, 91, 11, 7, 91, 8])


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("obs_factory", "expected_key"),
    [(_image_obs, "pixel_values"), (_video_obs, "pixel_values_videos")],
)
def test_authoritative_prompt_ids_keep_processed_media_and_next_token_mask(
    obs_factory: Any,
    expected_key: str,
) -> None:
    packer = _packer()
    obs = obs_factory()
    prompt_ids = [10, 90, 91, 91, 11] if expected_key == "pixel_values" else [10, 90, 92, 92, 11]
    completion_ids = torch.tensor([7, 8], dtype=torch.long)  # [C]

    processed = packer.get_policy_input(obs, completion_ids, prompt_ids=prompt_ids)

    assert isinstance(processed, Qwen3VLPolicyInput)
    assert processed.input_ids == prompt_ids + [7, 8]
    assert processed.logprob_masks == [0, 0, 0, 0, 1, 1, 0]
    assert expected_key in processed.model_inputs
    assert len(processed.attention_mask) == len(processed.input_ids)


@pytest.mark.L0
@pytest.mark.CPU
def test_shape_policy_input_always_calls_vlm_packer_with_stored_ids() -> None:
    packer = _packer()
    prompt_ids = torch.tensor([10, 90, 91, 91, 11], dtype=torch.long)  # [P]
    transition = Transition(
        obs=_image_obs(),
        action=torch.tensor([7], dtype=torch.long),  # [C]
        policy_info={"prompt_ids": prompt_ids},
    )

    processed = shape_policy_input(transition, packer)

    assert isinstance(processed, Qwen3VLPolicyInput)
    assert "pixel_values" in processed.model_inputs


@pytest.mark.L0
@pytest.mark.CPU
def test_authoritative_prompt_and_media_cardinality_fail_loud() -> None:
    with pytest.raises(ValueError, match="stored rollout prompt_ids differ"):
        _packer().get_policy_input(_image_obs(), [7], prompt_ids=[10, 90, 91, 99, 11])
    packer = _packer(processor=_Processor(bad_image_patch_count=True))
    with pytest.raises(ValueError, match="grid requires 8"):
        packer.get_policy_input(_image_obs(), [7], prompt_ids=[10, 90, 91, 91, 11])
    with pytest.raises(ValueError, match="prompt/media cardinality mismatch"):
        _packer().get_pretokenized_rollout_input(_image_obs(Image.new("RGB", (2, 2))), [10, 11])


@pytest.mark.L0
@pytest.mark.CPU
def test_qwen_policy_collate_right_pads_and_concatenates_media() -> None:
    packer = _packer()
    image = packer.get_policy_input(_image_obs(), [7, 8], prompt_ids=[10, 90, 91, 91, 11])
    text_obs = ObsBundle(
        representations={"messages": [{"role": "user", "content": "hello"}]},
        modality_tags=("text",),
        id="sample-text",
    )
    text = packer.get_policy_input(text_obs, [7], prompt_ids=[10, 11])
    max_len = packer.policy_compute_max_len([image, text])

    batch = packer.policy_collate_fn([image, text], max_len)

    assert batch["input_ids"].shape == (2, max_len)
    assert batch["attention_mask"].shape == (2, max_len)
    assert batch["logprob_masks"].shape == (2, max_len)
    assert not batch["attention_mask"][1, len(text.input_ids) :].any()
    assert batch["input_ids"][1, len(text.input_ids) :].eq(packer.pad_token_id).all()
    assert batch["pixel_values"].shape == (8, 3)
    assert batch["image_grid_thw"].shape == (1, 3)


@pytest.mark.L0
@pytest.mark.CPU
def test_multiframe_video_uses_timestamp_markers_and_flat_vllm_metadata() -> None:
    """One video handle may expand to T timestamped vision blocks; T is not a media count."""
    frames = [object(), object(), object()]
    packer = _packer(processor=_Processor(video_t=3))
    obs = _video_obs(frames)
    prompt_ids = [10, 90, 92, 92, 90, 92, 92, 90, 92, 92, 11]

    processed = packer.get_policy_input(obs, [7], prompt_ids=prompt_ids)
    request = packer.get_rollout_input(obs)

    assert processed.model_inputs["video_grid_thw"].tolist() == [[3, 2, 4]]
    assert request["mm_processor_kwargs"]["do_sample_frames"] is False
    assert "videos_kwargs" not in request["mm_processor_kwargs"]
    video, metadata = request["multi_modal_data"]["video"][0]
    assert video is frames
    assert metadata == {
        "fps": 12.0,
        "total_num_frames": 3,
        "frames_indices": [0, 1, 2],
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_packer_identity_protocols_declare_semantics_including_the_video_processor() -> None:
    """Validation and the OPD teacher bind a packer only through the declared identity protocols."""
    from cosmos_framework.posttrain.config.build import _cross_token_multimodal_teacher_identity
    from cosmos_framework.posttrain.data.packer.base import PackerProcessorIdentity, PackerSemanticsIdentity
    from cosmos_framework.posttrain.roles.rollout.validation import _packer_semantics_identity
    from cosmos_framework.posttrain.utils.identity import qualified_type

    packer = _packer()
    assert isinstance(packer, PackerSemanticsIdentity) and isinstance(packer, PackerProcessorIdentity)
    declared = _packer_semantics_identity(packer)["declared"]
    assert (declared["spatial_merge_size"], declared["video_token_id"]) == (2, 92)
    assert declared["processor"]["video_processor_type"] == qualified_type(packer.processor.video_processor)
    assert declared["processor"]["video_processor_sha256"]
    # Two processors differing only in video sampling never share an identity; text-only stays stable.
    resampled = _Processor()
    resampled.video_processor = _ProcessorComponent({**_VIDEO_PROCESSOR_CONFIG, "fps": 4})
    assert _packer(processor=resampled).validation_semantics_identity() != packer.validation_semantics_identity()
    text_only = _Processor()
    del text_only.video_processor
    assert _packer(processor=text_only).processor_semantics_identity()["video_processor_sha256"] is None

    class _Undeclared:
        def __init__(self) -> None:
            self.processor = object()
            self.model_config = SimpleNamespace(vision_config=object())

    with pytest.raises(TypeError, match="PackerSemanticsIdentity"):
        _packer_semantics_identity(_Undeclared())
    with pytest.raises(TypeError, match="PackerProcessorIdentity"):
        _cross_token_multimodal_teacher_identity(_Undeclared())


@pytest.mark.L0
@pytest.mark.CPU
def test_projected_conversation_prompt_renders_like_the_grpo_row_and_satisfies_the_ppo_rollout(
    tmp_path: Path,
) -> None:
    """A pool row's conversation list (text + image inside the content parts) reaches this packer the
    same way from both data roles -- `project_prompt_items` (PPO) and `jsonl_prompts` (GRPO) produce the
    same item -- and renders to the same request: `_messages_and_media` takes the list as the messages,
    `apply_chat_template` renders it, the image path inside the content becomes the snapshotted vLLM
    image. `PPOVLLMRollout._prepare_prompt` then reads that mapping as-is (the rendered ``prompt`` string
    and the media); it converts nothing to a string itself, and the policy-side shaping the PPO roles
    consume sees the image through the same list."""
    from cosmos_framework.posttrain.roles.env.dataset import jsonl_prompts, project_prompt_items
    from cosmos_framework.posttrain.roles.rollout.ppo import PPOVLLMRollout
    from cosmos_framework.posttrain.tests.helpers.conversation_rows import CONVERSATION_ROWS, write_conversation_jsonl

    image_path = tmp_path / "pool-row.png"
    Image.new("RGB", (2, 2), color=(44, 55, 66)).save(image_path, format="PNG")
    row = copy.deepcopy(CONVERSATION_ROWS[1])  # text + image: the exact-match shape
    row["messages"][0]["content"][1]["image"] = str(image_path)

    class _Rows:
        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int) -> dict[str, Any]:
            return row

        @staticmethod
        def cursor_fingerprint() -> str:
            return "sha256:pool-row"

    ppo_item = project_prompt_items(
        _Rows(),
        source_name="pool.jsonl",
        split="train",
        prompt_column="messages",
        response_column="reward_spec",
        seed=0,
        fingerprint_prefix="ppo-prompt",
    )[0]
    grpo_item = jsonl_prompts(write_conversation_jsonl(tmp_path, [row]), "messages", "reward_spec")[0]
    assert ppo_item == grpo_item == {"prompt": row["messages"], "reference": row["reward_spec"]}

    packer = _packer()
    observations = {
        name: ObsBundle(representations=dict(item), modality_tags=("text",), id=f"row-{name}")
        for name, item in (("ppo", ppo_item), ("grpo", grpo_item))
    }
    requests = {name: packer.get_rollout_input(obs) for name, obs in observations.items()}
    for request in requests.values():
        assert request["prompt"] == "<rendered>"
        assert request["multi_modal_data"]["image"][0].getpixel((0, 0)) == (44, 55, 66)
    assert requests["ppo"]["multi_modal_uuids"] == requests["grpo"]["multi_modal_uuids"]
    rendered_messages = packer.processor.last_messages
    assert rendered_messages[0]["content"][0] == {"type": "text", "text": row["messages"][0]["content"][0]["text"]}
    assert isinstance(rendered_messages[0]["content"][1]["image"], Image.Image)

    prepared = PPOVLLMRollout._prepare_prompt(SimpleNamespace(packer=packer), observations["ppo"])
    assert prepared.rendered_prompt == "<rendered>"
    assert prepared.prompt_token_ids == ()
    assert prepared.vllm_input["multi_modal_data"]["image"][0].getpixel((0, 0)) == (44, 55, 66)
    assert prepared.observation.representations["prompt"] == row["messages"]
    policy_input = packer.get_policy_input(prepared.observation, [7, 8], prompt_ids=[10, 90, 91, 91, 11])
    assert policy_input.model_inputs["image_grid_thw"].tolist() == [[1, 2, 4]]
    assert policy_input.input_ids == [10, 90, 91, 91, 11, 7, 8]
