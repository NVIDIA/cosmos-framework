# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import subprocess
import textwrap
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cosmos_framework.posttrain.schema import ObsBundle


def _bind_fake_processor(packer: Any, *, rendered: str) -> list[object]:
    """Bind the smallest processor needed by rollout-only tests."""

    seen: list[object] = []

    class _FakeProcessor:
        def apply_chat_template(
            self,
            sample: object,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            seen.append(sample)
            return rendered

    packer.hf_processor = _FakeProcessor()
    packer.image_token = None
    return seen


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_semantics_identity_binds_hf_vlm_generation_configuration() -> None:
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    class _Component:
        def __init__(self, value: int) -> None:
            self.value = value

        def to_dict(self) -> dict[str, int]:
            return {"value": self.value}

    class _Processor:
        chat_template = "{{ messages }}"
        image_processor = _Component(1)

    packer = HFVLMDataPacker()
    packer.hf_processor = _Processor()
    packer.hf_config = SimpleNamespace(to_dict=lambda: {"model_type": "qwen3_vl", "rope_theta": 1.0})
    packer.image_token = "<image>"
    packer.image_token_id = 11
    packer.model_type = "qwen3_vl"

    first = packer.validation_semantics_identity()

    assert first["model_type"] == "qwen3_vl"
    assert first["image_processor_sha256"]
    assert first["hf_config_sha256"]


@pytest.mark.L0
@pytest.mark.CPU
def test_importing_the_packer_does_not_import_qwen_vl_utils():
    """Exercise lazy imports without deleting shared modules from this pytest process."""
    code = textwrap.dedent(
        """
        import sys
        import cosmos_framework.posttrain.data.packer.hf_vlm  # noqa: F401
        assert "qwen_vl_utils" not in sys.modules
        """
    )
    result = subprocess.run(["python", "-c", code], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_input_mask_matches_the_text_packer_arithmetic():
    """The mask is the contract the objective indexes against; it must match DecoderOnlyLLMDataPacker
    exactly, or completion logprobs silently misalign."""
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    prompt_len, completion_len, n_ign = 5, 3, 1
    masks = HFVLMDataPacker._logprob_masks(prompt_len, completion_len, n_ign)
    assert masks == [0] * (prompt_len - 1 + n_ign) + [1] * (completion_len - n_ign) + [0]
    assert len(masks) == prompt_len + completion_len


@pytest.mark.L0
@pytest.mark.CPU
def test_rollout_input_carries_multi_modal_data():
    """vLLM needs {"prompt": ..., "multi_modal_data": {"image": [...]}} — a bare string cannot
    carry an image, which is the whole reason the text packer's `-> str` signature had to widen."""
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    packer = HFVLMDataPacker()
    _bind_fake_processor(packer, rendered="<im>PROMPT")
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": np.zeros((4, 4, 3), dtype=np.uint8)},
                {"type": "text", "text": "what is this?"},
            ],
        }
    ]
    # posttrain hands the packer an ObsBundle, NOT a bare conversation. Testing the bare list is
    # testing upstream's calling convention; the real path failed with
    # "'ObsBundle' object is not iterable" until `_as_conversation` bridged it.
    out = packer.get_rollout_input(ObsBundle(representations={"prompt": conv}, modality_tags=("text",), id="i"))
    assert isinstance(out, dict), f"rollout input must be a dict, got {type(out).__name__}"
    assert out["prompt"] == "<im>PROMPT"
    assert len(out["multi_modal_data"]["image"]) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_text_only_conversation_omits_multi_modal_data():
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    packer = HFVLMDataPacker()
    _bind_fake_processor(packer, rendered="PROMPT")
    out = packer.get_rollout_input(
        ObsBundle(
            representations={"prompt": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
            modality_tags=("text",),
            id="i",
        )
    )
    assert out == {"prompt": "PROMPT"}, "a text-only conversation must not claim multi_modal_data"


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_qwen_pretokenized_request_collapses_only_the_image_prompt_prefix() -> None:
    """HF Qwen and the native packer share marker normalization; other HF VLMs stay byte-identical."""

    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    processor_prompt_ids = [10, 90, 91, 91, 91, 92, 11]

    class _Processor:
        def __init__(self) -> None:
            self.calls = 0

        def apply_chat_template(
            self,
            _conversation: object,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert tokenize is False and add_generation_prompt is True
            return "rendered"

        def __call__(
            self,
            *,
            text: list[str],
            images: list[object],
            return_tensors: str,
        ) -> dict[str, list[list[int]]]:
            assert text == ["rendered"] and len(images) == 1 and return_tensors == "pt"
            self.calls += 1
            return {
                "input_ids": [list(processor_prompt_ids)],
                "attention_mask": [[1] * len(processor_prompt_ids)],
            }

    processor = _Processor()
    packer = HFVLMDataPacker()
    packer.hf_processor = processor
    packer.image_token = None
    packer.image_token_id = 91
    packer.model_type = "qwen3_vl"
    item = ObsBundle(
        representations={
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": np.zeros((2, 2, 3), dtype=np.uint8)},
                        {"type": "text", "text": "What is shown?"},
                    ],
                }
            ]
        },
        modality_tags=("text", "image"),
        id="hf-qwen-image",
    )
    completion_ids = [7, 91, 91, 8]
    exact_ids = [*processor_prompt_ids, *completion_ids]

    request = packer.get_pretokenized_rollout_input(item, exact_ids)

    assert request["prompt_token_ids"] == [10, 90, 91, 92, 11, *completion_ids]
    assert len(request["multi_modal_data"]["image"]) == 1
    assert processor.calls == 1

    packer.model_type = "other_vlm"
    unchanged = packer.get_pretokenized_rollout_input(item, exact_ids)
    assert unchanged["prompt_token_ids"] == exact_ids
    assert processor.calls == 1, "non-Qwen HF VLMs must not enter the Qwen processor-normalization path"

    packer.model_type = "qwen3_vl_moe"
    with pytest.raises(ValueError, match="processor's exact prompt IDs"):
        packer.get_pretokenized_rollout_input(item, [99, *exact_ids[1:]])


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("What is shown?", [{"role": "user", "content": [{"type": "text", "text": "What is shown?"}]}]),
        (
            [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "What is shown?"}],
            [
                {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
                {"role": "user", "content": [{"type": "text", "text": "What is shown?"}]},
            ],
        ),
    ],
)
def test_rollout_normalizes_text_prompts_to_hf_conversation_content(
    prompt: object,
    expected: list[dict[str, object]],
) -> None:
    """HF chat templates receive one canonical conversation shape on every supported input path."""
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    packer = HFVLMDataPacker()
    seen = _bind_fake_processor(packer, rendered="PROMPT")

    output = packer.get_rollout_input(
        ObsBundle(representations={"prompt": prompt}, modality_tags=("text",), id="sample")
    )

    assert output == {"prompt": "PROMPT"}
    assert seen == [expected]


@pytest.mark.L0
@pytest.mark.CPU
def test_non_qwen_policy_shaping_passes_conversation_images_to_the_hf_processor() -> None:
    """A generic HF VLM must score the same image that its rollout request carried."""
    import torch
    from PIL import Image as PILImage

    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    class _Tokenizer:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token_id = 2

    class _Processor:
        def __init__(self) -> None:
            self.images: list[object] | None = None

        def apply_chat_template(
            self,
            _messages: object,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            return "rendered prompt"

        def __call__(
            self,
            *,
            text: list[str],
            images: list[object],
            return_tensors: str,
        ) -> dict[str, torch.Tensor]:
            assert text == ["rendered prompt"]
            assert return_tensors == "pt"
            self.images = images
            return {
                "input_ids": torch.tensor([[10, 11]], dtype=torch.long),  # [B,L]
                "pixel_values": torch.ones((1, 3, 2, 2)),  # [N,C,H,W]
                "image_sizes": torch.tensor([[2, 2]], dtype=torch.long),  # [N,2]
            }

    processor = _Processor()
    packer = HFVLMDataPacker()
    packer.tokenizer = _Tokenizer()
    packer.hf_processor = processor
    packer.image_token = None
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": np.zeros((2, 2, 3), dtype=np.uint8)},
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]

    policy_input = packer.get_policy_input(
        ObsBundle(representations={"prompt": conversation}, modality_tags=("text", "image"), id="sample"),
        rollout_output=[12],
    )

    assert processor.images is not None
    assert len(processor.images) == 1
    assert isinstance(processor.images[0], PILImage.Image)
    assert policy_input.mm_kwargs is not None
    assert tuple(policy_input.mm_kwargs["pixel_values"].shape) == (1, 3, 2, 2)
    assert policy_input.mm_kwargs["image_sizes"].tolist() == [[2, 2]]


@pytest.mark.L0
@pytest.mark.CPU
def test_setup_builds_a_tokenizer(monkeypatch):
    """`setup` must call posttrain's base setup, which is what builds `self.tokenizer`.

    Regression: the port overrode `setup` without calling `super()`. Upstream's base only stored a
    config, so nothing looked wrong -- and every `self.tokenizer` use in the ported body (pad ids,
    collate padding, completion encoding) died only once a real run reached it. A GPU run found it;
    this pins it on CPU.
    """
    import cosmos_framework.posttrain.data.packer.hf_vlm as hv

    class _FakeImageProc:
        size = {"shortest_edge": 1, "longest_edge": 2}

    class _FakeProcessor:
        image_processor = _FakeImageProc()
        image_token = "<image>"

    class _FakeVision:
        model_type = "fake_vit"
        image_token_id = 7

    class _FakeCfg:
        model_type = "fake_vlm"
        vision_config = _FakeVision()
        image_token_id = 7

    # Patch the MODULE BINDINGS, not the AutoConfig/AutoProcessor classes. Patching the class
    # attribute is global and corrupts AutoTokenizer.from_pretrained's own config resolution --
    # the same trap test_unclassifiable_model_still_raises documents in test_data_packer_init.
    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(*a, **k):
            return _FakeProcessor()

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(*a, **k):
            return _FakeCfg()

    monkeypatch.setattr(hv, "AutoProcessor", _FakeAutoProcessor)
    monkeypatch.setattr(hv, "AutoConfig", _FakeAutoConfig)

    packer = hv.HFVLMDataPacker()
    packer.setup("gpt2")  # cached; the base builds a real tokenizer from it
    assert packer.tokenizer is not None, "setup must build self.tokenizer via super().setup()"
    assert packer.hf_processor is not None


@pytest.mark.L0
@pytest.mark.CPU
def test_obs_bundle_without_text_fails_loud():
    """A bundle with no conversation is a wiring error, not something to guess around."""
    from cosmos_framework.posttrain.data.packer.hf_vlm import _as_conversation

    with pytest.raises(ValueError, match=r"representations\['prompt'\]"):
        _as_conversation(ObsBundle(representations={"reference": "B"}, modality_tags=("text",), id="i"))
    # a bare conversation still passes through, so upstream-shaped callers keep working.
    # EQUAL, not identical: `_images_to_pil` rebuilds the content lists rather than mutating them,
    # so a cached dataset view handed to the packer is never modified in place.
    conv = [{"role": "user", "content": []}]
    assert _as_conversation(conv) == conv


@pytest.mark.L0
@pytest.mark.CPU
def test_collate_accepts_the_rl_policy_input_the_objective_builds():
    """The objective builds `RLPolicyInput` for BOTH packers so it keeps one code path.

    Regression: the ported collate indexed dicts (`x["input_ids"]`), so the real run died with
    "'RLPolicyInput' object is not subscriptable" at the first training step. Unit tests that call
    get_policy_input alone never reach the collate, which is why only a live run found it.
    """
    from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    items = [RLPolicyInput(input_ids=[1, 2, 3], logprob_masks=[0, 1, 0])]
    assert HFVLMDataPacker.policy_compute_max_len(HFVLMDataPacker(), items) == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_collate_preserves_image_processor_outputs() -> None:
    """Collation forwards the tensor fields emitted by an image processor.

    A real multimodal GRPO batch reached all eight policy ranks with ``pixel_values`` and
    ``image_grid_thw``. The generic packer must preserve both without assuming a fixed
    architecture-specific field set.
    """
    import torch

    from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    packer = HFVLMDataPacker()
    packer.tokenizer = SimpleNamespace(pad_token_id=0)
    items = [
        RLPolicyInput(
            input_ids=[1, 2, 3],
            logprob_masks=[0, 1, 0],
            mm_kwargs={
                "pixel_values": torch.ones((2, 4)),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
        )
    ]

    batch = packer.policy_collate_fn(items, computed_max_len=3)

    assert batch["pixel_values"].shape == (2, 4)
    assert batch["image_grid_thw"].tolist() == [[1, 1, 2]]


@pytest.mark.L0
@pytest.mark.CPU
def test_policy_collate_rejects_mixed_mm_token_type_id_schema() -> None:
    """A token-aligned processor field cannot silently lose one batch row."""
    import torch

    from cosmos_framework.posttrain.data.packer.decoder_only_llm import RLPolicyInput
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    packer = HFVLMDataPacker()
    packer.tokenizer = SimpleNamespace(pad_token_id=0)
    items = [
        RLPolicyInput(
            input_ids=[1, 2],
            logprob_masks=[0, 1],
            mm_kwargs={"mm_token_type_ids": torch.tensor([[0, 1]])},
        ),
        RLPolicyInput(input_ids=[1, 2], logprob_masks=[0, 1]),
    ]

    with pytest.raises(ValueError, match="present for every sample"):
        packer.policy_collate_fn(items, computed_max_len=2)


@pytest.mark.L0
@pytest.mark.CPU
def test_ndarray_images_are_converted_to_pil_for_the_vision_pipeline():
    """Rows use ndarrays for bulk transport; HF image processors receive PIL images."""
    from PIL import Image as PILImage

    from cosmos_framework.posttrain.data.packer.hf_vlm import _as_conversation

    arr = np.zeros((3, 4, 3), dtype=np.uint8)
    obs = ObsBundle(
        representations={
            "prompt": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": [{"type": "image", "image": arr}, {"type": "text", "text": "q"}]},
            ]
        },
        modality_tags=("text",),
        id="i",
    )
    conv = _as_conversation(obs)
    img = conv[1]["content"][0]["image"]
    assert isinstance(img, PILImage.Image), f"vision pipeline needs PIL, got {type(img).__name__}"
    assert img.size == (4, 3), "width x height must survive the conversion"
    assert conv[1]["content"][1] == {"type": "text", "text": "q"}, "text parts must pass through"
    # the caller's row must NOT be mutated -- it may be a cached dataset view
    assert isinstance(obs.representations["prompt"][1]["content"][0]["image"], np.ndarray)


@pytest.mark.L0
@pytest.mark.CPU
def test_shaping_does_not_mutate_the_caller_s_conversation():
    """The packer must normalize a MODEL VIEW, never rewrite the source observation.

    `get_rollout_input` shallow-copied the messages (`[dict(x) for x in sample]`) but then mutated
    the nested content parts in place, stripping the image token out of the caller's own text part.
    Rows can be cached views over memory-mapped Arrow, so that edit corrupts the dataset for every
    later reader -- and is invisible in a single-epoch run. Uses the REAL packer, not a fake.
    """
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    packer = HFVLMDataPacker()
    _bind_fake_processor(packer, rendered="PROMPT")
    packer.image_token = "<image>"

    original_text = "look <image> here"
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": np.zeros((4, 4, 3), dtype=np.uint8)},
                {"type": "text", "text": original_text},
            ],
        }
    ]
    obs = ObsBundle(representations={"prompt": conv}, modality_tags=("text",), id="i")
    packer.get_rollout_input(obs)

    assert conv[0]["content"][1]["text"] == original_text, (
        "the packer stripped the image token out of the CALLER's conversation; shaping must not "
        "write back into the source row"
    )
    assert isinstance(conv[0]["content"][0]["image"], np.ndarray), "the source image must stay an ndarray"


@pytest.mark.L0
@pytest.mark.CPU
def test_no_part_of_the_caller_s_conversation_is_aliased():
    """`_images_to_pil` must alias NOTHING back to the caller's row.

    The image and text parts were covered; the SYSTEM message was not -- its content is a `str`, so
    it takes an early branch. Copying every message keeps future shaping changes from writing into
    rows that may be cached views over memory-mapped Arrow.
    """
    from cosmos_framework.posttrain.data.packer.hf_vlm import _images_to_pil

    conv = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": [{"type": "image", "image": np.zeros((2, 2, 3), dtype=np.uint8)}]},
    ]
    out = _images_to_pil(conv)
    for i, (src, dst) in enumerate(zip(conv, out)):
        assert src is not dst, f"message {i} is aliased back to the caller's conversation"
    assert out[0] == conv[0], "a message with nothing to convert must still round-trip equal"


@pytest.mark.L0
@pytest.mark.CPU
def test_media_uuids_are_content_addressed_and_ordered():
    """The rollout's cross-rank identity check needs ids that agree without coordination.

    `roles/vllm_request.py` refuses image media that carries no `multi_modal_uuids`, because
    `_training_request_identity` compares WHICH media each rank sent and a bare tensor list cannot
    answer that. Content-addressing is what makes two ranks agree with no communication; a counter
    or `id()` would differ per process and fail the check it is supposed to feed.
    """
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    uuids = HFVLMDataPacker._image_content_uuids

    class _Img:
        def __init__(self, payload: bytes, size=(2, 2), mode="RGB"):
            self._payload, self.size, self.mode = payload, size, mode

        def tobytes(self):
            return self._payload

    a, b = _Img(b"AAA"), _Img(b"BBB")

    # Same content -> same id, computed independently. This is the cross-rank property.
    assert uuids([a]) == uuids([_Img(b"AAA")])
    # Different content -> different id.
    assert uuids([a]) != uuids([b])
    # ORDER matters: the same image twice must not collapse to one id, or a two-image request could
    # not be distinguished from a one-image one.
    twice = uuids([a, a])
    assert len(twice) == 2 and twice[0] != twice[1]
    # Reordering changes the ids, so a rank that shuffled its media is caught.
    assert uuids([a, b]) != uuids([b, a])


@pytest.mark.L0
@pytest.mark.CPU
def test_media_uuids_read_geometry_per_type_not_by_probing_size():
    """`.size` means three different things, and two of them break the digest.

    numpy's `.size` is the element COUNT, so `(1,2,3)` and `(2,1,3)` -- the same bytes, a different
    image -- would hash identically. torch's `.size` is a BOUND METHOD whose repr embeds the object
    address, so two equal tensors hash differently inside a single process, which is exactly
    backwards for an id whose whole job is to agree across ranks with no coordination. Only PIL's
    `.size` is the tuple the probe assumed.
    """
    import numpy as np
    import torch

    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    uuids = HFVLMDataPacker._image_content_uuids

    # numpy: same bytes, different geometry -> different ids.
    a = np.zeros((1, 2, 3), dtype=np.uint8)
    b = np.zeros((2, 1, 3), dtype=np.uint8)
    assert a.tobytes() == b.tobytes(), "the test is only meaningful while the bytes agree"
    assert uuids([a]) != uuids([b])

    # torch: two equal tensors -> the SAME id, in this process and in any other.
    t1, t2 = torch.zeros(2, 2), torch.zeros(2, 2)
    assert uuids([t1]) == uuids([t2])
    # ... and a different geometry with the same byte count still differs.
    assert uuids([torch.zeros(1, 4)]) != uuids([torch.zeros(2, 2)])


@pytest.mark.L0
@pytest.mark.CPU
def test_an_image_rollout_request_satisfies_the_vllm_identity_contract():
    """End to end: what the packer emits must pass the check the rollout actually runs.

    Pinned against the real `_training_request_identity` rather than a restatement of its rules --
    the packer emitting uuids and the rollout requiring them are two halves that were written
    apart, and only running one against the other proves they meet.
    """
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker
    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout

    class _Img:
        size, mode = (2, 2), "RGB"

        def tobytes(self):
            return b"PIXELS"

    images = [_Img()]
    request = {
        "prompt": "describe",
        "multi_modal_data": {"image": images},
        "multi_modal_uuids": {"image": HFVLMDataPacker._image_content_uuids(images)},
    }

    class _Obs:
        id = "sample-0"

    proof = VLLMRollout._training_request_identity([_Obs()], [request], {"max_tokens": 4}, 1)
    assert proof["requests"][0]["sample_id"] == "sample-0"
