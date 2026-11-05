# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `roles/teacher/vllm.py` — the held-engine teacher body, against a fake engine.

`VLLMTeacher` binds an already-built `vllm.LLM`, exactly as `VLLMRollout` does, so a stand-in with a
`generate` method exercises the whole body without vLLM, a GPU, or a model. The fake below encodes
the POSITION into the token ids it returns, which turns the row-alignment question — the one thing a
teacher can get wrong without anything raising — into a direct assertion.

The load-bearing tests pin scored-row alignment, atomic batch publication, and ranked-lane
convergence. Objective composition is tested independently in `test_algorithm_opd_objective.py`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import types
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.roles.teacher import TeacherContractError, VLLMTeacher
from cosmos_framework.posttrain.roles.teacher.base import TeacherRankStatus
from cosmos_framework.posttrain.roles.teacher.vllm import _ordered_request_digest
from cosmos_framework.posttrain.schema import ObsBundle, Status, TeacherSignal, Trajectory, Transition

TOP_K = 4


@pytest.fixture(autouse=True)
def _vllm_sampling_params(monkeypatch):
    """Let `score()` build its `SamplingParams` on an image with no vLLM.

    Every test here drives a FAKE engine, so vLLM is needed for exactly one line —
    `VLLMTeacher._sampling_params`'s `from vllm import SamplingParams`. Requiring the real package
    for that turned the row-alignment tests into `ModuleNotFoundError` on the CPU CI image, which is
    where they are most worth running: they are the guard against a teacher that is a token out of
    step, and that is a pure-Python question.

    Installs a stub ONLY when the real package is absent, and marks it, so
    `test_the_whole_batch_goes_in_one_engine_call` — the one test that asserts on the real
    `SamplingParams` contract rather than on alignment — can skip instead of asserting against the
    stand-in.
    """
    import sys

    if "vllm" in sys.modules:
        return
    try:
        # `find_spec` RAISES rather than returning None when an import hook rejects the name or a
        # parent package is missing, so "absent" has to cover both.
        present = importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        present = False
    if present:
        return

    class _SamplingParams:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    module = types.ModuleType("vllm")
    module.SamplingParams = _SamplingParams
    module.__opd_stub__ = True
    monkeypatch.setitem(sys.modules, "vllm", module)


class _Logprob:
    """Stands in for `vllm.sequence.Logprob` — the body reads `.logprob`."""

    def __init__(self, logprob: float, rank: int) -> None:
        self.logprob = logprob
        self.rank = rank
        self.decoded_token = "x"


class _Output:
    def __init__(self, prompt_logprobs: Any, prompt_token_ids: list[int] | None = None) -> None:
        self.prompt_logprobs = prompt_logprobs
        self.prompt_token_ids = prompt_token_ids


class _FakeEngine:
    """A `vllm.LLM` stand-in whose ids encode the position they describe.

    Position `i` returns ids `[i*100, i*100+1, ...]` with strictly decreasing logprobs, so
    `topk_ids[n][0] // 100` is exactly the full-sequence index row `n` came from."""

    def __init__(
        self,
        *,
        n_candidates: int = TOP_K,
        length_delta: int = 0,
        prompt_logprobs: bool = True,
        tokenizer_vocab: dict[str, int] | None = None,
        failures_before_success: int = 0,
    ) -> None:
        self.n_candidates = n_candidates
        self.length_delta = length_delta
        self.prompt_logprobs = prompt_logprobs
        self.failures_before_success = failures_before_success
        self.calls: list[dict] = []
        self.tokenizer = (
            None if tokenizer_vocab is None else types.SimpleNamespace(get_vocab=lambda: dict(tokenizer_vocab))
        )

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def generate(self, prompts: list[dict[str, list[int]]], sampling_params: Any) -> list[_Output]:
        self.calls.append({"prompts": prompts, "sampling_params": sampling_params})
        if self.failures_before_success:
            self.failures_before_success -= 1
            raise RuntimeError("synthetic retryable teacher failure")
        outputs = []
        for prompt in prompts:
            n_tokens = len(prompt["prompt_token_ids"]) + self.length_delta
            if not self.prompt_logprobs:
                outputs.append(_Output(None))
                continue
            entries = [None]  # nothing predicts the first token
            for position in range(1, n_tokens):
                entries.append(
                    {position * 100 + rank: _Logprob(-2.0 - rank, rank + 1) for rank in range(self.n_candidates)}
                )
            outputs.append(_Output(entries))
        return outputs


class _CharTokenizer:
    """Offset-preserving character codec with a deliberately different ID space per instance."""

    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None
    additional_special_tokens_ids: list[int] = []
    all_special_ids: list[int] = []
    chat_template: str = "fixture-chat-template-v1"

    def __init__(
        self,
        token_offset: int,
        *,
        supports_offsets: bool = True,
        mismatch_fallback_prefix: bool = False,
        disabled_think_prefix: bool = False,
    ) -> None:
        self.token_offset = token_offset
        self.supports_offsets = supports_offsets
        self.mismatch_fallback_prefix = mismatch_fallback_prefix
        self.disabled_think_prefix = disabled_think_prefix
        self.name_or_path = f"fixture-char-{token_offset}"
        self.tokenize_calls: list[tuple[str, bool]] = []
        self.render_calls: list[dict[str, Any]] = []
        self._fallback_calls = 0

    def get_vocab(self) -> dict[str, int]:
        return {"fixture-artifact": self.token_offset}

    def encode_text(self, text: str) -> list[int]:
        return [self.token_offset + ord(character) for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool = False,
    ) -> Any:
        assert add_special_tokens is False
        self.tokenize_calls.append((text, return_offsets_mapping))
        if return_offsets_mapping and not self.supports_offsets:
            raise NotImplementedError("fixture has no offsets")
        input_ids = self.encode_text(text)
        if not return_offsets_mapping:
            self._fallback_calls += 1
            if self.mismatch_fallback_prefix and self._fallback_calls == 1:
                input_ids = [input_ids[0] + 1, *input_ids[1:]]
        offsets = [(index, index + 1) for index in range(len(text))]
        return types.SimpleNamespace(
            input_ids=input_ids,
            offset_mapping=offsets if return_offsets_mapping else None,
        )

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id - self.token_offset) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        self.render_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
            }
        )
        rendered: list[str] = []
        for message in messages:
            content = message["content"]
            text = content if isinstance(content, str) else "".join(block["text"] for block in content)
            rendered.append(f"<{message['role']}>{text}</{message['role']}>")
        if add_generation_prompt:
            rendered.append("<assistant>")
            if not enable_thinking and self.disabled_think_prefix:
                rendered.append("<think>\n\n</think>\n\n")
        return "".join(rendered)


class _RealizedTokenEngine:
    """DPCA engine fixture returning only each forced prompt token's own logprob."""

    def __init__(self, tokenizer: _CharTokenizer, *, failures_before_success: int = 0) -> None:
        self.tokenizer = tokenizer
        self.failures_before_success = failures_before_success
        self.calls: list[dict[str, Any]] = []

    def get_tokenizer(self) -> _CharTokenizer:
        return self.tokenizer

    def generate(self, prompts: list[dict[str, list[int]]], sampling_params: Any) -> list[_Output]:
        self.calls.append({"prompts": prompts, "sampling_params": sampling_params})
        if self.failures_before_success:
            self.failures_before_success -= 1
            raise RuntimeError("synthetic DPCA retry")
        outputs: list[_Output] = []
        for prompt in prompts:
            token_ids = prompt["prompt_token_ids"]
            prompt_logprobs: list[Any] = [None]
            for position, token_id in enumerate(token_ids[1:], start=1):
                prompt_logprobs.append({token_id: _Logprob(-0.25 - 0.01 * position, 1)})
            outputs.append(_Output(prompt_logprobs))
        return outputs


class _ImageTeacherPacker:
    """Teacher-artifact packer fixture preserving ordered images and template semantics."""

    _VISION_START_MARKER = "\ue001"
    _IMAGE_MARKER = "\ue000"
    _VISION_END_MARKER = "\ue002"

    def __init__(self, tokenizer: _CharTokenizer) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def _messages(obs: ObsBundle) -> list[dict[str, Any]]:
        messages = obs.representations.get("messages")
        if not isinstance(messages, list):
            raise ValueError("fixture multimodal teacher requires messages")
        return messages

    def render_cross_token_multimodal_prompt(self, item: ObsBundle, *, enable_thinking: bool) -> str:
        rendered: list[str] = []
        for message in self._messages(item):
            content = message["content"]
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            text: list[str] = []
            for block in blocks:
                block_type = block.get("type")
                if block_type in {"image", "image_url"}:
                    text.append(self._VISION_START_MARKER + self._IMAGE_MARKER + self._VISION_END_MARKER)
                elif block_type in {"video", "video_frames"}:
                    raise ValueError(
                        "cross-token multimodal teacher supports ordered images only; video is unsupported"
                    )
                elif block_type == "text":
                    text.append(block["text"])
                else:
                    raise ValueError(f"unsupported fixture block {block_type!r}")
            rendered.append(f"<{message['role']}>{''.join(text)}</{message['role']}>")
        rendered.append("<assistant>")
        return "".join(rendered)

    def get_cross_token_multimodal_teacher_input(self, item: ObsBundle, token_ids: list[int]) -> Any:
        images: list[Any] = []
        for message in self._messages(item):
            content = message["content"]
            blocks = [] if isinstance(content, str) else content
            for block in blocks:
                block_type = block.get("type")
                if block_type in {"image", "image_url"}:
                    images.append(block.get("image", block.get("image_url")))
                elif block_type in {"video", "video_frames"}:
                    raise ValueError(
                        "cross-token multimodal teacher supports ordered images only; video is unsupported"
                    )
        request: dict[str, Any] = {"prompt_token_ids": list(token_ids)}
        mm_processor_kwargs: dict[str, Any] = {}
        image_uuids: list[str] = []
        for index, image in enumerate(images):
            content_sha256 = hashlib.sha256(str(image).encode("utf-8")).hexdigest()
            cache_seed = f"qwen3-vl-image-v1\0{index}\0{content_sha256}".encode("utf-8")
            image_uuids.append(hashlib.sha256(cache_seed).hexdigest())
        if images:
            request["multi_modal_data"] = {"image": images}
            mm_processor_kwargs = {"max_pixels": 4096}
            request["mm_processor_kwargs"] = mm_processor_kwargs
            request["multi_modal_uuids"] = {"image": image_uuids}
        return types.SimpleNamespace(
            request=request,
            image_count=len(images),
            image_token_id=self.tokenizer.token_offset + ord(self._IMAGE_MARKER),
            vision_start_token_id=self.tokenizer.token_offset + ord(self._VISION_START_MARKER),
            vision_end_token_id=self.tokenizer.token_offset + ord(self._VISION_END_MARKER),
            expected_image_marker_widths=(81,) * len(images),
            image_uuids=tuple(image_uuids),
        )


class _ImageExpandedRealizedTokenEngine(_RealizedTokenEngine):
    """Simulate vLLM expanding each image placeholder inside the prompt prefix."""

    def __init__(self, tokenizer: _CharTokenizer) -> None:
        super().__init__(tokenizer)
        self.image_token_id = tokenizer.token_offset + ord(_ImageTeacherPacker._IMAGE_MARKER)
        self.image_marker_width = 81

    def generate(self, prompts: list[dict[str, Any]], sampling_params: Any) -> list[_Output]:
        self.calls.append({"prompts": prompts, "sampling_params": sampling_params})
        outputs: list[_Output] = []
        for prompt in prompts:
            submitted_ids = list(prompt["prompt_token_ids"])
            expanded_ids: list[int] = []
            for token_id in submitted_ids:
                expanded_ids.append(token_id)
                if token_id == self.image_token_id:
                    expanded_ids.extend([token_id] * (self.image_marker_width - 1))
            prompt_logprobs: list[Any] = [None]
            for position, token_id in enumerate(expanded_ids[1:], start=1):
                prompt_logprobs.append({token_id: _Logprob(-0.25 - 0.01 * position, 1)})
            outputs.append(_Output(prompt_logprobs, prompt_token_ids=expanded_ids))
        return outputs


class _ImageExpandedTopKEngine(_FakeEngine):
    """Top-k analogue of `_ImageExpandedRealizedTokenEngine`.

    vLLM re-applies the Qwen3-VL placeholder replacement to token prompts: the i-th `<|image_pad|>`
    match becomes the i-th image's full run while every other pad stays. Support ids still encode the
    (expanded) position they describe, and `prompt_token_ids` reports the expanded prompt."""

    def __init__(
        self,
        *,
        image_token_id: int,
        image_marker_width: int,
        tokenizer_vocab: dict[str, int] | None = None,
    ) -> None:
        super().__init__(tokenizer_vocab=tokenizer_vocab)
        self.image_token_id = image_token_id
        self.image_marker_width = image_marker_width

    def generate(self, prompts: list[dict[str, Any]], sampling_params: Any) -> list[_Output]:
        self.calls.append({"prompts": prompts, "sampling_params": sampling_params})
        outputs: list[_Output] = []
        for prompt in prompts:
            submitted_ids = list(prompt["prompt_token_ids"])
            image_count = len(prompt.get("multi_modal_data", {}).get("image", []))
            matches = [index for index, token_id in enumerate(submitted_ids) if token_id == self.image_token_id]
            expanded_at = set(matches[:image_count])
            expanded_ids: list[int] = []
            for index, token_id in enumerate(submitted_ids):
                expanded_ids.extend([token_id] * (self.image_marker_width if index in expanded_at else 1))
            entries: list[Any] = [None]
            for position in range(1, len(expanded_ids)):
                entries.append(
                    {position * 100 + rank: _Logprob(-2.0 - rank, rank + 1) for rank in range(self.n_candidates)}
                )
            outputs.append(_Output(entries, prompt_token_ids=expanded_ids))
        return outputs


class _SameTokenImagePacker:
    """Student-packer fixture for same-token OPD.

    Attaches the image and, unless asked to reproduce the old behaviour, collapses each image
    placeholder run of the prompt to one marker exactly as `Qwen3VLDataPacker` does."""

    def __init__(
        self,
        *,
        tokenizer_vocab: dict[str, int],
        image_token_id: int,
        prompt_token_count: int,
        collapse: bool,
    ) -> None:
        self.tokenizer = types.SimpleNamespace(get_vocab=lambda: dict(tokenizer_vocab), chat_template=None)
        self.image_token_id = image_token_id
        self.prompt_token_count = prompt_token_count
        self.collapse = collapse

    def get_pretokenized_rollout_input(self, _obs: ObsBundle, token_ids: list[int]) -> dict[str, Any]:
        prompt_ids = list(token_ids[: self.prompt_token_count])
        if self.collapse:
            prompt_ids = [
                token_id
                for index, token_id in enumerate(prompt_ids)
                if token_id != self.image_token_id or index == 0 or prompt_ids[index - 1] != self.image_token_id
            ]
        return {
            "prompt_token_ids": [*prompt_ids, *token_ids[self.prompt_token_count :]],
            "multi_modal_data": {"image": ["image"]},
            "multi_modal_uuids": {"image": ["image-content-v1"]},
            "mm_processor_kwargs": {"max_pixels": 4096},
        }


class _OpaqueBulk:
    """A media/action stand-in that raises if a metadata-only teacher tries to inspect it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"metadata-only teacher touched opaque bulk attribute {name!r}")

    def __iter__(self) -> Any:
        raise AssertionError("metadata-only teacher iterated opaque bulk")

    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        del memo
        raise AssertionError("metadata-only teacher copied opaque bulk")


def _traj(
    prompt_ids: list[int],
    completion_ids: list[int],
    n_ignore: int = 0,
    *,
    trajectory_id: str = "g0_k0",
) -> Trajectory:
    """A trajectory carrying stored `prompt_ids`, so shaping needs no tokenizer (`packer=None`)."""
    transition = Transition(
        obs=ObsBundle(representations={"prompt": "unused — stored ids win"}, id="p0"),
        action=torch.tensor(completion_ids, dtype=torch.long),  # [T]
        policy_info={
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),  # [P]
            "completion_ids": list(completion_ids),
            "completion_boundary": {
                "prompt_token_count": len(prompt_ids),
                "completion_token_count": len(completion_ids),
                "full_sequence_token_count": len(prompt_ids) + len(completion_ids),
            },
        },
        algo_extra={"n_ignore_prefix_tokens": n_ignore} if n_ignore else {},
        done=True,
    )
    return Trajectory(transitions=[transition], id=trajectory_id, group_id="g0", status=Status.COMPLETED)


def _dpca_traj(
    student_tokenizer: _CharTokenizer,
    completion_text: str,
    *,
    trajectory_id: str = "g0_k0",
    media: Any | None = None,
    action: Any | None = None,
) -> Trajectory:
    media = _OpaqueBulk() if media is None else media
    action = _OpaqueBulk() if action is None else action
    completion_ids = student_tokenizer.encode_text(completion_text)
    obs = ObsBundle(
        representations={
            "messages": [
                {"role": "system", "content": "Keep every byte."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": media},
                        {"type": "text", "text": "What is shown?"},
                        {"type": "video", "video": media},
                        {"type": "text", "text": " Choose exactly."},
                    ],
                },
            ],
            "images": [media],
            "videos": [media],
        },
        modality_tags=("text", "image", "video"),
        id=f"prompt-{trajectory_id}",
    )
    transition = Transition(
        obs=obs,
        action=action,
        policy_info={
            "completion_ids": completion_ids,
            "completion_text": completion_text,
            "rollout_logprobs": [-0.7 - 0.01 * index for index in range(len(completion_ids))],
        },
        done=True,
    )
    return Trajectory(
        transitions=[transition],
        id=trajectory_id,
        group_id=trajectory_id.split("_", maxsplit=1)[0],
        status=Status.COMPLETED,
    )


def _dpca_image_traj(
    student_tokenizer: _CharTokenizer,
    completion_text: str,
    *,
    images: list[str],
    trajectory_id: str = "g0_k0",
) -> Trajectory:
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
    completion_ids = student_tokenizer.encode_text(completion_text)
    transition = Transition(
        obs=ObsBundle(
            representations={"messages": [{"role": "user", "content": content}]},
            modality_tags=("text", "image") if images else ("text",),
            id=f"prompt-{trajectory_id}",
        ),
        action=_OpaqueBulk(),
        policy_info={
            "completion_ids": completion_ids,
            "completion_text": completion_text,
            "rollout_logprobs": [-0.7 - 0.01 * index for index in range(len(completion_ids))],
        },
        done=True,
    )
    return Trajectory(
        transitions=[transition],
        id=trajectory_id,
        group_id=trajectory_id.split("_", maxsplit=1)[0],
        status=Status.COMPLETED,
    )


def _dpca_teacher(
    student_tokenizer: _CharTokenizer,
    teacher_tokenizer: _CharTokenizer,
    *,
    engine: _RealizedTokenEngine | None = None,
    max_attempts: int = 2,
) -> tuple[VLLMTeacher, _RealizedTokenEngine]:
    bound_engine = engine or _RealizedTokenEngine(teacher_tokenizer)
    teacher = VLLMTeacher(
        bound_engine,
        packer=types.SimpleNamespace(tokenizer=student_tokenizer),
        top_k=0,
        input_mode="cross_token_text",
        signal_mode="dpca",
        max_attempts=max_attempts,
        conditional_think_prefix="<think>\n\n</think>\n\n",
        max_alignment_chunk_tokens=64,
    )
    return teacher, bound_engine


def _dpca_multimodal_teacher(
    student_tokenizer: _CharTokenizer,
    teacher_tokenizer: _CharTokenizer,
) -> tuple[VLLMTeacher, _ImageExpandedRealizedTokenEngine]:
    engine = _ImageExpandedRealizedTokenEngine(teacher_tokenizer)
    teacher = VLLMTeacher(
        engine,
        packer=types.SimpleNamespace(tokenizer=student_tokenizer),
        teacher_packer=_ImageTeacherPacker(teacher_tokenizer),
        top_k=0,
        input_mode="cross_token_multimodal",
        signal_mode="dpca",
        conditional_think_prefix="<think>\n\n</think>\n\n",
        max_alignment_chunk_tokens=64,
    )
    return teacher, engine


@pytest.mark.L0
@pytest.mark.CPU
def test_signal_rows_line_up_with_the_scored_positions():
    """Row `n` must be the distribution over the `n`-th SCORED token, not the `n`-th input token.

    The ids encode their position, so this reads the mapping off directly. An implementation that
    forgot the next-token shift, or started at the prompt, or scored the whole sequence, produces
    different numbers here — none of which would raise anywhere."""
    prompt, completion = [11, 12, 13], [21, 22]
    trajs = [_traj(prompt, completion)]

    teacher = VLLMTeacher(_FakeEngine(), top_k=TOP_K)
    teacher.score(trajs)

    signal = trajs[0].transitions[0].targets
    assert isinstance(signal, TeacherSignal)
    assert signal.topk_ids.shape == (len(completion), TOP_K)
    assert [int(row[0]) // 100 for row in signal.topk_ids] == [3, 4]  # the completion's own indices
    assert torch.equal(signal.topk_logprobs[0], torch.tensor([-2.0, -3.0, -4.0, -5.0]))  # [K], ranked
    assert teacher.batch_stats() == {
        "train/opd_teacher_score_request_count": 1,
        "train/opd_teacher_score_success_count": 1,
        "train/opd_teacher_score_failure_count": 0,
        "train/opd_teacher_scored_sequences": 1,
        "train/opd_teacher_scored_tokens": 2,
        "train/opd_topk_support_mass_mean": pytest.approx(sum(math.exp(value) for value in (-2, -3, -4, -5))),
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_same_token_stored_ids_shape_media_request_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored rollout IDs define the span; only request attachment may invoke the packer."""
    vocab = {f"token-{token_id}": token_id for token_id in range(512)}
    tokenizer = types.SimpleNamespace(get_vocab=lambda: dict(vocab), chat_template=None)

    class _CountingPacker:
        def __init__(self) -> None:
            self.tokenizer = tokenizer
            self.policy_calls = 0
            self.request_calls = 0

        def get_policy_input(self, *_args: Any, **_kwargs: Any) -> Any:
            self.policy_calls += 1
            raise AssertionError("stored teacher span must not run policy/media shaping")

        def get_pretokenized_rollout_input(self, _obs: ObsBundle, token_ids: list[int]) -> dict[str, Any]:
            self.request_calls += 1
            return {
                "prompt_token_ids": list(token_ids),
                "multi_modal_data": {"image": ["image"]},
                "multi_modal_uuids": {"image": ["image-content-v1"]},
                "mm_processor_kwargs": {"max_pixels": 4096},
            }

    packer = _CountingPacker()
    engine = _FakeEngine(tokenizer_vocab=vocab)
    captured: list[dict[str, Any]] = []

    def _capture(identity: dict[str, Any]) -> str:
        captured.append(identity)
        return _ordered_request_digest(identity)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.teacher.vllm._ordered_request_digest",
        _capture,
    )

    VLLMTeacher(engine, packer=packer, top_k=TOP_K).score([_traj([11, 12], [21])])

    assert packer.policy_calls == 0
    assert packer.request_calls == 1
    request_identity = captured[0]["trajectories"][0]
    assert request_identity["media_cardinality"] == {"image": 1}
    assert request_identity["multi_modal_uuids"] == {"image": ["image-content-v1"]}
    assert request_identity["mm_processor_kwargs"] == {"max_pixels": 4096}


@pytest.mark.L0
@pytest.mark.CPU
def test_same_token_opd_hf_vlm_teacher_request_retains_the_original_image() -> None:
    """HF VLM OPD must attach media without changing or re-rendering the realized token sequence."""
    from cosmos_framework.posttrain.data.packer.hf_vlm import HFVLMDataPacker

    vocab = {f"token-{token_id}": token_id for token_id in range(32)}
    tokenizer = types.SimpleNamespace(get_vocab=lambda: dict(vocab), chat_template=None)
    packer = HFVLMDataPacker()
    packer.tokenizer = tokenizer
    packer.image_token = None
    image = "image://original"
    transition = Transition(
        obs=ObsBundle(
            representations={
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "What is shown?"},
                        ],
                    }
                ]
            },
            modality_tags=("text", "image"),
            id="image-prompt",
        ),
        action=torch.tensor([21], dtype=torch.long),  # [T]
        done=True,
    )
    exact_ids = [11, 12, 21]
    teacher = VLLMTeacher(
        _FakeEngine(tokenizer_vocab=vocab),
        packer=packer,
        top_k=TOP_K,
    )

    request = teacher._shared_vocab_request(transition, exact_ids, prompt_token_count=2)

    assert request["prompt_token_ids"] == exact_ids
    assert "prompt" not in request
    assert request["multi_modal_data"] == {"image": [image]}
    assert len(request["multi_modal_uuids"]["image"]) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_same_token_topk_submits_collapsed_image_markers_and_proves_exact_expansion() -> None:
    """The request carries one `<|image_pad|>` per image; the engine expands it back to the student's
    ids, and the signal rows line up with the student's completion positions.

    Uses the incident's numbers: one 448x448 image is 196 pads and the sequence is 460 tokens. Sending
    the already-expanded prompt (the old behaviour) makes vLLM expand the first pad again and return
    655 = 460 - 1 + 196 entries, which the parser must refuse by name instead of misaligning rows."""
    image_token_id, image_marker_width = 91, 196
    prompt = [*range(100, 133), 90, *([image_token_id] * image_marker_width), 92, *range(133, 166)]  # [264]
    completion = list(range(200, 396))  # [196]
    assert len(prompt) + len(completion) == 460
    vocab = {f"token-{token_id}": token_id for token_id in range(512)}

    def _teacher(*, collapse: bool) -> tuple[VLLMTeacher, _ImageExpandedTopKEngine]:
        engine = _ImageExpandedTopKEngine(
            image_token_id=image_token_id,
            image_marker_width=image_marker_width,
            tokenizer_vocab=vocab,
        )
        packer = _SameTokenImagePacker(
            tokenizer_vocab=vocab,
            image_token_id=image_token_id,
            prompt_token_count=len(prompt),
            collapse=collapse,
        )
        # Position-encoded support ids reach 459 * 100 + 3, beyond the 512-token fixture vocabulary.
        teacher = VLLMTeacher(engine, packer=packer, top_k=TOP_K, max_attempts=1, vocab_size=2**16)
        return teacher, engine

    trajs = [_traj(prompt, completion)]
    teacher, engine = _teacher(collapse=True)
    teacher.score(trajs)

    assert len(engine.calls) == 1
    request = engine.calls[0]["prompts"][0]
    assert request["prompt_token_ids"] == [*range(100, 133), 90, image_token_id, 92, *range(133, 166), *completion]
    assert request["multi_modal_data"] == {"image": ["image"]}
    signal = trajs[0].transitions[0].targets
    assert isinstance(signal, TeacherSignal)
    assert signal.topk_ids.shape == (len(completion), TOP_K)
    assert [int(row[0]) // 100 for row in signal.topk_ids] == list(range(len(prompt), 460))

    stale = [_traj(prompt, completion)]
    teacher, engine = _teacher(collapse=False)
    with pytest.raises(
        TeacherContractError,
        match=r"engine prompt has 655 tokens but the student sequence has 460 .* first mismatch at index 230",
    ):
        teacher.score(stale)
    assert len(engine.calls) == 1
    assert len(engine.calls[0]["prompts"][0]["prompt_token_ids"]) == 460
    assert stale[0].transitions[0].targets is None


@pytest.mark.L0
@pytest.mark.CPU
def test_same_token_request_hook_may_only_collapse_placeholder_runs() -> None:
    """Pre-engine invariant: submitted = the student's prompt with only placeholder repeats dropped,
    then the completion verbatim. A hook that re-tokenizes the prompt or touches the completion is
    refused before the engine is called; the untouched ids of a decoder-only packer still pass."""
    assert VLLMTeacher._is_placeholder_collapse([1, 90, 91, 91, 91, 92, 2], [1, 90, 91, 92, 2])
    assert VLLMTeacher._is_placeholder_collapse([1, 2, 3], [1, 2, 3])
    assert VLLMTeacher._is_placeholder_collapse([5, 5, 5], [5])
    assert not VLLMTeacher._is_placeholder_collapse([1, 2, 3], [1, 3])  # dropped a non-repeat
    assert not VLLMTeacher._is_placeholder_collapse([1, 2], [1, 2, 2])  # inserted
    assert not VLLMTeacher._is_placeholder_collapse([1, 91, 91], [1, 7])  # re-tokenized
    assert not VLLMTeacher._is_placeholder_collapse([1, 91, 91], [])

    vocab = {f"token-{token_id}": token_id for token_id in range(512)}
    transition = _traj([1, 90, 91, 91, 92], [21, 22]).transitions[0]
    student_ids = [1, 90, 91, 91, 92, 21, 22]

    def _request(hook_ids: list[int]) -> dict[str, Any]:
        packer = types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(get_vocab=lambda: dict(vocab), chat_template=None),
            get_pretokenized_rollout_input=lambda _obs, _ids: {"prompt_token_ids": list(hook_ids)},
        )
        teacher = VLLMTeacher(_FakeEngine(tokenizer_vocab=vocab), packer=packer, top_k=TOP_K)
        return teacher._shared_vocab_request(transition, student_ids, prompt_token_count=5)

    assert _request([1, 90, 91, 92, 21, 22])["prompt_token_ids"] == [1, 90, 91, 92, 21, 22]
    assert _request(student_ids)["prompt_token_ids"] == student_ids
    for hook_ids in (
        [1, 90, 92, 21, 22],  # dropped the whole image run
        [7, 90, 91, 92, 21, 22],  # re-tokenized prompt
        [1, 90, 91, 92, 21],  # truncated completion
        [1, 90, 91, 92, 21, 23],  # changed completion
        [21, 22],  # no prompt at all
    ):
        with pytest.raises(TeacherContractError, match="never re-rendering or re-tokenizing"):
            _request(hook_ids)


@pytest.mark.L0
@pytest.mark.CPU
def test_the_whole_batch_goes_in_one_engine_call():
    """Batching is the reason this body holds the engine instead of calling a service.

    Also pins the scoring `SamplingParams`: `prompt_logprobs=top_k` makes vLLM report a distribution
    at every position, while `max_tokens=1` bounds the dummy continuation used for prompt scoring.

    The one test here that needs the REAL vLLM: asserting these three fields against the stub the
    autouse fixture installs would assert that the stub stores what it was given."""
    vllm = pytest.importorskip("vllm")
    if getattr(vllm, "__opd_stub__", False):
        pytest.skip("no vLLM on this image; the SamplingParams contract needs the real package")
    engine = _FakeEngine()
    trajs = [
        _traj([11, 12], [21], trajectory_id="g0_k0"),
        _traj([31, 32, 33], [41, 42], trajectory_id="g1_k0"),
    ]

    VLLMTeacher(engine, top_k=TOP_K).score(trajs)

    assert len(engine.calls) == 1
    assert [p["prompt_token_ids"] for p in engine.calls[0]["prompts"]] == [[11, 12, 21], [31, 32, 33, 41, 42]]
    params = engine.calls[0]["sampling_params"]
    assert params.prompt_logprobs == TOP_K
    assert params.max_tokens == 1
    assert params.temperature == 1.0
    assert params.top_p == 1.0 and params.top_k == -1
    assert all(t.transitions[0].targets is not None for t in trajs)


@pytest.mark.L0
@pytest.mark.CPU
def test_n_ignore_prefix_tokens_shifts_the_span():
    """The shared-prefix filter shortens the scored span; the signal must shorten with it.

    This is the case `_scored_signal` refuses to guess at — a signal built for the full completion
    would pair every position with its neighbour's teacher."""
    trajs = [_traj([11, 12, 13], [21, 22, 23], n_ignore=1)]

    VLLMTeacher(_FakeEngine(), top_k=TOP_K).score(trajs)

    signal = trajs[0].transitions[0].targets
    assert signal.topk_ids.shape == (2, TOP_K)
    assert [int(row[0]) // 100 for row in signal.topk_ids] == [4, 5]  # first completion token skipped


@pytest.mark.L0
@pytest.mark.CPU
def test_too_few_candidates_names_the_cap():
    """The teacher must name ``max_logprobs`` when the engine returns less than configured ``top_k``."""
    with pytest.raises(TeacherContractError, match="max_logprobs"):
        VLLMTeacher(_FakeEngine(n_candidates=TOP_K - 1), top_k=TOP_K).score([_traj([11, 12], [21])])


@pytest.mark.L0
@pytest.mark.CPU
def test_malformed_engine_and_tokenizer_contracts_are_refused() -> None:
    """Engine responses and shared-vocabulary identity both fail loud instead of misaligning rows."""
    with pytest.raises(TeacherContractError, match="no prompt_logprobs"):
        VLLMTeacher(_FakeEngine(prompt_logprobs=False), top_k=TOP_K).score([_traj([11, 12], [21])])

    with pytest.raises(TeacherContractError, match="cannot be aligned"):
        VLLMTeacher(_FakeEngine(length_delta=-1), top_k=TOP_K).score([_traj([11, 12], [21])])

    with pytest.raises(ValueError, match="requires top_k > 0"):
        VLLMTeacher(_FakeEngine(), top_k=0)
    with pytest.raises(ValueError, match="top_k must be a non-bool integer"):
        VLLMTeacher(_FakeEngine(), top_k=True)
    with pytest.raises(ValueError, match="forced-sequence teacher reads raw"):
        VLLMTeacher(_FakeEngine(), top_k=TOP_K, scoring_temperature=0.5)

    student_vocab = {"alpha": 0, "beta": 1}
    packer = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(get_vocab=lambda: dict(student_vocab)),
    )
    VLLMTeacher(_FakeEngine(tokenizer_vocab=student_vocab), packer=packer, top_k=TOP_K)

    # Equal size is not enough: a token-id permutation scores the wrong student logits silently.
    with pytest.raises(TeacherContractError, match=r"identical.*get_vocab\(\).*remapped"):
        VLLMTeacher(_FakeEngine(tokenizer_vocab={"alpha": 1, "beta": 0}), packer=packer, top_k=TOP_K)

    with pytest.raises(TeacherContractError, match="does not expose get_tokenizer"):
        VLLMTeacher(object(), packer=packer, top_k=TOP_K)
    with pytest.raises(TeacherContractError, match=r"student packer.*get_vocab\(\)"):
        VLLMTeacher(
            _FakeEngine(tokenizer_vocab=student_vocab),
            packer=types.SimpleNamespace(tokenizer=object()),
            top_k=TOP_K,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_rank_digest_mismatch_fails_before_any_engine_lane_enters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank-local shaping drift converges before the collective-bearing vLLM call."""
    engine = _FakeEngine()

    def _mismatched(digest: str, *, context: str) -> dict[int, str]:
        del context
        return {0: digest, 1: "different-rank-digest"}

    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.teacher.vllm.gather_rank_states",
        _mismatched,
    )
    with pytest.raises(TeacherContractError, match="differs across teacher TP ranks"):
        VLLMTeacher(engine, top_k=TOP_K).score([_traj([11, 12], [21])])
    assert engine.calls == []


@pytest.mark.L0
@pytest.mark.CPU
def test_nonleader_returns_status_without_committing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cosmos_framework.posttrain.roles.teacher.vllm._rank_world", lambda: (1, 2))
    trajectory = _traj([11, 12], [21])

    result = VLLMTeacher(_FakeEngine(), top_k=TOP_K).score([trajectory])

    assert isinstance(result, TeacherRankStatus)
    assert (result.rank, result.world_size, result.scored_sequences, result.scored_tokens) == (1, 2, 1, 1)
    assert trajectory.transitions[0].targets is None
    assert not any(isinstance(value, torch.Tensor) for value in vars(result).values())


@pytest.mark.L0
@pytest.mark.CPU
def test_support_rejects_duplicate_nonfinite_out_of_range_and_overfull_rows() -> None:
    """Every top-k row is finite, unique, in-range, and a valid probability sub-mass."""

    class _DuplicateSupport(dict):
        def items(self) -> list[tuple[int, _Logprob]]:
            return [(1, _Logprob(-2.0, 1)), (1, _Logprob(-3.0, 2))]

    teacher = VLLMTeacher(_FakeEngine(), top_k=2, vocab_size=5)
    with pytest.raises(TeacherContractError, match="duplicate support id"):
        teacher._support(_DuplicateSupport({1: _Logprob(-2.0, 1)}), 1)
    with pytest.raises(TeacherContractError, match="not finite/in range"):
        teacher._support({1: _Logprob(float("nan"), 1), 2: _Logprob(-2.0, 2)}, 1)
    with pytest.raises(TeacherContractError, match=r"outside \[0, 5\)"):
        teacher._support({1: _Logprob(-2.0, 1), 5: _Logprob(-3.0, 2)}, 1)
    with pytest.raises(TeacherContractError, match="support mass"):
        teacher._support({1: _Logprob(-0.1, 1), 2: _Logprob(-0.1, 2)}, 1)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("disabled_think_prefix", [False, True])
@pytest.mark.parametrize(
    ("completion_text", "expected_prefix"),
    [
        ("plain answer", "<think>\n\n</think>\n\n"),
        ("<think>reason</think>\nanswer", ""),
        ("reason already in flight</think>\nanswer", "<think>\n\n"),
    ],
)
def test_dpca_think_boundaries_are_normalized_exactly_once(
    disabled_think_prefix: bool,
    completion_text: str,
    expected_prefix: str,
) -> None:
    """The three settled think cases add exactly one context after the assistant header."""
    student_tokenizer = _CharTokenizer(1_000)
    teacher_tokenizer = _CharTokenizer(10_000, disabled_think_prefix=disabled_think_prefix)
    teacher, engine = _dpca_teacher(student_tokenizer, teacher_tokenizer)
    media = _OpaqueBulk()
    action = _OpaqueBulk()
    trajectory = _dpca_traj(
        student_tokenizer,
        completion_text,
        media=media,
        action=action,
    )
    original_messages = trajectory.transitions[0].obs.representations["messages"]

    result = teacher.score([trajectory])

    assert result == [trajectory]
    assert trajectory.transitions[0].action is action
    assert trajectory.transitions[0].obs.representations["messages"] is original_messages
    render = teacher_tokenizer.render_calls[-1]
    assert render["add_generation_prompt"] is True
    assert render["enable_thinking"] is False
    assert render["messages"] == [
        {"role": "system", "content": "Keep every byte."},
        {"role": "user", "content": "What is shown? Choose exactly."},
    ]
    joint_ids = engine.calls[0]["prompts"][0]["prompt_token_ids"]
    rendered_joint = teacher_tokenizer.decode(
        joint_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    rendered_prompt = "<system>Keep every byte.</system><user>What is shown? Choose exactly.</user><assistant>"
    assert rendered_joint == rendered_prompt + expected_prefix + completion_text
    assert rendered_joint.count("<think>\n\n</think>\n\n") <= 1
    assert engine.calls[0]["sampling_params"].prompt_logprobs == 0
    assert teacher_tokenizer.tokenize_calls == [(rendered_joint, True)]
    signal = trajectory.transitions[0].targets
    assert isinstance(signal, TeacherSignal)
    assert signal.topk_ids is None and signal.topk_logprobs is None
    assert signal.chunk_bounds.shape == (len(completion_text), 2)
    assert torch.count_nonzero(signal.chunk_teacher_logprob).item() == len(completion_text)


@pytest.mark.L0
@pytest.mark.CPU
def test_duplicate_template_think_prefills_fail_closed() -> None:
    prefix = "<think>\n\n</think>\n\n"
    with pytest.raises(TeacherContractError, match="duplicate disabled-thinking prefills"):
        VLLMTeacher._normalize_template_think_prefix("<assistant>" + prefix + prefix)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("mismatched", [False, True])
def test_cross_token_multimodal_cached_tokenizer_wrapper_preserves_semantics(mismatched: bool) -> None:
    """vLLM's dynamic Cached* subclass must compare equal to the packer's original HF tokenizer."""

    class CachedCharTokenizer(_CharTokenizer):
        pass

    student_tokenizer = _CharTokenizer(1_000)
    hf_teacher_tokenizer = _CharTokenizer(10_000)
    cached_teacher_tokenizer = CachedCharTokenizer(10_000)
    if mismatched:
        cached_teacher_tokenizer.name_or_path = "different-artifact"
    engine = _ImageExpandedRealizedTokenEngine(cached_teacher_tokenizer)

    def build() -> VLLMTeacher:
        return VLLMTeacher(
            engine,
            packer=types.SimpleNamespace(tokenizer=student_tokenizer),
            teacher_packer=_ImageTeacherPacker(hf_teacher_tokenizer),
            top_k=0,
            input_mode="cross_token_multimodal",
            signal_mode="dpca",
        )

    if mismatched:
        with pytest.raises(TeacherContractError, match="tokenizer differs"):
            build()
        return

    teacher = build()

    assert teacher.teacher_codec is not None
    assert type(teacher.teacher_codec.tokenizer) is CachedCharTokenizer
    assert type(teacher.teacher_packer.tokenizer) is _CharTokenizer


@pytest.mark.L0
@pytest.mark.CPU
def test_multimodal_prefix_collapse_accepts_only_grid_expansion() -> None:
    submitted = [1, 90, 91, 92, 2]
    realized = [1, 90, *([91] * 81), 92, 2]

    collapsed, widths = VLLMTeacher._collapse_image_marker_runs(
        realized,
        image_token_id=91,
        vision_start_token_id=90,
        vision_end_token_id=92,
        expected_widths=(81,),
    )

    assert collapsed == submitted
    assert widths == [81]
    with pytest.raises(TeacherContractError, match=r"widths \(161,\) differ from expected \(81,\)"):
        VLLMTeacher._collapse_image_marker_runs(
            [1, 90, *([91] * 161), 92, 2],
            image_token_id=91,
            vision_start_token_id=90,
            vision_end_token_id=92,
            expected_widths=(81,),
        )
    with pytest.raises(TeacherContractError, match="outside a vision block"):
        VLLMTeacher._collapse_image_marker_runs(
            [1, 91, 2],
            image_token_id=91,
            vision_start_token_id=90,
            vision_end_token_id=92,
            expected_widths=(),
        )
    with pytest.raises(TeacherContractError, match="not followed by vision-end"):
        VLLMTeacher._collapse_image_marker_runs(
            [1, 90, 91, 2],
            image_token_id=91,
            vision_start_token_id=90,
            vision_end_token_id=92,
            expected_widths=(1,),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_token_multimodal_dpca_preserves_ordered_images_and_expanded_prefix_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teacher-owned image shaping must preserve order and keep vLLM expansion outside completion."""
    student_tokenizer = _CharTokenizer(1_000)
    teacher_tokenizer = _CharTokenizer(10_000)
    teacher, engine = _dpca_multimodal_teacher(student_tokenizer, teacher_tokenizer)
    image_trajectory = _dpca_image_traj(
        student_tokenizer,
        "image answer",
        images=["/media/first.png", "/media/second.png"],
        trajectory_id="g0_k0",
    )
    text_trajectory = _dpca_image_traj(
        student_tokenizer,
        "text answer",
        images=[],
        trajectory_id="g1_k0",
    )
    captured: list[dict[str, Any]] = []

    def _capture(identity: dict[str, Any]) -> str:
        captured.append(identity)
        return _ordered_request_digest(identity)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.teacher.vllm._ordered_request_digest",
        _capture,
    )

    teacher.score([image_trajectory, text_trajectory])

    assert len(engine.calls) == 1
    image_request, text_request = engine.calls[0]["prompts"]
    assert image_request["multi_modal_data"]["image"] == ["/media/first.png", "/media/second.png"]
    assert image_request["mm_processor_kwargs"] == {"max_pixels": 4096}
    assert "multi_modal_data" not in text_request
    assert captured[0]["version"] == 3
    image_identity, text_identity = captured[0]["trajectories"]
    assert image_identity["images"]["count"] == 2
    assert image_identity["images"]["expected_marker_widths"] == (81, 81)
    assert len(image_identity["images"]["uuids"]) == 2
    assert text_identity["images"]["count"] == 0
    for trajectory in (image_trajectory, text_trajectory):
        signal = trajectory.transitions[0].targets
        assert isinstance(signal, TeacherSignal)
        assert signal.chunk_bounds.shape[0] > 0

    image_signal = image_trajectory.transitions[0].targets
    submitted_ids = image_request["prompt_token_ids"]
    expanded_length = len(submitted_ids) + 80 * len(image_request["multi_modal_data"]["image"])
    completion_width = len(image_identity["teacher_completion_token_ids"])
    completion_start = expanded_length - completion_width
    expected_logprobs = torch.tensor(  # [T]
        [-0.25 - 0.01 * position for position in range(completion_start, expanded_length)],
        dtype=torch.float32,
    )
    assert torch.allclose(image_signal.chunk_teacher_logprob, expected_logprobs)


@pytest.mark.L0
@pytest.mark.CPU
def test_dpca_offset_free_fallback_requires_exact_joint_prefix_and_tail() -> None:
    """A slow tokenizer may fall back only with exact prefix IDs and byte-exact decoded tail."""
    student_tokenizer = _CharTokenizer(1_000)
    teacher_tokenizer = _CharTokenizer(10_000, supports_offsets=False)
    teacher, engine = _dpca_teacher(student_tokenizer, teacher_tokenizer)
    trajectory = _dpca_traj(student_tokenizer, "fallback answer")

    teacher.score([trajectory])

    assert len(engine.calls) == 1
    assert [with_offsets for _text, with_offsets in teacher_tokenizer.tokenize_calls] == [True, False, False]
    assert trajectory.transitions[0].targets.chunk_bounds.shape[0] == len("fallback answer")

    mismatched_tokenizer = _CharTokenizer(
        20_000,
        supports_offsets=False,
        mismatch_fallback_prefix=True,
    )
    failing_teacher, failing_engine = _dpca_teacher(student_tokenizer, mismatched_tokenizer)
    untouched = _dpca_traj(student_tokenizer, "fallback answer")
    with pytest.raises(TeacherContractError, match="joint token sequence.*prefix exactly"):
        failing_teacher.score([untouched])
    assert failing_engine.calls == []
    assert untouched.transitions[0].targets is None


@pytest.mark.L0
@pytest.mark.CPU
def test_dpca_uses_exact_completion_ids_when_vllm_display_text_hides_stop_tokens() -> None:
    """The realized student codec, not vLLM's special-token-stripped display text, owns alignment."""

    class _StopTokenizer(_CharTokenizer):
        eos_token_id = 9_999
        additional_special_tokens_ids = [9_999]
        all_special_ids = [9_999]

        def decode(
            self,
            token_ids: list[int],
            *,
            skip_special_tokens: bool,
            clean_up_tokenization_spaces: bool,
        ) -> str:
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return "".join(
                "<|im_end|>" if token_id == self.eos_token_id else chr(token_id - self.token_offset)
                for token_id in token_ids
            )

    student_tokenizer = _StopTokenizer(1_000)
    teacher_tokenizer = _CharTokenizer(10_000)
    teacher, engine = _dpca_teacher(student_tokenizer, teacher_tokenizer)
    trajectory = _dpca_traj(student_tokenizer, "answer")
    policy_info = trajectory.transitions[0].policy_info
    policy_info["completion_ids"].append(student_tokenizer.eos_token_id)
    policy_info["rollout_logprobs"].append(-0.2)
    assert policy_info["completion_text"] == "answer"

    teacher.score([trajectory])

    assert len(engine.calls) == 1
    assert trajectory.transitions[0].targets is not None


@pytest.mark.L0
@pytest.mark.CPU
def test_dpca_signal_retry_atomic_commit_and_consumed_diagnostics() -> None:
    """Cross-token retries one immutable batch, commits together, and reports bounded stats."""
    student_tokenizer = _CharTokenizer(1_000)
    teacher_tokenizer = _CharTokenizer(10_000)
    engine = _RealizedTokenEngine(teacher_tokenizer, failures_before_success=1)
    teacher, _engine = _dpca_teacher(
        student_tokenizer,
        teacher_tokenizer,
        engine=engine,
        max_attempts=2,
    )
    trajectories = [
        _dpca_traj(student_tokenizer, "abc", trajectory_id="g0_k0"),
        _dpca_traj(student_tokenizer, "de", trajectory_id="g1_k0"),
    ]

    teacher.score(trajectories)

    assert len(engine.calls) == 2
    assert engine.calls[0]["prompts"] == engine.calls[1]["prompts"]
    assert all(isinstance(trajectory.transitions[0].targets, TeacherSignal) for trajectory in trajectories)
    stats = teacher.batch_stats()
    assert stats == {
        "train/opd_teacher_score_request_count": 2,
        "train/opd_teacher_score_success_count": 1,
        "train/opd_teacher_score_failure_count": 1,
        "train/opd_teacher_scored_sequences": 2,
        "train/opd_teacher_scored_tokens": 5,
        "train/opd_dpca_chunk_count": 5,
        "train/opd_dpca_mean_chunk_length": 1.0,
        "train/opd_dpca_max_chunk_length": 1,
        "train/opd_dpca_exact_alignment_coverage": 1.0,
        "train/opd_dpca_chunk_teacher_logprob_mean": pytest.approx(
            sum(
                float(value)
                for trajectory in trajectories
                for value in trajectory.transitions[0].targets.chunk_teacher_logprob.tolist()
            )
            / 5
        ),
        "train/opd_dpca_chunk_student_logprob_mean": pytest.approx((-0.7 - 0.71 - 0.72 - 0.7 - 0.71) / 5),
    }
    assert not any(isinstance(value, torch.Tensor) for value in stats.values())
    with pytest.raises(RuntimeError, match="score must succeed first"):
        teacher.batch_stats()


@pytest.mark.L0
@pytest.mark.CPU
def test_dpca_late_parse_failure_never_partially_commits() -> None:
    class _LateMalformedEngine(_RealizedTokenEngine):
        def generate(self, prompts: list[dict[str, list[int]]], sampling_params: Any) -> list[_Output]:
            outputs = super().generate(prompts, sampling_params)
            outputs[1].prompt_logprobs[-1] = {123: _Logprob(-1.0, 1)}
            return outputs

    student_tokenizer = _CharTokenizer(1_000)
    teacher_tokenizer = _CharTokenizer(10_000)
    teacher, _engine = _dpca_teacher(
        student_tokenizer,
        teacher_tokenizer,
        engine=_LateMalformedEngine(teacher_tokenizer),
        max_attempts=1,
    )
    trajectories = [
        _dpca_traj(student_tokenizer, "abc", trajectory_id="g0_k0"),
        _dpca_traj(student_tokenizer, "de", trajectory_id="g1_k0"),
    ]
    original_targets = [object(), object()]
    for trajectory, target in zip(trajectories, original_targets, strict=True):
        trajectory.transitions[0].targets = target

    with pytest.raises(TeacherContractError, match="failed after 1 replica-wide attempt"):
        teacher.score(trajectories)

    assert [trajectory.transitions[0].targets for trajectory in trajectories] == original_targets
    with pytest.raises(RuntimeError, match="score must succeed first"):
        teacher.batch_stats()


@pytest.mark.L0
@pytest.mark.CPU
def test_dpca_signal_digest_must_match_every_teacher_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rank-local chunk drift poisons the attempt before the leader publishes tensors."""
    student_tokenizer = _CharTokenizer(1_000)
    teacher_tokenizer = _CharTokenizer(10_000)
    teacher, engine = _dpca_teacher(student_tokenizer, teacher_tokenizer, max_attempts=1)
    trajectory = _dpca_traj(student_tokenizer, "answer")

    def _mismatch_signal_only(digest: str, *, context: str) -> dict[int, str]:
        return {0: digest, 1: "different-signal-digest" if "signal digest" in context else digest}

    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.teacher.vllm.gather_rank_states",
        _mismatch_signal_only,
    )
    with pytest.raises(TeacherContractError, match="ranked teacher signal differs"):
        teacher.score([trajectory])

    assert len(engine.calls) == 1
    assert trajectory.transitions[0].targets is None


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_rollout_converges_lanes_and_keeps_boundary_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every lane proves request/output identity, while only the leader publishes trajectories."""
    from cosmos_framework.posttrain.roles.rollout import VLLMRollout

    class _Packer:
        def adds_special_tokens(self) -> bool:
            return False

        def get_rollout_input(self, _obs: ObsBundle) -> dict[str, Any]:
            return {
                "prompt": "rendered prompt",
                "multi_modal_data": {"image": ["image"]},
                "multi_modal_uuids": {"image": ["image-content-v1"]},
                "mm_processor_kwargs": {"max_pixels": 4096},
            }

    completion = types.SimpleNamespace(
        token_ids=[21, 22],
        text="answer",
        logprobs=[{21: _Logprob(-0.2, 1)}, {22: _Logprob(-0.3, 1)}],
    )
    request = types.SimpleNamespace(prompt_token_ids=[11, 12, 13], outputs=[completion])
    engine = types.SimpleNamespace(generate=lambda _prompts, _params: [request])
    contexts: list[str] = []
    identities: list[dict[str, Any]] = []

    def _capture_identity(identity: dict[str, Any], *, context: str) -> None:
        identities.append(identity)
        contexts.append(context)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.rollout.vllm._require_identical_rank_identity",
        _capture_identity,
    )
    rollout = VLLMRollout(
        engine,
        packer=_Packer(),
        sampling={"temperature": 1.0, "top_p": 1.0, "max_tokens": 2},
    )

    trajectories = rollout.generate([ObsBundle(representations={"prompt": "x"}, id="p0")], group=1)

    policy_info = trajectories[0].transitions[0].policy_info
    assert policy_info["completion_ids"] == [21, 22]
    assert policy_info["completion_boundary"] == {
        "prompt_token_count": 3,
        "completion_token_count": 2,
        "full_sequence_token_count": 5,
    }
    assert contexts == ["vLLM training request identity", "vLLM training output identity"]
    request_identity = identities[0]["requests"][0]
    assert request_identity["media_cardinality"] == {"image": 1}
    assert request_identity["multi_modal_uuids"] == {"image": ["image-content-v1"]}
    assert request_identity["mm_processor_kwargs"] == {"max_pixels": 4096}

    contexts.clear()
    monkeypatch.setattr("cosmos_framework.posttrain.roles.rollout.vllm._is_replica_leader", lambda: False)
    assert rollout.generate([ObsBundle(representations={"prompt": "x"}, id="p0")], group=1) == []
    assert contexts == ["vLLM training request identity", "vLLM training output identity"]


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_request_media_identity_is_ordered_and_image_fail_closed() -> None:
    """Content order is part of TP convergence; legacy TP1 video stays cardinality-only."""
    from cosmos_framework.posttrain.roles.vllm_request import vllm_request_media_identity

    base = {
        "multi_modal_data": {"image": ["first", "second"]},
        "multi_modal_uuids": {"image": ["first-uuid", "second-uuid"]},
    }
    identity = vllm_request_media_identity(base, context="test request")
    reversed_identity = vllm_request_media_identity(
        {**base, "multi_modal_uuids": {"image": ["second-uuid", "first-uuid"]}},
        context="test request",
    )
    assert identity != reversed_identity
    with pytest.raises(ValueError, match="image media requires ordered content UUIDs"):
        vllm_request_media_identity({"multi_modal_data": base["multi_modal_data"]}, context="test request")
    with pytest.raises(ValueError, match="media/UUID cardinality mismatch"):
        vllm_request_media_identity(
            {**base, "multi_modal_uuids": {"image": ["first-uuid"]}},
            context="test request",
        )
    with pytest.raises(ValueError, match="no matching media"):
        vllm_request_media_identity(
            {**base, "multi_modal_uuids": {"image": ["first-uuid", "second-uuid"], "video": ["extra"]}},
            context="test request",
        )
    assert vllm_request_media_identity(
        {"multi_modal_data": {"video": ["legacy-video"]}},
        context="test request",
    )["cardinality_only_modalities"] == ["video"]
    with pytest.raises(ValueError, match="video media requires ordered content UUIDs"):
        vllm_request_media_identity(
            {"multi_modal_data": {"video": ["legacy-video"]}},
            context="test request",
            require_all_content_uuids=True,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_completion_is_skipped_and_never_reaches_the_engine():
    """The objective skips empty completions; the teacher must not invent a signal for them, or its
    row-count check would reject a batch the objective was happy to ignore. An all-empty batch must
    also not call the engine at all — vLLM rejects an empty prompt list."""
    engine = _FakeEngine()
    trajs = [_traj([11, 12], [])]

    VLLMTeacher(engine, top_k=TOP_K).score(trajs)

    assert trajs[0].transitions[0].targets is None
    assert engine.calls == []

    # A mixed batch scores only the non-empty one, and still in a single call.
    mixed = [
        _traj([11, 12], [], trajectory_id="g0_k0"),
        _traj([31, 32], [41], trajectory_id="g1_k0"),
    ]
    VLLMTeacher(engine, top_k=TOP_K).score(mixed)
    assert len(engine.calls) == 1
    assert [p["prompt_token_ids"] for p in engine.calls[0]["prompts"]] == [[31, 32, 41]]
    assert mixed[0].transitions[0].targets is None
    assert mixed[1].transitions[0].targets is not None
