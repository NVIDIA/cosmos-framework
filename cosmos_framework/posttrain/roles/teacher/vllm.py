# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronous ranked vLLM teacher scoring.

The teacher scores the realized student sequence; it never generates a replacement answer. Every
tensor-parallel lane receives the same request and enters the same ``vllm.LLM.generate`` call under
``external_launcher``. Request shaping, request identity, engine execution, response parsing, and
signal validation are rank-converged phases so one bad lane cannot strand its peers in the next
collective.

Only the replica leader publishes the trajectories carrying :class:`TeacherSignal` tensors.
Non-leaders return a small :class:`TeacherRankStatus`, avoiding duplicate ``[T,K]`` bulk in replay.
Signals are parsed into temporary objects and committed only after the entire batch and every rank
have succeeded, which also makes a bounded retry safe.

All settled scoring modes live behind the same ranked, atomic commit protocol:
``shared_vocab_multimodal`` + ``topk`` scores the student's exact token IDs with its original media,
while ``cross_token_text`` + ``dpca`` renders only canonical text with the teacher tokenizer and
``cross_token_multimodal`` + ``dpca`` adds the teacher's own image processor/request semantics. Both
cross-token modes align the realized completion back to immutable student-token coordinates.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any

import torch

from cosmos_framework.posttrain.algorithm.dpca import build_dpca_signal
from cosmos_framework.posttrain.data.packer import shape_policy_input
from cosmos_framework.posttrain.roles.teacher.base import (
    TeacherContractError,
    TeacherRankStatus,
    TeacherScoreResult,
)
from cosmos_framework.posttrain.roles.vllm_request import vllm_request_media_identity
from cosmos_framework.posttrain.schema import TeacherSignal, Trajectory
from cosmos_framework.posttrain.utils.dist import run_rank_converged
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

if TYPE_CHECKING:
    from cosmos_framework.posttrain.data.packer.base import DataPacker
    from cosmos_framework.posttrain.schema import Transition


_SHARED_VOCAB_MULTIMODAL = "shared_vocab_multimodal"
_CROSS_TOKEN_TEXT = "cross_token_text"
_CROSS_TOKEN_MULTIMODAL = "cross_token_multimodal"
_TOPK = "topk"
_DPCA = "dpca"
_LOGPROB_TOLERANCE = 1.0e-6
_SUPPORT_MASS_TOLERANCE = 1.0e-4
_MAX_RETRY_BACKOFF_S = 60.0
_MEDIA_CONTENT_TYPES = frozenset({"image", "image_url", "video", "video_frames"})
_TEXT_CONTENT_TYPE = "text"
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_QWEN_DISABLED_THINK_PREFIX = "<think>\n\n</think>\n\n"
_DECODE_SETTINGS: dict[str, bool] = {
    "skip_special_tokens": False,
    "clean_up_tokenization_spaces": False,
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class _PreparedTopKRequest:
    """One exact pretokenized teacher request plus its completion-scoring boundary."""

    trajectory: Trajectory
    request: dict[str, Any]
    token_ids: list[int]
    scored_indices: list[int]
    prompt_token_count: int
    completion_token_count: int
    n_ignore_prefix_tokens: int
    media_identity: dict[str, Any]

    def identity(self) -> dict[str, Any]:
        """Return the small ordered identity compared immediately before the engine call."""
        return {
            "trajectory_id": self.trajectory.id,
            "prompt_token_ids": self.token_ids,
            "completion_boundary": {
                "prompt_token_count": self.prompt_token_count,
                "completion_token_count": self.completion_token_count,
                "n_ignore_prefix_tokens": self.n_ignore_prefix_tokens,
                "scored_indices": self.scored_indices,
            },
            **self.media_identity,
        }


@dataclass(frozen=True)
class _PreparedDPCARequest:
    """One cross-token joint request plus both completion coordinate systems."""

    trajectory: Trajectory
    request: dict[str, Any]
    token_ids: list[int]
    scored_indices: list[int]
    student_completion_ids: list[int]
    teacher_completion_ids: list[int]
    rollout_logprobs: list[float]
    image_count: int
    image_token_id: int | None
    vision_start_token_id: int | None
    vision_end_token_id: int | None
    expected_image_marker_widths: tuple[int, ...]
    image_uuids: tuple[str, ...]

    def identity(self) -> dict[str, Any]:
        """Return the complete ordered cross-token request/alignment identity."""
        return {
            "trajectory_id": self.trajectory.id,
            "student_completion_ids": self.student_completion_ids,
            "teacher_prompt_token_ids": self.token_ids,
            "teacher_completion_token_ids": self.teacher_completion_ids,
            "teacher_completion_indices": self.scored_indices,
            "images": {
                "count": self.image_count,
                "image_token_id": self.image_token_id,
                "vision_start_token_id": self.vision_start_token_id,
                "vision_end_token_id": self.vision_end_token_id,
                "expected_marker_widths": self.expected_image_marker_widths,
                "uuids": self.image_uuids,
            },
            "mm_processor_kwargs": self.request.get("mm_processor_kwargs", {}),
        }


@dataclass(frozen=True)
class _ParsedTeacherSignal:
    """A temporary signal that is not attached to its trajectory until global success."""

    trajectory: Trajectory
    signal: TeacherSignal
    scored_tokens: int
    engine_prompt_token_ids: list[int]
    rollout_logprobs: tuple[float, ...] | None = None


@dataclass(frozen=True)
class _FixedTokenizerCodec:
    """A tokenizer decode capability pinned to one artifact and explicit decode settings."""

    owner: str
    tokenizer: Any
    decode_fn: Callable[..., Any]
    identity: dict[str, Any]

    @classmethod
    def build(cls, tokenizer: Any, *, owner: str) -> "_FixedTokenizerCodec":
        if tokenizer is None:
            raise TeacherContractError(f"{owner} tokenizer is unavailable")
        decode_fn = getattr(tokenizer, "decode", None)
        if not callable(decode_fn):
            raise TeacherContractError(f"{owner} tokenizer exposes no decode() capability")
        return cls(
            owner=owner,
            tokenizer=tokenizer,
            decode_fn=decode_fn,
            identity=_tokenizer_identity(tokenizer, owner=owner),
        )

    def require_unchanged(self) -> None:
        """Reject mutable tokenizer state before it can redefine alignment coordinates."""
        current = _tokenizer_identity(self.tokenizer, owner=self.owner)
        if current != self.identity:
            raise TeacherContractError(f"{self.owner} tokenizer identity changed after teacher construction")

    def decode(self, token_ids: list[int]) -> str:
        """Decode with the fixed, non-cleaning settings used by every alignment comparison."""
        try:
            text = self.decode_fn(list(token_ids), **_DECODE_SETTINGS)
        except Exception as error:  # noqa: BLE001 -- normalize arbitrary tokenizer backend failures
            raise TeacherContractError(f"{self.owner} fixed decode failed: {error}") from error
        if not isinstance(text, str):
            raise TeacherContractError(f"{self.owner} tokenizer.decode() must return str")
        return text


def _tokenizer_identity(tokenizer: Any, *, owner: str) -> dict[str, Any]:
    """Bind tokenizer semantics while ignoring transparent vLLM cached-wrapper classes."""
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise TeacherContractError(f"{owner} tokenizer exposes no get_vocab(); its codec cannot be frozen")
    try:
        vocab = get_vocab()
    except Exception as error:  # noqa: BLE001 -- normalize arbitrary tokenizer backend failures
        raise TeacherContractError(f"{owner} tokenizer.get_vocab() failed: {error}") from error
    if not isinstance(vocab, dict) or any(
        not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int)
        for token, token_id in vocab.items()
    ):
        raise TeacherContractError(f"{owner} tokenizer.get_vocab() must return dict[str, int]")
    if any(token_id < 0 for token_id in vocab.values()):
        raise TeacherContractError(f"{owner} tokenizer.get_vocab() contains a negative token id")

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is not None and not (
        isinstance(chat_template, str)
        or (
            isinstance(chat_template, dict)
            and all(isinstance(name, str) and isinstance(template, str) for name, template in chat_template.items())
        )
    ):
        raise TeacherContractError(f"{owner} tokenizer.chat_template must be str, dict[str, str], or None")
    special_ids: dict[str, int | list[int] | None] = {}
    for field in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "additional_special_tokens_ids",
        "all_special_ids",
    ):
        value = getattr(tokenizer, field, None)
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            special_ids[field] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in value
        ):
            special_ids[field] = [int(token_id) for token_id in value]
        else:
            raise TeacherContractError(f"{owner} tokenizer.{field} is not a stable integer ID setting")
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_sha256": canonical_json_sha256(vocab, context=f"{owner} tokenizer vocabulary"),
        "chat_template_sha256": canonical_json_sha256(
            chat_template,
            context=f"{owner} tokenizer chat template",
        ),
        "special_ids": special_ids,
        "decode_settings": dict(_DECODE_SETTINGS),
    }


def _rank_world() -> tuple[int, int]:
    """Return this teacher replica's TP rank/world, degenerating to ``(0, 1)`` locally."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return 0, 1
    return int(dist.get_rank()), int(dist.get_world_size())


def _ordered_request_digest(identity: dict[str, Any]) -> str:
    """Hash a JSON-safe request identity without sorting its trajectory list."""
    return canonical_json_sha256(identity, context="ranked teacher request identity")


def _require_identical_digest(digest: str, *, context: str) -> None:
    """Fail on every rank when one lane prepared or parsed different content."""
    by_rank = gather_rank_states(digest, context=f"{context} digest preparation")
    if any(candidate != digest for candidate in by_rank.values()):
        raise TeacherContractError(f"{context} differs across teacher TP ranks: {by_rank!r}")


def _entry_logprob(value: Any) -> float:
    """Read vLLM's ``Logprob`` object or the dictionary form used by test/HTTP adapters."""
    logprob = getattr(value, "logprob", None)
    if logprob is None and isinstance(value, dict):
        logprob = value.get("logprob")
    if logprob is None:
        raise TeacherContractError(f"teacher entry carries no logprob: {value!r}")
    try:
        return float(logprob)
    except (TypeError, ValueError) as error:
        raise TeacherContractError(f"teacher entry carries a non-numeric logprob: {value!r}") from error


class VLLMTeacher:
    """Score realized sequences with a held, optionally tensor-parallel vLLM engine.

    Args:
        model: A built ``vllm.LLM``. Under TP, every actor builds one facade against the already-live
            per-replica process group.
        packer: Student packer. The same-token path scores exact rollout IDs and only asks the packer,
            through ``get_pretokenized_rollout_input`` when available, to attach original media and
            collapse each image placeholder run to the one marker vLLM re-expands.
        teacher_packer: Teacher-artifact packer used only by ``cross_token_multimodal`` to render the
            teacher prompt, validate image placeholders, and attach ordered images.
        top_k: Teacher support width per scored token. Must be positive in ``topk`` mode and zero in
            ``dpca`` mode.
        input_mode: ``shared_vocab_multimodal``, ``cross_token_text``, or ``cross_token_multimodal``.
        signal_mode: ``topk`` or ``dpca``.
        logprobs_mode: vLLM engine logprob convention. OPD fixes this to ``processed_logprobs`` for
            generated-token consistency, although forced-sequence ``prompt_logprobs`` remain raw.
        scoring_temperature: Declared likelihood temperature. The current forced-sequence path
            supports exactly one because vLLM does not temperature-process prompt logprobs.
        max_attempts: Replica-wide attempts for one immutable request.
        retry_backoff_s: Delay between converged failed attempts, bounded to at most 60 seconds.
        vocab_size: Optional explicit token-id upper bound for test doubles or unusual tokenizers.
        conditional_think_prefix: Manual prefix inserted after the text teacher's assistant header
            according to the three DPCA think-boundary cases.
        max_alignment_chunk_tokens: Exact-alignment rejection guard on either tokenizer's chunk.
    """

    model: Any
    packer: DataPacker | None
    teacher_packer: DataPacker | None
    top_k: int
    input_mode: str
    signal_mode: str
    logprobs_mode: str
    scoring_temperature: float
    max_attempts: int
    retry_backoff_s: float
    vocab_size: int | None
    conditional_think_prefix: str
    max_alignment_chunk_tokens: int
    student_codec: _FixedTokenizerCodec | None
    teacher_codec: _FixedTokenizerCodec | None
    _teacher_encode: Callable[..., Any] | None
    _teacher_apply_chat_template: Callable[..., Any] | None
    _last_batch_stats: dict[str, float | int] | None

    def __init__(
        self,
        model: Any,
        *,
        packer: "DataPacker | None" = None,
        teacher_packer: "DataPacker | None" = None,
        top_k: int = 32,
        input_mode: str = _SHARED_VOCAB_MULTIMODAL,
        signal_mode: str = _TOPK,
        logprobs_mode: str = "processed_logprobs",
        scoring_temperature: float = 1.0,
        max_attempts: int = 2,
        retry_backoff_s: float = 0.0,
        vocab_size: int | None = None,
        conditional_think_prefix: str = "<think>\n\n</think>\n\n",
        max_alignment_chunk_tokens: int = 64,
    ) -> None:
        self.model = model
        self.packer = packer
        self.teacher_packer = teacher_packer
        self.input_mode = self._input_mode(input_mode)
        self.signal_mode = self._signal_mode(signal_mode)
        if logprobs_mode != "processed_logprobs":
            raise ValueError(
                "logprobs_mode must be 'processed_logprobs' for the declared teacher likelihood "
                f"contract, got {logprobs_mode!r}"
            )
        self.logprobs_mode = logprobs_mode
        self.top_k = self._top_k(top_k, self.signal_mode)
        self.scoring_temperature = self._positive_finite_real(
            scoring_temperature,
            field="scoring_temperature",
        )
        if self.scoring_temperature != 1.0:
            raise ValueError(
                "scoring_temperature must be 1.0: the current forced-sequence teacher reads raw "
                "vLLM prompt_logprobs, which are not temperature-processed"
            )
        self.max_attempts = self._positive_int(max_attempts, field="max_attempts")
        self.retry_backoff_s = self._retry_backoff(retry_backoff_s)
        self.vocab_size = self._optional_positive_int(vocab_size, field="vocab_size")
        if not isinstance(conditional_think_prefix, str):
            raise ValueError("conditional_think_prefix must be a string")
        self.conditional_think_prefix = conditional_think_prefix
        self.max_alignment_chunk_tokens = self._positive_int(
            max_alignment_chunk_tokens,
            field="max_alignment_chunk_tokens",
        )
        self.student_codec = None
        self.teacher_codec = None
        self._teacher_encode: Callable[..., Any] | None = None
        self._teacher_apply_chat_template: Callable[..., Any] | None = None
        self._last_batch_stats: dict[str, float | int] | None = None
        self._validate_mode_pair()
        if self.input_mode == _SHARED_VOCAB_MULTIMODAL:
            inferred_vocab_size = self._validate_shared_vocab()
            if self.vocab_size is None:
                self.vocab_size = inferred_vocab_size
            elif inferred_vocab_size is not None and self.vocab_size < inferred_vocab_size:
                raise TeacherContractError(
                    f"configured vocab_size={self.vocab_size} is below the shared tokenizer's required "
                    f"id bound {inferred_vocab_size}"
                )
        else:
            self._initialize_cross_token_codecs()

    @staticmethod
    def _input_mode(value: str) -> str:
        if value not in (_SHARED_VOCAB_MULTIMODAL, _CROSS_TOKEN_TEXT, _CROSS_TOKEN_MULTIMODAL):
            raise ValueError(
                f"input_mode must be shared_vocab_multimodal|cross_token_text|cross_token_multimodal, got {value!r}"
            )
        return value

    @staticmethod
    def _signal_mode(value: str) -> str:
        if value not in (_TOPK, _DPCA):
            raise ValueError(f"signal_mode must be topk|dpca, got {value!r}")
        return value

    @staticmethod
    def _positive_int(value: int, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field} must be a non-bool positive integer, got {value!r}")
        return value

    @classmethod
    def _optional_positive_int(cls, value: int | None, *, field: str) -> int | None:
        return None if value is None else cls._positive_int(value, field=field)

    @staticmethod
    def _positive_finite_real(value: float, *, field: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a non-bool finite real > 0, got {value!r}")
        normalized = float(value)
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{field} must be a non-bool finite real > 0, got {value!r}")
        return normalized

    @staticmethod
    def _retry_backoff(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"retry_backoff_s must be a non-bool finite real >= 0, got {value!r}")
        normalized = float(value)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= _MAX_RETRY_BACKOFF_S:
            raise ValueError(f"retry_backoff_s must be finite and in [0, {_MAX_RETRY_BACKOFF_S}], got {value!r}")
        return normalized

    @staticmethod
    def _top_k(value: int, signal_mode: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"top_k must be a non-bool integer >= 0, got {value!r}")
        if signal_mode == _TOPK and value < 1:
            raise ValueError("signal_mode='topk' requires top_k > 0")
        if signal_mode == _DPCA and value != 0:
            raise ValueError("signal_mode='dpca' requires top_k=0")
        return value

    def _validate_mode_pair(self) -> None:
        if self.signal_mode == _TOPK and self.input_mode != _SHARED_VOCAB_MULTIMODAL:
            raise ValueError("signal_mode='topk' requires input_mode='shared_vocab_multimodal'")
        if self.signal_mode == _DPCA and self.input_mode not in (_CROSS_TOKEN_TEXT, _CROSS_TOKEN_MULTIMODAL):
            raise ValueError("signal_mode='dpca' requires a cross-token input mode")
        if self.input_mode == _CROSS_TOKEN_MULTIMODAL and self.teacher_packer is None:
            raise ValueError("input_mode='cross_token_multimodal' requires teacher_packer")

    def _model_tokenizer(self) -> Any:
        """Return the held teacher engine's tokenizer with backend failures normalized."""
        get_teacher_tokenizer = getattr(self.model, "get_tokenizer", None)
        if not callable(get_teacher_tokenizer):
            raise TeacherContractError("teacher model does not expose get_tokenizer()")
        try:
            return get_teacher_tokenizer()
        except Exception as error:  # noqa: BLE001 -- normalize arbitrary vLLM backend failures
            raise TeacherContractError(f"teacher model.get_tokenizer() failed: {error}") from error

    def _initialize_cross_token_codecs(self) -> None:
        """Freeze student/teacher completion codecs and the teacher rendering capabilities."""
        student_tokenizer = getattr(self.packer, "tokenizer", None)
        if student_tokenizer is None:
            raise TeacherContractError(
                "cross-token DPCA requires a student packer with its immutable completion tokenizer"
            )
        teacher_tokenizer = self._model_tokenizer()
        encode = getattr(teacher_tokenizer, "__call__", None)
        apply_chat_template = getattr(teacher_tokenizer, "apply_chat_template", None)
        if not callable(encode):
            raise TeacherContractError("teacher tokenizer exposes no tokenization capability")
        if not callable(apply_chat_template):
            raise TeacherContractError("teacher tokenizer exposes no apply_chat_template() capability")
        self.student_codec = _FixedTokenizerCodec.build(student_tokenizer, owner="student completion codec")
        self.teacher_codec = _FixedTokenizerCodec.build(teacher_tokenizer, owner="teacher completion codec")
        self._teacher_encode = encode
        self._teacher_apply_chat_template = apply_chat_template
        if self.input_mode != _CROSS_TOKEN_MULTIMODAL:
            return
        teacher_packer_tokenizer = getattr(self.teacher_packer, "tokenizer", None)
        packer_identity = _tokenizer_identity(
            teacher_packer_tokenizer,
            owner="cross-token multimodal teacher packer",
        )
        if packer_identity != self.teacher_codec.identity:
            raise TeacherContractError(
                "cross-token multimodal teacher packer tokenizer differs from the held vLLM teacher tokenizer"
            )

    @staticmethod
    def _vocab_mapping(tokenizer: Any, *, owner: str) -> dict[str, int]:
        """Return one proven token-to-id mapping for the shared-vocabulary contract."""
        if tokenizer is None:
            raise TeacherContractError(f"{owner} tokenizer is unavailable; shared-vocabulary OPD cannot be proven safe")
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            raise TeacherContractError(
                f"{owner} tokenizer does not expose get_vocab(); shared-vocabulary OPD requires the exact "
                "token-to-id mapping"
            )
        try:
            vocab = get_vocab()
        except Exception as error:  # noqa: BLE001 -- normalize arbitrary tokenizer backend failures
            raise TeacherContractError(f"{owner} tokenizer.get_vocab() failed: {error}") from error
        if not isinstance(vocab, dict) or any(
            not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int)
            for token, token_id in vocab.items()
        ):
            raise TeacherContractError(f"{owner} tokenizer.get_vocab() must return dict[str, int]")
        if any(token_id < 0 for token_id in vocab.values()):
            raise TeacherContractError(f"{owner} tokenizer.get_vocab() contains a negative token id")
        return vocab

    def _validate_shared_vocab(self) -> int | None:
        """Prove teacher and student token IDs match exactly and return their upper id bound."""
        if self.packer is None:
            return None  # explicit test-double path
        student_tokenizer = getattr(self.packer, "tokenizer", None)
        try:
            teacher_tokenizer = self._model_tokenizer()
        except TeacherContractError as error:
            raise TeacherContractError(
                f"{error}; shared-vocabulary OPD cannot prove the teacher token ids match the student packer"
            ) from error
        teacher_vocab = self._vocab_mapping(teacher_tokenizer, owner="teacher")
        student_vocab = self._vocab_mapping(student_tokenizer, owner="student packer")
        if teacher_vocab != student_vocab:
            teacher_tokens = set(teacher_vocab)
            student_tokens = set(student_vocab)
            missing = sorted(student_tokens - teacher_tokens)[:3]
            extra = sorted(teacher_tokens - student_tokens)[:3]
            remapped = sorted(
                token for token in teacher_tokens & student_tokens if teacher_vocab[token] != student_vocab[token]
            )[:3]
            raise TeacherContractError(
                "shared-vocabulary OPD requires identical teacher/student tokenizer get_vocab() mappings, "
                f"got teacher={len(teacher_vocab)} student={len(student_vocab)} entries; "
                f"missing_from_teacher={missing}, extra_on_teacher={extra}, remapped={remapped}"
            )
        teacher_template = getattr(teacher_tokenizer, "chat_template", None)
        student_template = getattr(student_tokenizer, "chat_template", None)
        if teacher_template != student_template:
            raise TeacherContractError(
                "shared-vocabulary OPD requires identical teacher/student chat templates; token "
                "IDs alone do not prove the same multimodal prompt semantics"
            )
        return max(student_vocab.values(), default=-1) + 1

    def _sampling_kwargs(self) -> dict[str, Any]:
        """Return the complete forced-token scoring contract used in request identity."""
        return {
            "prompt_logprobs": self.top_k if self.signal_mode == _TOPK else 0,
            "max_tokens": 1,
            "temperature": self.scoring_temperature,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

    def _sampling_params(self) -> Any:
        from vllm import SamplingParams

        return SamplingParams(**self._sampling_kwargs())

    def _validate_token_ids(self, value: Any, *, owner: str) -> list[int]:
        try:
            raw_ids = value.tolist() if hasattr(value, "tolist") else list(value)
        except (TypeError, ValueError) as error:
            raise TeacherContractError(f"{owner} must be an iterable of token ids") from error
        ids: list[int] = []
        for position, token_id in enumerate(raw_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TeacherContractError(f"{owner}[{position}] must be a non-bool integer token id, got {token_id!r}")
            if token_id < 0 or (self.vocab_size is not None and token_id >= self.vocab_size):
                bound = "unbounded" if self.vocab_size is None else str(self.vocab_size)
                raise TeacherContractError(
                    f"{owner}[{position}]={token_id} is outside the valid token range [0, {bound})"
                )
            ids.append(token_id)
        return ids

    def _completion_ids(self, transition: Transition) -> list[int]:
        if isinstance(transition.action, str):
            raise TeacherContractError(
                "shared-vocabulary teacher scoring requires realized completion token ids, not decoded text"
            )
        action_ids = self._validate_token_ids(transition.action, owner="transition.action")
        inline = transition.policy_info.get("completion_ids") if transition.policy_info else None
        if inline is not None:
            inline_ids = self._validate_token_ids(inline, owner="policy_info['completion_ids']")
            if inline_ids != action_ids:
                raise TeacherContractError(
                    "policy_info['completion_ids'] differs from transition.action; the teacher cannot "
                    "choose which realized sequence to score"
                )
        return action_ids

    @staticmethod
    def _inline_student_token_ids(value: Any, *, owner: str) -> list[int]:
        """Validate inline student IDs without applying the unrelated teacher-vocabulary bound."""
        if isinstance(value, (str, bytes)):
            raise TeacherContractError(f"{owner} must be an inline iterable of student token ids")
        try:
            raw_ids = list(value)
        except (TypeError, ValueError) as error:
            raise TeacherContractError(f"{owner} must be an inline iterable of student token ids") from error
        token_ids: list[int] = []
        for position, token_id in enumerate(raw_ids):
            if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                raise TeacherContractError(
                    f"{owner}[{position}] must be a non-bool non-negative integer, got {token_id!r}"
                )
            token_ids.append(token_id)
        return token_ids

    @staticmethod
    def _inline_rollout_logprobs(value: Any, *, expected: int, owner: str) -> list[float]:
        """Validate DPCA's inline rollout likelihood witness without materializing tensor bulk."""
        if isinstance(value, (str, bytes)):
            raise TeacherContractError(f"{owner} must be an inline iterable of finite logprobs")
        try:
            raw_logprobs = list(value)
        except (TypeError, ValueError) as error:
            raise TeacherContractError(f"{owner} must be an inline iterable of finite logprobs") from error
        if len(raw_logprobs) != expected:
            raise TeacherContractError(
                f"{owner} has {len(raw_logprobs)} value(s), expected one per completion token ({expected})"
            )
        logprobs: list[float] = []
        for position, value_at_position in enumerate(raw_logprobs):
            if isinstance(value_at_position, bool) or not isinstance(value_at_position, Real):
                raise TeacherContractError(f"{owner}[{position}] must be a non-bool finite real")
            logprob = float(value_at_position)
            if not math.isfinite(logprob) or logprob > _LOGPROB_TOLERANCE:
                raise TeacherContractError(f"{owner}[{position}] is not a finite logprob <= 0")
            logprobs.append(logprob)
        return logprobs

    @staticmethod
    def _text_only_messages(obs: Any) -> list[dict[str, Any]]:
        """Copy role/text structure while deliberately never reading media payload fields."""
        representations = getattr(obs, "representations", None)
        if not isinstance(representations, Mapping):
            raise TeacherContractError("cross-token teacher ObsBundle.representations must be a mapping")
        payload = representations.get("messages", representations.get("prompt"))
        if isinstance(payload, Mapping) and "messages" in payload:
            payload = payload["messages"]
        if not isinstance(payload, list) or not all(isinstance(message, Mapping) for message in payload):
            raise TeacherContractError("cross-token teacher requires canonical list-valued messages in ObsBundle")

        stripped: list[dict[str, Any]] = []
        for message_index, message in enumerate(payload):
            role = message.get("role")
            if not isinstance(role, str) or not role:
                raise TeacherContractError(f"teacher message {message_index} requires a non-empty string role")
            content = message.get("content")
            if isinstance(content, str):
                stripped.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                raise TeacherContractError(
                    f"teacher message {message_index} content must be str or a list of content blocks"
                )
            text_blocks: list[str] = []
            for block_index, block in enumerate(content):
                if not isinstance(block, Mapping):
                    raise TeacherContractError(
                        f"teacher message {message_index} content block {block_index} must be a mapping"
                    )
                block_type = block.get("type")
                if block_type in _MEDIA_CONTENT_TYPES:
                    continue
                if block_type != _TEXT_CONTENT_TYPE:
                    raise TeacherContractError(
                        f"teacher message {message_index} content block {block_index} has unsupported "
                        f"text-only type {block_type!r}"
                    )
                text = block.get("text")
                if not isinstance(text, str):
                    raise TeacherContractError(
                        f"teacher message {message_index} text block {block_index} requires string text"
                    )
                text_blocks.append(text)
            # A text-only Qwen tokenizer consumes the ordinary role/content-string schema. Joining
            # without a separator preserves every text block's bytes and order while adding none.
            stripped.append({"role": role, "content": "".join(text_blocks)})
        return stripped

    @staticmethod
    def _normalize_template_think_prefix(rendered: str) -> str:
        """Remove Qwen3.5's disabled-thinking prefill so conditional context is inserted once."""
        if rendered.endswith(_QWEN_DISABLED_THINK_PREFIX):
            base = rendered[: -len(_QWEN_DISABLED_THINK_PREFIX)]
            if not base:
                raise TeacherContractError("teacher chat template contains no assistant header before think prefill")
            if base.endswith(_QWEN_DISABLED_THINK_PREFIX):
                raise TeacherContractError("teacher chat template emitted duplicate disabled-thinking prefills")
            return base
        return rendered

    def _render_teacher_prompt(self, transition: Transition) -> str:
        """Render one cross-token prompt without a template-owned think prefill."""
        try:
            if self.input_mode == _CROSS_TOKEN_MULTIMODAL:
                render = getattr(self.teacher_packer, "render_cross_token_multimodal_prompt", None)
                if not callable(render):
                    raise TeacherContractError(
                        "cross-token multimodal teacher packer exposes no render_cross_token_multimodal_prompt()"
                    )
                rendered = render(transition.obs, enable_thinking=False)
            else:
                if self._teacher_apply_chat_template is None:
                    raise TeacherContractError("cross-token teacher chat-template capability was not initialized")
                messages = self._text_only_messages(transition.obs)
                rendered = self._teacher_apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
        except TeacherContractError:
            raise
        except Exception as error:  # noqa: BLE001 -- normalize arbitrary tokenizer/processor template failures
            raise TeacherContractError(f"teacher cross-token chat rendering failed: {error}") from error
        if not isinstance(rendered, str) or not rendered:
            raise TeacherContractError("teacher apply_chat_template() must return a non-empty string")
        return self._normalize_template_think_prefix(rendered)

    def _think_prefix(self, completion_text: str) -> str:
        """Choose the configured/full, absent, or opening-only think context."""
        if completion_text.startswith(_THINK_OPEN):
            return ""
        if _THINK_CLOSE not in completion_text:
            return self.conditional_think_prefix
        opening_context, separator, _closed_tail = self.conditional_think_prefix.partition(_THINK_CLOSE)
        if not separator or not opening_context.startswith(_THINK_OPEN):
            raise TeacherContractError(
                "a completion with a later </think> needs conditional_think_prefix to contain its "
                "configured opening context"
            )
        return opening_context

    @staticmethod
    def _tokenizer_field(encoded: Any, field: str) -> Any:
        if isinstance(encoded, Mapping):
            return encoded.get(field)
        return getattr(encoded, field, None)

    def _teacher_tokenize_without_offsets(self, text: str, *, owner: str) -> list[int]:
        if self._teacher_encode is None:
            raise TeacherContractError("cross-token teacher tokenizer capability was not initialized")
        try:
            encoded = self._teacher_encode(text, add_special_tokens=False)
        except Exception as error:  # noqa: BLE001 -- normalize arbitrary tokenizer backend failures
            raise TeacherContractError(f"{owner} tokenization failed: {error}") from error
        raw_ids = self._tokenizer_field(encoded, "input_ids")
        if raw_ids is None:
            raise TeacherContractError(f"{owner} tokenization returned no input_ids")
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        if raw_ids and isinstance(raw_ids[0], (list, tuple)):
            if len(raw_ids) != 1:
                raise TeacherContractError(f"{owner} tokenization unexpectedly returned a batch")
            raw_ids = raw_ids[0]
        return self._validate_token_ids(raw_ids, owner=f"{owner} input_ids")

    def _teacher_joint_tokenization(self, text: str) -> tuple[list[int], list[tuple[int, int]] | None]:
        """Tokenize once with offsets when the backend exposes them, else mark fallback needed."""
        if self._teacher_encode is None:
            raise TeacherContractError("cross-token teacher tokenizer capability was not initialized")
        try:
            encoded = self._teacher_encode(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
        except NotImplementedError:
            return self._teacher_tokenize_without_offsets(text, owner="joint teacher sequence"), None
        except TypeError as error:
            if "return_offsets_mapping" not in str(error):
                raise TeacherContractError(f"joint teacher tokenization failed: {error}") from error
            return self._teacher_tokenize_without_offsets(text, owner="joint teacher sequence"), None
        except Exception as error:  # noqa: BLE001 -- normalize arbitrary tokenizer backend failures
            raise TeacherContractError(f"joint teacher tokenization failed: {error}") from error

        raw_ids = self._tokenizer_field(encoded, "input_ids")
        if raw_ids is None:
            raise TeacherContractError("joint teacher tokenization returned no input_ids")
        if hasattr(raw_ids, "tolist"):
            raw_ids = raw_ids.tolist()
        if raw_ids and isinstance(raw_ids[0], (list, tuple)):
            if len(raw_ids) != 1:
                raise TeacherContractError("joint teacher tokenization unexpectedly returned a batch")
            raw_ids = raw_ids[0]
        token_ids = self._validate_token_ids(raw_ids, owner="joint teacher input_ids")

        raw_offsets = self._tokenizer_field(encoded, "offset_mapping")
        if raw_offsets is None:
            return token_ids, None
        if hasattr(raw_offsets, "tolist"):
            raw_offsets = raw_offsets.tolist()
        if (
            raw_offsets
            and len(raw_offsets) == 1
            and isinstance(raw_offsets[0], (list, tuple))
            and raw_offsets[0]
            and isinstance(raw_offsets[0][0], (list, tuple))
        ):
            raw_offsets = raw_offsets[0]
        if len(raw_offsets) != len(token_ids):
            raise TeacherContractError(
                f"joint teacher offset_mapping has {len(raw_offsets)} row(s) for {len(token_ids)} token(s)"
            )
        offsets: list[tuple[int, int]] = []
        for position, offset in enumerate(raw_offsets):
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise TeacherContractError(f"joint teacher offset_mapping[{position}] must be a pair")
            start, end = offset
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or end > len(text)
            ):
                raise TeacherContractError(
                    f"joint teacher offset_mapping[{position}]={offset!r} is outside the rendered text"
                )
            offsets.append((start, end))
        return token_ids, offsets

    @staticmethod
    def _offset_completion_start(offsets: list[tuple[int, int]], boundary: int) -> int:
        """Locate a token-edge completion boundary without assigning a crossing token."""
        for position, (start, end) in enumerate(offsets):
            if start < boundary < end:
                raise TeacherContractError(
                    f"joint teacher token {position} crosses the completion character boundary {boundary}"
                )
            if end > boundary:
                if start < boundary:
                    raise TeacherContractError(
                        f"joint teacher token {position} cannot be assigned wholly to the completion"
                    )
                if any(previous_end > boundary for _previous_start, previous_end in offsets[:position]):
                    raise TeacherContractError("joint teacher offsets are not ordered around the completion boundary")
                if any(next_start < boundary for next_start, _next_end in offsets[position:]):
                    raise TeacherContractError("joint teacher offsets return to the prefix after completion starts")
                return position
        return len(offsets)

    def _teacher_completion_tokens(
        self,
        rendered_prefix: str,
        completion_text: str,
    ) -> tuple[list[int], list[int], list[int]]:
        """Joint-tokenize and prove one exact teacher completion tail."""
        joint_text = rendered_prefix + completion_text
        joint_ids, offsets = self._teacher_joint_tokenization(joint_text)
        if not joint_ids:
            raise TeacherContractError("joint teacher prompt+completion tokenization is empty")
        if offsets is not None:
            completion_start = self._offset_completion_start(offsets, len(rendered_prefix))
        else:
            prefix_ids = self._teacher_tokenize_without_offsets(rendered_prefix, owner="teacher prefix")
            if joint_ids[: len(prefix_ids)] != prefix_ids:
                raise TeacherContractError(
                    "offset-free teacher fallback requires the joint token sequence to begin with the "
                    "separately encoded prefix exactly"
                )
            completion_start = len(prefix_ids)
        if completion_start < 1:
            raise TeacherContractError("teacher completion has no preceding prompt token to score it")
        teacher_completion_ids = joint_ids[completion_start:]
        if self.teacher_codec is None:
            raise TeacherContractError("teacher completion codec was not initialized")
        decoded_tail = self.teacher_codec.decode(teacher_completion_ids)
        if decoded_tail.encode("utf-8") != completion_text.encode("utf-8"):
            raise TeacherContractError(
                "joint teacher completion tail does not decode to the canonical completion byte-for-byte"
            )
        scored_indices = list(range(completion_start, len(joint_ids)))
        return joint_ids, teacher_completion_ids, scored_indices

    @staticmethod
    def _is_placeholder_collapse(student_ids: list[int], submitted_ids: list[int]) -> bool:
        """Return whether ``submitted_ids`` is ``student_ids`` with only repeats of a predecessor dropped.

        Collapsing an image placeholder run to one marker drops exactly such repeats. Re-rendering or
        re-tokenizing the prompt changes, inserts, or drops some other token and is refused.
        """
        cursor = 0
        for index, token_id in enumerate(student_ids):
            if cursor < len(submitted_ids) and submitted_ids[cursor] == token_id:
                cursor += 1
            elif index == 0 or student_ids[index - 1] != token_id:
                return False
        return cursor == len(submitted_ids)

    def _shared_vocab_request(
        self,
        transition: Transition,
        token_ids: list[int],
        *,
        prompt_token_count: int,
    ) -> dict[str, Any]:
        """Attach media to exact rollout IDs without re-rendering the prompt.

        Invariant: submitted = the student's prompt prefix with each image placeholder run collapsed
        to one marker, then the completion verbatim; scored = the student's ids. vLLM re-applies the
        Qwen3-VL placeholder replacement to token prompts, so an already-expanded prompt would come
        back ``N - 1`` tokens longer per image; the collapsed marker expands back to exactly the
        student's ids, which :meth:`_topk_engine_prompt_ids` proves before any position is scored.
        Qwen3-VL's ``get_pretokenized_rollout_input`` performs that collapse; the decoder-only fallback
        and packers whose engine does not re-expand return the ids untouched. Either way the hook may
        only drop placeholder repeats inside the prompt, never re-render or re-tokenize the sequence.
        """
        hook = getattr(self.packer, "get_pretokenized_rollout_input", None)
        if callable(hook):
            request = hook(transition.obs, list(token_ids))
        else:
            request = {"prompt_token_ids": list(token_ids)}
        if not isinstance(request, dict):
            raise TeacherContractError("get_pretokenized_rollout_input must return a vLLM request dictionary")
        request_ids = request.get("prompt_token_ids")
        if request_ids is None:
            raise TeacherContractError("teacher vLLM request is missing prompt_token_ids")
        submitted_ids = self._validate_token_ids(request_ids, owner="teacher request prompt_token_ids")
        completion_ids = token_ids[prompt_token_count:]
        submitted_prompt_count = len(submitted_ids) - len(completion_ids)
        if (
            submitted_prompt_count < 1
            or submitted_ids[submitted_prompt_count:] != completion_ids
            or not self._is_placeholder_collapse(token_ids[:prompt_token_count], submitted_ids[:submitted_prompt_count])
        ):
            raise TeacherContractError(
                "get_pretokenized_rollout_input must return the student's prompt with at most its image "
                "placeholder runs collapsed, followed by the completion verbatim; same-token OPD permits "
                "attaching media but never re-rendering or re-tokenizing the sequence"
            )
        return {**request, "prompt_token_ids": submitted_ids}

    def _cross_token_multimodal_request(
        self,
        transition: Transition,
        token_ids: list[int],
    ) -> tuple[dict[str, Any], int, int, int, int, tuple[int, ...], tuple[str, ...]]:
        """Attach ordered images with the teacher packer while preserving exact teacher IDs."""
        hook = getattr(self.teacher_packer, "get_cross_token_multimodal_teacher_input", None)
        if not callable(hook):
            raise TeacherContractError(
                "cross-token multimodal teacher packer exposes no get_cross_token_multimodal_teacher_input()"
            )
        try:
            prepared = hook(transition.obs, list(token_ids))
        except Exception as error:  # noqa: BLE001 -- normalize arbitrary packer/media failures
            raise TeacherContractError(f"cross-token multimodal teacher request shaping failed: {error}") from error
        request = getattr(prepared, "request", None)
        image_count = getattr(prepared, "image_count", None)
        image_token_id = getattr(prepared, "image_token_id", None)
        vision_start_token_id = getattr(prepared, "vision_start_token_id", None)
        vision_end_token_id = getattr(prepared, "vision_end_token_id", None)
        expected_image_marker_widths = getattr(prepared, "expected_image_marker_widths", None)
        image_uuids = getattr(prepared, "image_uuids", None)
        if not isinstance(request, dict):
            raise TeacherContractError("cross-token multimodal teacher packer returned no request dictionary")
        if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count < 0:
            raise TeacherContractError("cross-token multimodal teacher packer returned an invalid image count")
        if isinstance(image_token_id, bool) or not isinstance(image_token_id, int) or image_token_id < 0:
            raise TeacherContractError("cross-token multimodal teacher packer returned an invalid image token ID")
        if (
            isinstance(vision_start_token_id, bool)
            or not isinstance(vision_start_token_id, int)
            or vision_start_token_id < 0
        ):
            raise TeacherContractError("cross-token multimodal teacher packer returned an invalid vision-start ID")
        if isinstance(vision_end_token_id, bool) or not isinstance(vision_end_token_id, int) or vision_end_token_id < 0:
            raise TeacherContractError("cross-token multimodal teacher packer returned an invalid vision-end ID")
        if not isinstance(expected_image_marker_widths, tuple) or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in expected_image_marker_widths
        ):
            raise TeacherContractError("cross-token multimodal teacher packer returned invalid marker widths")
        if len(expected_image_marker_widths) != image_count:
            raise TeacherContractError("cross-token multimodal teacher image count and marker widths disagree")
        if not isinstance(image_uuids, tuple) or any(not _is_sha256(value) for value in image_uuids):
            raise TeacherContractError("cross-token multimodal teacher packer returned invalid image UUIDs")
        request_ids = request.get("prompt_token_ids")
        if request_ids is None:
            raise TeacherContractError("cross-token multimodal teacher request is missing prompt_token_ids")
        normalized_ids = self._validate_token_ids(request_ids, owner="multimodal teacher request prompt_token_ids")
        if normalized_ids != token_ids:
            raise TeacherContractError(
                "cross-token multimodal teacher packer changed the exact teacher-tokenizer sequence"
            )
        media = request.get("multi_modal_data", {})
        if not isinstance(media, Mapping) or set(media) - {"image"}:
            raise TeacherContractError("cross-token multimodal teacher request supports image media only")
        raw_images = media.get("image", [])
        if not isinstance(raw_images, (list, tuple)):
            raise TeacherContractError("cross-token multimodal teacher image payload must be an ordered sequence")
        if len(raw_images) != image_count or len(image_uuids) != image_count:
            raise TeacherContractError("cross-token multimodal teacher image payload/count/UUIDs disagree")
        mm_processor_kwargs = request.get("mm_processor_kwargs", {})
        if not isinstance(mm_processor_kwargs, Mapping):
            raise TeacherContractError("cross-token multimodal teacher mm_processor_kwargs must be a mapping")
        request_uuids = request.get("multi_modal_uuids", {})
        if image_count:
            if not isinstance(request_uuids, Mapping) or request_uuids.get("image") != list(image_uuids):
                raise TeacherContractError("cross-token multimodal teacher request UUIDs do not match ordered images")
        elif request_uuids not in ({}, None):
            raise TeacherContractError("text-only cross-token multimodal request must not carry media cache UUIDs")
        return (
            {**request, "prompt_token_ids": normalized_ids},
            image_count,
            image_token_id,
            vision_start_token_id,
            vision_end_token_id,
            expected_image_marker_widths,
            image_uuids,
        )

    @staticmethod
    def _validate_boundary_metadata(
        transition: Transition,
        *,
        prompt_token_count: int,
        completion_token_count: int,
        full_sequence_token_count: int,
    ) -> None:
        boundary = transition.policy_info.get("completion_boundary") if transition.policy_info else None
        if boundary is None:
            return  # legacy trajectories remain scoreable; live ranked rollout records this proof
        if not isinstance(boundary, Mapping):
            raise TeacherContractError("policy_info['completion_boundary'] must be a mapping")
        expected = {
            "prompt_token_count": prompt_token_count,
            "completion_token_count": completion_token_count,
            "full_sequence_token_count": full_sequence_token_count,
        }
        for key, value in expected.items():
            if boundary.get(key) != value:
                raise TeacherContractError(
                    f"policy_info['completion_boundary'][{key!r}]={boundary.get(key)!r}, expected {value}"
                )

    def _prepare_topk_batch(self, trajectories: list[Trajectory]) -> list[_PreparedTopKRequest]:
        prepared: list[_PreparedTopKRequest] = []
        seen_ids: set[str] = set()
        for trajectory_index, trajectory in enumerate(trajectories):
            if not isinstance(trajectory.id, str) or not trajectory.id:
                raise TeacherContractError(
                    f"teacher trajectory at batch index {trajectory_index} needs a non-empty string id"
                )
            if trajectory.id in seen_ids:
                raise TeacherContractError(f"teacher batch contains duplicate trajectory id {trajectory.id!r}")
            seen_ids.add(trajectory.id)
            if len(trajectory.transitions) != 1:
                raise TeacherContractError(
                    f"trajectory {trajectory.id!r} must contain exactly one transition, "
                    f"got {len(trajectory.transitions)}"
                )
            transition = trajectory.transitions[0]
            completion_ids = self._completion_ids(transition)
            n_ignore = int(transition.algo_extra.get("n_ignore_prefix_tokens", 0))
            # Live rollouts already carry the exact generation prefix. Derive only the scored span
            # from those IDs here; the packer attaches/validates media once in the request below.
            # Legacy trajectories without stored IDs retain the tokenizer-backed fallback.
            stored_prompt_ids = transition.policy_info.get("prompt_ids") if transition.policy_info else None
            span_packer = None if stored_prompt_ids is not None else self.packer
            policy_input = shape_policy_input(transition, span_packer, n_ignore)
            token_ids = self._validate_token_ids(policy_input.input_ids, owner="shaped policy input_ids")
            if len(completion_ids) > len(token_ids):
                raise TeacherContractError(
                    f"trajectory {trajectory.id!r} has {len(completion_ids)} completion ids but only "
                    f"{len(token_ids)} full-sequence ids"
                )
            prompt_token_count = len(token_ids) - len(completion_ids)
            if prompt_token_count < 1:
                raise TeacherContractError(f"trajectory {trajectory.id!r} has no prompt token")
            scored_indices = [index + 1 for index, keep in enumerate(policy_input.logprob_masks[:-1]) if keep]
            expected_scored = list(range(prompt_token_count + n_ignore, len(token_ids)))
            if scored_indices != expected_scored:
                raise TeacherContractError(
                    f"trajectory {trajectory.id!r} scored indices {scored_indices} do not exactly cover "
                    f"the completion boundary {expected_scored}"
                )
            self._validate_boundary_metadata(
                transition,
                prompt_token_count=prompt_token_count,
                completion_token_count=len(completion_ids),
                full_sequence_token_count=len(token_ids),
            )
            request = self._shared_vocab_request(transition, token_ids, prompt_token_count=prompt_token_count)
            try:
                media_identity = vllm_request_media_identity(
                    request,
                    context=f"trajectory {trajectory.id!r} teacher request",
                    require_all_content_uuids=_rank_world()[1] > 1,
                )
            except (TypeError, ValueError) as error:
                raise TeacherContractError(str(error)) from error
            prepared.append(
                _PreparedTopKRequest(
                    trajectory=trajectory,
                    request=request,
                    token_ids=token_ids,
                    scored_indices=scored_indices,
                    prompt_token_count=prompt_token_count,
                    completion_token_count=len(completion_ids),
                    n_ignore_prefix_tokens=n_ignore,
                    media_identity=media_identity,
                )
            )
        return prepared

    def _prepare_cross_token_batch(self, trajectories: list[Trajectory]) -> list[_PreparedDPCARequest]:
        """Shape a complete text-only or image-conditioned DPCA batch before vLLM."""
        if self.student_codec is None or self.teacher_codec is None:
            raise TeacherContractError("cross-token completion codecs were not initialized")
        self.student_codec.require_unchanged()
        self.teacher_codec.require_unchanged()

        prepared: list[_PreparedDPCARequest] = []
        seen_ids: set[str] = set()
        for trajectory_index, trajectory in enumerate(trajectories):
            if not isinstance(trajectory.id, str) or not trajectory.id:
                raise TeacherContractError(
                    f"teacher trajectory at batch index {trajectory_index} needs a non-empty string id"
                )
            if trajectory.id in seen_ids:
                raise TeacherContractError(f"teacher batch contains duplicate trajectory id {trajectory.id!r}")
            seen_ids.add(trajectory.id)
            if len(trajectory.transitions) != 1:
                raise TeacherContractError(
                    f"trajectory {trajectory.id!r} must contain exactly one transition, "
                    f"got {len(trajectory.transitions)}"
                )
            transition = trajectory.transitions[0]
            policy_info = transition.policy_info
            completion_ids = self._inline_student_token_ids(
                policy_info.get("completion_ids"),
                owner=f"trajectory {trajectory.id!r} policy_info['completion_ids']",
            )
            display_completion_text = policy_info.get("completion_text")
            if not isinstance(display_completion_text, str):
                raise TeacherContractError(
                    f"trajectory {trajectory.id!r} policy_info['completion_text'] must be a string"
                )
            rollout_logprobs = self._inline_rollout_logprobs(
                policy_info.get("rollout_logprobs"),
                expected=len(completion_ids),
                owner=f"trajectory {trajectory.id!r} policy_info['rollout_logprobs']",
            )
            student_decoded = self.student_codec.decode(completion_ids)
            # vLLM's display text intentionally hides stop/EOS special tokens while token_ids retain
            # them. DPCA aligns the realized sequence, so the fixed student codec is authoritative.
            completion_text = student_decoded

            rendered_prompt = self._render_teacher_prompt(transition)
            applied_think_prefix = self._think_prefix(completion_text)
            rendered_prefix = rendered_prompt + applied_think_prefix
            joint_ids, teacher_completion_ids, scored_indices = self._teacher_completion_tokens(
                rendered_prefix, completion_text
            )
            if len(teacher_completion_ids) != len(scored_indices):
                raise TeacherContractError("teacher completion IDs and score indices disagree")
            image_count = 0
            image_token_id: int | None = None
            vision_start_token_id: int | None = None
            vision_end_token_id: int | None = None
            expected_image_marker_widths: tuple[int, ...] = ()
            image_uuids: tuple[str, ...] = ()
            request: dict[str, Any] = {"prompt_token_ids": joint_ids}
            if self.input_mode == _CROSS_TOKEN_MULTIMODAL:
                (
                    request,
                    image_count,
                    image_token_id,
                    vision_start_token_id,
                    vision_end_token_id,
                    expected_image_marker_widths,
                    image_uuids,
                ) = self._cross_token_multimodal_request(transition, joint_ids)
            prepared.append(
                _PreparedDPCARequest(
                    trajectory=trajectory,
                    request=request,
                    token_ids=joint_ids,
                    scored_indices=scored_indices,
                    student_completion_ids=completion_ids,
                    teacher_completion_ids=teacher_completion_ids,
                    rollout_logprobs=rollout_logprobs,
                    image_count=image_count,
                    image_token_id=image_token_id,
                    vision_start_token_id=vision_start_token_id,
                    vision_end_token_id=vision_end_token_id,
                    expected_image_marker_widths=expected_image_marker_widths,
                    image_uuids=image_uuids,
                )
            )
        return prepared

    def _prepare_batch(
        self,
        trajectories: list[Trajectory],
    ) -> list[_PreparedTopKRequest] | list[_PreparedDPCARequest]:
        if self.input_mode == _SHARED_VOCAB_MULTIMODAL and self.signal_mode == _TOPK:
            return self._prepare_topk_batch(trajectories)
        return self._prepare_cross_token_batch(trajectories)

    def _support(self, entry: Any, position: int) -> tuple[list[int], list[float]]:
        """Validate and return the teacher's own top-k at one scored position."""
        if not isinstance(entry, Mapping) or not entry:
            raise TeacherContractError(f"teacher returned no distribution at position {position}")
        candidates: list[tuple[int, float]] = []
        seen_ids: set[int] = set()
        for raw_token_id, raw_logprob in entry.items():
            if isinstance(raw_token_id, bool) or not isinstance(raw_token_id, int):
                raise TeacherContractError(
                    f"teacher support id at position {position} must be a non-bool integer, got {raw_token_id!r}"
                )
            token_id = int(raw_token_id)
            if token_id in seen_ids:
                raise TeacherContractError(f"teacher returned duplicate support id {token_id} at position {position}")
            if token_id < 0 or (self.vocab_size is not None and token_id >= self.vocab_size):
                bound = "unbounded" if self.vocab_size is None else str(self.vocab_size)
                raise TeacherContractError(
                    f"teacher support id {token_id} at position {position} is outside [0, {bound})"
                )
            logprob = _entry_logprob(raw_logprob)
            if not math.isfinite(logprob) or logprob > _LOGPROB_TOLERANCE:
                raise TeacherContractError(
                    f"teacher logprob {logprob!r} at position {position}, token {token_id} is not finite/in range"
                )
            seen_ids.add(token_id)
            candidates.append((token_id, logprob))
        ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))[: self.top_k]
        if len(ranked) < self.top_k:
            raise TeacherContractError(
                f"teacher returned {len(ranked)} candidates at position {position}, fewer than "
                f"top_k={self.top_k}. Lower top_k, or raise the engine's max_logprobs "
                "(`vllm.LLM(..., max_logprobs=...)`; the OpenAI server flag is --max-logprobs)."
            )
        support_mass = math.fsum(math.exp(logprob) for _token_id, logprob in ranked)
        if not math.isfinite(support_mass) or support_mass <= 0.0 or support_mass > 1.0 + _SUPPORT_MASS_TOLERANCE:
            raise TeacherContractError(
                f"teacher top-k support mass at position {position} is {support_mass!r}, outside (0, 1]"
            )
        return [token_id for token_id, _logprob in ranked], [logprob for _token_id, logprob in ranked]

    def _parse_topk_outputs(
        self,
        prepared: list[_PreparedTopKRequest],
        outputs: Any,
    ) -> list[_ParsedTeacherSignal]:
        scorable = [item for item in prepared if item.scored_indices]
        try:
            output_list = list(outputs)
        except TypeError as error:
            raise TeacherContractError("teacher engine output is not iterable") from error
        if len(output_list) != len(scorable):
            raise TeacherContractError(
                f"teacher returned {len(output_list)} outputs for {len(scorable)} scored sequences"
            )
        parsed: list[_ParsedTeacherSignal] = []
        for item, output in zip(scorable, output_list):
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if prompt_logprobs is None:
                raise TeacherContractError(
                    "vLLM returned no prompt_logprobs; SamplingParams(prompt_logprobs=...) must be set"
                )
            engine_prompt_token_ids = self._topk_engine_prompt_ids(item, output)
            if len(prompt_logprobs) != len(engine_prompt_token_ids):
                raise TeacherContractError(
                    f"teacher returned {len(prompt_logprobs)} logprob entries for {len(engine_prompt_token_ids)} "
                    f"input tokens on trajectory {item.trajectory.id!r}; the response cannot be aligned"
                )
            ids: list[list[int]] = []
            logprobs: list[list[float]] = []
            for position in item.scored_indices:
                row_ids, row_logprobs = self._support(prompt_logprobs[position], position)
                ids.append(row_ids)
                logprobs.append(row_logprobs)
            topk_ids = torch.tensor(ids, dtype=torch.int32)  # [T,K]
            topk_logprobs = torch.tensor(logprobs, dtype=torch.float32)  # [T,K]
            finite = torch.isfinite(topk_logprobs)  # [T,K]
            expected_shape = (len(item.scored_indices), self.top_k)
            if tuple(topk_ids.shape) != expected_shape or tuple(topk_logprobs.shape) != expected_shape:
                raise TeacherContractError(
                    f"teacher signal for trajectory {item.trajectory.id!r} has shapes "
                    f"{tuple(topk_ids.shape)} and {tuple(topk_logprobs.shape)}, expected {expected_shape}"
                )
            if not bool(finite.all().item()):
                raise TeacherContractError(
                    f"teacher signal for trajectory {item.trajectory.id!r} contains non-finite logprobs"
                )
            parsed.append(
                _ParsedTeacherSignal(
                    trajectory=item.trajectory,
                    signal=TeacherSignal(topk_ids=topk_ids, topk_logprobs=topk_logprobs),
                    scored_tokens=len(item.scored_indices),
                    engine_prompt_token_ids=engine_prompt_token_ids,
                )
            )
        return parsed

    def _topk_engine_prompt_ids(self, item: _PreparedTopKRequest, output: Any) -> list[int]:
        """Prove vLLM's post-multimodal-processing prompt is exactly the student's token ids.

        The request carries each image placeholder run collapsed to one marker and the engine expands
        it back. ``scored_indices`` address the student's ids, so the expanded prompt must equal them
        position for position before ``prompt_logprobs`` may be indexed with them.
        """
        raw_output_ids = getattr(output, "prompt_token_ids", None)
        if raw_output_ids is None:
            if item.request.get("prompt_token_ids") != item.token_ids:
                raise TeacherContractError(
                    "same-token teacher requires vLLM output.prompt_token_ids to prove the collapsed image "
                    "markers expanded back to the student's token ids"
                )
            return list(item.token_ids)
        output_ids = self._validate_token_ids(raw_output_ids, owner="teacher output prompt_token_ids")
        if output_ids != item.token_ids:
            mismatch = next(
                (
                    index
                    for index, (engine_id, student_id) in enumerate(zip(output_ids, item.token_ids))
                    if engine_id != student_id
                ),
                min(len(output_ids), len(item.token_ids)),
            )
            raise TeacherContractError(
                f"teacher engine prompt has {len(output_ids)} tokens but the student sequence has "
                f"{len(item.token_ids)} on trajectory {item.trajectory.id!r}; first mismatch at index "
                f"{mismatch}. The submitted image markers must expand back to exactly the student's ids"
            )
        return output_ids

    @staticmethod
    def _collapse_image_marker_runs(
        token_ids: list[int],
        *,
        image_token_id: int | None,
        vision_start_token_id: int | None,
        vision_end_token_id: int | None,
        expected_widths: tuple[int, ...],
    ) -> tuple[list[int], list[int]]:
        if image_token_id is None or vision_start_token_id is None or vision_end_token_id is None:
            raise TeacherContractError("multimodal teacher request is missing image marker boundary IDs")
        collapsed: list[int] = []
        widths: list[int] = []
        index = 0
        while index < len(token_ids):
            token_id = token_ids[index]
            if token_id == image_token_id:
                raise TeacherContractError("multimodal teacher prefix contains an image token outside a vision block")
            if token_id == vision_end_token_id:
                raise TeacherContractError("multimodal teacher prefix contains a stray vision-end token")
            collapsed.append(token_id)
            index += 1
            if token_id != vision_start_token_id:
                continue
            if index >= len(token_ids) or token_ids[index] != image_token_id:
                raise TeacherContractError("multimodal teacher vision-start token is not followed by an image run")
            end = index
            while end < len(token_ids) and token_ids[end] == image_token_id:
                end += 1
            widths.append(end - index)
            collapsed.append(image_token_id)
            if end >= len(token_ids) or token_ids[end] != vision_end_token_id:
                raise TeacherContractError("multimodal teacher image run is not followed by vision-end")
            collapsed.append(vision_end_token_id)
            index = end + 1
        if tuple(widths) != expected_widths:
            raise TeacherContractError(
                f"multimodal teacher image marker widths {tuple(widths)} differ from expected {expected_widths}"
            )
        return collapsed, widths

    def _dpca_engine_prompt_ids(self, item: _PreparedDPCARequest, output: Any) -> list[int]:
        """Resolve vLLM's post-multimodal-processing IDs and prove the completion tail is unchanged."""
        raw_output_ids = getattr(output, "prompt_token_ids", None)
        if raw_output_ids is None:
            if self.input_mode == _CROSS_TOKEN_MULTIMODAL:
                raise TeacherContractError(
                    "multimodal DPCA requires vLLM output.prompt_token_ids to prove image expansion "
                    "stayed in the prefix"
                )
            return list(item.token_ids)
        output_ids = self._validate_token_ids(raw_output_ids, owner="teacher output prompt_token_ids")
        if self.input_mode != _CROSS_TOKEN_MULTIMODAL:
            if output_ids != item.token_ids:
                raise TeacherContractError("text-only teacher output prompt_token_ids differ from the submitted IDs")
            return output_ids
        completion_width = len(item.teacher_completion_ids)
        if completion_width < 1 or len(output_ids) <= completion_width:
            raise TeacherContractError("multimodal teacher output has no scored completion tail after its image prefix")
        if output_ids[-completion_width:] != item.teacher_completion_ids:
            raise TeacherContractError("multimodal teacher processing changed the realized completion token tail")
        submitted_prefix = item.token_ids[:-completion_width]
        collapsed_submitted, _submitted_widths = self._collapse_image_marker_runs(
            submitted_prefix,
            image_token_id=item.image_token_id,
            vision_start_token_id=item.vision_start_token_id,
            vision_end_token_id=item.vision_end_token_id,
            expected_widths=(1,) * item.image_count,
        )
        if collapsed_submitted != submitted_prefix:
            raise TeacherContractError(
                "multimodal teacher request must contain one logical image marker per ordered image"
            )
        realized_prefix = output_ids[:-completion_width]
        collapsed_realized, _realized_widths = self._collapse_image_marker_runs(
            realized_prefix,
            image_token_id=item.image_token_id,
            vision_start_token_id=item.vision_start_token_id,
            vision_end_token_id=item.vision_end_token_id,
            expected_widths=item.expected_image_marker_widths,
        )
        if collapsed_realized != submitted_prefix:
            raise TeacherContractError(
                "multimodal teacher realized prefix differs outside grid-derived image marker expansion"
            )
        return output_ids

    def _parse_dpca_outputs(
        self,
        prepared: list[_PreparedDPCARequest],
        outputs: Any,
    ) -> list[_ParsedTeacherSignal]:
        """Parse realized-token likelihoods and build exact student-coordinate chunk signals."""
        scorable = [item for item in prepared if item.scored_indices]
        try:
            output_list = list(outputs)
        except TypeError as error:
            raise TeacherContractError("teacher engine output is not iterable") from error
        if len(output_list) != len(scorable):
            raise TeacherContractError(
                f"teacher returned {len(output_list)} outputs for {len(scorable)} scored sequences"
            )
        if self.student_codec is None or self.teacher_codec is None:
            raise TeacherContractError("cross-token completion codecs were not initialized")

        parsed: list[_ParsedTeacherSignal] = []
        for item, output in zip(scorable, output_list):
            engine_prompt_token_ids = self._dpca_engine_prompt_ids(item, output)
            if self.input_mode == _CROSS_TOKEN_MULTIMODAL:
                completion_start = len(engine_prompt_token_ids) - len(item.teacher_completion_ids)
                scored_indices = list(range(completion_start, len(engine_prompt_token_ids)))
            else:
                scored_indices = item.scored_indices
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if prompt_logprobs is None:
                raise TeacherContractError(
                    "vLLM returned no prompt_logprobs; DPCA requires prompt_logprobs=0 realized-token scoring"
                )
            if len(prompt_logprobs) != len(engine_prompt_token_ids):
                raise TeacherContractError(
                    f"teacher returned {len(prompt_logprobs)} logprob entries for {len(engine_prompt_token_ids)} "
                    f"input tokens on trajectory {item.trajectory.id!r}; the response cannot be aligned"
                )
            realized_logprobs: list[float] = []
            for position in scored_indices:
                entry = prompt_logprobs[position]
                if not isinstance(entry, Mapping):
                    raise TeacherContractError(
                        f"teacher returned no realized-token logprob mapping at position {position}"
                    )
                realized_token_id = engine_prompt_token_ids[position]
                if realized_token_id not in entry:
                    raise TeacherContractError(
                        f"teacher realized token {realized_token_id} is absent from prompt_logprobs "
                        f"at position {position}"
                    )
                logprob = _entry_logprob(entry[realized_token_id])
                if not math.isfinite(logprob) or logprob > _LOGPROB_TOLERANCE:
                    raise TeacherContractError(
                        f"teacher realized-token logprob {logprob!r} at position {position} is not finite/in range"
                    )
                realized_logprobs.append(logprob)
            try:
                signal = build_dpca_signal(
                    item.student_completion_ids,
                    item.teacher_completion_ids,
                    realized_logprobs,
                    self.student_codec.decode,
                    self.teacher_codec.decode,
                    max_chunk_tokens=self.max_alignment_chunk_tokens,
                )
            except Exception as error:  # noqa: BLE001 -- normalize algorithm/tokenizer alignment failures
                raise TeacherContractError(
                    f"trajectory {item.trajectory.id!r} DPCA alignment failed: {error}"
                ) from error
            bounds = signal.chunk_bounds  # [C,2]
            chunk_logprobs = signal.chunk_teacher_logprob  # [C]
            if bounds is None or chunk_logprobs is None:
                raise TeacherContractError("DPCA builder returned a signal without its chunk pair")
            expected_start = 0
            for chunk_index, (start, end) in enumerate(bounds.tolist()):
                if start != expected_start or end <= start or end > len(item.student_completion_ids):
                    raise TeacherContractError(
                        f"trajectory {item.trajectory.id!r} DPCA chunk {chunk_index} does not form an exact cover"
                    )
                expected_start = end
            if expected_start != len(item.student_completion_ids):
                raise TeacherContractError(
                    f"trajectory {item.trajectory.id!r} DPCA chunks do not cover the full student completion"
                )
            parsed.append(
                _ParsedTeacherSignal(
                    trajectory=item.trajectory,
                    signal=signal,
                    scored_tokens=len(item.student_completion_ids),
                    engine_prompt_token_ids=engine_prompt_token_ids,
                    rollout_logprobs=tuple(item.rollout_logprobs),
                )
            )
        return parsed

    @staticmethod
    def _signal_digest(parsed: list[_ParsedTeacherSignal]) -> str:
        identity: list[dict[str, Any]] = []
        for item in parsed:
            ids = item.signal.topk_ids
            logprobs = item.signal.topk_logprobs
            bounds = item.signal.chunk_bounds
            chunk_logprobs = item.signal.chunk_teacher_logprob
            if ids is not None and logprobs is not None:
                payload = {
                    "topk_ids": ids.tolist(),
                    # Hex spelling preserves the exact float32 value; parsing already proved finite.
                    "topk_logprobs_hex": [[float(value).hex() for value in row] for row in logprobs.tolist()],
                }
            elif bounds is not None and chunk_logprobs is not None:
                payload = {
                    "chunk_bounds": bounds.tolist(),
                    "chunk_teacher_logprob_hex": [float(value).hex() for value in chunk_logprobs.tolist()],
                }
            else:
                raise TeacherContractError("ranked teacher parser produced an incomplete signal payload")
            identity.append(
                {
                    "trajectory_id": item.trajectory.id,
                    "engine_prompt_token_ids": item.engine_prompt_token_ids,
                    **payload,
                }
            )
        return canonical_json_sha256(identity, context="ranked teacher signal identity")

    def _score_with_retry(
        self,
        prepared: list[_PreparedTopKRequest] | list[_PreparedDPCARequest],
    ) -> tuple[list[_ParsedTeacherSignal], str, int]:
        scorable = [item for item in prepared if item.scored_indices]
        if not scorable:
            empty_digest = canonical_json_sha256([], context="empty ranked teacher signal identity")
            return [], empty_digest, 0
        engine_inputs = [item.request for item in scorable]
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                outputs = run_rank_converged(
                    lambda: self.model.generate(engine_inputs, self._sampling_params()),
                    f"ranked teacher engine scoring attempt {attempt}",
                )
                if self.signal_mode == _TOPK:
                    parsed = run_rank_converged(
                        lambda: self._parse_topk_outputs(prepared, outputs),
                        f"ranked teacher top-k output parsing attempt {attempt}",
                    )
                else:
                    parsed = run_rank_converged(
                        lambda: self._parse_dpca_outputs(prepared, outputs),
                        f"ranked teacher DPCA output parsing attempt {attempt}",
                    )
                signal_digest = run_rank_converged(
                    lambda: self._signal_digest(parsed),
                    f"ranked teacher signal digest attempt {attempt}",
                )
                run_rank_converged(
                    lambda: _require_identical_digest(signal_digest, context="ranked teacher signal"),
                    f"ranked teacher signal convergence attempt {attempt}",
                )
                return parsed, signal_digest, attempt
            except Exception as error:  # noqa: BLE001 -- every rank retries the converged attempt together
                last_error = error
                if attempt < self.max_attempts and self.retry_backoff_s:
                    time.sleep(self.retry_backoff_s)
        raise TeacherContractError(
            f"ranked teacher scoring failed after {self.max_attempts} replica-wide attempt(s): {last_error}"
        ) from last_error

    def _make_batch_stats(
        self,
        parsed: list[_ParsedTeacherSignal],
        attempts: int,
    ) -> dict[str, float | int]:
        """Build bounded tensor-free diagnostics for the most recent successful batch."""
        scored_tokens = sum(item.scored_tokens for item in parsed)
        stats: dict[str, float | int] = {
            "train/opd_teacher_score_request_count": attempts,
            "train/opd_teacher_score_success_count": int(attempts > 0),
            "train/opd_teacher_score_failure_count": max(0, attempts - 1),
            "train/opd_teacher_scored_sequences": len(parsed),
            "train/opd_teacher_scored_tokens": scored_tokens,
        }
        if self.signal_mode == _TOPK:
            support_masses: list[float] = []
            for item in parsed:
                topk_logprobs = item.signal.topk_logprobs  # [T,K]
                if topk_logprobs is None:
                    raise TeacherContractError("top-k diagnostics received a signal without topk_logprobs")
                support_masses.extend(
                    math.fsum(math.exp(float(logprob)) for logprob in row) for row in topk_logprobs.tolist()
                )
            stats["train/opd_topk_support_mass_mean"] = (
                math.fsum(support_masses) / len(support_masses) if support_masses else 0.0
            )
            return stats

        chunk_lengths: list[int] = []
        teacher_chunk_logprobs: list[float] = []
        student_chunk_logprobs: list[float] = []
        covered_student_tokens = 0
        for item in parsed:
            bounds = item.signal.chunk_bounds  # [C,2]
            chunk_logprobs = item.signal.chunk_teacher_logprob  # [C]
            if bounds is None or chunk_logprobs is None:
                raise TeacherContractError("DPCA diagnostics received a signal without its chunk pair")
            rollout_logprobs = item.rollout_logprobs
            if rollout_logprobs is None or len(rollout_logprobs) != item.scored_tokens:
                raise TeacherContractError(
                    "DPCA diagnostics received rollout logprobs that no longer match the scored completion"
                )
            for start, end in bounds.tolist():
                width = int(end) - int(start)
                chunk_lengths.append(width)
                covered_student_tokens += width
                student_chunk_logprobs.append(math.fsum(rollout_logprobs[int(start) : int(end)]))
            teacher_chunk_logprobs.extend(float(value) for value in chunk_logprobs.tolist())
        chunk_count = len(chunk_lengths)
        stats.update(
            {
                "train/opd_dpca_chunk_count": chunk_count,
                "train/opd_dpca_mean_chunk_length": (math.fsum(chunk_lengths) / chunk_count if chunk_count else 0.0),
                "train/opd_dpca_max_chunk_length": max(chunk_lengths, default=0),
                "train/opd_dpca_exact_alignment_coverage": (
                    covered_student_tokens / scored_tokens if scored_tokens else 1.0
                ),
                "train/opd_dpca_chunk_teacher_logprob_mean": (
                    math.fsum(teacher_chunk_logprobs) / chunk_count if chunk_count else 0.0
                ),
                "train/opd_dpca_chunk_student_logprob_mean": (
                    math.fsum(student_chunk_logprobs) / chunk_count if chunk_count else 0.0
                ),
            }
        )
        return stats

    def _request_identity(
        self,
        prepared: list[_PreparedTopKRequest] | list[_PreparedDPCARequest],
    ) -> dict[str, Any]:
        """Build the exact pre-engine identity shared by training and observation calls."""
        return {
            "version": 3,
            "input_mode": self.input_mode,
            "signal_mode": self.signal_mode,
            "logprobs_mode": self.logprobs_mode,
            "sampling": self._sampling_kwargs(),
            "cross_token_contract": (
                {
                    "media_contract": (
                        {"preserve": ["image"], "reject": ["video", "video_frames"]}
                        if self.input_mode == _CROSS_TOKEN_MULTIMODAL
                        else {"strip": sorted(_MEDIA_CONTENT_TYPES)}
                    ),
                    "add_generation_prompt": True,
                    "enable_thinking": False,
                    "normalized_template_think_prefix": _QWEN_DISABLED_THINK_PREFIX,
                    "conditional_think_prefix": self.conditional_think_prefix,
                    "max_alignment_chunk_tokens": self.max_alignment_chunk_tokens,
                    "student_codec": self.student_codec.identity if self.student_codec is not None else None,
                    "teacher_codec": self.teacher_codec.identity if self.teacher_codec is not None else None,
                }
                if self.signal_mode == _DPCA
                else None
            ),
            "trajectories": [item.identity() for item in prepared],
        }

    def batch_stats(self) -> dict[str, float | int]:
        """Consume tensor-free diagnostics from the latest successful :meth:`score` call."""
        if self._last_batch_stats is None:
            raise RuntimeError("no completed teacher report; score must succeed first")
        stats = self._last_batch_stats
        self._last_batch_stats = None
        return dict(stats)

    def score(self, trajectories: list[Trajectory]) -> TeacherScoreResult:
        """Attach one valid teacher signal per non-empty trajectory, atomically on rank 0."""
        self._last_batch_stats = None
        prepared = run_rank_converged(
            lambda: self._prepare_batch(trajectories),
            "ranked teacher batch preparation",
        )
        request_identity = self._request_identity(prepared)
        request_digest = run_rank_converged(
            lambda: _ordered_request_digest(request_identity),
            "ranked teacher request digest",
        )
        run_rank_converged(
            lambda: _require_identical_digest(request_digest, context="ranked teacher request"),
            "ranked teacher request convergence",
        )
        parsed, signal_digest, attempts = self._score_with_retry(prepared)
        rank, world_size = _rank_world()
        scored_tokens = sum(item.scored_tokens for item in parsed)
        self._last_batch_stats = self._make_batch_stats(parsed, attempts)
        if rank == 0:
            # Commit only after every row, every trajectory, and every TP lane has succeeded. Until
            # this loop, retries and failures leave the caller's trajectories byte-for-byte untouched.
            for item in parsed:
                item.trajectory.transitions[0].targets = item.signal
            return trajectories
        return TeacherRankStatus(
            rank=rank,
            world_size=world_size,
            request_digest=request_digest,
            signal_digest=signal_digest,
            attempts=attempts,
            scored_sequences=len(parsed),
            scored_tokens=scored_tokens,
        )
