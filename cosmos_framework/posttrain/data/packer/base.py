# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""DataPacker contract + model_type registry (design 2026-07-28).

A DataPacker is the single, model-agnostic, weight-free owner of raw-sample ->
model-ready-tensor shaping (tokenization, packing, collation). Rollout bodies and
algorithms call it instead of tokenizing/tensorizing inline. Ported from cosmos-rl
`cosmos_rl/dispatcher/data/packer/base.py`, minus the multi-turn / tool-agent path.

Second shaping seam, for objectives whose unit is not a token sequence: a batch-assembly FUNCTION
over `Trajectory` lists, handed to the objective instead of registered by `model_type`
(`preference.py` for DPO's winner/loser sides; `gr00t.py` for GR00T-N1.7 flow-BC, where Isaac's own
collator owns tokenization, patchifying and padding). Same invariants as a DataPacker: weight-free,
CPU-side, one owner per model x objective. Use a DataPacker when the unit is a token sequence (the
abstract methods below assume prompt ids and a padded max_len); otherwise a batch-assembly function.
DESIGN.md v3 "Batch-assembly seam" is the architecture statement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from cosmos_framework.posttrain.schema import ObsBundle, SFTTarget

if TYPE_CHECKING:
    from cosmos_framework.posttrain.data.packer.sft import SFTPolicyInput

_FUTURE = "SFT on this packer is future work; use data.packer.sft.build_sft_packer"


def collapse_contiguous_token_runs(token_ids: list[int], token_id: int | None) -> list[int]:
    """Collapse each contiguous run of ``token_id`` to one token.

    Multimodal packers apply this only to the processor-owned prompt prefix when their serving
    engine re-expands logical media markers. Completion tokens must remain outside this helper.
    """
    exact_ids = [int(value) for value in token_ids]
    if token_id is None:
        return exact_ids
    return [
        value
        for index, value in enumerate(exact_ids)
        if value != token_id or index == 0 or exact_ids[index - 1] != token_id
    ]


@runtime_checkable
class PackerSemanticsIdentity(Protocol):
    """Declared generation semantics of a configured packer instance.

    Held-out validation (``roles/rollout/validation.py``) binds every packer by class source hash
    and REQUIRES this declaration from any instance that carries configured fields, because those
    fields shape prompts and media without leaving a trace in the source. The declaration is
    JSON-compatible and part of the exact-resume identity: a change to any declared value refuses
    to resume a pending validation.
    """

    def validation_semantics_identity(self) -> dict[str, Any]: ...


@runtime_checkable
class PackerProcessorIdentity(Protocol):
    """Immutable processor contract of a multimodal packer.

    Consumed by the OPD cross-token teacher identity (``config/build.py``), which must refuse to
    resume when the teacher tokenizer, chat template, image processor, or video processor changed.
    """

    def processor_semantics_identity(self) -> dict[str, Any]: ...


class BaseDataPacker(ABC):
    """Root shaping contract + a class-level `model_type -> packer class` registry."""

    # Registered classes are `DataPacker`s, not merely `BaseDataPacker`s: the registry exists to
    # serve `init_data_packer`, which calls `setup(name_or_path)` on whatever it selects, and
    # `setup` is declared on `DataPacker`. A bare `BaseDataPacker` subclass would type-check here
    # and then `AttributeError` at setup time.
    _REGISTRY: dict[str, type[DataPacker]] = {}

    @classmethod
    def register(
        cls, model_types: str | list[str], packer_cls: type[DataPacker], *, allow_override: bool = False
    ) -> None:
        """Register `packer_cls` as the default for one or more HF `model_type`s.

        Args:
            model_types: a `model_type` string or list of them.
            packer_cls: the `DataPacker` subclass to select for those types (it must carry `setup`,
                which `init_data_packer` calls on the selected packer).
            allow_override: permit replacing an existing distinct registration.

        Raises:
            ValueError: a different class is already registered and `allow_override` is False.
        """
        types = [model_types] if isinstance(model_types, str) else list(model_types)
        for mt in types:
            existing = BaseDataPacker._REGISTRY.get(mt)
            if not allow_override and existing is not None and existing is not packer_cls:
                raise ValueError(f"data packer for model_type {mt!r} already registered ({existing.__name__})")
            BaseDataPacker._REGISTRY[mt] = packer_cls

    @classmethod
    def get_default_data_packer(cls, model_type: str) -> DataPacker:
        """Instantiate the registered default packer for `model_type`.

        Raises:
            ValueError: no packer is registered for `model_type`.
        """
        if model_type not in BaseDataPacker._REGISTRY:
            raise ValueError(f"no data packer registered for model_type {model_type!r}")
        return BaseDataPacker._REGISTRY[model_type]()

    # --- rollout side ---
    @abstractmethod
    def get_rollout_input(self, item: ObsBundle) -> Any: ...

    def get_pretokenized_rollout_input(self, item: ObsBundle, token_ids: list[int]) -> dict[str, Any]:
        """Attach model-owned side inputs to an exact pretokenized vLLM request.

        Text models need only the ids. Multimodal packers override this hook to
        attach the original media handles without re-rendering the prompt; when
        the engine re-expands media placeholders on token prompts (Qwen3-VL), the
        override submits each placeholder run collapsed to one marker so the
        expansion reproduces exactly ``token_ids``, keeping the completion tail
        verbatim. The same-token teacher and deterministic validation paths
        share this seam.
        """
        del item
        return {"prompt_token_ids": [int(token_id) for token_id in token_ids]}

    def rollout_collate_fn(self, items: list) -> list:
        return items

    def get_rollout_output(
        self, completions: list, token_ids: list | None = None, logprobs: list | None = None, **kw
    ) -> list:
        # Optional output-shaping hook (token decode / logprob attach); default passthrough.
        # NO external offload: posttrain's codec owns the metadata/bulk split (design §0).
        return completions

    # --- policy (RL) side ---
    @abstractmethod
    def get_policy_input(
        self,
        sample: ObsBundle,
        rollout_output: Any,
        n_ignore_prefix_tokens: int = 0,
        *,
        prompt_ids: list[int] | None = None,
    ) -> Any:
        """Shape policy inputs, treating supplied rollout prompt ids as authoritative."""
        ...

    @abstractmethod
    def policy_compute_max_len(self, processed_samples: list) -> int: ...

    @abstractmethod
    def policy_collate_fn(self, processed_samples: list, computed_max_len: int) -> dict: ...

    # --- SFT side (implemented by data.packer.sft; optional for other packers) ---
    def sft_process_sample(self, sample: SFTTarget) -> SFTPolicyInput:
        raise NotImplementedError(_FUTURE)

    def sft_compute_max_len(self, processed_samples: list[SFTPolicyInput]) -> int:
        raise NotImplementedError(_FUTURE)

    def sft_collate_fn(
        self, sub_batch: list[SFTPolicyInput], computed_max_len: int, ignore_label_id: int = -100
    ) -> dict[str, Any]:
        raise NotImplementedError(_FUTURE)

    # --- misc hooks ---
    def batch_size(self, batch) -> int:
        return len(batch)

    def slice_batch(self, batch, start, end):
        return batch[start:end]

    def save_state(self, dest_path: str) -> None:
        pass


class DataPacker(BaseDataPacker, ABC):
    """A tokenizer-backed `BaseDataPacker`. `setup` builds the HF tokenizer once."""

    def setup(self, name_or_path: str, trust_remote_code: bool = False) -> None:
        """Build `self.tokenizer` from `name_or_path`; fall back pad->eos if unset.

        Args:
            name_or_path: HF model id/path the tokenizer is loaded from.
            trust_remote_code: execute custom tokenizer code shipped by the repo. Defaults to
                FALSE, matching `HFModelConfig.trust_remote_code` — enabling the packer must not
                silently widen what a run is allowed to execute. Opt in explicitly per model.

        Raises:
            ValueError: `name_or_path` is empty (cannot build a tokenizer).
        """
        from transformers import AutoTokenizer

        if not name_or_path:
            raise ValueError("DataPacker.setup: empty name_or_path — cannot build a tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def adds_special_tokens(self, probe: str = "abc 123") -> bool:
        """Whether this tokenizer adds special tokens (e.g. a BOS) at `add_special_tokens=True`.

        The non-raising predicate behind `assert_no_special_tokens`. `VLLMRollout` uses THIS (and
        warns) rather than the assert, because a BOS-adding tokenizer is no longer a correctness
        failure on that path — see `assert_no_special_tokens` for the full argument.

        Args:
            probe: a short string used to compare the two tokenization modes.

        Returns:
            True if `add_special_tokens=True` and `=False` disagree on the probe.
        """
        with_special = self.tokenizer(probe, add_special_tokens=True).input_ids
        without = self.tokenizer(probe, add_special_tokens=False).input_ids
        return list(with_special) != list(without)

    def assert_no_special_tokens(self, probe: str = "abc 123") -> None:
        """Raise if this tokenizer adds special tokens. NOT called on any production path.

        HISTORY — why this is no longer enforced at vLLM construction. The guard was written when
        the objective ALWAYS re-tokenized the prompt: vLLM tokenizes the prompt string internally
        while `get_policy_input` re-tokenizes with `add_special_tokens=False`, and those agree only
        for a tokenizer that adds nothing, so a BOS-adding tokenizer (llama) meant generating from
        one prefix and scoring under another. That premise no longer holds — the objective now
        PREFERS `policy_info["prompt_ids"]` (design 2026-07-28 §10), and `VLLMRollout` records
        `req.prompt_token_ids`, vLLM's OWN tokenization, so the scored prefix is exactly the
        generated one whatever the tokenizer adds. Enforcing it at construction only blocked
        `llama`, which `data/packer/__init__.py` explicitly REGISTERS as supported.

        It is not enforced on the objective's fallback branch either: a producer that records no
        `prompt_ids` has no engine tokenization to disagree with, so the packer's re-derivation is
        by definition the prefix — raising there would wrongly reject synthetic/test producers.

        Kept as an explicit opt-in for callers that genuinely need the stricter promise (e.g. a
        future path that feeds vLLM prompt strings AND scores via re-tokenization).

        Args:
            probe: a short string used to compare the two tokenization modes.

        Raises:
            ValueError: the tokenizer adds special tokens on the probe.
        """
        if self.adds_special_tokens(probe):
            raise ValueError(
                "packer tokenizer adds special tokens (add_special_tokens=True != False); the "
                "vLLM+packer path scores completions under a different prompt prefix than vLLM "
                "generated from. Use a no-BOS tokenizer (e.g. Qwen) or feed vLLM pre-tokenized "
                "prompts (spec §9.6)."
            )

    def save_state(self, dest_path: str) -> None:
        self.tokenizer.save_pretrained(dest_path)
