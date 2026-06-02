# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Abstract interfaces for persistent memory in the MoT transformer stack.

``MemoryState`` is a *mutable* Python object that lives **outside** the
``torch.compile`` boundary.  It is responsible for reading cached tensors
(``read_for_layer``) and writing new tensors back (``write_for_layer``).

``MemoryValue`` is a *read-only* tensor container that is safe to pass
**into** a compiled decoder layer.  Concrete implementations are plain
dataclasses whose fields are tensors (or None).  No methods on
``MemoryValue`` should mutate state.

``KVToStore`` is a type alias for the 4-tuple of tensors
``(gen_k, gen_v, und_k, und_v)`` returned by each compiled layer so
the caller can write them back into the cache outside the compile boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

# (gen_k, gen_v, und_k, und_v) returned by each compiled layer for the caller
# to write back into the cache outside the torch.compile boundary.
KVToStore = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class MemoryValue(ABC):
    """Read-only tensor container safe to pass into ``torch.compile``.

    Concrete subclasses (e.g. ``ARMemoryValue``, ``KVTrainMemoryValue``)
    are plain dataclasses of tensors.  No methods on this class should
    mutate state or perform non-trivial computation.
    """

    @property
    def supports_context_parallel_attention(self) -> bool:
        """Whether this memory value is compatible with context-parallel attention.

        Overridden by ``KVTrainMemoryValue`` to return ``False``.  Used by
        ``ContextParallelDispatch`` to reject an unsupported combination
        without importing the concrete subclass.
        """
        return True


class MemoryState(ABC):
    """Mutable persistent memory that lives outside ``torch.compile``.

    The outer loop in ``_impl_forward`` calls ``read_for_layer`` before
    each decoder layer and ``write_for_layer`` after.  The ``MemoryState``
    object itself is **never** passed into a compiled region.
    """

    @abstractmethod
    def init(self, hidden_states: dict, device: torch.device) -> None:
        """Initialization per training step.

        Called once before any transformer layers are processed.

        Args:
            hidden_states: The packed sequence (``FactoredSequencePack``).
            device: Target device for any new tensors.
        """

    @abstractmethod
    def read_for_layer(self, layer_idx: int) -> MemoryValue:
        """Produce a read-only tensor snapshot for *layer_idx*.

        Used to retrieve KV values from the cache.
        The returned ``MemoryValue`` is passed into the compiled decoder
        layer as ``memory_value``.
        """

    @abstractmethod
    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        """Store the K/V tensors produced by *layer_idx* back into the cache.

        Called outside the ``torch.compile`` boundary.
        """

    @abstractmethod
    def is_gen_only(self) -> bool:
        """Return ``True`` when only the generation pathway should run.

        When ``True``, the decoder layer assumes that the text caption has
        already been processed and cached in the MemoryState object.
        Used for autoregressive frame-by-frame generation of video.
        """

    def requires_natten_metadata(self) -> bool:
        """Whether the packed-sequence builder should create NATTEN metadata.

        Memory paths whose attention implementation handles temporal
        visibility itself return ``False``.
        """
        return True

    def is_und_only(self) -> bool:
        """Return ``True`` when only the understanding (reasoner) pathway should run.

        When ``True``, the decoder layer runs the reasoner prefill: it computes
        the understanding pathway over the text tokens and caches the per-layer
        K/V, skipping the generation pathway entirely. Defaults to ``False`` so
        existing memory states (which only toggle gen-only) are unaffected.
        """
        return False


@dataclass
class ReasonerMemoryValue(MemoryValue):
    """Read-only per-layer understanding (reasoner) K/V snapshot.

    Carries the post-RoPE understanding-pathway keys/values cached during the
    one-time reasoner prefill, so the generator-only denoise pass can attend to
    them without recomputing the understanding pathway. Shapes follow the
    ``KVToStore`` contract produced by ``PackedAttentionMoT.forward``:
    ``[1, und_len, num_kv_heads, head_dim]``.
    """

    # ``None`` only during the prefill pass (cache not yet populated); the
    # generator-only attention path asserts these are present before use.
    und_k: torch.Tensor | None
    und_v: torch.Tensor | None


class ReasonerMemoryState(MemoryState):
    """Per-layer understanding-K/V cache for the reasoner/generator split.

    Single-sample inference only (the offloaded single-GPU path). Drives a
    one-time understanding prefill followed by generator-only denoise steps via
    a three-valued ``mode``:

    - ``"prefill"`` (``is_und_only()`` is ``True``): each decoder layer runs the
      reasoner pathway only and ``write_for_layer`` stores the understanding K/V.
    - ``"gen"`` (``is_gen_only()`` is ``True``): each decoder layer runs the
      generation pathway only and ``read_for_layer`` returns the cached
      understanding K/V for the gen->und cross-attention.
    - ``None`` (default): both ``is_und_only()`` and ``is_gen_only()`` are
      ``False`` so the decoder runs the joint forward.

    The understanding tokens are the (fixed) text prompt and the understanding
    pathway attends causally over understanding tokens only, so its per-layer
    K/V are independent of the diffusion timestep and are valid across all
    denoise steps.
    """

    _VALID_MODES = (None, "prefill", "gen")

    def __init__(self, num_layers: int) -> None:
        self._num_layers = num_layers
        self._und_k: list[torch.Tensor | None] = [None] * num_layers
        self._und_v: list[torch.Tensor | None] = [None] * num_layers
        self._mode: str | None = None

    def init(self, hidden_states: dict, device: torch.device) -> None:  # noqa: D401 - see base
        # Nothing to allocate up front; entries are filled by write_for_layer.
        return None

    def set_mode(self, mode: str | None) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"Invalid ReasonerMemoryState mode {mode!r}; expected one of {self._VALID_MODES}.")
        self._mode = mode

    def is_gen_only(self) -> bool:
        return self._mode == "gen"

    def is_und_only(self) -> bool:
        return self._mode == "prefill"

    @property
    def is_initialized(self) -> bool:
        return all(k is not None for k in self._und_k)

    def read_for_layer(self, layer_idx: int) -> ReasonerMemoryValue:
        # Never raises: the decoder loop calls this on every layer, including
        # during the prefill pass when the cache is still empty. During prefill
        # the returned (None) K/V are ignored by the attention dispatch (the
        # generation sequence is empty, so the und-only prefill path runs); only
        # the generator-only path consumes the cached K/V (and asserts they exist).
        return ReasonerMemoryValue(und_k=self._und_k[layer_idx], und_v=self._und_v[layer_idx])

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        # kv_to_store == (gen_k, gen_v, und_k, und_v); only the understanding
        # K/V are persisted, and only during the prefill pass. On generator-only
        # steps the understanding sequence is empty, so skip (keep the cache).
        if self._mode != "prefill":
            return
        _, _, und_k, und_v = kv_to_store
        self._und_k[layer_idx] = und_k
        self._und_v[layer_idx] = und_v
