# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Diffusion-time inference cache for Cosmos3 VFM.

Based on SeaCache: "Spectral-Evolution-Aware Cache for Accelerating Diffusion
Models" (https://arxiv.org/pdf/2602.18993).  See :class:`DiffusionCache`.

Installs three hooks:

* ``model.generate_samples_from_batch`` — resets cache state and adopts the
  call's local diffusion-step count (so ``cutoff_from_end`` is correct when
  samples use fewer steps than the install-time max).
* ``model.denoise`` — tracks step / sample / CFG-pass, computes the SEA
  indicator, and decides skip-vs-full.
* ``net.language_model.forward`` — on full, caches und output as-is and the gen
  residual (``out - in``); on skip, returns the cached und and ``input + gen_residual``.

The decode heads run after ``language_model`` in ``net.forward``, so they re-run
every step and a skipped step still reflects the fresh input.  Caching is
auto-disabled for autoregressive / KV-cache generation (see ``_disables_caching``);
the static text-K/V reuse of ``InferenceTextKVMemoryState`` stays compatible, with
skipped steps reproducing its gen-only output shape (see ``_is_gen_only``).

The SEA indicator and step tracker use noisy vision only
(``_batch_supports_diffusion_cache``). Joint video+sound/action still caches;
packs without noisy vision disable it.

Under FSDP ``dp_shard > 1`` each rank denoises a different sample, so the skip
decision is reduced across the ``dp_shard`` group (run a full eval if *any* rank
needs one) to keep the collective ``language_model`` all-gather deadlock-free.

"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import torch

from cosmos_framework.utils import log
from cosmos_framework.model.generator.mot.inference_text_kv_memory import InferenceTextKVMemoryState

CalculationType: TypeAlias = Literal["full", "cache"]
"""Per-step decision: either run the network in full (and refresh the
cache) or serve the most recent cached prediction."""

DenoiseOutput: TypeAlias = dict[str, Any]
"""Output of ``model.denoise``: ``preds_vision`` (``list[Tensor[C,T,H,W]]``),
optional ``preds_action`` / ``preds_sound``, plus passthrough
``lbl_metadata_*`` entries."""


# -----------------------------------------------------------------------------
# helper functions: rank-0 logging, denoise-call tracking.
# -----------------------------------------------------------------------------


def _is_rank0() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _resolve_dp_shard_group(parallel_dims: Any) -> Any:
    """Resolve the FSDP ``dp_shard`` process group, or ``None`` when disabled.

    Returns the 1-D ``dp_shard`` sub-mesh's process group when
    ``parallel_dims.dp_shard_enabled`` (``dp_shard > 1``); otherwise ``None``,
    in which case the skip decision stays purely local.  The group is used to
    reduce the skip-vs-full decision across dp_shard ranks so the collective
    ``language_model`` parameter all-gather can never deadlock.
    """
    if parallel_dims is None or not getattr(parallel_dims, "dp_shard_enabled", False):
        return None
    mesh = getattr(parallel_dims, "dp_shard_mesh", None)
    if mesh is None:
        return None
    return mesh.get_group()


def _modality_has_noisy_tokens(modality: Any) -> bool:
    """Whether ``modality`` has noised tokens (non-empty ``timesteps``).

    Packing only records timesteps for noised tokens, so a non-empty tensor is
    the inference-time signal that this modality is under diffusion.
    """
    if modality is None:
        return False
    timesteps = getattr(modality, "timesteps", None)
    return isinstance(timesteps, torch.Tensor) and timesteps.numel() > 0


def _batch_supports_diffusion_cache(data_batch_packed: Any) -> bool:
    """Whether this denoise batch can drive vision-based SeaCache decisions.

    Requires noisy vision (``vision.timesteps``). Action-/sound-only or
    conditioning-only vision packs are unsupported; joint vision+action /
    vision+sound is supported via the noisy vision branch.
    """
    if data_batch_packed is None:
        return False
    vision = getattr(data_batch_packed, "vision", None)
    return _modality_has_noisy_tokens(vision)


def _unsupported_modality_reason(data_batch_packed: Any) -> str:
    """Short reason string for the unsupported-modality disable log."""
    if data_batch_packed is None:
        return "missing packed batch"
    vision = getattr(data_batch_packed, "vision", None)
    action = getattr(data_batch_packed, "action", None)
    sound = getattr(data_batch_packed, "sound", None)
    if _modality_has_noisy_tokens(action) and not _modality_has_noisy_tokens(vision):
        return "action diffusion without noisy vision"
    if _modality_has_noisy_tokens(sound) and not _modality_has_noisy_tokens(vision):
        return "sound diffusion without noisy vision"
    if vision is None:
        return "no vision modality"
    return "vision not under diffusion (conditioning-only or empty)"


def _extract_timestep_key(data_batch_packed: Any) -> float | None:
    """Best-effort scalar key for the current **vision** diffusion timestep.

    Used to detect step boundaries (timestep changes) and sample boundaries
    (timestep direction reversals) from successive ``denoise`` calls.
    Returns ``None`` if the vision timestep cannot be located.  Callers must
    gate on ``_batch_supports_diffusion_cache`` first so action-/sound-only
    packs never reach :meth:`_DenoiseStepTracker.advance` with ``None``.
    """
    if data_batch_packed is None:
        return None
    vision = getattr(data_batch_packed, "vision", None)
    if vision is None:
        return None
    timesteps = getattr(vision, "timesteps", None)
    if not isinstance(timesteps, torch.Tensor) or timesteps.numel() == 0:
        return None
    return float(timesteps.flatten()[0].item())


def _disables_caching(memory: Any) -> bool:
    """Whether this ``MemoryState`` makes residual caching unsafe.

    Autoregressive / KV-cache generation carries per-frame state across denoise
    calls, so the language_model must run every step.  ``InferenceTextKVMemoryState``
    is the exception: it only reuses the *static* text K/V within a single diffusion
    request (see ``inference_text_kv_memory.py``) and is populated by the forced-full
    warmup step, so skipping later steps neither reads nor corrupts it.
    """
    return memory is not None and not isinstance(memory, InferenceTextKVMemoryState)


def _resolve_generation_num_steps(model: Any, kwargs: dict[str, Any]) -> int:
    """Resolve the local diffusion-step count for one ``generate_samples_from_batch`` call.

    Mirrors ``OmniMoTModel.generate_samples_from_batch``: prefer the call's
    ``num_steps`` kwarg (default 35), but when a ``FixedStepSampler`` is active
    use ``len(t_list) - 1`` instead.  Duck-typed on ``t_list`` + class name to
    avoid importing the sampler module into the cache.
    """
    num_steps = int(kwargs.get("num_steps", 35))
    sampler = kwargs.get("sampler")
    if sampler is None:
        sampler = getattr(model, "sampler", None)
    t_list = getattr(sampler, "t_list", None)
    if t_list is not None and type(sampler).__name__ == "FixedStepSampler":
        num_steps = len(t_list) - 1
    return num_steps


@dataclass(slots=True)
class _DenoiseStepTracker:
    """Tracks step / sample / CFG-pass boundaries from successive denoise calls.

    Assumes a flow-matching sampler that decreases the timestep
    monotonically within a sample (1 → 0); an upward jump in timestep
    indicates the previous sample finished and a new one started.  Repeated
    calls at the same timestep are treated as successive CFG passes within the
    step and indexed positionally (``pass_idx`` 0, 1, ...).  The passes are
    kept deliberately generic: which physical branch (conditional /
    unconditional) maps to which index depends on the sampler and on CFG
    parallelism, so the tracker only guarantees that a given index consistently
    identifies the same pass within a run, not that index 0 is "conditional".
    """

    last_timestep_key: float | None = None
    step: int = -1
    pass_idx: int = 0  # positional CFG-pass index within the current step

    def advance(self, timestep_key: float | None) -> tuple[bool, bool]:
        """Advance the tracker by one denoise call (mutates internal state).

        Returns ``(is_new_step, is_new_sample)`` describing what kind of
        boundary, if any, this call crossed.  A repeated ``timestep_key``
        is interpreted as the next CFG pass within the current step (no
        boundary); a fresh value steps the counter forward, and an upward
        jump additionally flags a new sample.

        ``timestep_key`` must be a resolved **vision** timestep.  A ``None``
        (no ``vision.timesteps``) is a programming error and raises
        ``ValueError`` up front rather than being silently absorbed; the denoise
        hook must not call ``advance`` unless ``_batch_supports_diffusion_cache``
        is true.  Exact float equality identifies same-step CFG passes, which is
        reliable because the sampler reuses the identical timestep tensor across
        the CFG passes of a step.
        """
        if timestep_key is None:
            raise ValueError("_DenoiseStepTracker.advance requires a resolved timestep, got None")

        # First-ever call: no previous timestep to compare against.
        if self.last_timestep_key is None:
            self.last_timestep_key = timestep_key
            self.step += 1
            self.pass_idx = 0
            return True, False

        # Same timestep ⇒ another CFG pass within the current step.
        if timestep_key == self.last_timestep_key:
            self.pass_idx += 1
            return False, False

        # Different timestep ⇒ a new step; an upward jump means the previous
        # sample finished (flow-matching decreases the timestep 1 → 0).
        is_new_sample = timestep_key > self.last_timestep_key
        self.last_timestep_key = timestep_key
        self.step += 1
        self.pass_idx = 0
        return True, is_new_sample

    def reset_for_new_sample(self) -> None:
        """Re-seed the step counter for a new sample (timestep history kept)."""
        self.step = 0

    @property
    def pass_name(self) -> str:
        # Generic positional key ("cfg0", "cfg1";
        return f"cfg{self.pass_idx}"


# -----------------------------------------------------------------------------
# language_model residual cache + Spectral-Evolution-Aware Wiener filter.
# -----------------------------------------------------------------------------
LMCacheEntry: TypeAlias = tuple[torch.Tensor | None, torch.Tensor]
"""Cached language_model state ``(und_out, gen_delta)`` in hidden space.

* ``und_out`` — absolute und/causal hidden states, reused verbatim on a skip
  (und is not a diffusion prediction). ``None`` when the forward ran gen-only
  (``memory.is_gen_only()``) and emitted a length-0 und split.
* ``gen_delta`` — SeaCache residual ``out_full - in_full``, optionally
  extrapolated across steps via ``residual_order``.
"""


def _is_gen_only(memory: Any) -> bool:
    """Whether this forward skips the und pathway and emits a length-0 und sequence.

    Mirrors ``_impl_forward``'s ``memory.is_gen_only()`` gate: once the request-local
    text K/V are populated, the decoder layers reuse them and return an empty und
    split, so cached-and-reused steps must reproduce that same shape.
    """
    return memory is not None and bool(memory.is_gen_only())


def _lm_cache_entry(
    in_causal: torch.Tensor,
    in_full: torch.Tensor,
    out_causal: torch.Tensor,
    out_full: torch.Tensor,
) -> LMCacheEntry:
    """Snapshot und as-is and the gen residual from a full language_model eval."""
    # Gen-only forwards emit a length-0 und split; there is nothing absolute to store.
    und_out = None if out_causal.shape != in_causal.shape else out_causal.detach().clone()
    return (und_out, (out_full - in_full).detach().clone())


def _cache_entry_matches(
    entry: LMCacheEntry,
    in_causal: torch.Tensor,
    in_full: torch.Tensor,
    gen_only: bool,
) -> bool:
    """Whether a cached entry can be reused for the current language_model input.

    A gen-only step needs only the gen delta (its und output is empty regardless);
    a standard step additionally needs a stored und tensor of the matching shape.
    """
    und_out, gen_delta = entry
    if gen_delta.shape != in_full.shape:
        return False
    if gen_only:
        return True
    return und_out is not None and und_out.shape == in_causal.shape


def _extrapolate_gen(history: list[tuple[int, LMCacheEntry]], step: int, order: int) -> torch.Tensor:
    """Extrapolate the generation residual to ``step`` from cached full-eval entries.

    ``history`` is the recent ``(step_index, (und_out, gen_delta))`` full evals,
    chronological.  A Newton polynomial (divided differences) is fit through the last
    ``order + 1`` gen deltas at their actual (non-uniform) step indices and evaluated
    at ``step``.  ``order == 0`` (or too little history) reduces to constant reuse of
    the last gen delta (plain SeaCache).  Only the generation split is extrapolated;
    und is reused verbatim by the caller.
    """
    k = max(0, min(order, len(history) - 1))
    window = history[-(k + 1) :]
    steps = [s for s, _ in window]
    # Newton divided-difference table over the gen deltas (in-place, high→low index).
    coeffs = [gen.clone() for _, (_und, gen) in window]
    for level in range(1, k + 1):
        for j in range(k, level - 1, -1):
            coeffs[j] = (coeffs[j] - coeffs[j - 1]) / float(steps[j] - steps[j - level])
    # Horner evaluation of the Newton form at ``step``.
    result = coeffs[k]
    for j in range(k - 1, -1, -1):
        result = result * float(step - steps[j]) + coeffs[j]
    return result


def apply_sea_from_ab(
    x: torch.Tensor,
    a: float,
    b: float,
    power_exp: float = 2.0,
    power_const: float = 1.0,
    dims: tuple[int, ...] | None = None,
    eps: float = 1e-16,
) -> torch.Tensor:
    """Apply an N-D separable Spectral-Evolution-Aware Wiener filter.

    The filter gain per axis is ``H = a·Sx0 / (a²·Sx0 + b² + eps)`` where
    ``Sx0 = power_const / (|f|^power_exp + eps)`` is a ``1/f``-style prior on
    the clean-signal power spectrum and ``(a, b)`` are the timestep-dependent
    signal / noise mixing coefficients (see :meth:`DiffusionCache._sea_ab`).
    Low frequencies (small ``|f|``, large ``Sx0``) are preserved while high
    frequencies are suppressed; the gain is then normalized to unit mean, as in
    the SeaCache paper, leaving the indicator's overall scale intact.

    ``dims`` selects the axes to filter (e.g. videos ``T,H,W`` = ``(-4,-3,-2)``
    for a ``[T,H,W,C]`` tensor); remaining axes (batch, channel) pass through.
    """
    orig_dtype = x.dtype
    x32 = x.contiguous().to(torch.float32)

    if dims is None:
        dims = tuple(range(x32.ndim)) if x32.ndim <= 2 else tuple(range(-2, -x32.ndim, -1))

    X = torch.fft.fftn(x32, dim=dims)

    # Build the separable N-D gain as a product of per-axis 1D gains.
    H: torch.Tensor | None = None
    for ax in dims:
        f = torch.fft.fftfreq(x32.shape[ax], device=x32.device, dtype=torch.float32)
        Sx0 = power_const / ((torch.abs(f) ** power_exp) + eps)
        H1 = (a * Sx0) / (a * a * Sx0 + (b * b) + eps)

        shape_i = [1] * x32.ndim
        shape_i[ax] = H1.shape[0]
        H = H1.reshape(shape_i) if H is None else (H * H1.reshape(shape_i))

    assert H is not None

    # Normalize to unit mean gain so filtered-feature energy is comparable across
    # timesteps (SeaCache density normalization).
    normv = torch.mean(H)
    if torch.isfinite(normv) and normv > 0:
        H = H / normv

    return torch.fft.ifftn(X * H, dim=dims).real.to(orig_dtype)


def rel_l1_dist(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-16) -> float:
    """Relative L1 distance ``mean|a-b| / (mean|b| + eps)`` as a Python float."""
    num = (a - b).abs().mean()
    den = b.abs().mean() + eps
    return float((num / den).detach().cpu())


# -----------------------------------------------------------------------------
# DiffusionCache: language_model-residual SeaCache.
# -----------------------------------------------------------------------------


class DiffusionCache:
    """Diffusion-time inference cache based on
    "SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models"
    (https://arxiv.org/pdf/2602.18993).

    Skips ``net.language_model`` on steps whose accumulated, SEA-filtered
    relative-L1 change stays below ``diffusion_cache_thresh``; on a skip the cached
    und hidden is reused as-is and the cached gen residual is re-applied to the
    current step's fresh gen input, so the decode heads still produce an
    input-adaptive prediction.  Three hooks are
    installed (see :meth:`install`), with no model-source changes:

    * ``model.generate_samples_from_batch`` resets cache state and adopts the
      call's local ``num_steps`` (needed for ``cutoff_from_end``).
    * ``model.denoise`` decides skip-vs-full per call (step / sample / CFG-pass
      tracking + SEA indicator + accumulated rel-L1).
    * ``net.language_model.forward`` executes it: on a full step cache und as-is
      and ``gen_out - gen_in``; on a skip return the cached und and
      ``gen_in + gen_residual``.

    Each CFG pass of a step is tracked as an independent pathway, keyed
    positionally (``cfg0``, ``cfg1``).
    Caching is disabled for AR generation memory states (see ``_disables_caching``)
    and for packs without noisy vision (see ``_batch_supports_diffusion_cache``).
    """

    @dataclass(frozen=True, slots=True)
    class Config:
        diffusion_cache_thresh: float = 0.25
        """Accumulated relative-L1 budget before a full eval is forced (shared across
        the per-step CFG-pass pathways).  Larger ⇒ more skipping ⇒ faster but
        lower fidelity."""
        residual_order: int = 1
        """Polynomial order for extrapolating the *generation* residual on a skipped step
        via Newton divided differences over the cached full-eval residuals:
        0 = constant reuse (as in SeaCache), 1 = linear, 2 = quadratic."""
        ret_steps: int = 1
        """Retention: always run full for the first ``ret_steps`` steps (warmup)."""
        cutoff_from_end: int = 1
        """Always run full for the last ``cutoff_from_end`` steps (0 disables)."""
        max_consecutive_cached: int = 2
        """Maximum consecutive residual reuses per CFG pathway before forcing a
        full evaluation. ``0`` disables the limit."""
        power_exp: float = 3.0
        """Exponent of the ``1/|f|^power_exp`` clean-signal power prior in the SEA filter."""
        timestep_max: float = 1000.0
        """Divisor used to map a raw timestep (>1) to sigma∈(0,1) for the SEA (a,b)."""

        def __post_init__(self) -> None:
            if self.residual_order < 0:
                raise ValueError(f"residual_order must be >= 0, got {self.residual_order}")
            if (
                isinstance(self.max_consecutive_cached, bool)
                or not isinstance(self.max_consecutive_cached, int)
                or self.max_consecutive_cached < 0
            ):
                raise ValueError(
                    f"max_consecutive_cached must be a nonnegative integer, got {self.max_consecutive_cached}"
                )

        @classmethod
        def from_overrides(cls, overrides: dict[str, Any] | None) -> "DiffusionCache.Config":
            if not overrides:
                return cls()
            unknown = set(overrides) - {f for f in cls.__dataclass_fields__}
            if unknown:
                raise ValueError(f"Unsupported diffusion-time inference cache config keys: {sorted(unknown)}")
            return cls(**overrides)

    @dataclass(slots=True)
    class _PathwayState:
        """Per-CFG-pass (``cfg0`` / ``cfg1`` ) diffusion-time inference cache bookkeeping.

        ``history`` holds the most recent full-eval cache entries as
        ``(step_index, (und_out, gen_delta))``, chronologically, bounded to the
        length the configured ``residual_order`` needs.  Und is reused verbatim
        from the last entry; the gen residual is extrapolated across the
        (non-uniform) step indices.
        """

        accumulated: float = 0.0
        prev_indicator: list[torch.Tensor] | None = None
        history: list[tuple[int, LMCacheEntry]] = field(default_factory=list)
        consecutive_cached: int = 0

    @dataclass(slots=True)
    class State:
        step: int = 0
        num_steps: int = 0
        calc_type: CalculationType = "full"
        pathway: str = "cfg0"
        """CFG-pass key for the denoise call in flight (``cfg0`` / ``cfg1``)."""

    def __init__(self, num_steps: int, config: dict[str, Any] | None = None) -> None:
        self.num_steps = num_steps
        self.config = self.Config.from_overrides(config)
        self.state = self.State(num_steps=num_steps)
        self._tracker = _DenoiseStepTracker()
        # Per-pathway ("cfg0" / "cfg1") diffusion-time inference cache state.
        self._pathways: dict[str, DiffusionCache._PathwayState] = {}
        # FSDP ``dp_shard`` process group. When set (dp_shard > 1), the skip
        # decision is reduced across the group so every rank runs / skips the
        # language_model in lockstep (see :meth:`_synchronize_compute`).
        self._dp_shard_group: Any = None
        # True only when this denoise call may skip / refresh residual history.
        # Cleared for unsupported modalities and AR memory before ``language_model``.
        self._cache_active = False
        # Per-sample step counters (reset on sample boundaries and end-of-run).
        self._step_full = 0
        self._step_skipped = 0
        # Survives reset(): the "caching disabled" notice is per-run, not per-sample.
        self._disabled_notice_logged = False

    def install(self, pipe: Any) -> None:
        """Install the diffusion-time inference cache hooks on ``pipe.model``.

        ``model.generate_samples_from_batch`` is patched to call
        :meth:`begin_generation` with the call's local step count;
        ``model.denoise`` is patched to drive a :class:`_DenoiseStepTracker`
        (step / sample / CFG-pass boundaries), compute the SEA indicator, and
        record the skip-vs-full decision; ``net.language_model.forward`` is patched
        to execute that decision via residual reuse.  No model source is modified.
        """
        model = getattr(pipe, "model", None)
        if model is None:
            raise ValueError("Inference pipe has no model; cannot apply diffusion cache")
        if getattr(model, "_diffusion_cache_installed", False):
            raise RuntimeError("A diffusion cache is already installed on this model")
        net = getattr(model, "net", None)
        language_model = getattr(net, "language_model", None) if net is not None else None
        if language_model is None:
            raise ValueError("Model has no net.language_model; cannot apply diffusion cache")
        if not hasattr(model, "generate_samples_from_batch"):
            raise ValueError("Model has no generate_samples_from_batch; cannot apply diffusion cache")

        self._dp_shard_group = _resolve_dp_shard_group(getattr(model, "parallel_dims", None))

        original_generate = model.generate_samples_from_batch
        original_denoise = model.denoise
        original_lm_forward = language_model.forward

        def patched_generate(self_model: Any, *args: Any, **kwargs: Any) -> Any:
            self.begin_generation(_resolve_generation_num_steps(self_model, kwargs))
            return original_generate(*args, **kwargs)

        def patched_denoise(self_model: Any, *args: Any, **kwargs: Any) -> Any:
            del self_model
            batch = kwargs.get("data_batch_packed")
            memory = kwargs.get("memory")
            # Default off until the vision-diffusing path proves caching is safe.
            self._cache_active = False

            # AR / KV-cache first: bypass before any vision.timestep tracking so a
            # missing vision timestep cannot raise before the documented disable.
            if _disables_caching(memory):
                self.state.calc_type = "full"
                self._log_disabled_once(
                    f"denoise carries {type(memory).__name__} "
                    "(autoregressive / KV-cache generation runs the language_model every step)"
                )
                return original_denoise(*args, **kwargs)

            if not _batch_supports_diffusion_cache(batch):
                # Action-/sound-only (or conditioning-only vision): do not read
                # vision.timesteps / vision.tokens for step tracking or SEA.
                self.state.calc_type = "full"
                self._log_disabled_once(_unsupported_modality_reason(batch))
                return original_denoise(*args, **kwargs)

            tk = _extract_timestep_key(batch)
            is_new_step, is_new_sample = self._tracker.advance(tk)
            if is_new_step:
                if is_new_sample or self._tracker.step >= self.num_steps:
                    self.reset()
                    self._tracker.reset_for_new_sample()
                self.state.step = self._tracker.step

            pathway = self._tracker.pass_name
            self.state.pathway = pathway
            self._cache_active = True
            indicator = self._extract_indicator(batch, tk)
            compute = self._should_compute(pathway, indicator)
            compute = self._synchronize_compute(compute, pathway)
            if compute:
                self.state.calc_type = "full"
                self._step_full += 1
            else:
                self.state.calc_type = "cache"
                self._step_skipped += 1
            return original_denoise(*args, **kwargs)

        def patched_lm_forward(self_lm: Any, pack: Any, *args: Any, **kwargs: Any) -> Any:
            del self_lm
            memory = kwargs.get("memory")
            pathway = self.state.pathway
            ps = self._pathways.setdefault(pathway, self._PathwayState())
            in_causal, in_full = pack["causal_seq"], pack["full_only_seq"]
            caching = self._cache_active and not _disables_caching(memory)
            gen_only = caching and _is_gen_only(memory)

            reuse = (
                caching
                and self.state.calc_type == "cache"
                and ps.history
                and _cache_entry_matches(ps.history[-1][1], in_causal, in_full, gen_only)
            )
            if reuse:
                ps.consecutive_cached += 1
                und_out = ps.history[-1][1][0]  # understanding: absolute reuse (not a residual)
                gen_delta = _extrapolate_gen(ps.history, self.state.step, self.config.residual_order)
                out_pack = dict(pack)
                # Match the shape the real forward would have produced: gen-only layers
                # emit an empty und split instead of an updated one.
                out_pack["causal_seq"] = in_causal.new_empty(0, in_causal.shape[-1]) if gen_only else und_out
                out_pack["full_only_seq"] = in_full + gen_delta
                return out_pack, {}

            outputs = original_lm_forward(pack, *args, **kwargs)
            if caching:
                ps.consecutive_cached = 0
                out_pack = outputs[0] if isinstance(outputs, tuple) else outputs
                entry = _lm_cache_entry(in_causal, in_full, out_pack["causal_seq"], out_pack["full_only_seq"])
                ps.history.append((self.state.step, entry))
                max_history = self.config.residual_order + 1
                if len(ps.history) > max_history:
                    ps.history = ps.history[-max_history:]
            return outputs

        model.generate_samples_from_batch = types.MethodType(patched_generate, model)
        model.denoise = types.MethodType(patched_denoise, model)
        language_model.forward = types.MethodType(patched_lm_forward, language_model)
        model._diffusion_cache_installed = True
        model._diffusion_cache = self

    def begin_generation(self, num_steps: int) -> None:
        """Reset state for a generation and use its local diffusion-step count."""
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        self.reset()
        self.num_steps = num_steps
        self.state = self.State(num_steps=num_steps)
        self._tracker = _DenoiseStepTracker()

    def reset(self) -> None:
        """Drop all cached state.  Called on sample boundaries and end-of-run."""
        self._log_sample_summary()
        self._pathways = {}
        self.state = self.State(num_steps=self.num_steps)
        self._cache_active = False
        self._step_full = 0
        self._step_skipped = 0

    # ----- SeaCache decision --------------------------------------------------

    def _should_compute(self, pathway: str, indicator: list[torch.Tensor] | None) -> bool:
        """Decide whether to run the language_model (vs reuse its residual).

        Runs full during the retention / cutoff windows, when no cached residual or
        indicator is available, or once the accumulated relative-L1 change of the SEA
        indicator exceeds ``diffusion_cache_thresh``.  Otherwise the language_model is skipped
        and its cached residual reused.  Updates the accumulator / stored indicator.
        """
        ps = self._pathways.setdefault(pathway, self._PathwayState())
        step = self.state.step
        cfg = self.config

        max_consecutive_forced = bool(
            cfg.max_consecutive_cached and ps.consecutive_cached >= cfg.max_consecutive_cached
        )
        forced = (
            step < cfg.ret_steps
            or step >= self.num_steps - cfg.cutoff_from_end
            or max_consecutive_forced
            or not ps.history
            or indicator is None
            or ps.prev_indicator is None
        )
        if forced:
            ps.accumulated = 0.0
            ps.prev_indicator = indicator
            return True

        distance = self._indicator_distance(indicator, ps.prev_indicator)
        ps.accumulated += distance
        ps.prev_indicator = indicator
        if ps.accumulated < cfg.diffusion_cache_thresh:
            return False
        ps.accumulated = 0.0
        return True

    def _synchronize_compute(self, compute: bool, pathway: str) -> bool:
        """Reduce the skip-vs-full decision across the FSDP ``dp_shard`` group.

        Under FSDP ``dp_shard > 1`` each rank denoises a different sample, so
        their per-rank skip decisions can diverge.  The ``language_model``
        parameter all-gather is collective, so a rank that skips (no all-gather)
        while a peer computes (all-gather) would deadlock.  We make the decision
        global with an OR reduction: run a full eval if *any* rank needs one, and
        skip only when *all* ranks agree to skip — the conservative choice that
        never reuses a residual a peer considers stale.

        When a rank is overridden from skip to full it performs a real eval and
        refreshes its residual, so its accumulator is reset to 0 like any other
        full step.  No-op (returns ``compute`` unchanged) when ``dp_shard`` is
        disabled.
        """
        group = self._dp_shard_group
        if group is None:
            return compute
        device = "cuda" if torch.cuda.is_available() else "cpu"
        flag = torch.tensor([1 if compute else 0], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX, group=group)
        global_compute = bool(flag.item() > 0)
        if global_compute and not compute:
            # Locally wanted to skip but a peer forces a full eval; this refreshes
            # our residual, so reset the accumulator like any other full step.
            self._pathways.setdefault(pathway, self._PathwayState()).accumulated = 0.0
        return global_compute

    def _indicator_distance(self, cur: list[torch.Tensor] | None, prev: list[torch.Tensor] | None) -> float:
        """Mean per-sample relative-L1 distance between two SEA indicators.

        Returns ``inf`` (⇒ force a full eval) when the indicators are missing
        or their per-sample shapes disagree, so a shape change can never be
        silently cached over.
        """
        if cur is None or prev is None or len(cur) != len(prev) or not cur:
            return float("inf")
        total = 0.0
        for c, p in zip(cur, prev, strict=True):
            if c.shape != p.shape:
                return float("inf")
            total += rel_l1_dist(c, p)
        return total / len(cur)

    def _extract_indicator(self, data_batch_packed: Any, timestep_key: float | None) -> list[torch.Tensor] | None:
        """SEA-filtered noisy-latent indicator, one filtered tensor per sample.

        Reads the per-sample noisy vision latents ``[C,T,H,W]`` from the packed
        batch, moves the channel to the last axis, and applies the SEA Wiener
        filter over the ``(T,H,W)`` axes.  Returns ``None`` (⇒ force a full
        eval) when vision latents are unavailable.
        """
        vision = getattr(data_batch_packed, "vision", None)
        if vision is None:
            return None
        tokens = getattr(vision, "tokens", None)
        shapes = getattr(vision, "token_shapes", None)
        if tokens is None or not shapes:
            return None

        if timestep_key is None:
            raise ValueError(
                "Diffusion cache requires a resolved timestep to build the SEA indicator, "
                "but none could be located in the denoise batch."
            )
        a, b = self._sea_ab(timestep_key)
        filtered: list[torch.Tensor] = []
        for latent in tokens:
            if not isinstance(latent, torch.Tensor):
                return None
            lat = latent.squeeze(0) if latent.dim() == 5 else latent  # [C,T,H,W]
            if lat.dim() != 4:
                return None
            thwc = lat.movedim(0, -1)  # [T,H,W,C] — leave channel unfiltered
            filt = apply_sea_from_ab(
                thwc,
                a,
                b,
                power_exp=self.config.power_exp,
                dims=(-4, -3, -2),
            )
            filtered.append(filt)
        return filtered

    def _sea_ab(self, timestep_key: float) -> tuple[float, float]:
        """Flow-matching signal / noise mixing coefficients ``(a, b) = (1-σ, σ)``.

        The timestep is interpreted as ``σ``; raw timesteps ``> 1`` are scaled
        by ``timestep_max`` into ``(0, 1)`` and the result is clamped away from
        the singular endpoints.  A missing (``None``) timestep is a programming
        error and raises ``ValueError`` rather than being silently substituted
        with a neutral filter; callers must resolve the timestep beforehand.
        """
        if timestep_key is None:
            raise ValueError("_sea_ab requires a resolved timestep, got None")
        eps = 1e-6
        sigma = float(timestep_key)
        if sigma > 1.0:
            sigma = sigma / self.config.timestep_max
        sigma = max(eps, min(1.0 - eps, sigma))
        return 1.0 - sigma, sigma

    # ----- logging --------------------------------------------------------

    def _log_disabled_once(self, reason: str) -> None:
        """Warn once that caching is bypassed for this run (AR memory or modality).

        Without this the run looks identical to a baseline run with no diagnostics
        at all, since the skipped/full counters never advance.
        """
        if self._disabled_notice_logged:
            return
        self._disabled_notice_logged = True
        if _is_rank0():
            log.warning(f"[Diffusion Cache] disabled for this run: {reason}")

    def _log_sample_summary(self) -> None:
        if not _is_rank0() or (self._step_full + self._step_skipped) == 0:
            return
        log.info(f"[Diffusion Cache] sample done: full={self._step_full} skipped={self._step_skipped}")


def install_diffusion_cache(
    pipe: Any,
    enabled: bool,
    sample_args_list: list[Any],
    config_overrides: dict[str, Any] | None = None,
) -> DiffusionCache | None:
    """Install the SeaCache language_model-residual cache on ``pipe.model``.

    When ``enabled`` is ``False`` this is a no-op and returns ``None``.
    When ``True`` a :class:`DiffusionCache` is constructed (using the
    defaults in :class:`DiffusionCache.Config` unless overridden via
    ``config_overrides``), installed, and returned.  The cache is also
    accessible after installation via ``pipe.model._diffusion_cache``.

    FSDP ``dp_shard > 1`` is supported: different ``dp_shard`` ranks denoise
    different samples, so :meth:`DiffusionCache._synchronize_compute` reduces the
    skip decision across the ``dp_shard`` group (run a full eval if *any* rank
    needs one), keeping the collective ``language_model`` all-gather deadlock-free.
    Context / tensor / CFG parallelism need no reduction (their inputs are
    replicated across the collective group, so the decision is already identical
    on every rank).
    """
    if not enabled:
        return None

    max_num_steps = max((int(getattr(s, "num_steps", 1)) for s in sample_args_list), default=1)
    cache = DiffusionCache(num_steps=max_num_steps, config=config_overrides)
    cache.install(pipe)
    cfg = cache.config
    log.info(
        f"Enabled diffusion cache (diffusion-time inference cache) "
        f"max_num_steps={max_num_steps} "
        f"diffusion_cache_thresh={cfg.diffusion_cache_thresh} ret_steps={cfg.ret_steps} "
        f"cutoff_from_end={cfg.cutoff_from_end} "
        f"max_consecutive_cached={cfg.max_consecutive_cached} power_exp={cfg.power_exp} "
        f"residual_order={cfg.residual_order}"
    )
    return cache
