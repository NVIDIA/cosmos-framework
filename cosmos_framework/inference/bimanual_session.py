# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""One eager Bimanual rollout with decoded feedback and request-owned state."""

from __future__ import annotations

import copy
import uuid
import weakref
from contextlib import ExitStack
from typing import Any

import torch
import torch.distributed as dist

from cosmos_framework.model.generator.utils.action_stream import ARHistorySnapshot, StreamedActionChunk


def require_idle_bimanual_model(model: Any) -> None:
    owner = getattr(model, "_active_bimanual_session", None)
    if isinstance(owner, weakref.ReferenceType):
        session = owner()
        if session is not None and not session.closed:
            raise RuntimeError("Close the active Bimanual session before starting another generation")


class BimanualSession:
    """Use ``OmniInference.open_bimanual_session`` to prepare an initial image.

    Actions are already normalized schema-5 sidecars, one complete latent chunk
    at a time. Only the latest decoded RGB chunk is held here; the model owns
    retained latent history. Close explicitly or use a ``with`` block.
    """

    def __init__(self, model: Any, data_batch: dict[str, Any], *, domain_name: str, fps: int, seed: int = 0):
        require_idle_bimanual_model(model)
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError(
                "Bimanual sessions currently require one process; distributed control delivery is unsupported"
            )
        if (
            not model.config.video_temporal_causal
            or model.config.compile.enabled
            or model.config.compile.use_cuda_graphs
        ):
            raise ValueError("Bimanual sessions require a causal model with torch.compile and CUDA graphs disabled")
        if model.config.fixed_step_sampler_config is None:
            raise ValueError("Bimanual sessions require a distilled checkpoint with its fixed sampling schedule")
        self.contract = getattr(model, "_bimanual_action_contract", None)
        if self.contract is None or self.contract.model_action_dim != model.config.max_action_dim:
            raise ValueError("Bimanual sessions require the checkpoint's matching schema-5 action contract")
        if domain_name not in self.contract.domain_ids:
            raise ValueError(f"Unsupported initial action domain {domain_name!r}")
        if type(fps) is not int or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self.model = model
        # Native preprocessing mutates the batch (normalization, flattened
        # items and condition indexes). Every reset must start from raw inputs.
        self._initial_data_batch = copy.deepcopy(data_batch)
        self.domain_name = domain_name
        self.fps = fps
        self.seed = seed
        self.tcf = model.tokenizer_vision_gen.temporal_compression_factor
        self.chunk_size = model.config.teacher_forcing_frames_per_chunk
        self.action_rows = self.tcf * self.chunk_size
        self.closed = True
        self._open()

    def _open(self) -> None:
        require_idle_bimanual_model(self.model)
        self.session_id = uuid.uuid4().hex
        self._history = ARHistorySnapshot(0, (), ())
        self.generated_actions = 0
        self.output_frames = 0
        self.control_source: str | None = None
        self.observation: torch.Tensor | None = None
        self._stack = ExitStack()
        self._iterator = None
        self.closed = False
        self.model._active_bimanual_session = weakref.ref(self)
        try:
            self._stack.enter_context(self.model.tokenizer_vision_gen.use_cached_decoder())
            data_batch = copy.deepcopy(self._initial_data_batch)
            self._iterator = self.model.iter_samples_from_batch_autoregressive(
                data_batch,
                mode="forward_dynamics",
                guidance=1.0,
                seed=self.seed,
                sampler_mode="distilled",
                on_kv_cache_update=self._record_history,
                has_negative_prompt=f"neg_{self.model.input_caption_key}" in data_batch,
            )
            self.observation = self._decode(next(self._iterator)["vision"])
        except BaseException:
            self.close()
            raise

    def _record_history(self, snapshot: ARHistorySnapshot) -> None:
        self._history = snapshot

    @torch.no_grad()
    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.model.decode(latent)
        if not torch.isfinite(decoded).all():
            raise ValueError("Bimanual decoder produced non-finite RGB values")
        rgb = ((1.0 + decoded) / 2).clamp(0, 1).detach().cpu()
        self.output_frames += rgb.shape[2]
        return rgb

    @torch.no_grad()
    def step(self, payload: dict[str, Any], *, control_source: str = "policy") -> torch.Tensor:
        """Apply the latest control and return its decoded observation chunk.

        Choose a policy/human adapter in the caller using ``observation``. The
        source label records that choice; it does not load or execute a policy.
        Invalid controls fail before advancing the generator and can be retried.
        """
        if self.closed or self._iterator is None:
            raise RuntimeError("Bimanual session is closed")
        if not isinstance(control_source, str) or not control_source.strip():
            raise ValueError("control_source must be a nonempty label")
        actions, domains = self.contract.validate(payload, action_count=self.action_rows, domain_name=self.domain_name)
        control = torch.zeros(self.action_rows, self.contract.model_action_dim, dtype=torch.float32)
        control[:, : self.contract.action_dim] = torch.tensor(actions, dtype=torch.float32)
        if not torch.isfinite(control).all():
            raise ValueError("Bimanual controls overflow float32")
        message = StreamedActionChunk(control, torch.tensor(domains, dtype=torch.long))
        try:
            output = self._iterator.send(message)
            self.observation = self._decode(output["vision"])
        except BaseException:
            self.close()
            raise
        self.generated_actions += self.action_rows
        self.control_source = control_source
        return self.observation

    def inspect(self) -> dict[str, Any]:
        """JSON-compatible counters and actual populated KV history per layer."""
        retained = self._history.conditional
        span = min(((ids[-1] - ids[0]) * self.tcf / self.fps if ids else 0.0 for ids in retained), default=0.0)
        return {
            "session_id": self.session_id,
            "closed": self.closed,
            "seed": self.seed,
            "control_source": self.control_source,
            "action_rows_per_step": self.action_rows,
            "generated_actions": self.generated_actions,
            "decoded_frames": self.output_frames,
            "rollout_seconds": max(0, self.output_frames - 1) / self.fps,
            "next_latent_frame": self._history.next_frame_idx,
            "conditional_history": [list(ids) for ids in retained],
            "unconditional_history": [list(ids) for ids in self._history.unconditional],
            "minimum_retained_history_span_seconds": span,
        }

    def reset(self, *, seed: int | None = None) -> torch.Tensor:
        """Start again from the original image/prompt with fresh KV and VAE state."""
        self.close()
        if seed is not None:
            self.seed = seed
        self._open()
        assert self.observation is not None
        return self.observation

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self._iterator is not None:
                self._iterator.close()
        finally:
            try:
                self._stack.close()
            finally:
                self._iterator = None
                self._history = ARHistorySnapshot(0, (), ())
                self.observation = None
                owner = getattr(self.model, "_active_bimanual_session", None)
                if isinstance(owner, weakref.ReferenceType) and owner() is self:
                    del self.model._active_bimanual_session

    def __enter__(self) -> BimanualSession:
        if self.closed:
            raise RuntimeError("Bimanual session is closed")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
