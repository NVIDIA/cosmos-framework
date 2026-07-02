# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tokens-per-second throughput callback for VLM training.

Logs a single, directly-comparable throughput number per logging window so that
optimization ablations (compile, sharding strategy, comm dtype, packing budget)
can be A/B'd on equal footing.

Why this exists instead of ``MFUCallback`` (``cosmos_framework/callbacks/mfu.py``):
``MFUCallback`` is written for the OmniMoT/VFM network -- it reads
``model.net.language_model.config`` and per-modality token counts
(``output_batch["und_token_length"]`` etc.). ``VLMModel``
(``cosmos_framework/model/generator/vlm_model.py``) is a plain HF wrapper: it exposes
``self.model`` (no ``.net``) and its ``training_step`` returns only
``{"loss", "loss_avg", "labels"}``. ``MFUCallback`` would therefore silently
no-op (token length is ``None``) or fail on ``model.net``. This callback instead
counts the real input tokens off ``data_batch["input_ids"]`` -- the one tensor
the VLM forward always consumes -- so it is correct for the VLM path and
self-normalizes for any residual per-step token variation.

Measurement notes:
  - Reports rank-0 *per-GPU* throughput. With token-based packing (each rank
    packs to the same ``max_tokens`` budget) this is representative of every
    rank; no cross-rank collective is taken, so the callback adds no
    communication and does not contaminate timing.
  - ``input_ids.numel()`` is a metadata read (tensor shape), so it triggers no
    device sync.
  - Content (non-pad) tokens are read from the loader-emitted CPU int
    ``data_batch["content_tokens"]`` (set in ``collate_fn.custom_collate``), so the
    packing-efficiency metric also triggers no per-step device sync. The legacy
    ``(input_ids != pad_id).sum()`` device reduction remains only as a fallback when
    that key is absent.
  - ``hit_thres`` warm-up steps are skipped so compilation / allocator warm-up
    don't pollute the first window (mirrors ``IterSpeed`` / ``MFUCallback``).

Packing-aware telemetry. On top of the headline
throughput rate, this callback reports the "iteration speed per sample at a given packing
density" view, in four groups:
  - per-sample / per-step: ``sec_per_sample``, ``samples_per_step``, ``tokens_per_step``.
  - density: ``useful_util`` (= U/padded), supervision density ``rho_sup`` (= U*/U),
    ``attention_quadratic_waste`` (the O(L^2) work true packing would remove), and the
    unpadded ``seq_max_len`` (l_max) mean/max.
  - utilization / comm amortization: ``useful_mfu`` (+ ``compute_bound`` flag) tells us when
    raising the batch budget stops paying; ``comm_bytes_per_useful_token`` makes the
    fixed-per-step FSDP collective volume amortized over useful work explicit.
  - cost-model calibration: ``realized`` vs packer-``predicted`` step time (FLOP path only).

All inputs are loader-emitted CPU ints/floats (``content_tokens``, ``supervised_tokens``,
``sum_len_sq``, ``seq_max_len``, ``predicted_runtime_ms`` from ``collate_fn.custom_collate``)
plus tensor-shape metadata and a one-time parameter count, so the callback adds NO per-step
device sync. ``COSMOS_VLM_TELEMETRY=0`` turns the whole callback into a no-op for the
telemetry-overhead A/B. We intentionally do not log ``sec_per_useful_token`` (it is the exact
reciprocal of ``useful_tokens_per_sec_per_gpu``).
"""

import os
import time

import torch
import wandb
from torch import Tensor

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import log
from cosmos_framework.utils.distributed import is_rank0, rank0_only


class VLMTokensPerSec(EveryN):
    """Log per-GPU tokens/sec over each logging window.

    Args:
        hit_thres: Number of warm-up steps to skip before timing begins.
        length_key: Key in ``data_batch`` whose tensor holds the packed token
            ids for this rank (``input_ids`` for the VLM path).
    """

    def __init__(
        self,
        *args,
        hit_thres: int = 50,
        length_key: str = "input_ids",
        peak_flops_per_gpu: float | None = None,
        comm_factor: float = 3.0,
        comm_dtype_bytes: float = 2.0,
        compute_bound_mfu_thresh: float = 0.5,
        fwd_bwd_flop_coeff: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.hit_thres = hit_thres
        self.length_key = length_key
        # A/B toggle for the telemetry-overhead validation:
        # COSMOS_VLM_TELEMETRY=0 makes the whole callback a no-op so a "telemetry off" run
        # measures the true cost of the callback (base + extended) against a "telemetry on" run.
        # Defaults to enabled when the env var is unset (normal training).
        self._enabled: bool = os.environ.get("COSMOS_VLM_TELEMETRY", "1") != "0"
        # Hardware/structural constants for the comm-amortization and utilization metrics.
        #   peak_flops_per_gpu: dense BF16 peak; None => resolve from torch.cuda device name.
        #   comm_factor: per-step FSDP collective volume in units of the (global) parameter
        #     bytes -- 2 all-gathers (fwd + activation-ckpt bwd recompute) + 1 grad
        #     reduce-scatter ~= 3x. comm_dtype_bytes: 2 for BF16 collectives.
        self._peak_flops: float | None = peak_flops_per_gpu
        self._comm_factor: float = comm_factor
        self._comm_dtype_bytes: float = comm_dtype_bytes
        self._compute_bound_mfu_thresh: float = compute_bound_mfu_thresh
        # fwd+bwd FLOPs per parameter per token for useful_mfu: 6 without activation
        # recomputation, 8 with full activation checkpointing (the +2 is the recompute
        # forward run during backward). None => auto-resolve from the model's AC mode.
        self._fwd_bwd_flop_coeff: float | None = fwd_bwd_flop_coeff
        self._flop_coeff_resolved: float | None = None  # cached auto-resolved coeff
        self._num_params: int | None = None  # lazily filled from model.parameters() (global/DTensor)
        self._hit_counter: int = 0
        self._tokens_in_window: int = 0
        self._useful_tokens_in_window: int = 0
        self._samples_in_window: int = 0
        self._singleton_steps_in_window: int = 0
        self._steps_in_window: int = 0
        self._window_start_time: float | None = None
        # Extended packing telemetry (all from loader-emitted CPU ints -> sync-free).
        self._supervised_tokens_in_window: int = 0  # U* = #(labels != ignore_index)
        self._padded_attn_in_window: int = 0  # Σ_step k * padded_len^2 (attention work paid)
        self._content_attn_in_window: int = 0  # Σ_step Σ_i L_i^2 (attention work needed)
        self._seq_max_len_sum: int = 0  # for mean l_max over the window
        self._seq_max_len_max: int = 0  # max l_max over the window
        self._seq_max_len_steps: int = 0  # #steps that carried seq_max_len (for the mean)
        # Cost-model calibration: packer-predicted per-step runtime (FLOP path only).
        self._predicted_runtime_ms_in_window: float = 0.0
        self._predicted_steps_in_window: int = 0

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        # Telemetry-off A/B arm: complete no-op (no accumulation, no periodic report).
        if not self._enabled:
            return
        # Skip warm-up steps entirely (no timing, no accumulation).
        if self._hit_counter < self.hit_thres:
            self._hit_counter += 1
            return

        # Only rank 0 accumulates AND reports. every_n_impl (which resets the window) is
        # @rank0_only, so accumulating on the other ranks would never reset -- their window
        # counters (notably _padded_attn_in_window ~ Σ k·l_max²) would grow for the entire
        # run. This callback takes no cross-rank collective (it reports rank-0 per-GPU
        # throughput), so scoping accumulation to rank 0 matches the report/reset scope.
        # Non-rank0 still delegates to EveryN below so its periodic distributed barrier
        # (barrier_after_run) stays collective and in lockstep.
        if not is_rank0():
            super().on_training_step_end(model, data_batch, output_batch, loss, iteration)
            return

        # Open the timing window on the first post-warm-up step.
        if self._window_start_time is None:
            self._window_start_time = time.time()

        ids = data_batch.get(self.length_key)
        if ids is not None:
            # numel() is the PADDED token count (dynamic batching pads each batch to
            # its longest member); shape[0] is the number of packed samples this step.
            n_samples = int(ids.shape[0]) if ids.dim() > 1 else 1
            self._tokens_in_window += int(ids.numel())
            self._samples_in_window += n_samples
            if n_samples == 1:
                self._singleton_steps_in_window += 1
            # USEFUL (non-pad) tokens. Prefer the loader-emitted CPU int
            # data_batch["content_tokens"] (= Σ len(input_ids) captured PRE-pad in
            # collate_fn.custom_collate). Reading it is sync-free, so this throughput
            # callback no longer perturbs the very step timing it measures. Only if the
            # key is absent (e.g. a non-VLM collate) do we fall back to a device
            # reduction off pad_token_id -- correct, but it forces a per-step D2H sync.
            # useful/padded ratio = packing efficiency -> whether the padded budget is
            # real content or padding (i.e. is true no-pad packing worth it).
            content = data_batch.get("content_tokens")
            if content is not None:
                self._useful_tokens_in_window += int(content)
            else:
                pad = data_batch.get("pad_token_id")
                if pad is not None:
                    pad_id = int(pad.flatten()[0])
                    self._useful_tokens_in_window += int((ids != pad_id).sum())

            # Extended packing telemetry. All reads are loader-emitted CPU ints (or tensor
            # shape metadata), so this adds NO device sync to the timed step. Each is guarded
            # so a non-VLM collate that omits a key simply skips that derived metric.
            sup = data_batch.get("supervised_tokens")
            if sup is not None:
                self._supervised_tokens_in_window += int(sup)
            slq = data_batch.get("sum_len_sq")
            if slq is not None:
                self._content_attn_in_window += int(slq)
            smax = data_batch.get("seq_max_len")
            if smax is not None:
                smax = int(smax)
                self._seq_max_len_sum += smax
                self._seq_max_len_max = max(self._seq_max_len_max, smax)
                self._seq_max_len_steps += 1
            # Padded attention work this step = n_samples * padded_len^2. padded_len is the
            # row length the model actually attends over (shape metadata -> no sync).
            padded_len = int(ids.shape[1]) if ids.dim() > 1 else int(ids.shape[0])
            self._padded_attn_in_window += n_samples * padded_len * padded_len
        # Packer-predicted per-step runtime (FLOP cost model), a CPU float emitted by
        # collate_fn only on the FLOP-batching path -> sync-free; absent => skipped.
        pred = data_batch.get("predicted_runtime_ms")
        if pred is not None:
            self._predicted_runtime_ms_in_window += float(pred)
            self._predicted_steps_in_window += 1
        self._steps_in_window += 1

        # Delegate to EveryN for the periodic reporting cadence.
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    def _get_num_params(self, model: ImaginaireModel) -> int | None:
        """Global parameter count, cached. For FSDP2/DTensor params ``.numel()`` returns the
        logical (unsharded) size, so this is the true model size regardless of shard degree."""
        if self._num_params is None:
            try:
                self._num_params = int(sum(p.numel() for p in model.parameters()))
            except Exception:  # pragma: no cover - defensive; MFU/comm metrics just skip
                self._num_params = None
        return self._num_params

    def _resolve_peak_flops(self) -> float | None:
        """Dense BF16 peak FLOP/s per GPU. Explicit override wins; else map the device name.
        Unknown device -> None so MFU is skipped rather than reported against a wrong peak."""
        if self._peak_flops is not None:
            return self._peak_flops
        if not torch.cuda.is_available():
            return None
        name = torch.cuda.get_device_name()
        if any(k in name for k in ("B200", "GB200", "Blackwell")):
            return 2.45e15  # GB200 NVL72 dense BF16 ~2.45 PFLOP/s/GPU (no sparsity)
        if "H200" in name or "H100" in name:
            return 989e12  # SXM dense BF16
        if "A100" in name:
            return 312e12
        log.warning(f"VLMTokensPerSec: unknown device '{name}', MFU not reported (set peak_flops_per_gpu)")
        return None

    def _resolve_flop_coeff(self, model: ImaginaireModel) -> float:
        """fwd+bwd FLOPs per parameter per token for useful_mfu. 6 without activation
        recomputation; 8 with full activation checkpointing (the +2 is the recompute
        forward during backward). Explicit override (``fwd_bwd_flop_coeff``) wins; else
        read the model's AC mode. On the VLM/HF path ONLY ``mode == "full"`` actually
        recomputes -- ``"selective"`` degrades to a no-op because the HF backbone has no
        per-op SAC (see vlm_model.py / activation_checkpointing.py), so we key on "full"
        rather than the looser ``!= "none"`` MFUCallback uses. Cached (config is static)."""
        if self._fwd_bwd_flop_coeff is not None:
            return self._fwd_bwd_flop_coeff
        if self._flop_coeff_resolved is None:
            ac_cfg = getattr(getattr(model, "config", None), "activation_checkpointing", None)
            ac_mode = getattr(ac_cfg, "mode", "none")
            self._flop_coeff_resolved = 8.0 if ac_mode == "full" else 6.0
        return self._flop_coeff_resolved

    @rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, Tensor],
        output_batch: dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        if self._window_start_time is None or self._steps_in_window == 0:
            return

        elapsed = time.time() - self._window_start_time
        if elapsed <= 0 or self._tokens_in_window == 0:
            return

        tokens_per_sec_per_gpu = self._tokens_in_window / elapsed
        tokens_per_step = self._tokens_in_window / self._steps_in_window
        samples_per_step = self._samples_in_window / self._steps_in_window
        singleton_rate = self._singleton_steps_in_window / self._steps_in_window

        # Packing efficiency: useful (non-pad) tokens vs the padded budget actually
        # paid for. useful_util ~1.0 => packing is dense (true no-pad packing would
        # buy little); useful_util << 1 => padding waste (pool_size / scoring / true
        # sequence packing are the levers). useful_tokens_per_sec is the REAL training
        # throughput (what loss-vs-tokens should be measured against).
        useful_tokens_per_sec_per_gpu = self._useful_tokens_in_window / elapsed
        useful_util = self._useful_tokens_in_window / self._tokens_in_window if self._tokens_in_window else 0.0

        # --- Extended packing telemetry (the "iteration speed per sample at a density" view) ---
        # sec_per_sample is only fair alongside the density terms below, so they are logged
        # together. NOTE: we intentionally do NOT log a
        # sec_per_useful_token -- it is the exact reciprocal of useful_tokens_per_sec_per_gpu.
        sec_per_sample = elapsed / self._samples_in_window if self._samples_in_window else 0.0
        # rho_sup = U*/U : fraction of CONTENT tokens that carry a loss signal.
        rho_sup = (
            self._supervised_tokens_in_window / self._useful_tokens_in_window if self._useful_tokens_in_window else 0.0
        )
        useful_supervised_tokens_per_sec_per_gpu = self._supervised_tokens_in_window / elapsed
        # attention-quadratic waste = 1 - Σ L_i^2 / Σ k*l_max^2 : the part of attention cost
        # that true sequence packing removes but the linear useful_util cannot see.
        attention_quadratic_waste = (
            1.0 - self._content_attn_in_window / self._padded_attn_in_window if self._padded_attn_in_window else 0.0
        )
        seq_max_len_mean = self._seq_max_len_sum / self._seq_max_len_steps if self._seq_max_len_steps else 0.0
        seq_max_len_max = self._seq_max_len_max

        # --- Amortized communication per useful token ---
        # Per-step FSDP collective volume per rank is a FIXED structural constant (independent of
        # batch size): comm_factor * N_params * comm_dtype_bytes. Dividing the fixed per-step bytes
        # by useful tokens/step makes the amortization explicit -- this is exactly why denser packing
        # lowers comm-per-sample (more useful work under the same collectives). Structural estimate,
        # not a measured NCCL byte count.
        num_params = self._get_num_params(model)
        useful_tokens_per_step = self._useful_tokens_in_window / self._steps_in_window
        comm_bytes_per_useful_token = 0.0
        if num_params and useful_tokens_per_step > 0:
            comm_bytes_per_step = self._comm_factor * num_params * self._comm_dtype_bytes
            comm_bytes_per_useful_token = comm_bytes_per_step / useful_tokens_per_step

        # --- Useful-MFU + compute-bound flag ---
        # useful_mfu = coeff * N * useful_tokens/s / peak (conventional linear-term MFU, but on
        # USEFUL tokens so padding is not credited). coeff is 6 (no recompute) or 8 (full activation
        # checkpointing), resolved from the model's AC mode so recompute runs are not underestimated
        # by ~25%. It is a conservative lower bound (ignores the attention quadratic term -- which is
        # separately visible as attn_quadratic_waste). It answers "when to stop raising the budget":
        # useful_mfu near peak => compute-bound => denser packing buys little more throughput; far
        # from peak => overhead/comm-bound => packing still pays.
        peak_flops = self._resolve_peak_flops()
        useful_mfu = 0.0
        compute_bound = 0
        if num_params and peak_flops:
            flop_coeff = self._resolve_flop_coeff(model)
            useful_mfu = (flop_coeff * num_params * useful_tokens_per_sec_per_gpu) / peak_flops
            compute_bound = 1 if useful_mfu >= self._compute_bound_mfu_thresh else 0

        # --- Cost-model calibration: realized vs packer-predicted step time ---
        # The FLOP packer sizes each step to ~target_runtime using estimate_runtime_ms; logging the
        # realized/predicted ratio calibrates that cost model. ~1.x is expected (predicted models
        # compute only, realized includes optimizer/comm/dataloader); a drifting or large ratio means
        # the packer is mis-sizing steps (oversized => comm under-amortized; undersized => throughput
        # left on the table). Populated only on the FLOP-batching path.
        realized_step_ms = elapsed * 1000.0 / self._steps_in_window
        predicted_step_ms = 0.0
        realized_over_predicted = 0.0
        if self._predicted_steps_in_window > 0:
            predicted_step_ms = self._predicted_runtime_ms_in_window / self._predicted_steps_in_window
            if predicted_step_ms > 0:
                realized_over_predicted = realized_step_ms / predicted_step_ms

        # Cumulative device peak (NOT reset) -> OOM-headroom gate for Nmax/Tmax sweeps.
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        log.info(
            f"{iteration} : tokens_per_sec_per_gpu {tokens_per_sec_per_gpu:.1f} | "
            f"useful_tokens_per_sec_per_gpu {useful_tokens_per_sec_per_gpu:.1f} | "
            f"useful_supervised_tokens_per_sec_per_gpu {useful_supervised_tokens_per_sec_per_gpu:.1f} | "
            f"tokens_per_step {tokens_per_step:.1f} | samples_per_step {samples_per_step:.2f} | "
            f"sec_per_sample {sec_per_sample:.4f} | "
            f"useful_util {useful_util:.3f} | rho_sup {rho_sup:.3f} | "
            f"attn_quadratic_waste {attention_quadratic_waste:.3f} | "
            f"seq_max_len_mean {seq_max_len_mean:.0f} | seq_max_len_max {seq_max_len_max} | "
            f"useful_mfu {useful_mfu:.3f} | compute_bound {compute_bound} | "
            f"comm_bytes_per_useful_token {comm_bytes_per_useful_token:.1f} | "
            f"realized_step_ms {realized_step_ms:.0f} | predicted_step_ms {predicted_step_ms:.0f} | "
            f"realized_over_predicted {realized_over_predicted:.2f} | "
            f"singleton_rate {singleton_rate:.3f} | "
            f"peak_mem_gb {peak_mem_gb:.1f} | steps_in_window {self._steps_in_window}",
            rank0_only=False,
        )

        if wandb.run:
            wandb.log(
                {
                    "throughput/tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
                    "throughput/useful_tokens_per_sec_per_gpu": useful_tokens_per_sec_per_gpu,
                    "throughput/useful_supervised_tokens_per_sec_per_gpu": useful_supervised_tokens_per_sec_per_gpu,
                    "throughput/tokens_per_step": tokens_per_step,
                    "throughput/samples_per_step": samples_per_step,
                    "throughput/sec_per_sample": sec_per_sample,
                    "throughput/useful_mfu": useful_mfu,
                    "throughput/compute_bound": compute_bound,
                    "throughput/singleton_rate": singleton_rate,
                    "throughput/peak_mem_gb": peak_mem_gb,
                    "packing/useful_util": useful_util,
                    "packing/rho_sup": rho_sup,
                    "packing/attention_quadratic_waste": attention_quadratic_waste,
                    "packing/seq_max_len_mean": seq_max_len_mean,
                    "packing/seq_max_len_max": seq_max_len_max,
                    "comm/comm_bytes_per_useful_token": comm_bytes_per_useful_token,
                    "cost_model/predicted_step_ms": predicted_step_ms,
                    "cost_model/realized_step_ms": realized_step_ms,
                    "cost_model/realized_over_predicted": realized_over_predicted,
                },
                step=iteration,
            )

        # Reset the window (peak mem is intentionally left cumulative).
        self._tokens_in_window = 0
        self._useful_tokens_in_window = 0
        self._samples_in_window = 0
        self._singleton_steps_in_window = 0
        self._steps_in_window = 0
        self._supervised_tokens_in_window = 0
        self._padded_attn_in_window = 0
        self._content_attn_in_window = 0
        self._seq_max_len_sum = 0
        self._seq_max_len_max = 0
        self._seq_max_len_steps = 0
        self._predicted_runtime_ms_in_window = 0.0
        self._predicted_steps_in_window = 0
        self._window_start_time = time.time()
