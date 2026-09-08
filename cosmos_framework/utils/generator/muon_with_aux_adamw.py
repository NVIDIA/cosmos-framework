# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
MuonWithAuxAdamW optimizer implementation.

Muon (MomentUm Orthogonalized by Newton-schulz) for nn.Linear weight matrices,
with auxiliary AdamW for embeddings, biases, norms, and output layers (lm_head).

This implementation combines elements from:

1. FusedAdam (cosmos_framework/utils/generator/fused_adam.py):
   - DTensor handling via `get_local_tensor_if_DTensor` for FSDP/TP compatibility
   - Distributed step synchronization in `load_state_dict` for checkpoint compatibility

2. KellerJordan/Muon (https://github.com/KellerJordan/Muon):
   - Newton-Schulz orthogonalization algorithm (`zeropower_via_newtonschulz5`)
   - Quintic iteration coefficients (a=3.4445, b=-4.7750, c=2.0315)
   - Nesterov momentum formulation

3. MoonshotAI/Moonlight (https://arxiv.org/pdf/2502.16982):
   - SGD-style momentum: `buf = momentum * buf + grad` (not EMA-style)
   - Learning rate scaling by matrix size: `adjusted_lr = 0.2 * sqrt(max(A, B)) * lr`
   - `@torch.compile` decoration for kernel fusion
   - Parameter separation: Muon for nn.Linear weights only, AdamW for everything else
   - Distributed Newton-Schulz: all-gather gradients, NS on full matrix, scatter back
     (required because NS is a matrix-level operation, not element-wise)

Sharding -> orthogonalization (the core idea)
---------------------------------------------
Under FSDP each weight matrix is a DTensor split across GPUs, but Newton-Schulz
(NS) is a *whole-matrix* op and needs the full matrix in one place. Muon handles
each matrix one at a time with an all-gather: every GPU rebuilds the *same* full
matrix and runs NS on it.

Weight A sharded over 4 GPUs (one row-slice each):

    GPU0:[A0]  GPU1:[A1]  GPU2:[A2]  GPU3:[A3]

    --- all_gather(A) --->  every GPU holds the full matrix:

    GPU0:[A0 A1 A2 A3]   GPU1:[A0 A1 A2 A3]   GPU2:[...]   GPU3:[...]
         |                    |                    |          |
       NS(A)                NS(A)                NS(A)      NS(A)     (all identical)
         |                    |                    |          |
    --- each GPU slices back its own rows and applies the update to its shard ---

This is simple and correct, but the NS compute is duplicated world_size times
(every rank orthogonalizes the same matrix). The sibling ``Dion2WithAuxAdamW``
removes that redundancy by trading shards with all-to-all so each GPU
orthogonalizes a *different* matrix in parallel (see its module docstring).
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine as te
import transformer_engine_torch as tex
from torch.distributed.tensor import DTensor, Replicate

from cosmos_framework.utils import log
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor
from cosmos_framework.utils.generator.aux_optimizer_utils import (
    compute_pre_ns_update,
    create_moe_megabatches,
    pair_moe_gate_up_down_params,
    split_orthogonalizable_params,
    step_stacked_expert_params,
    validate_split_expert_ns_config,
    zeropower_via_newtonschulz5,
)
from cosmos_framework.utils.generator.optimizer import require_fp32_param_group


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """
    MuonWithAuxAdamW optimizer.

    Uses Muon (MomentUm Orthogonalized by Newton-schulz) for nn.Linear hidden weight matrices,
    and AdamW for embeddings, biases, layer norms, and output heads (lm_head).

    See module docstring for full attribution of borrowed components.

    Args:
        params: Iterable of parameters to optimize.
        lr: Base learning rate. Muon scales this by muon_lr_scale*sqrt(max(A,B)), AdamW uses directly.
        muon_momentum: Momentum coefficient for Muon.
        muon_lr_scale: Scale factor for Muon LR adjustment. Final LR = muon_lr_scale * sqrt(max(A,B)) * lr.
        ns_steps: Number of Newton-Schulz iterations.
        nesterov: Whether to use Nesterov momentum for Muon.
        weight_decay: Weight decay for all parameters.
        adam_betas: Beta coefficients for the auxiliary AdamW side (matches VFM convention).
        eps: Epsilon for AdamW numerical stability.
        use_distributed: Whether to sync step counters across ranks when loading checkpoints.

    Parameter precision: every parameter must be FP32, which
    :meth:`add_param_group` enforces. Both the Muon and the auxiliary AdamW update are
    applied to the parameter in place in FP32, so there are no FP32 master weights: for
    an FP32 parameter a master would be a bit-identical duplicate of it, costing 4 bytes
    per element and an extra copy per step while computing the same numbers. It also
    removes the master weights' checkpoint problem -- the FP32 weight is now the
    parameter itself, which the checkpoint already stores losslessly, rather than a copy
    that has to be re-derived from a rounded parameter on resume. To keep the
    forward/backward in BF16, wrap the model in FSDP2 with
    ``MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)``:
    FSDP down-casts for compute and all-gather while the sharded parameter this
    optimizer steps stays FP32. Do not cast the parameters themselves.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        muon_momentum: float = 0.95,
        muon_lr_scale: float = 0.2,
        ns_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        use_distributed: bool = True,
        capturable: bool = False,
        expert_param_keywords: tuple[str, ...] | None = None,
        orthogonalize_skip_patterns: tuple[str, ...] | None = None,
        split_expert_gate_up: bool = False,
        batch_split_expert_ns: bool = False,
        max_moe_expert_ns_matrices: int = 0,
        **kwargs,  # Absorb VFM-specific args (fused, keys_to_select, etc.)
    ):
        if "master_weights" in kwargs:
            # Not silently ignored: a caller asking for master weights is asking for
            # precision this optimizer no longer provides that way, and would otherwise
            # get a warning buried in the log.
            raise ValueError(
                "MuonWithAuxAdamW no longer maintains FP32 master weights -- its parameters "
                "must already be FP32, which makes a master a bit-identical duplicate (see the "
                "class docstring). Drop the master_weights argument."
            )

        # Log any ignored kwargs for debugging
        if kwargs:
            ignored_keys = list(kwargs.keys())
            # These are expected VFM args that we silently ignore
            expected_ignored = {"fused", "keys_to_select", "adamw_betas", "adamw_eps"}
            unexpected = set(ignored_keys) - expected_ignored
            if unexpected:
                import warnings

                warnings.warn(f"MuonWithAuxAdamW ignoring unexpected kwargs: {unexpected}")

        validate_split_expert_ns_config(split_expert_gate_up, batch_split_expert_ns)
        if max_moe_expert_ns_matrices < 0:
            raise ValueError(f"max_moe_expert_ns_matrices must be >= 0, got {max_moe_expert_ns_matrices}")

        # Store shared hyperparameters
        # Note: lr is accessed via property that reads from param_groups
        # to support LR schedulers (which update param_groups[X]["lr"])
        self.wd = weight_decay

        # Store Muon-specific hyperparameters
        self.muon_momentum = muon_momentum
        self.muon_lr_scale = muon_lr_scale
        # Shape -> LR scaling ratio; see _get_adjusted_lr.
        self._adjusted_lr_ratios: dict[tuple[int, ...], float] = {}
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.split_expert_gate_up = split_expert_gate_up
        self.batch_split_expert_ns = batch_split_expert_ns
        self.max_moe_expert_ns_matrices = max_moe_expert_ns_matrices

        # Name substrings that route stacked MoE expert params ([E, M, N]) to the
        # Muon side (each expert slice orthogonalized). Empty = experts stay on
        # AdamW (no behavior change).
        self.expert_param_keywords = tuple(expert_param_keywords) if expert_param_keywords else ()
        # Regex patterns (matched against param names) that force matching 2D Linear
        # weights onto the auxiliary AdamW side (skip Muon orthogonalization), in
        # addition to the default head detection. E.g. r"\.gate\.weight$" for MoE
        # routers. Empty = nothing extra skipped (no behavior change).
        self.orthogonalize_skip_patterns = tuple(orthogonalize_skip_patterns) if orthogonalize_skip_patterns else ()

        # Store AdamW-specific hyperparameters
        self.adam_betas = tuple(adam_betas) if isinstance(adam_betas, list) else adam_betas
        self.eps = eps

        # Distributed settings
        self.use_distributed = use_distributed and dist.is_initialized()

        # ``capturable`` is also read by external code that duck-types on it to detect the
        # group-level step convention (see ``DistillationTrainer._uses_group_step``).
        self.capturable = capturable

        # Parameter lists (populated by categorize_params)
        self.muon_params: list[nn.Parameter] = []
        self.adamw_params: list[nn.Parameter] = []
        # Stacked MoE expert params ([E, M, N]); orthogonalized per expert slice.
        self.stacked_muon_params: list[nn.Parameter] = []
        self.param_to_name: dict[nn.Parameter, str] = {}
        # Split-expert pair tracking (gate_up + down pairs for multi-layer NS batching).
        self._split_expert_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        self._split_expert_param_ids: set[int] = set()
        self._moe_megabatches: list[list[tuple[nn.Parameter, nn.Parameter]]] = []

        # Transformer Engine fused Adam for AdamW params. The zero buffer is the
        # noop flag required as the second argument of TE's multi_tensor_applier;
        # it is a fixed constant here (no AMP overflow handling).
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self._multi_tensor_adam = tex.multi_tensor_adam
        self._multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable

        # Initialize base optimizer. betas / eps go in the defaults so each param
        # group carries them (FusedAdam convention); the AdamW step reads them
        # per-group, enabling per-group overrides and exact FusedAdam parity.
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=self.adam_betas, eps=eps)
        super().__init__(params, defaults)

        # Convert LR to tensor for capturable mode
        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group["params"]) == 0:
                    continue
                device = group["params"][0].device
                if isinstance(group["lr"], float):
                    group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)
                self.param_groups[idx]["lr"] = group["lr"].to(device=device)

        # id(param) -> owning param_group, so the Muon and AdamW updates can read
        # the *per-group* lr / weight_decay. This is what lets the factory's
        # lr_multipliers and disable_weight_decay_for_1d_params flow through. With
        # a single param group it degenerates to a single global lr/wd, matching
        # the original (reference) behavior. Populated by categorize_params.
        self._param_group_map: dict[int, dict] = {}
        self._adamw_param_ids: set[int] = set()

        log.info(f"MuonWithAuxAdamW capturable: {capturable}")

    def add_param_group(self, param_group: dict) -> None:
        """Register a param group, rejecting any parameter that is not FP32.

        ``torch.optim.Optimizer.__init__`` funnels every group through here, so this
        doubles as the construction-time check. See the class docstring for why FP32
        parameters are required in place of FP32 master weights.

        Validates before calling ``super()`` -- ``Optimizer.add_param_group`` appends the
        group to ``self.param_groups`` unconditionally, so validating after the call would
        leave a rejected group registered (with mixed-dtype params `categorize_params` is
        unaware of) if a caller caught the ``ValueError`` and kept stepping. See
        :func:`require_fp32_param_group`.
        """
        require_fp32_param_group(
            param_group,
            "MuonWithAuxAdamW",
            "it applies both the Muon and the AdamW update to the parameter in place in "
            "FP32 and keeps no master weights",
        )
        super().add_param_group(param_group)

    def categorize_params(self, model: nn.Module) -> None:
        """
        Categorize parameters into Muon and AdamW groups.

        Muon is used for hidden ``nn.Linear`` weights only. Embeddings, the output
        head (``lm_head`` / tied / vocab-shaped projection), biases, norms, and any
        non-Linear parameter go to AdamW. See
        :func:`split_orthogonalizable_params` for the architecture-agnostic
        embedding / output-head detection.

        Args:
            model: The model (typically the trainable ``net``) to categorize.
        """
        # Only categorize params that are in this optimizer's param_groups. This
        # respects keys_to_select filtering - params not passed to __init__ should
        # not be categorized, otherwise state_dict() would fail with KeyError when
        # mapping state entries to param indices.
        optimizer_param_ids = {id(p) for group in self.param_groups for p in group["params"]}

        orthogonalizable, self.adamw_params, self.param_to_name = split_orthogonalizable_params(
            model,
            optimizer_param_ids,
            expert_param_keywords=self.expert_param_keywords,
            orthogonalize_skip_patterns=self.orthogonalize_skip_patterns,
        )

        # Split orthogonalizable params into 2-D (regular Linear weights -> Muon) and
        # stacked 3-D MoE experts ([E, M, N] -> orthogonalized per expert slice).
        self.muon_params = [p for p in orthogonalizable if p.ndim == 2]
        self.stacked_muon_params = [p for p in orthogonalizable if p.ndim >= 3]

        # Sort Muon params by size (largest first) for distributed load balancing
        self.muon_params = sorted(self.muon_params, key=lambda x: x.numel(), reverse=True)

        # Build gate_up/down pairs for split-expert NS (if requested).
        self._split_expert_pairs = []
        self._split_expert_param_ids = set()
        if self.split_expert_gate_up and self.stacked_muon_params:
            self._split_expert_pairs = pair_moe_gate_up_down_params(self.stacked_muon_params, self.param_to_name)
            for gate_up_p, down_p in self._split_expert_pairs:
                self._split_expert_param_ids.add(id(gate_up_p))
                self._split_expert_param_ids.add(id(down_p))

        # Build MoE megabatch plan (K pairs per NS call).
        self._create_moe_megabatches()

        # Check if using DTensor (FSDP) - determines whether we use distributed NS
        self._params_are_dtensor = isinstance(self.muon_params[0], DTensor) if self.muon_params else False

        muon_numel = sum(p.numel() for p in self.muon_params)
        adamw_numel = sum(p.numel() for p in self.adamw_params)
        stacked_muon_numel = sum(p.numel() for p in self.stacked_muon_params)

        log.info(
            f"MuonWithAuxAdamW: {len(self.muon_params)} Muon params ({muon_numel:,} elements), "
            f"{len(self.stacked_muon_params)} stacked-expert params ({stacked_muon_numel:,} elements), "
            f"{len(self.adamw_params)} AdamW params ({adamw_numel:,} elements)"
            f"{', using distributed NS (DTensor/FSDP detected)' if self._params_are_dtensor else ''}"
        )

        # Log Muon param details (layer names and shapes)
        log.info("Muon parameters (layer name -> shape):")
        for p in self.muon_params:
            name = self.param_to_name.get(p, "unknown")
            log.info(f"  {name}: {tuple(p.shape)}")

        # Build the param -> owning group map for per-group lr / weight_decay
        # lookups during the Muon and AdamW steps.
        self._param_group_map = {}
        for group in self.param_groups:
            for p in group["params"]:
                self._param_group_map[id(p)] = group
        self._adamw_param_ids = {id(p) for p in self.adamw_params}

    def _base_lr_for(self, p: nn.Parameter) -> float | torch.Tensor:
        """Per-group base learning rate for ``p`` (honors lr_multipliers)."""
        return self._param_group_map[id(p)]["lr"]

    def _wd_for(self, p: nn.Parameter) -> float:
        """Per-group weight decay for ``p`` (honors disable_weight_decay_for_1d_params)."""
        return self._param_group_map[id(p)]["weight_decay"]

    def _get_adjusted_lr(self, param_shape: tuple[int, ...], base_lr: float | torch.Tensor) -> float | torch.Tensor:
        """
        Compute adjusted learning rate based on parameter matrix size.

        Based on Moonlight: adjusted_lr = muon_lr_scale * sqrt(max(A, B)) * base_lr

        Args:
            param_shape: Shape of the parameter tensor.
            base_lr: The owning param-group's learning rate (after any
                lr_multiplier). Muon's matrix-size scaling layers on top of it.

        Returns:
            Adjusted learning rate for this parameter.
        """
        # Memoized: the ratio depends only on the shape and muon_lr_scale, both fixed
        # for the run, but the MoE megabatch path asks for it once per matrix per step.
        adjusted_ratio = self._adjusted_lr_ratios.get(param_shape)
        if adjusted_ratio is None:
            A, B = param_shape[:2]
            adjusted_ratio = self.muon_lr_scale * math.sqrt(max(A, B))
            self._adjusted_lr_ratios[param_shape] = adjusted_ratio
        return base_lr * adjusted_ratio

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Muon updates (with distributed NS for FSDP)
        self._step_muon()

        # Stacked MoE expert updates (orthogonalize each expert slice)
        self._step_stacked_muon()

        # AdamW updates
        self._step_adamw()

        return loss

    def _create_moe_megabatches(self) -> None:
        """Group split expert pairs into K-layer NS batches (see
        :func:`create_moe_megabatches`)."""
        world_size = dist.get_world_size() if (self.use_distributed and dist.is_initialized()) else 1
        self._moe_megabatches = create_moe_megabatches(
            self._split_expert_pairs, world_size, self.max_moe_expert_ns_matrices
        )

    def _step_stacked_muon(self) -> None:
        """Orthogonalize stacked MoE expert params via step_stacked_expert_params.

        Dispatches to split-expert or megabatch path when ``split_expert_gate_up``
        is True; falls back to the historical whole-param per-expert NS otherwise.

        See :func:`step_stacked_expert_params` for the full dispatch logic.
        """
        if not self.stacked_muon_params:
            return

        moe_megabatches = self._moe_megabatches if self.split_expert_gate_up else None

        step_stacked_expert_params(
            self.stacked_muon_params,
            self._split_expert_pairs,
            self._split_expert_param_ids,
            optimizer_state=self.state,
            param_to_name=self.param_to_name,
            # No FP32 masters: this optimizer's params are already FP32 and are updated in
            # place (see the class docstring), so the helper's master path is unused and it
            # writes the update straight into the parameter.
            param_to_master={},
            master_weights=False,
            momentum=self.muon_momentum,
            nesterov=self.nesterov,
            ns_steps=self.ns_steps,
            batch_split_expert_ns=self.batch_split_expert_ns,
            base_lr_for=self._base_lr_for,
            weight_decay_for=self._wd_for,
            adjusted_lr_for=self._get_adjusted_lr,
            moe_megabatches=moe_megabatches,
            # profile_phases is intentionally left at its default (off): the NVTX
            # megabatch ranges are gated on DION2's dion2_profile_phases knob, and
            # Muon has no equivalent. Add one here if the Muon stacked path ever
            # needs phase-level profiling.
        )

    def _step_muon(self) -> None:
        """
        Muon step with distributed Newton-Schulz (following Moonlight paper).

        For DTensor/FSDP parameters:
        1. Apply momentum + Nesterov on local shards (element-wise, shard-safe)
        2. All-gather shards to reconstruct full pre-NS gradient
        3. Run Newton-Schulz on full matrix (must be done globally)
        4. Scatter back to get local orthogonalized update
        5. Apply weight decay and update to the local shard, in place, in FP32

        For non-DTensor parameters, runs single-device Muon.
        """
        for p in self.muon_params:
            if p.grad is None:
                continue

            # Sharding info drives the routing below: a non-empty list means the
            # param is FSDP-sharded (-> distributed NS); an empty list (a plain
            # tensor, or a fully-replicated DTensor) takes the single-device path.
            shard_info: list[tuple[int, int]] = []

            if isinstance(p, DTensor):
                # Local shard of the parameter (used to write the update in Step 5).
                # The gradient / momentum stay as DTensors and are handled below.
                local_param = p._local_tensor

                # Get sharding info from DTensor (FSDP typically shards dim 0 / rows).
                # Each entry: (mesh_dim, tensor_shard_dim)
                device_mesh = p.device_mesh
                placements = p.placements
                for mesh_dim_idx, placement in enumerate(placements):
                    if placement.is_shard():
                        shard_info.append((mesh_dim_idx, placement.dim))

                if not shard_info:
                    # Fully replicated DTensor (e.g. a DDP-like config where
                    # placements == (Replicate(),)). Legitimate -- fall back to
                    # single-device NS and log at info level (a warning here would
                    # needlessly alarm anyone reading the logs).
                    param_name = self.param_to_name.get(p, "unknown")
                    log.info(
                        f"DTensor '{param_name}' has no Shard placement (placements={placements}); "
                        "running single-device Newton-Schulz (replicated param)."
                    )

            if shard_info:
                # Distributed Muon: all-gather → NS → scatter
                # Handles both 1D sharding (FSDP) and 2D sharding (FSDP + TP)

                # Initialize state with local shard shape
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p).float()

                # Step 1: Momentum + Nesterov, kept in DTensor space so the buffer
                # stays sharded (memory-efficient) and the gather below is
                # padding-aware. These ops are element-wise, so DTensor runs them on
                # the local shards under the hood. compute_pre_ns_update does not
                # mutate the gradient, so p.grad can be passed directly.
                pre_ns_dtensor = compute_pre_ns_update(
                    p.grad,
                    state["momentum_buffer"],
                    momentum=self.muon_momentum,
                    nesterov=self.nesterov,
                )

                # Step 2: Reconstruct the full matrix from its shards.
                #
                # ``full_tensor()`` is DTensor's own all-gather (a Shard -> Replicate
                # redistribute). Because the DTensor carries the logical shape, it
                # rebuilds exactly ``p.shape`` and strips any FSDP2 padding of a shard
                # dim that is not divisible by the mesh size. A hand-rolled
                # ``all_gather`` + ``cat`` would keep those padding rows and feed a
                # wrong-sized / misaligned matrix into Newton-Schulz (the uneven-shard
                # bug this path used to have).
                #
                # PERF TODO: this is a *synchronous* all-gather (we block until the
                # full matrix is reconstructed before running Newton-Schulz). If the
                # optimizer step ever becomes a bottleneck, explore pipelining: overlap
                # the gather for parameter (i) with the Newton-Schulz compute for
                # parameter (i-1) across the muon_params loop.
                #
                # Cast to bf16 BEFORE the all-gather so the collective moves bf16 (not
                # fp32), halving its communication volume (matches the dion2 path). This
                # is numerically identical: Newton-Schulz casts to bf16 anyway, and bf16
                # rounding commutes with the reconstruction (round-then-gather ==
                # gather-then-round elementwise). Keep in sync with the input dtype of
                # ``zeropower_via_newtonschulz5``. The fp32 momentum buffer is untouched
                # (only this transient gathered copy is downcast).
                full_pre_ns = pre_ns_dtensor.bfloat16().full_tensor()

                # Sanity check: full_tensor() must return the logical (unpadded) shape.
                expected_shape = tuple(p.shape)
                actual_shape = tuple(full_pre_ns.shape)
                assert actual_shape == expected_shape, (
                    f"Failed to reconstruct full matrix for '{self.param_to_name.get(p, 'unknown')}': "
                    f"got {actual_shape}, expected {expected_shape}"
                )

                # Log shapes on first step to confirm distributed NS is working
                if "logged_shapes" not in state:
                    state["logged_shapes"] = True
                    log.info(
                        f"Muon distributed NS: '{self.param_to_name.get(p, 'unknown')}' "
                        f"local={tuple(local_param.shape)} → full={actual_shape} (verified)"
                    )

                # Step 3: Newton-Schulz on full matrix
                full_ortho = zeropower_via_newtonschulz5(full_pre_ns, steps=self.ns_steps)

                # Step 4: Scatter the orthogonalized full matrix back to p's sharding.
                #
                # full_ortho is identical on every rank, so we treat it as Replicate
                # and redistribute to p's placements. Replicate -> Shard is pure local
                # slicing (no communication) and re-applies the exact same (possibly
                # padded) shard layout p has, so local_update lines up with the local
                # shard.
                ortho_dtensor = DTensor.from_local(
                    full_ortho,
                    device_mesh,
                    [Replicate()] * device_mesh.ndim,
                    run_check=False,
                )
                local_update = ortho_dtensor.redistribute(device_mesh, placements).to_local()

                # Step 5: Apply weight decay and update
                # Use full shape for LR scaling (not local shard shape) and the
                # owning param-group's lr/wd (honors lr_multipliers / WD grouping).
                base_lr = self._base_lr_for(p)
                wd = self._wd_for(p)
                adjusted_lr = self._get_adjusted_lr(full_pre_ns.shape, base_lr)

                # NOTE: adjusted_lr may be a tensor (capturable mode), so we scale via
                # multiply rather than ``alpha=`` (Tensor.add_ only accepts a
                # Python-number alpha). ``local_update`` comes out of Newton-Schulz in
                # bf16, so upcast before scaling: a Python-float lr would not promote it
                # and the product would be rounded to bf16 before reaching the param.
                local_param.mul_(1 - base_lr * wd)
                local_param.add_(local_update.to(local_param.dtype) * (-adjusted_lr))

            else:
                # Single-device Muon (non-FSDP or non-sharded)
                grad = get_local_tensor_if_DTensor(p.grad)
                param = get_local_tensor_if_DTensor(p)

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p).float()

                # Compute pre-NS update (momentum + Nesterov). compute_pre_ns_update
                # does not mutate the gradient, so grad can be passed directly.
                pre_ns = compute_pre_ns_update(
                    grad,
                    get_local_tensor_if_DTensor(state["momentum_buffer"]),
                    momentum=self.muon_momentum,
                    nesterov=self.nesterov,
                )

                # Newton-Schulz orthogonalization
                update = zeropower_via_newtonschulz5(pre_ns, steps=self.ns_steps)

                # Get adjusted LR based on matrix size (Moonlight scaling), using
                # the owning param-group's lr/wd.
                base_lr = self._base_lr_for(p)
                wd = self._wd_for(p)
                adjusted_lr = self._get_adjusted_lr(p.shape, base_lr)

                # Apply weight decay (using base LR) and update (using adjusted LR).
                # adjusted_lr may be a tensor (capturable), so scale via multiply, not
                # ``alpha=``; ``update`` is bf16 out of Newton-Schulz, so upcast it first
                # (see the distributed branch above).
                param.mul_(1 - base_lr * wd)
                param.add_(update.to(param.dtype) * (-adjusted_lr))

    def _step_adamw(self) -> None:
        """
        AdamW step using Transformer Engine's fused multi-tensor Adam kernel.

        Iterates over param groups so each group's lr / betas / eps / weight_decay
        (set by the factory's lr_multipliers and disable_weight_decay_for_1d_params)
        is honored. The per-group step counter lives on ``group["step"]``
        (FusedAdam-style) so it is round-tripped by the distributed-checkpoint
        optimizer state dict.

        Params and moments are all FP32 (see :meth:`add_param_group`), so a group is one
        dtype batch, and the kernel updates the param in place -- there is no master
        weight for it to maintain alongside.
        """
        if not self.adamw_params:
            return

        adam_w_mode = 1  # Decoupled weight decay
        bias_correction = 1

        for group in self.param_groups:
            # Only the AdamW-categorized params of this group are handled here;
            # the Muon-categorized params were already updated in _step_muon.
            group_params = [p for p in group["params"] if id(p) in self._adamw_param_ids]
            if not group_params:
                continue

            device = group_params[0].device

            # Per-group step counter stored on the group (FusedAdam convention) so
            # DCP round-trips it on resume.
            if group.get("step", None) is not None:
                if self.capturable and not isinstance(group["step"], torch.Tensor):
                    group["step"] = torch.tensor(group["step"], dtype=torch.int32, device=device)
                group["step"] = (
                    group["step"].to(device=device) if isinstance(group["step"], torch.Tensor) else group["step"]
                )
                group["step"] += 1
            else:
                group["step"] = torch.tensor([1], dtype=torch.int32, device=device) if self.capturable else 1

            if self.capturable and not isinstance(group["lr"], torch.Tensor):
                group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device=device)

            lr = group["lr"]
            wd = group["weight_decay"]
            step = group["step"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            grads, params, exp_avgs, exp_avg_sqs = [], [], [], []

            for p in group_params:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p).float()
                    state["exp_avg_sq"] = torch.zeros_like(p).float()

                grads.append(get_local_tensor_if_DTensor(p.grad))
                params.append(get_local_tensor_if_DTensor(p))
                exp_avgs.append(get_local_tensor_if_DTensor(state["exp_avg"]))
                exp_avg_sqs.append(get_local_tensor_if_DTensor(state["exp_avg_sq"]))

            if not grads:
                continue

            tensor_lists = [grads, params, exp_avgs, exp_avg_sqs]
            if self.capturable:
                # The capturable kernel requires a trailing inverse-scale argument;
                # bf16-only training has no grad scaler, so it is a constant one.
                te.pytorch.optimizers.multi_tensor_applier(
                    self._multi_tensor_adam_capturable,
                    self._dummy_overflow_buf,
                    tensor_lists,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    step,
                    adam_w_mode,
                    bias_correction,
                    wd,
                    torch.ones((1,), device=device, dtype=torch.float32),
                )
            else:
                te.pytorch.optimizers.multi_tensor_applier(
                    self._multi_tensor_adam,
                    self._dummy_overflow_buf,
                    tensor_lists,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    step,
                    adam_w_mode,
                    bias_correction,
                    wd,
                )

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state.

        The optimizer state (per-param momentum / exp_avg / exp_avg_sq and the
        per-group ``step``) round-trips through the base ``torch.optim.Optimizer``
        state dict, so the distributed-checkpoint container can save/restore it
        with FSDP2 resharding just like FusedAdam. There is no master weight to
        restore: the FP32 weight is the parameter itself, which the model checkpoint
        carries.

        This direct-call path (used outside the OptimizersContainer / DCP flow)
        normalizes the capturable LR/step tensors. The state needs no dtype fix-up:
        ``super().load_state_dict`` casts every floating-point state tensor to its
        param's dtype, which the FP32-param invariant makes FP32 -- so the moments and
        the momentum buffer come back FP32 whatever precision the checkpoint stored
        them in.
        """
        super().load_state_dict(state_dict)

        for group in self.param_groups:
            device = group["params"][0].device if group["params"] else "cuda"
            if self.capturable:
                if isinstance(group["lr"], torch.Tensor):
                    group["lr"] = group["lr"].to(device=device)
                else:
                    group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device=device)
                if group.get("step", None) is not None and not isinstance(group["step"], torch.Tensor):
                    group["step"] = torch.tensor(group["step"], dtype=torch.int32, device=device)
