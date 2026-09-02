# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""
Dion2WithAuxAdamW optimizer implementation.

DION2 (Distributed Orthogonalization) for nn.Linear weight matrices,
with auxiliary AdamW for embeddings, biases, norms, and output layers (lm_head).

This implementation combines elements from:

1. Microsoft DION2 (https://github.com/microsoft/dion):
   - All-to-all communication pattern for efficient distributed orthogonalization
   - Submatrix selection (top-k rows/columns by L1 norm)
   - Error feedback for unselected parts
   - Async operations for overlapping communication with computation
     (TBD: not realized yet -- see the all-to-all sites in
     ``_process_dion2_batch_distributed``, which currently ``wait()`` immediately)

2. KellerJordan/Muon (https://github.com/KellerJordan/Muon):
   - Newton-Schulz orthogonalization algorithm
   - Quintic iteration coefficients (a=3.4445, b=-4.7750, c=2.0315)

3. FusedAdam (cosmos_framework/utils/generator/fused_adam.py):
   - DTensor handling for FSDP/TP compatibility
   - Transformer Engine fused AdamW kernel

Key differences from MuonWithAuxAdamW:
- Uses all-to-all instead of all-gather (no redundant NS computation)
- Megabatches params in groups of world_size * K for efficient distribution
- Supports submatrix selection (fraction parameter) for sparse orthogonalization
- Each rank computes NS for exactly one param per batch (truly parallel)

Sharding -> orthogonalization (the core idea)
---------------------------------------------
Under FSDP each weight matrix is a DTensor split across GPUs, but Newton-Schulz
(NS) is a *whole-matrix* op and needs the full matrix in one place. The two
optimizers assemble it differently:

Weight A sharded over 4 GPUs (one row-slice each):

    GPU0:[A0]  GPU1:[A1]  GPU2:[A2]  GPU3:[A3]

Muon -- all-gather: every GPU rebuilds the *same* full A and runs NS on it, so
the NS work is duplicated world_size times:

    all_gather(A) -> each GPU holds full A -> every GPU runs NS(A)   (N x redundant)

DION2 -- all-to-all: process world_size matrices (A,B,C,D) together and give each
GPU one complete matrix, so the N matrices are orthogonalized in parallel with no
redundant compute:

    before (each GPU has one slice of every matrix):

             A    B    C    D
      GPU0 [A0] [B0] [C0] [D0]
      GPU1 [A1] [B1] [C1] [D1]
      GPU2 [A2] [B2] [C2] [D2]
      GPU3 [A3] [B3] [C3] [D3]

      --- all_to_all #1 (transpose) --->  each GPU now owns one FULL matrix:

      GPU0: [A0 A1 A2 A3] = A  -> NS(A)
      GPU1: [B0 B1 B2 B3] = B  -> NS(B)
      GPU2: [C0 C1 C2 C3] = C  -> NS(C)
      GPU3: [D0 D1 D2 D3] = D  -> NS(D)

      --- all_to_all #2 (transpose back) --->  each GPU gets its own
          orthogonalized slice of every matrix, then applies the update.

Matrices are grouped by global shape and dtype into batches of ``world_size * K``
so the all-to-all tensors are uniform (see ``_create_dion2_batches``). With
``fraction < 1.0`` only the top-k rows/cols (by L1 norm) are sent through this
dance, and the unselected part is carried forward via error feedback.

TODO(hsdp-replicate-redundancy): eliminate redundant orthogonalization work in
the replicate dimension under HSDP. ``world_size`` above is the *shard* mesh dim
only -- the all-to-all is confined to the FSDP shard group and the replicate
(data-parallel) dim keeps its ``Replicate`` placement. That is correct, but each
replica group independently reruns the full Newton-Schulz on identical (post
all-reduce) gradients, so the NS *compute* is duplicated ``R`` times (R =
replicate degree; e.g. 2x for the 30B-A3B run at shard_degree=64 on 128 GPUs).
This is the standard data-parallel optimizer redundancy (Adam has it too) but is
pricier here because NS is several matmuls per matrix rather than an element-wise
step. It could be removed by distributing the matrices across the *full* 2-D mesh
(shard x replicate) so every rank orthogonalizes a distinct matrix, then
broadcasting the results back across the replicate dim -- trading the duplicate
compute for an extra cross-replica collective. Worth doing only if R is large or
the optimizer step becomes a step-time bottleneck.
"""

import math
from collections.abc import Callable, Iterable
from typing import ParamSpec, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine as te
import transformer_engine_torch as tex
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Shard

from cosmos_framework.utils import log
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor
from cosmos_framework.utils.generator.aux_optimizer_utils import (
    compute_pre_ns_update,
    compute_pre_ns_updates_and_pack,
    create_moe_megabatches,
    pair_moe_gate_up_down_params,
    split_orthogonalizable_params,
    step_stacked_expert_params,
    validate_split_expert_ns_config,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)
from cosmos_framework.utils.generator.optimizer import require_fp32_param_group

_Dion2PhaseArgs = ParamSpec("_Dion2PhaseArgs")
_Dion2PhaseResult = TypeVar("_Dion2PhaseResult")


@torch.compile(fullgraph=True)
def _apply_dion2_distributed_updates_compiled(
    local_params: list[torch.Tensor],
    back_local: torch.Tensor,
    active_indices: list[int],
    base_lrs: list[float | torch.Tensor],
    weight_decays: list[float],
    adjusted_lr_ratios: list[float],
) -> None:  # local_params: [M,N] each, back_local: [S,...], returns None
    """Apply one active distributed DION2 batch to the FP32 parameters in place."""
    for local_param, active_index, base_lr, weight_decay, adjusted_lr_ratio in zip(
        local_params, active_indices, base_lrs, weight_decays, adjusted_lr_ratios
    ):
        update_local = back_local[active_index]  # matching local update shard
        local_param.mul_(1 - base_lr * weight_decay)  # local parameter shard
        local_param.add_(
            update_local.to(local_param.dtype) * base_lr, alpha=-adjusted_lr_ratio
        )  # local parameter shard


class Dion2WithAuxAdamW(torch.optim.Optimizer):
    """
    Dion2WithAuxAdamW optimizer.

    Uses DION2 (Distributed Orthogonalization) for nn.Linear hidden weight matrices,
    and AdamW for embeddings, biases, layer norms, and output heads (lm_head).

    Key features:
    - All-to-all communication for efficient distributed NS (no redundant compute)
    - Submatrix selection: only orthogonalize top-k rows/columns
    - Error feedback: maintains momentum for unselected parts
    - Megabatched processing: handles world_size * K same-shape params per redistribution

    Parameter precision: every parameter must be FP32, which
    :meth:`add_param_group` enforces. Both the DION2 and the auxiliary AdamW update
    are applied to the parameter in place in FP32, so there are no FP32 master
    weights. To keep the forward/backward in BF16, wrap the model in FSDP2 with
    ``MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)``:
    FSDP down-casts for compute and all-gather while the sharded parameter this
    optimizer steps stays FP32. Do not cast the parameters themselves.

    Args:
        params: Iterable of parameters to optimize.
        lr: Base learning rate.
        muon_momentum: Momentum coefficient for Muon/DION2.
        muon_lr_scale: Scale factor for Muon LR adjustment.
        ns_steps: Number of Newton-Schulz iterations.
        nesterov: Whether to use Nesterov momentum.
        fraction: Fraction of rows/columns to orthogonalize (0 < fraction <= 1).
        ef_decay: Error feedback decay factor for selected submatrix.
        weight_decay: Weight decay for all parameters.
        adam_betas: Beta coefficients for the auxiliary AdamW side.
        eps: Epsilon for AdamW numerical stability.
        use_distributed: Whether to use distributed operations.
        capturable: Must be ``True`` (the default); passing ``False`` raises. Only the
            CUDA-graph-capturable update is implemented: ``lr`` and the per-group ``step``
            are kept as device tensors and dispatched to Transformer Engine's
            ``multi_tensor_adam_capturable``. The keyword is retained because the
            factory in ``utils/optimizer.py`` passes it explicitly.
        max_dion2_megabatch_width: Maximum number of same-shape matrices processed per rank.
        dion2_profile_phases: Whether to emit Torch Profiler and NVTX phase ranges.
        split_expert_gate_up: When True, both routed gate_up_proj [E,H,2I]
            parameters and fused 2-D shared-expert gate_up_proj [2I,H] parameters
            are split into separate gate and up matrices before Newton-Schulz.
            Enables finer-grained orthogonalization and routed-expert megabatching.
        batch_split_expert_ns: When True (and split_expert_gate_up=True), gate+up+down
            matrices for one layer are batched into a single NS call. Superseded by
            the megabatch path when max_moe_expert_ns_matrices > 0.
        max_moe_expert_ns_matrices: Maximum total number of [E, H, I] expert matrices
            per NS call across K layers. K = max(1, value // (3 * E_local)). 0 (default)
            means K=1 (one layer per NS call). Requires split_expert_gate_up=True.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter | dict[str, object]],
        lr: float = 1e-4,
        muon_momentum: float = 0.95,
        muon_lr_scale: float = 0.2,
        ns_steps: int = 5,
        nesterov: bool = True,
        fraction: float = 1.0,  # 1.0 = full matrix, <1.0 = submatrix selection
        ef_decay: float = 0.95,
        weight_decay: float = 0.1,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        use_distributed: bool = True,
        capturable: bool = True,
        expert_param_keywords: tuple[str, ...] | None = None,
        orthogonalize_skip_patterns: tuple[str, ...] | None = None,
        max_dion2_megabatch_width: int = 25,
        dion2_profile_phases: bool = False,
        split_expert_gate_up: bool = False,
        batch_split_expert_ns: bool = False,
        max_moe_expert_ns_matrices: int = 0,
        **kwargs: object,
    ) -> None:
        if "dion2_megabatch_width" in kwargs:
            raise TypeError("dion2_megabatch_width has been removed; use max_dion2_megabatch_width instead")
        if "master_weights" in kwargs:
            # Not silently ignored: a caller asking for master weights is asking for
            # precision this optimizer no longer provides that way, and would otherwise
            # get a warning buried in the log.
            raise ValueError(
                "Dion2WithAuxAdamW no longer maintains FP32 master weights -- its parameters "
                "must already be FP32, which makes a master a bit-identical duplicate (see the "
                "class docstring). Drop the master_weights argument."
            )

        if kwargs:
            ignored_keys = list(kwargs.keys())
            expected_ignored = {"fused", "keys_to_select", "adamw_betas", "adamw_eps"}
            unexpected = set(ignored_keys) - expected_ignored
            if unexpected:
                import warnings

                warnings.warn(f"Dion2WithAuxAdamW ignoring unexpected kwargs: {unexpected}")

        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if isinstance(max_dion2_megabatch_width, bool) or not isinstance(max_dion2_megabatch_width, int):
            raise TypeError(
                f"max_dion2_megabatch_width must be a positive non-boolean integer, got {max_dion2_megabatch_width!r}"
            )
        if max_dion2_megabatch_width < 1:
            raise ValueError(f"max_dion2_megabatch_width must be at least 1, got {max_dion2_megabatch_width}")
        if not isinstance(dion2_profile_phases, bool):
            raise TypeError(f"dion2_profile_phases must be a bool, got {dion2_profile_phases!r}")
        validate_split_expert_ns_config(split_expert_gate_up, batch_split_expert_ns, fraction=fraction)
        if max_moe_expert_ns_matrices < 0:
            raise ValueError(f"max_moe_expert_ns_matrices must be >= 0, got {max_moe_expert_ns_matrices}")

        # Only the capturable update is implemented: lr and the per-group step are held as
        # device tensors and dispatched to TE's multi_tensor_adam_capturable. The factory in
        # utils/optimizer.py always passes True, so the non-capturable branches were dead
        # and have been removed.
        if not capturable:
            raise ValueError("Dion2WithAuxAdamW only supports capturable=True.")

        # Store hyperparameters
        # Note: lr is accessed via property that reads from param_groups
        # to support LR schedulers (which update param_groups[X]["lr"])
        self.wd = weight_decay
        self.muon_momentum = muon_momentum
        self.muon_lr_scale = muon_lr_scale
        # Shape -> LR scaling ratio; see _get_adjusted_lr_ratio.
        self._adjusted_lr_ratios: dict[tuple[int, ...], float] = {}
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.fraction = fraction
        self.ef_decay = ef_decay
        self.adam_betas = tuple(adam_betas) if isinstance(adam_betas, list) else adam_betas
        self.eps = eps
        self.max_dion2_megabatch_width: int = max_dion2_megabatch_width
        self.dion2_profile_phases = dion2_profile_phases
        self.split_expert_gate_up = split_expert_gate_up
        self.batch_split_expert_ns = batch_split_expert_ns
        self.max_moe_expert_ns_matrices = max_moe_expert_ns_matrices

        # Name substrings that route stacked MoE expert params ([E, M, N]) to the
        # DION2 side (each expert slice orthogonalized). Empty = experts stay on
        # AdamW (no behavior change).
        self.expert_param_keywords = tuple(expert_param_keywords) if expert_param_keywords else ()
        # Regex patterns (matched against param names) that force matching 2D Linear
        # weights onto the auxiliary AdamW side (skip DION2 orthogonalization), in
        # addition to the default head detection. E.g. r"\.gate\.weight$" for MoE
        # routers. Empty = nothing extra skipped (no behavior change).
        self.orthogonalize_skip_patterns = tuple(orthogonalize_skip_patterns) if orthogonalize_skip_patterns else ()

        # Distributed settings
        self.use_distributed = use_distributed and dist.is_initialized()
        self._world_size = 1
        self._device_rank = 0
        self._process_group = None
        self._device_mesh: DeviceMesh | None = None

        # ``capturable`` is always True (enforced above) and is kept as an attribute
        # because external code duck-types on it to detect the group-level step
        # convention (see ``DistillationTrainer._uses_group_step``).
        self.capturable = capturable

        # Parameter lists
        self.dion2_params: list[nn.Parameter] = []
        self.adamw_params: list[nn.Parameter] = []
        # Stacked MoE expert params ([E, M, N]); orthogonalized per expert slice.
        self.stacked_dion2_params: list[nn.Parameter] = []
        self.param_to_name: dict[nn.Parameter, str] = {}
        self._dion2_batches: list[list[nn.Parameter]] = []
        # Frozen during parameter categorization so optimizer steps never need to
        # inspect parameter names. Batch membership is frozen when batches are built.
        self._shared_expert_gate_up_param_ids: set[int] = set()
        self._split_shared_gate_up_batch_ids: set[int] = set()
        # Split-expert pair tracking (gate_up + down pairs for multi-layer NS batching).
        self._split_expert_pairs: list[tuple[nn.Parameter, nn.Parameter]] = []
        self._split_expert_param_ids: set[int] = set()
        self._moe_megabatches: list[list[tuple[nn.Parameter, nn.Parameter]]] = []

        # Transformer Engine fused Adam. The zero buffer is the noop flag required
        # as the second argument of TE's multi_tensor_applier; it is a fixed
        # constant here (no AMP overflow handling).
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self._multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable

        # Initialize base optimizer. betas / eps go in the defaults so each param
        # group carries them (FusedAdam convention); the AdamW step reads them
        # per-group, enabling per-group overrides and exact FusedAdam parity.
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=self.adam_betas, eps=eps)
        super().__init__(params, defaults)

        # Convert LR to a device tensor: the capturable kernels read it from device memory
        # (and an LR scheduler then updates it in place rather than rebinding a float).
        for idx, group in enumerate(self.param_groups):
            if len(group["params"]) == 0:
                continue
            device = group["params"][0].device
            if isinstance(group["lr"], float):
                group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)
            self.param_groups[idx]["lr"] = group["lr"].to(device=device)

        # id(param) -> owning param_group, so the DION2 and AdamW updates can read
        # the *per-group* lr / weight_decay (honors lr_multipliers and
        # disable_weight_decay_for_1d_params). With a single group it degenerates
        # to a single global lr/wd, matching the original (reference) behavior.
        self._param_group_map: dict[int, dict] = {}
        self._adamw_param_ids: set[int] = set()

        log.info(f"Dion2WithAuxAdamW capturable: {capturable}")

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
            "Dion2WithAuxAdamW",
            "it applies both the DION2 and the AdamW update to the parameter in place in "
            "FP32 and keeps no master weights",
        )
        super().add_param_group(param_group)

    def categorize_params(self, model: nn.Module) -> None:
        """
        Categorize parameters into DION2 and AdamW groups; also set up distributed
        configuration from the DTensor DeviceMesh.

        DION2 is used for hidden ``nn.Linear`` weights only. Embeddings, the output
        head (``lm_head`` / tied / vocab-shaped projection), biases, norms, and any
        non-Linear parameter go to AdamW. See
        :func:`split_orthogonalizable_params` for the architecture-agnostic
        embedding / output-head detection.
        """
        optimizer_param_ids = {id(p) for group in self.param_groups for p in group["params"]}

        orthogonalizable, self.adamw_params, self.param_to_name = split_orthogonalizable_params(
            model,
            optimizer_param_ids,
            expert_param_keywords=self.expert_param_keywords,
            orthogonalize_skip_patterns=self.orthogonalize_skip_patterns,
        )

        # Every optimizer param lands in exactly ONE of three disjoint buckets,
        # each updated by a different function in step():
        #   1. self.dion2_params    -> _step_dion2   (dense 2D Linear weights)
        #   2. self.stacked_dion2_params -> _step_stacked_dion2 (3D MoE experts [E, M, N])
        #   3. self.adamw_params   -> _step_adamw   (embeddings/head/norms/biases/1D)
        # split_orthogonalizable_params separates the orthogonalizable Linear
        # weights (buckets 1+2) from everything else (bucket 3); here we further
        # split the orthogonalizable set into 2D (DION2 all-to-all path) vs stacked
        # 3D MoE experts (orthogonalized per expert slice in _step_stacked_dion2).
        self.dion2_params = [p for p in orthogonalizable if p.ndim == 2]
        self.stacked_dion2_params = [p for p in orthogonalizable if p.ndim >= 3]

        self._freeze_shared_expert_gate_up_params()

        # Sort by size for load balancing
        self.dion2_params = sorted(self.dion2_params, key=lambda x: x.numel(), reverse=True)

        # Build gate_up/down pairs for split-expert NS (if requested).
        self._split_expert_pairs = []
        self._split_expert_param_ids = set()
        if self.split_expert_gate_up and self.stacked_dion2_params:
            self._split_expert_pairs = pair_moe_gate_up_down_params(self.stacked_dion2_params, self.param_to_name)
            for gate_up_p, down_p in self._split_expert_pairs:
                self._split_expert_param_ids.add(id(gate_up_p))
                self._split_expert_param_ids.add(id(down_p))

        # Setup distributed from first DTensor param
        self._setup_distributed_from_params()

        # Create same-shape batches using a per-group width capped by max_dion2_megabatch_width.
        self._create_dion2_batches()

        # Build MoE megabatch plan (K pairs per NS call).
        self._create_moe_megabatches()

        dion2_numel = sum(p.numel() for p in self.dion2_params)
        adamw_numel = sum(p.numel() for p in self.adamw_params)
        stacked_dion2_numel = sum(p.numel() for p in self.stacked_dion2_params)

        log.info(
            f"Dion2WithAuxAdamW: {len(self.dion2_params)} Muon params ({dion2_numel:,} elements), "
            f"{len(self.stacked_dion2_params)} stacked-expert params ({stacked_dion2_numel:,} elements), "
            f"{len(self.adamw_params)} AdamW params ({adamw_numel:,} elements), "
            f"world_size={self._world_size}, max_megabatch_width={self.max_dion2_megabatch_width}, "
            f"{len(self._dion2_batches)} redistribution rounds"
        )

        # Log Muon param details
        log.info("DION2 parameters (layer name -> shape):")
        for p in self.dion2_params:
            name = self.param_to_name.get(p, "unknown")
            log.info(f"  {name}: {tuple(p.shape)}")

        # Build the param -> owning group map for per-group lr / weight_decay
        # lookups during the DION2 and AdamW steps.
        self._param_group_map = {}
        for group in self.param_groups:
            for p in group["params"]:
                self._param_group_map[id(p)] = group
        self._adamw_param_ids = {id(p) for p in self.adamw_params}

    def _assert_homogeneous_sharding(self) -> None:
        """Verify every DION2 param shares the first param's mesh and sharding.

        The all-to-all path caches a single distributed config (device_mesh, shard
        mesh dim, shard tensor dim, world_size, process group) derived from
        ``dion2_params[0]`` and applies it to *every* DION2 param. That is only
        correct if all params are sharded identically. Heterogeneous sharding -- a
        param on a different mesh, sharded on a different dim, or replicated while
        others are sharded -- would be processed with the wrong layout and silently
        corrupt the update, so it is rejected up front (once, at setup). VFM's FSDP2
        shards every 2-D weight on dim 0 with one mesh, so this is a no-op today; the
        guard fails loudly if TP/EP or mixed replication is ever introduced.
        """
        p0 = self.dion2_params[0]
        for p in self.dion2_params[1:]:
            mismatch = isinstance(p, DTensor) != isinstance(p0, DTensor) or (
                isinstance(p0, DTensor) and (p.device_mesh != p0.device_mesh or p.placements != p0.placements)
            )
            if mismatch:
                name = self.param_to_name.get(p, "unknown")
                raise NotImplementedError(
                    f"DION2 requires all params to share the first param's sharding, but '{name}' "
                    f"differs (placements={getattr(p, 'placements', None)}). "
                    f"Heterogeneous sharding is unsupported."
                )

    def _setup_distributed_from_params(self) -> None:
        """Extract distributed config from DTensor DeviceMesh."""
        if not self.dion2_params:
            return

        # The config below is derived from dion2_params[0] and reused for every
        # param, so first confirm they are all sharded identically.
        self._assert_homogeneous_sharding()

        first_param = self.dion2_params[0]
        if isinstance(first_param, DTensor):
            device_mesh = first_param.device_mesh
            placements = first_param.placements

            # Find the shard dimension in the mesh
            for mesh_dim_idx, placement in enumerate(placements):
                if placement.is_shard():
                    self._device_mesh = device_mesh
                    self._world_size = device_mesh.size(mesh_dim=mesh_dim_idx)
                    self._device_rank = device_mesh.get_local_rank(mesh_dim=mesh_dim_idx)
                    self._process_group = device_mesh.get_group(mesh_dim=mesh_dim_idx)
                    self._shard_mesh_dim = mesh_dim_idx
                    self._shard_tensor_dim = placement.dim
                    log.info(
                        f"DION2 distributed setup: world_size={self._world_size}, "
                        f"rank={self._device_rank}, shard_dim={self._shard_tensor_dim}"
                    )
                    return

        # Fallback: not distributed or not sharded
        self._world_size = 1
        self._device_rank = 0

    def _create_dion2_batches(self) -> None:
        """
        Group Muon params by GLOBAL shape and dtype, then batch within each group.

        The distributed step (``_process_dion2_batch_distributed``) stacks a batch of
        params into a single DTensor and redistributes it, so the batches must be
        identical ACROSS ranks. Group by the *global* shape (same on every rank), NOT
        the local shard shape -- under uneven sharding the local shard shape differs
        per rank and would make ranks build inconsistent batches. Each group uses
        ``min(ceil(group_size / world_size),
        max_dion2_megabatch_width)`` matrices per rank. A partial tail uses the
        smallest effective width that fits it, so its capacity is the nearest
        multiple of ``world_size`` and each rank still receives complete matrices.
        """
        self._dion2_batches = []
        self._split_shared_gate_up_batch_ids = set()

        # Step 1: Group params by global shape and dtype (identical on all ranks).
        shape_groups: dict[tuple[tuple[int, ...], torch.dtype, bool], list[nn.Parameter]] = {}
        for p in self.dion2_params:
            # Split shared-expert gate/up params must occupy their own redistribution
            # batches so every full matrix received by a rank has the same NS layout.
            group_key = (tuple(p.shape), p.dtype, self._is_shared_expert_gate_up(p))
            if group_key not in shape_groups:
                shape_groups[group_key] = []
            shape_groups[group_key].append(p)

        # Step 2: Create batches using the independently capped width of each shape group.
        for params in shape_groups.values():
            effective_width = self._dion2_group_megabatch_width(len(params))
            group_capacity = self._world_size * effective_width
            for i in range(0, len(params), group_capacity):
                batch = params[i : i + group_capacity]
                self._dion2_batches.append(batch)

        # Classification is invariant after categorization. Verify the batching
        # invariant once here, outside the per-step collective path, and cache the
        # split status on the batch object itself.
        for batch in self._dion2_batches:
            split_shared_gate_up = self._is_shared_expert_gate_up(batch[0])
            if any(self._is_shared_expert_gate_up(p) != split_shared_gate_up for p in batch[1:]):
                raise RuntimeError("DION2 dense batch mixed split and unsplit shared-expert gate/up parameters")
            if split_shared_gate_up:
                self._split_shared_gate_up_batch_ids.add(id(batch))

        # Log batch info
        num_shape_groups = len(shape_groups)
        num_batches = len(self._dion2_batches)
        if self._dion2_batches:
            # Count batches that need padding
            padded_batches = sum(1 for batch in self._dion2_batches if len(batch) % self._world_size != 0)
            log.info(
                f"DION2: {len(self.dion2_params)} params grouped into {num_shape_groups} shape groups, "
                f"{num_batches} redistribution rounds (world_size={self._world_size}, "
                f"max_megabatch_width={self.max_dion2_megabatch_width}, "
                f"{padded_batches} need padding)"
            )
            # Log shape group details
            for (shape, dtype, split_shared_gate_up), params in shape_groups.items():
                natural_width = math.ceil(len(params) / self._world_size)
                effective_width = self._dion2_group_megabatch_width(len(params))
                group_capacity = self._world_size * effective_width
                rounds = math.ceil(len(params) / group_capacity)
                padded_slots = (-len(params)) % self._world_size
                log.info(
                    f"  Shape {shape}, dtype={dtype}, split_shared_gate_up={split_shared_gate_up}: "
                    f"{len(params)} params, "
                    f"natural_width={natural_width}, effective_width={effective_width}, "
                    f"capacity={group_capacity}, {rounds} rounds, {padded_slots} padded slots"
                )

    def _dion2_group_megabatch_width(self, group_size: int) -> int:
        """Return the independently capped per-rank width for one shape group."""
        if group_size < 1:
            raise ValueError(f"DION2 shape group size must be positive, got {group_size}")
        return min(math.ceil(group_size / self._world_size), self.max_dion2_megabatch_width)

    def _effective_dion2_batch_width(self, actual_batch_size: int) -> int:
        """Return the smallest per-rank matrix count that can fit a dense batch."""
        max_batch_size = self._world_size * self.max_dion2_megabatch_width
        if actual_batch_size < 1 or actual_batch_size > max_batch_size:
            raise ValueError(f"DION2 batch size must be in [1, {max_batch_size}], got {actual_batch_size}")
        return math.ceil(actual_batch_size / self._world_size)

    @staticmethod
    def _validate_redistributed_matrix_shape(
        local_matrices: torch.Tensor,
        effective_width: int,
        expected_matrix_shape: tuple[int, ...],
    ) -> None:
        """Verify that redistribution produced complete logical matrices."""
        if local_matrices.shape[0] != effective_width:
            raise RuntimeError(
                f"DION2 expected {effective_width} full matrices per rank after redistribution, "
                f"got local shape {tuple(local_matrices.shape)}"
            )
        if tuple(local_matrices.shape[1:]) != expected_matrix_shape:
            raise RuntimeError(
                f"DION2 expected full matrix shape {expected_matrix_shape} after redistribution, "
                f"got local shape {tuple(local_matrices.shape)}"
            )

    def _base_lr_for(self, p: nn.Parameter) -> float | torch.Tensor:
        """Per-group base learning rate for ``p`` (honors lr_multipliers)."""
        return self._param_group_map[id(p)]["lr"]

    def _wd_for(self, p: nn.Parameter) -> float:
        """Per-group weight decay for ``p`` (honors disable_weight_decay_for_1d_params)."""
        return self._param_group_map[id(p)]["weight_decay"]

    def _get_adjusted_lr(self, param_shape: tuple[int, ...], base_lr: float | torch.Tensor) -> float | torch.Tensor:
        """Compute adjusted learning rate based on parameter matrix size and the
        owning param-group's base lr."""
        return base_lr * self._get_adjusted_lr_ratio(param_shape)  # [] when base_lr is a tensor

    def _get_adjusted_lr_ratio(self, param_shape: tuple[int, ...]) -> float:
        """Compute the shape-dependent scalar applied to the base learning rate.

        Memoized: this depends only on the shape and ``muon_lr_scale``, both fixed for
        the run, but the MoE megabatch path asks for it once per matrix per step.
        """
        ratio = self._adjusted_lr_ratios.get(param_shape)
        if ratio is None:
            A, B = param_shape[:2]
            ratio = self.muon_lr_scale * math.sqrt(max(A, B))
            self._adjusted_lr_ratios[param_shape] = ratio
        return ratio

    def _matches_shared_expert_gate_up_name(self, p: nn.Parameter) -> bool:
        """Whether ``p`` has the configured shared-expert gate/up name."""
        if not self.split_expert_gate_up:
            return False
        name_parts = self.param_to_name.get(p, "").split(".")
        return name_parts[-3:] == ["shared_expert", "gate_up_proj", "weight"]

    def _freeze_shared_expert_gate_up_params(self) -> None:
        """Resolve name-based shared-expert classification once at setup time."""
        shared_expert_gate_up_params = [p for p in self.dion2_params if self._matches_shared_expert_gate_up_name(p)]
        self._shared_expert_gate_up_param_ids = {id(p) for p in shared_expert_gate_up_params}
        for p in shared_expert_gate_up_params:
            self._validate_shared_expert_gate_up_shape(p)
            log.info(f"DION2 split shared-expert gate/up parameter: {self.param_to_name[p]} shape={tuple(p.shape)}")

    def _is_shared_expert_gate_up(self, p: nn.Parameter) -> bool:
        """Whether ``p`` was classified as a fused shared-expert gate/up projection."""
        return id(p) in self._shared_expert_gate_up_param_ids

    def _validate_shared_expert_gate_up_shape(self, p: nn.Parameter) -> None:
        """Validate the physical [2I,H] layout required for virtual splitting."""
        if p.ndim != 2 or p.shape[0] % 2 != 0:
            name = self.param_to_name.get(p, "unknown")
            raise ValueError(
                f"split_expert_gate_up expected shared expert '{name}' to have shape [2I,H], got {tuple(p.shape)}"
            )

    def _get_adjusted_lr_ratio_for_param(self, p: nn.Parameter) -> float:
        """Return LR scaling using a shared gate/up half's logical shape."""
        shape = tuple(p.shape)
        if self._is_shared_expert_gate_up(p):
            shape = (shape[0] // 2, shape[1])
        return self._get_adjusted_lr_ratio(shape)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Params are split into three disjoint buckets at init (see
        # categorize_params) and each is updated by a different function below:
        #   1. self.dion2_params    (dense 2D linears)     -> _step_dion2
        #   2. self.stacked_dion2_params (3D MoE experts [E,M,N]) -> _step_stacked_dion2
        #   3. self.adamw_params   (embeddings/head/norms/1D) -> _step_adamw
        # Every parameter belongs to exactly one bucket, so exactly one of these
        # updates it. Order does not matter (buckets are disjoint).

        # 1. Dense 2D linears: orthogonalized via all-to-all distributed Newton-Schulz.
        self._step_dion2()

        # 2. MoE expert weights: each [E, M, N] param orthogonalized one expert
        #    slice at a time (local NS; per-expert masking for inactive experts).
        self._step_stacked_dion2()

        # 3. Everything else (embeddings, lm_head, norms, biases, 1D): fused AdamW.
        self._step_adamw()

        return loss

    def _step_stacked_dion2(self) -> None:
        """Orthogonalize stacked MoE expert params via step_stacked_expert_params.

        Dispatches to split-expert or megabatch path when ``split_expert_gate_up``
        is True; falls back to the historical whole-param per-expert NS otherwise.

        See :func:`step_stacked_expert_params` for the full dispatch logic.
        """
        if not self.stacked_dion2_params:
            return

        moe_megabatches = self._moe_megabatches if self.split_expert_gate_up else None

        step_stacked_expert_params(
            self.stacked_dion2_params,
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
            profile_phases=self.dion2_profile_phases,
        )

    def _create_moe_megabatches(self) -> None:
        """Group split expert pairs into K-layer NS batches (see
        :func:`create_moe_megabatches`)."""
        self._moe_megabatches = create_moe_megabatches(
            self._split_expert_pairs, self._world_size, self.max_moe_expert_ns_matrices
        )

    def _step_dion2(self) -> None:
        """
        DION2 step with all-to-all distributed Newton-Schulz.

        For each batch of world_size * K params:
        1. Compute momentum + select submatrix on local shards
        2. All-to-all to redistribute shards (each rank gets full submatrix for its param)
        3. Newton-Schulz on K full submatrices per rank (2-D specialization for K=1)
        4. All-to-all to scatter results back
        5. Apply weight decay and update (in place, in FP32, on the param)
        """
        if not self.dion2_params:
            return

        for batch in self._dion2_batches:
            self._process_dion2_batch(batch)

    def _run_dion2_phase(
        self,
        name: str,
        callback: Callable[_Dion2PhaseArgs, _Dion2PhaseResult],
        *args: _Dion2PhaseArgs.args,
        **kwargs: _Dion2PhaseArgs.kwargs,
    ) -> _Dion2PhaseResult:
        """Run one optimizer phase, adding profiler and NVTX ranges only when requested."""
        if not self.dion2_profile_phases:
            return callback(*args, **kwargs)
        with torch.profiler.record_function(name), torch.cuda.nvtx.range(name):
            return callback(*args, **kwargs)

    def _dion2_orthogonalize(self, matrices: torch.Tensor) -> torch.Tensor:  # matrices/returns: [K,M,N]
        if matrices.shape[0] == 1:
            ortho = zeropower_via_newtonschulz5(matrices[0], steps=self.ns_steps)  # [M,N]
            return ortho.unsqueeze(0)  # [1,M,N]
        return zeropower_via_newtonschulz5_batched(matrices, steps=self.ns_steps)  # [K,M,N]

    def _dion2_orthogonalize_dense_batch(
        self,
        matrices: torch.Tensor,
        *,
        split_shared_gate_up: bool,
    ) -> torch.Tensor:  # matrices/returns: [K,M,N]
        """Orthogonalize reconstructed dense matrices, virtually splitting shared gate/up."""
        if not split_shared_gate_up:
            return self._dion2_orthogonalize(matrices)
        if matrices.ndim != 3 or matrices.shape[1] % 2 != 0:
            raise ValueError(
                "split_expert_gate_up expected reconstructed shared-expert matrices with shape [K,2I,H], "
                f"got {tuple(matrices.shape)}"
            )
        batch_size, fused_rows, hidden_size = matrices.shape
        logical_matrices = matrices.reshape(batch_size * 2, fused_rows // 2, hidden_size)  # [2K,I,H]
        logical_ortho = self._dion2_orthogonalize(logical_matrices)  # [2K,I,H]
        return logical_ortho.reshape(batch_size, fused_rows, hidden_size)  # [K,2I,H]

    def _dion2_pre_ns_updates_and_pack(
        self,
        grads: list[torch.Tensor],
        momentum_buffers: list[torch.Tensor],
        active_indices: list[int],
        local_param_shard: torch.Tensor,
        actual_batch_size: int,
        batch_capacity: int,
    ) -> torch.Tensor:  # grads/momentum_buffers/local_param_shard: [M,N] each, returns [S,M,N]
        """Run and pack local batched pre-NS updates inside one compiled graph."""
        if any(isinstance(tensor, DTensor) for tensor in (*grads, *momentum_buffers, local_param_shard)):
            raise TypeError("DION2 compiled pre-NS packing requires plain local tensors, not DTensors")
        return compute_pre_ns_updates_and_pack(
            grads,
            momentum_buffers,
            active_indices,
            local_param_shard,
            actual_batch_size,
            batch_capacity,
            momentum=self.muon_momentum,
            nesterov=self.nesterov,
            output_dtype=torch.bfloat16,
        )  # [S,M,N]

    def _dion2_reverse_redistribute(
        self,
        ortho_local: torch.Tensor,
        fwd_placements: list[Placement],
        back_placements: list[Placement],
    ) -> DTensor:  # ortho_local: [K,M,N], returns [S,...]
        if self._device_mesh is None:
            raise RuntimeError("DION2 distributed processing requires an initialized device mesh")
        ortho_dt = DTensor.from_local(  # [S,M,N]
            ortho_local,
            self._device_mesh,
            fwd_placements,
            run_check=False,
        )
        return ortho_dt.redistribute(self._device_mesh, back_placements)  # [S,...]

    def _apply_dion2_distributed_updates(
        self,
        batch: list[nn.Parameter],
        actual_batch_size: int,
        active: list[bool],
        back_local: torch.Tensor,
    ) -> None:  # back_local: [S,...]
        local_params: list[torch.Tensor] = []
        active_indices: list[int] = []
        base_lrs: list[float | torch.Tensor] = []
        weight_decays: list[float] = []
        adjusted_lr_ratios: list[float] = []
        for i in range(actual_batch_size):
            if not active[i]:
                # No gradient this step: momentum was frozen above; skip weight
                # decay and the update entirely (matches the reference, which
                # leaves a None-grad param completely untouched).
                continue
            p = batch[i]
            local_param = p._local_tensor  # local parameter shard
            base_lr = self._base_lr_for(p)
            wd = self._wd_for(p)
            adjusted_lr_ratio = self._get_adjusted_lr_ratio_for_param(p)
            local_params.append(local_param)
            active_indices.append(i)
            base_lrs.append(base_lr)
            weight_decays.append(wd)
            adjusted_lr_ratios.append(adjusted_lr_ratio)

        if not local_params:
            return
        _apply_dion2_distributed_updates_compiled(
            local_params,
            back_local,
            active_indices,
            base_lrs,
            weight_decays,
            adjusted_lr_ratios,
        )

    def _process_dion2_batch(self, batch: list[nn.Parameter]) -> None:
        """Process a single batch of params using DION2 all-to-all pattern."""
        world_size = self._world_size

        actual_batch_size = len(batch)
        split_shared_gate_up = id(batch) in self._split_shared_gate_up_batch_ids

        # Check if using DTensor (FSDP)
        is_dtensor = isinstance(batch[0], DTensor)

        if is_dtensor and world_size > 1:
            self._process_dion2_batch_distributed(
                batch,
                actual_batch_size,
                split_shared_gate_up=split_shared_gate_up,
            )
        else:
            self._process_dion2_batch_single(batch)

    def _process_dion2_batch_distributed(
        self,
        batch: list[nn.Parameter],
        actual_batch_size: int,
        *,
        split_shared_gate_up: bool,
    ) -> None:
        """Process a batch via DTensor collectives (the "each rank orthogonalizes one
        whole param" transpose), correct for uneven / non-divisible shard dims.

        Rather than a hand-rolled ``all_to_all`` + ``cat``/``narrow`` (which assumed
        FSDP2 padded every local shard uniformly -- it does not; ``_local_tensor`` is
        unpadded and uneven), the gather/scatter is expressed through DTensor:

          1. Momentum + Nesterov per param on plain local tensors, then pack the
             world_size*K local shards into one ``[W*K, ...]`` buffer.
          2. Wrap the packed buffer as a DTensor with explicit global shape/stride;
             the original shard tensor dim shifts to ``shard_dim + 1``.
          3. ``redistribute`` so the PARAM axis is sharded on the FSDP shard mesh dim
             -> each rank owns K whole params (forward all-to-all).
          4. Newton-Schulz on those K whole params (2-D specialization for K=1).
          5. ``redistribute`` back to shard the data axis (backward all-to-all),
             unstack, and apply the update to each local shard.

        DTensor owns the (possibly uneven) per-rank size bookkeeping, so this is
        correct regardless of divisibility. The four-rank transpose, uneven-shard,
        padded-batch, BF16, shard-dimension, and empty-local-shard cases are covered
        by ``dion2_with_aux_adamw_distributed_test.py``. The test owns its
        worker-process launch so ordinary pytest runs the distributed path.

        Submatrix selection (``fraction < 1``) is not supported on this path -- it is
        unused (every config runs fraction=1.0) -- and is rejected up front.
        """
        # distributed path. Will need to wire it in is a
        # well-scoped change: per-param DTensor select (norms via
        # ``pre.abs().sum(sharded_dim).full_tensor()`` -> top-k -> ``index_select``
        # with a Replicate index), error-feedback decay on the momentum buffer, and a
        # branched apply that reuses ``_apply_submatrix_update`` (index_add
        # into the selected indices). Deferred for now: every config runs fraction=1.0,
        # so this path is unused and not worth the added complexity yet. Single-device
        # fraction<1 still works via ``_process_dion2_batch_single``.
        if self.fraction != 1.0:
            raise NotImplementedError(
                "DION2 distributed path supports only fraction=1.0 (full-matrix "
                f"orthogonalization); got fraction={self.fraction}. Submatrix selection "
                "under FSDP is not implemented yet (see TODO(dion2-fsdp-fraction))."
            )

        world_size = self._world_size
        effective_width = self._effective_dion2_batch_width(actual_batch_size)
        batch_capacity = world_size * effective_width
        if self._device_mesh is None:
            raise RuntimeError("DION2 distributed processing requires an initialized device mesh")
        shard_mesh_dim = self._shard_mesh_dim
        shard_dim = self._shard_tensor_dim
        first_param = batch[0]
        if not isinstance(first_param, DTensor):
            raise RuntimeError("DION2 distributed processing requires DTensor parameters")
        expected_param_placement = Shard(shard_dim)
        if first_param.placements[shard_mesh_dim] != expected_param_placement:
            raise RuntimeError(
                f"DION2 expected parameter placement {expected_param_placement} on mesh dim {shard_mesh_dim}, "
                f"got {first_param.placements}."
            )

        # Step 1: momentum + Nesterov on plain local tensors. This keeps DTensor
        # subclass dispatch outside torch.compile while mutating the local views of
        # each active sharded momentum buffer in place.
        #
        # None-grad handling: a param with no gradient this step is sat out --
        # momentum frozen (excluded from the compiled pre-NS update, so no mu-decay) and no
        # update applied (skipped in the apply loop via ``active``). We cannot just
        # drop the slot: the pack + redistribute all-to-alls are a fixed-size
        # collective every rank must enter identically, so inactive and padding slots
        # share one zero placeholder before the local stack.
        # Newton-Schulz on zeros stays finite (norm+1e-7)
        # and the result is discarded on apply. This relies on ``p.grad is None``
        # being identical across ranks -- true for dense params, where a missing grad
        # is structural (an unused param is None on every rank), not data-dependent.
        active_grads: list[torch.Tensor] = []
        active_momentum_buffers: list[torch.Tensor] = []
        active_indices: list[int] = []
        active: list[bool] = []
        for i in range(actual_batch_size):
            p = batch[i]
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p).float()  # [M,N]
            if p.grad is None:
                active.append(False)
                continue
            active.append(True)
            active_grads.append(get_local_tensor_if_DTensor(p.grad))  # [local_M,N]
            active_momentum_buffers.append(get_local_tensor_if_DTensor(state["momentum_buffer"]))  # [local_M,N]
            active_indices.append(i)

        local_param_shard = get_local_tensor_if_DTensor(batch[0])  # [local_M,N]
        packed_local = self._run_dion2_phase(  # [S,local_M,N]
            "dion2.megabatch.pre_ns",
            self._dion2_pre_ns_updates_and_pack,
            active_grads,
            active_momentum_buffers,
            active_indices,
            local_param_shard,
            actual_batch_size,
            batch_capacity,
        )
        del active_grads, active_momentum_buffers, active_indices, local_param_shard

        # Step 2: attach DTensor metadata once, after packing. Explicit shape and
        # stride are required because uneven local shards cannot be inferred when
        # run_check=False.
        back_placements: list[Placement] = [
            Shard(placement.dim + 1) if isinstance(placement, Shard) else placement
            for placement in first_param.placements
        ]
        stacked_global_shape = torch.Size((batch_capacity, *first_param.shape))
        stacked_meta = torch.empty(stacked_global_shape, device="meta")  # [S,M,N]
        stacked_global_stride = stacked_meta.stride()
        del stacked_meta
        stacked = DTensor.from_local(  # [S,M,N]
            packed_local,
            self._device_mesh,
            back_placements,
            run_check=False,
            shape=stacked_global_shape,
            stride=stacked_global_stride,
        )
        del packed_local

        # Step 3: forward all-to-all -- shard the PARAM axis on the FSDP shard mesh dim
        # (keep any other mesh-dim placements, e.g. Replicate under HSDP). Each rank
        # then owns one whole param.
        fwd_placements = list(stacked.placements)
        fwd_placements[shard_mesh_dim] = Shard(0)
        per_matrix = self._run_dion2_phase(  # [S,M,N]
            "dion2.megabatch.forward",
            stacked.redistribute,
            self._device_mesh,
            fwd_placements,
        )
        local_matrices = per_matrix.to_local()  # [K,M,N]
        del stacked, per_matrix
        self._validate_redistributed_matrix_shape(local_matrices, effective_width, tuple(first_param.shape))

        # Step 4: Newton-Schulz keeps a unified [K,M,N] interface. The helper
        # selects the specialized 2-D kernel internally when K=1.
        ortho_p = self._run_dion2_phase(  # [K,M,N]
            "dion2.megabatch.ns",
            self._dion2_orthogonalize_dense_batch,
            local_matrices,
            split_shared_gate_up=split_shared_gate_up,
        )
        del local_matrices

        # Step 5: backward all-to-all -- re-shard the data axis, then unstack.
        back = self._run_dion2_phase(  # [S,M,N]
            "dion2.megabatch.reverse",
            self._dion2_reverse_redistribute,
            ortho_p,
            fwd_placements,
            back_placements,
        )
        del ortho_p
        back_local = back.to_local()  # [S,<local shard on shard_dim>,...]
        del back

        self._run_dion2_phase(
            "dion2.megabatch.apply",
            self._apply_dion2_distributed_updates,
            batch,
            actual_batch_size,
            active,
            back_local,
        )
        del back_local

    def _process_dion2_batch_single(self, batch: list[nn.Parameter]) -> None:
        """Process batch on single device (no distribution)."""
        for p in batch:
            # No gradient this step -> sit the param out entirely: momentum buffer
            # stays frozen (not created/decayed) and no update is applied. Newton-
            # Schulz renormalizes any nonzero input back to unit norm, so feeding a
            # stale/zero momentum would emit a full-strength spurious update; hence
            # we skip rather than treat a missing grad as grad=0. Matches the
            # reference (microsoft/dion), which filters None-grad params up front.
            if p.grad is None:
                continue

            grad = get_local_tensor_if_DTensor(p.grad)
            param = get_local_tensor_if_DTensor(p)

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p).float()

            mom = get_local_tensor_if_DTensor(state["momentum_buffer"])

            # Momentum + submatrix selection. Single-device fallback is not
            # sharded; pass shard_dim=1 so the selection dim is rows (-2),
            # matching the apply call below. (Fixes the original ``select_dim=``
            # keyword-name bug, which raised TypeError on any single-GPU /
            # non-DTensor run.)
            if self.fraction == 1.0:
                # Muon baseline: pre-decayed momentum (M <- mu*M + G) + Nesterov,
                # full-matrix NS. compute_pre_ns_update does not mutate the
                # gradient, so grad can be passed directly.
                pre_ns = compute_pre_ns_update(
                    grad,
                    mom,
                    momentum=self.muon_momentum,
                    nesterov=self.nesterov,
                )
                submatrix, indices = self._select_submatrix(pre_ns, state, shard_dim=1)
            else:
                # Dion2 Algorithm 1 (fractional): pure error-feedback accumulation
                # with selective decay. No whole-buffer decay and no Nesterov -- NS
                # runs on the accumulated M[K], and only the selected slice is
                # decayed (by ef_decay) inside _select_submatrix. This keeps the
                # unselected rows as a running residual, matching the reference.
                mom.add_(grad)
                submatrix, indices = self._select_submatrix(mom, state, shard_dim=1)

            # Newton-Schulz. A fused shared-expert gate/up projection is split
            # only after its full logical matrix is available on this device.
            ortho = self._dion2_orthogonalize_dense_batch(
                submatrix.unsqueeze(0),
                split_shared_gate_up=self._is_shared_expert_gate_up(p),
            )[0]

            # Get adjusted LR / wd from the owning param-group.
            base_lr = self._base_lr_for(p)
            wd = self._wd_for(p)
            adjusted_lr = base_lr * self._get_adjusted_lr_ratio_for_param(p)

            # Apply weight decay
            param.mul_(1 - base_lr * wd)
            # Apply update to selected indices
            self._apply_submatrix_update(param, ortho, indices, adjusted_lr, select_dim=-2)

    def _select_submatrix(
        self,
        tensor: torch.Tensor,
        state: dict,
        shard_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select submatrix based on L1 norm (DION2 style).

        Args:
            tensor: Input tensor (local shard or full matrix)
            state: Optimizer state dict
            shard_dim: Dimension along which tensor is sharded (for FSDP)

        Returns:
            submatrix: Selected rows/columns
            indices: Indices of selected rows/columns
        """
        if self.fraction == 1.0:
            # Full matrix, no selection
            if tensor.ndim == 2:
                indices = torch.arange(tensor.size(0), device=tensor.device)
            else:
                indices = None
            # Convert to BF16 for all_to_all compatibility (tensor may be FP32 from momentum)
            return tensor.to(torch.bfloat16), indices

        # Determine selection dimension (opposite of shard dim for efficiency)
        # If sharded along rows, select columns; if sharded along cols, select rows
        if shard_dim == 0:
            select_dim = -1  # Select columns
            norm_dim = -2  # Compute norm over rows
        else:
            select_dim = -2  # Select rows
            norm_dim = -1  # Compute norm over columns

        num_select = tensor.size(select_dim)
        k = max(1, int(math.ceil(self.fraction * num_select)))

        # Compute L1 norm along norm_dim
        slice_norms = tensor.abs().sum(dim=norm_dim)

        # All-reduce norms across ranks so all ranks select the same indices
        # This is critical for FSDP where each rank has different rows/cols
        # The all-reduce sums the partial norms to get global norms
        if self._process_group is not None:
            dist.all_reduce(slice_norms, group=self._process_group)

        # Top-k selection (now deterministic across all ranks)
        _, indices = torch.topk(slice_norms, k, dim=-1, sorted=False)

        # Extract selected submatrix
        if select_dim == -2:
            submatrix = tensor.index_select(dim=0, index=indices)
        else:
            submatrix = tensor.index_select(dim=1, index=indices)

        # Apply error feedback decay to the selected rows/cols of the momentum buffer.
        # Operate on the LOCAL shard (not the DTensor wrapper): ``indices`` are
        # computed from the local pre-NS slice, and the selection dimension is the
        # non-sharded one (opposite of shard_dim), so local and global indices
        # coincide on that axis. This keeps the buffer in the same (local) coordinate
        # frame as the update applied later, and avoids an unsupported in-place
        # index_copy_ on a DTensor.
        if "momentum_buffer" in state and self.ef_decay < 1.0:
            momentum = get_local_tensor_if_DTensor(state["momentum_buffer"])
            dim = 0 if select_dim == -2 else 1
            selected = momentum.index_select(dim=dim, index=indices)
            momentum.index_copy_(dim=dim, index=indices, source=selected * self.ef_decay)

        return submatrix.to(torch.bfloat16), indices

    def _apply_submatrix_update(
        self,
        param: torch.Tensor,
        ortho: torch.Tensor,
        indices: torch.Tensor | None,
        lr: float | torch.Tensor,
        select_dim: int,
    ) -> None:
        """Apply orthogonalized update to selected indices."""
        ortho = ortho.to(param.dtype)

        if indices is None or self.fraction == 1.0:
            # Full matrix update. lr may be a tensor (capturable), so scale via
            # multiply rather than ``alpha=`` (Tensor.add_ only accepts a Number).
            param.add_(ortho * (-lr))
        else:
            # Submatrix update at selected indices
            scaled_ortho = -lr * ortho
            if select_dim == -2 or select_dim == 0:
                param.index_add_(dim=0, index=indices, source=scaled_ortho)
            else:
                param.index_add_(dim=1, index=indices, source=scaled_ortho)

    def _step_adamw(self) -> None:
        """
        AdamW step using Transformer Engine's fused kernel.

        Iterates over param groups so each group's lr / betas / eps / weight_decay
        (set by the factory's lr_multipliers and disable_weight_decay_for_1d_params)
        is honored. The per-group step counter lives on ``group["step"]``
        (FusedAdam-style) so it is round-tripped by the distributed-checkpoint
        optimizer state dict.

        Params and moments are all FP32 (see :meth:`add_param_group`), so a group is one
        dtype batch for ``multi_tensor_adam_capturable``, which updates the param in
        place -- there is no master weight for it to maintain alongside.
        """
        if not self.adamw_params:
            return

        adam_w_mode = 1
        bias_correction = 1

        for group in self.param_groups:
            # Only the AdamW-categorized params of this group are handled here;
            # the DION2-categorized params were already updated in _step_dion2.
            group_params = [p for p in group["params"] if id(p) in self._adamw_param_ids]
            if not group_params:
                continue

            device = group_params[0].device

            # Per-group step counter stored on the group (FusedAdam convention) so
            # DCP round-trips it on resume.
            if group.get("step", None) is not None:
                if not isinstance(group["step"], torch.Tensor):
                    group["step"] = torch.tensor(group["step"], dtype=torch.int32, device=device)
                group["step"] = group["step"].to(device=device)
                group["step"] += 1
            else:
                group["step"] = torch.tensor([1], dtype=torch.int32, device=device)

            # An LR scheduler may have written a plain float into the group.
            if not isinstance(group["lr"], torch.Tensor):
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

            # The capturable kernel requires an inverse-scale argument; bf16-only
            # training has no grad scaler, so it is a constant one.
            kernel_inv_scale = torch.ones((1,), device=device, dtype=torch.float32)
            te.pytorch.optimizers.multi_tensor_applier(
                self._multi_tensor_adam_capturable,
                self._dummy_overflow_buf,
                [grads, params, exp_avgs, exp_avg_sqs],
                lr,
                beta1,
                beta2,
                eps,
                step,
                adam_w_mode,
                bias_correction,
                wd,
                kernel_inv_scale,
            )

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state.

        The optimizer state (per-param momentum / exp_avg / exp_avg_sq and the
        per-group ``step``) round-trips through the base ``torch.optim.Optimizer``
        state dict, so the distributed-checkpoint container can save/restore it
        with FSDP2 resharding just like FusedAdam. There is no master weight to
        restore: the FP32 weight is the parameter itself, which the model checkpoint
        carries.

        The state needs no dtype fix-up either. ``super().load_state_dict`` casts every
        floating-point state tensor to its param's dtype, which the FP32-param invariant
        makes FP32 -- so the moments and the momentum buffer come back FP32 whatever
        precision the checkpoint stored them in.
        """
        super().load_state_dict(state_dict)

        for group in self.param_groups:
            device = group["params"][0].device if group["params"] else "cuda"
            # load_state_dict copies param_groups from the checkpoint verbatim, so restore
            # the device tensors the capturable kernels require.
            if isinstance(group["lr"], torch.Tensor):
                group["lr"] = group["lr"].to(device=device)
            else:
                group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device=device)
            if group.get("step", None) is not None and not isinstance(group["step"], torch.Tensor):
                group["step"] = torch.tensor(group["step"], dtype=torch.int32, device=device)

        # ``super().load_state_dict`` replaces every param_group dict (it rebuilds them
        # from the checkpoint and re-attaches the current params), so the map built in
        # categorize_params now points at the discarded dicts -- the DION2 and AdamW
        # updates would read a pre-resume lr / weight_decay that no scheduler updates.
        self._param_group_map = {id(p): group for group in self.param_groups for p in group["params"]}
