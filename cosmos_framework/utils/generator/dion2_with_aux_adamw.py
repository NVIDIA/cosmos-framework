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
- Batches params in groups of world_size for efficient distribution
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

Matrices are grouped by local shard shape into batches of world_size so the
all-to-all tensors are uniform (see ``_create_dion2_batches``). With
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

import torch
import torch.distributed as dist
import torch.nn as nn
import transformer_engine as te
import transformer_engine_torch as tex
from torch.distributed.tensor import DTensor, Shard

from cosmos_framework.utils import log
from cosmos_framework.utils.misc import get_local_tensor_if_DTensor
from cosmos_framework.utils.generator.aux_optimizer_utils import (
    DEFAULT_ADAMW_MODULE_KEYWORDS,
    compute_pre_ns_update,
    compute_pre_ns_update_moe_expert,
    split_orthogonalizable_params,
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_batched,
)


class Dion2WithAuxAdamW(torch.optim.Optimizer):
    """
    Dion2WithAuxAdamW optimizer.

    Uses DION2 (Distributed Orthogonalization) for nn.Linear hidden weight matrices,
    and AdamW for embeddings, biases, layer norms, and output heads (lm_head).

    Key features:
    - All-to-all communication for efficient distributed NS (no redundant compute)
    - Submatrix selection: only orthogonalize top-k rows/columns
    - Error feedback: maintains momentum for unselected parts
    - Batched processing: handles world_size params per batch

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
    """

    def __init__(
        self,
        params,
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
        capturable: bool = False,
        master_weights: bool = False,
        adamw_module_keywords: tuple[str, ...] | None = None,
        expert_param_keywords: tuple[str, ...] | None = None,
        **kwargs,
    ):
        if kwargs:
            ignored_keys = list(kwargs.keys())
            expected_ignored = {"fused", "keys_to_select", "adamw_betas", "adamw_eps"}
            unexpected = set(ignored_keys) - expected_ignored
            if unexpected:
                import warnings

                warnings.warn(f"Dion2WithAuxAdamW ignoring unexpected kwargs: {unexpected}")

        if not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

        # Master weights requires capturable mode
        if master_weights and not capturable:
            raise RuntimeError("Master weights is currently only supported with capturable=True.")

        # Store hyperparameters
        # Note: lr is accessed via property that reads from param_groups
        # to support LR schedulers (which update param_groups[X]["lr"])
        self.wd = weight_decay
        self.muon_momentum = muon_momentum
        self.muon_lr_scale = muon_lr_scale
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.fraction = fraction
        self.ef_decay = ef_decay
        self.adam_betas = tuple(adam_betas) if isinstance(adam_betas, list) else adam_betas
        self.eps = eps

        # Name substrings that route an nn.Linear to AdamW (output heads). Used by
        # categorize_params alongside tied-weight / vocab-shape detection.
        self.adamw_module_keywords = (
            tuple(adamw_module_keywords) if adamw_module_keywords else DEFAULT_ADAMW_MODULE_KEYWORDS
        )
        # Name substrings that route stacked MoE expert params ([E, M, N]) to the
        # DION2 side (each expert slice orthogonalized). Empty = experts stay on
        # AdamW (no behavior change).
        self.expert_param_keywords = tuple(expert_param_keywords) if expert_param_keywords else ()

        # Distributed settings
        self.use_distributed = use_distributed and dist.is_initialized()
        self._world_size = 1
        self._device_rank = 0
        self._process_group = None
        self._device_mesh = None

        # Master weights settings (for mixed-precision training stability)
        self.capturable = capturable
        self.master_weights = master_weights

        # Parameter lists
        self.dion2_params: list[nn.Parameter] = []
        self.adamw_params: list[nn.Parameter] = []
        # Stacked MoE expert params ([E, M, N]); orthogonalized per expert slice.
        self.stacked_dion2_params: list[nn.Parameter] = []
        self.param_to_name: dict[nn.Parameter, str] = {}
        self._dion2_batches: list[list[nn.Parameter]] = []

        # Master weight copies (populated by _create_master_weights after categorize_params)
        self._dion2_masters: list[torch.Tensor] = []
        self._adamw_masters: list[torch.Tensor] = []

        # Transformer Engine fused Adam. The zero buffer is the noop flag required
        # as the second argument of TE's multi_tensor_applier; it is a fixed
        # constant here (no AMP overflow handling).
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self._multi_tensor_adam = tex.multi_tensor_adam
        self._multi_tensor_adam_capturable = tex.multi_tensor_adam_capturable
        self._multi_tensor_adam_capturable_master = tex.multi_tensor_adam_capturable_master

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

        # id(param) -> owning param_group, so the DION2 and AdamW updates can read
        # the *per-group* lr / weight_decay (honors lr_multipliers and
        # disable_weight_decay_for_1d_params). With a single group it degenerates
        # to a single global lr/wd, matching the original (reference) behavior.
        self._param_group_map: dict[int, dict] = {}
        self._adamw_param_ids: set[int] = set()
        # Master weights are created lazily on the first step() (FusedAdam-style),
        # so that after a checkpoint resume they are rebuilt from the *restored*
        # params rather than the freshly-initialized ones.
        self._masters_initialized = False
        self._param_to_master: dict[int, torch.Tensor] = {}

        log.info(f"Dion2WithAuxAdamW master_weights: {master_weights} capturable: {capturable}")

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
            adamw_module_keywords=self.adamw_module_keywords,
            expert_param_keywords=self.expert_param_keywords,
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

        # Sort by size for load balancing
        self.dion2_params = sorted(self.dion2_params, key=lambda x: x.numel(), reverse=True)

        # Setup distributed from first DTensor param
        self._setup_distributed_from_params()

        # Create batches of world_size params
        self._create_dion2_batches()

        dion2_numel = sum(p.numel() for p in self.dion2_params)
        adamw_numel = sum(p.numel() for p in self.adamw_params)
        stacked_dion2_numel = sum(p.numel() for p in self.stacked_dion2_params)

        log.info(
            f"Dion2WithAuxAdamW: {len(self.dion2_params)} Muon params ({dion2_numel:,} elements), "
            f"{len(self.stacked_dion2_params)} stacked-expert params ({stacked_dion2_numel:,} elements), "
            f"{len(self.adamw_params)} AdamW params ({adamw_numel:,} elements), "
            f"world_size={self._world_size}, {len(self._dion2_batches)} batches"
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

        # NOTE: master weights are intentionally NOT created here. They are
        # created lazily on the first step() (see _maybe_init_master_weights) so
        # that a checkpoint resume rebuilds them from the restored params.

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
        Group Muon params by GLOBAL shape, then batch within each shape group.

        The distributed step (``_process_dion2_batch_distributed``) stacks a batch of
        params into a single DTensor and redistributes it, so the batches must be
        identical ACROSS ranks. Group by the *global* shape (same on every rank), NOT
        the local shard shape -- under uneven sharding the local shard shape differs
        per rank and would make ranks build inconsistent batches. ``batch_size =
        world_size`` ensures each rank orthogonalizes exactly one param per batch.
        """
        self._dion2_batches = []
        batch_size = self._world_size

        # Step 1: Group params by global shape (identical on all ranks).
        shape_groups: dict[tuple, list[nn.Parameter]] = {}
        for p in self.dion2_params:
            shape = tuple(p.shape)
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append(p)

        # Step 2: Create batches within each shape group
        for shape, params in shape_groups.items():
            for i in range(0, len(params), batch_size):
                batch = params[i : i + batch_size]
                self._dion2_batches.append(batch)

        # Log batch info
        num_shape_groups = len(shape_groups)
        num_batches = len(self._dion2_batches)
        if self._dion2_batches:
            # Count batches that need padding
            padded_batches = sum(1 for b in self._dion2_batches if len(b) < batch_size)
            log.info(
                f"DION2: {len(self.dion2_params)} params grouped into {num_shape_groups} shape groups, "
                f"{num_batches} batches (world_size={batch_size}, {padded_batches} need padding)"
            )
            # Log shape group details
            for shape, params in shape_groups.items():
                log.info(f"  Shape {shape}: {len(params)} params")

    def _base_lr_for(self, p: nn.Parameter) -> float | torch.Tensor:
        """Per-group base learning rate for ``p`` (honors lr_multipliers)."""
        return self._param_group_map[id(p)]["lr"]

    def _wd_for(self, p: nn.Parameter) -> float:
        """Per-group weight decay for ``p`` (honors disable_weight_decay_for_1d_params)."""
        return self._param_group_map[id(p)]["weight_decay"]

    def _get_adjusted_lr(self, param_shape: tuple[int, ...], base_lr: float | torch.Tensor) -> float | torch.Tensor:
        """Compute adjusted learning rate based on parameter matrix size and the
        owning param-group's base lr."""
        A, B = param_shape[:2]
        adjusted_ratio = self.muon_lr_scale * math.sqrt(max(A, B))
        return base_lr * adjusted_ratio

    def _maybe_init_master_weights(self) -> None:
        """Create FP32 master weights on first use (FusedAdam-style lazy init)."""
        if self.master_weights and not self._masters_initialized:
            self._create_master_weights()

    def _create_master_weights(self) -> None:
        """
        Create FP32 master weight copies for mixed-precision training stability.

        Creates param_groups_master (for LowPrecisionCallback compatibility) and
        indexed lists for efficient lookup during DION2/AdamW steps.
        """
        # Create param_groups_master mirroring param_groups (like FusedAdam)
        # This enables LowPrecisionCallback to copy masters -> params periodically
        self.param_groups_master = []
        for pg in self.param_groups:
            param_list = pg["params"]
            self.param_groups_master.append(
                {
                    "params": [p.clone().detach().float() if self.master_weights else None for p in param_list],
                }
            )

        # Build param_id -> master mapping for efficient lookup
        self._param_to_master: dict[int, torch.Tensor] = {}
        for group, group_master in zip(self.param_groups, self.param_groups_master):
            for p, p_master in zip(group["params"], group_master["params"]):
                if p_master is not None:
                    self._param_to_master[id(p)] = p_master

        # Create indexed lists for DION2/AdamW step() methods
        self._dion2_masters = [self._param_to_master[id(p)] for p in self.dion2_params]
        self._adamw_masters = [self._param_to_master[id(p)] for p in self.adamw_params]

        muon_master_numel = sum(m.numel() for m in self._dion2_masters)
        adamw_master_numel = sum(m.numel() for m in self._adamw_masters)
        log.info(
            f"Created FP32 master weights: {len(self._dion2_masters)} DION2 ({muon_master_numel:,} elements), "
            f"{len(self._adamw_masters)} AdamW ({adamw_master_numel:,} elements)"
        )
        self._masters_initialized = True

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Lazily build FP32 master weights from the current (possibly
        # checkpoint-restored) params before the first update.
        self._maybe_init_master_weights()

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
        """Orthogonalize stacked MoE expert params, one expert slice at a time.

        Each param has shape ``[E, M, N]`` (E experts, each an M x N matrix). Under
        FSDP2 these are sharded on the expert dim (dim 0), so every rank holds whole
        expert matrices -- Newton-Schulz is therefore fully local (no all-to-all),
        and is batched across the local experts via ``zeropower_via_newtonschulz5_batched``.

        NOTE (sharding assumption): the "no communication" property relies on the
        expert tensor being sharded on dim 0 (the expert axis). This holds for the
        FSDP2 ``fully_shard`` path used by LLM/VFM here, because FSDP2 shards every
        parameter on dim 0. It is NOT guaranteed in general -- e.g. tensor/expert
        parallelism could shard *within* an expert matrix (dim 1/2). That case is
        unsupported and is rejected by the placement check below (fails loudly
        rather than silently computing a wrong update); supporting it would require
        a per-expert gather. The assumption was not exhaustively audited against
        every parallelization config, which is exactly why it is enforced here.
        """
        for p in self.stacked_dion2_params:
            if p.grad is None:
                continue

            # Validate sharding: only the expert axis (tensor dim 0) may be sharded.
            if isinstance(p, DTensor):
                for placement in p.placements:
                    if placement.is_shard() and placement.dim != 0:
                        raise NotImplementedError(
                            "Stacked-expert orthogonalization requires sharding on the expert "
                            f"dim (0); got placement {placement} for "
                            f"'{self.param_to_name.get(p, 'unknown')}'."
                        )

            local_grad = get_local_tensor_if_DTensor(p.grad)
            local_param = get_local_tensor_if_DTensor(p)
            if local_grad.ndim != 3:
                raise NotImplementedError(
                    f"Stacked-expert orthogonalization supports 3D params [E, M, N]; "
                    f"got shape {tuple(local_grad.shape)} for '{self.param_to_name.get(p, 'unknown')}'."
                )

            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p).float()

            # Per-expert masked momentum + Nesterov (element-wise over [E, M, N]).
            # Active experts follow the standard mu*M + G recurrence; inactive
            # experts (no gradient this step) keep their momentum frozen. ``active``
            # ([E] bool) is used below to zero the update and skip weight decay for
            # inactive experts -- required because Newton-Schulz would otherwise
            # renormalize their stale momentum into a full-strength spurious update.
            pre_ns, active = compute_pre_ns_update_moe_expert(
                local_grad,
                get_local_tensor_if_DTensor(state["momentum_buffer"]),
                momentum=self.muon_momentum,
                nesterov=self.nesterov,
            )

            # Batched Newton-Schulz over the local experts, then zero the update for
            # inactive experts (their NS result is a bogus unit-norm matrix).
            ortho = zeropower_via_newtonschulz5_batched(pre_ns, steps=self.ns_steps)
            ortho = ortho * active.view(-1, 1, 1).to(ortho.dtype)

            # LR scaling uses the per-expert matrix shape (M, N), shared across experts.
            base_lr = self._base_lr_for(p)
            wd = self._wd_for(p)
            adjusted_lr = self._get_adjusted_lr(tuple(p.shape[-2:]), base_lr)

            # Per-expert weight-decay factor: (1 - base_lr*wd) for active experts,
            # 1.0 (no decay) for inactive ones. Combined with the zeroed update
            # above, inactive experts are left completely untouched.
            if self.master_weights:
                master = get_local_tensor_if_DTensor(self._param_to_master[id(p)])
                a_wd = active.view(-1, 1, 1).to(master.dtype)
                master.mul_(1 - a_wd * (base_lr * wd))
                master.add_(ortho.float() * (-adjusted_lr))
                local_param.copy_(master)
            else:
                a_wd = active.view(-1, 1, 1).to(local_param.dtype)
                local_param.mul_(1 - a_wd * (base_lr * wd))
                local_param.add_(ortho.to(local_param.dtype) * (-adjusted_lr))

    def _step_dion2(self) -> None:
        """
        DION2 step with all-to-all distributed Newton-Schulz.

        For each batch of world_size params:
        1. Compute momentum + select submatrix on local shards
        2. All-to-all to redistribute shards (each rank gets full submatrix for its param)
        3. Newton-Schulz on full submatrix (each rank does different param)
        4. All-to-all to scatter results back
        5. Apply weight decay and update (to FP32 master if enabled)
        """
        if not self.dion2_params:
            return

        for batch in self._dion2_batches:
            self._process_dion2_batch(batch)

    def _process_dion2_batch(self, batch: list[nn.Parameter]) -> None:
        """Process a single batch of params using DION2 all-to-all pattern."""
        world_size = self._world_size

        # Pad batch if needed
        actual_batch_size = len(batch)
        if actual_batch_size < world_size:
            # Pad with the last param (will be masked out)
            padding = [batch[-1]] * (world_size - actual_batch_size)
            batch = batch + padding

        # Check if using DTensor (FSDP)
        is_dtensor = isinstance(batch[0], DTensor)

        if is_dtensor and world_size > 1:
            self._process_dion2_batch_distributed(batch, actual_batch_size)
        else:
            self._process_dion2_batch_single(batch, actual_batch_size)

    def _process_dion2_batch_distributed(self, batch: list[nn.Parameter], actual_batch_size: int) -> None:
        """Process a batch via DTensor collectives (the "each rank orthogonalizes one
        whole param" transpose), correct for uneven / non-divisible shard dims.

        Rather than a hand-rolled ``all_to_all`` + ``cat``/``narrow`` (which assumed
        FSDP2 padded every local shard uniformly -- it does not; ``_local_tensor`` is
        unpadded and uneven), the gather/scatter is expressed through DTensor:

          1. Momentum + Nesterov per param, kept as a DTensor so its shard metadata
             (including uneven, unpadded local sizes) is preserved.
          2. ``torch.stack`` the world_size params -> a ``[W, ...]`` DTensor; the shard
             tensor dim shifts to ``shard_dim + 1``.
          3. ``redistribute`` so the PARAM axis is sharded on the FSDP shard mesh dim
             -> each rank owns one whole param (forward all-to-all).
          4. Newton-Schulz on that whole param (each rank a different one).
          5. ``redistribute`` back to shard the data axis (backward all-to-all),
             unstack, and apply the update to each local shard.

        DTensor owns the (possibly uneven) per-rank size bookkeeping, so this is
        correct regardless of divisibility. This transpose has been validated across
        even/uneven/partial/bf16/shard-dim configs by the standalone multi-GPU script
        ``unit_tests/dion2_uneven_shard_dtensor_check.py``.

        TODO(dion2-optionb-unittest): provide a standalone torchrun script into a
        committed, CI-run multi-GPU unit test (it currently must be launched manually
        via ``torchrun`` and is not part of the automated suite).

        Submatrix selection (``fraction < 1``) is not supported on this path -- it is
        unused (every config runs fraction=1.0) -- and is rejected up front.
        """
        # distributed path. Will need to wire it in is a
        # well-scoped change: per-param DTensor select (norms via
        # ``pre.abs().sum(sharded_dim).full_tensor()`` -> top-k -> ``index_select``
        # with a Replicate index), error-feedback decay on the momentum buffer, and a
        # branched apply that reuses ``_apply_submatrix_update[_master]`` (index_add
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
        mesh = self._device_mesh
        shard_mesh_dim = self._shard_mesh_dim
        shard_dim = self._shard_tensor_dim
        stack_axis = shard_dim + 1  # torch.stack adds a leading param axis

        # Step 1: momentum + Nesterov, kept in DTensor space (metadata preserved).
        # compute_pre_ns_update does not mutate the gradient, so p.grad is passed
        # directly; it mutates the (sharded DTensor) momentum buffer in place.
        #
        # None-grad handling: a param with no gradient this step is sat out --
        # momentum frozen (compute_pre_ns_update NOT called, so no mu-decay) and no
        # update applied (skipped in the apply loop via ``active``). We cannot just
        # drop the slot: the stack + redistribute all-to-alls are a fixed-size
        # collective every rank must enter identically, so an inactive slot instead
        # contributes a zero placeholder (same DTensor sharding/dtype) to keep the
        # collective shapes uniform. Newton-Schulz on zeros stays finite (norm+1e-7)
        # and the result is discarded on apply. This relies on ``p.grad is None``
        # being identical across ranks -- true for dense params, where a missing grad
        # is structural (an unused param is None on every rank), not data-dependent.
        pre_ns_list = []
        active = []
        for i in range(actual_batch_size):
            p = batch[i]
            state = self.state[p]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(p).float()
            if p.grad is None:
                active.append(False)
                pre_ns_list.append(torch.zeros_like(state["momentum_buffer"]).to(torch.bfloat16))
                continue
            active.append(True)
            pre_ns = compute_pre_ns_update(
                p.grad, state["momentum_buffer"], momentum=self.muon_momentum, nesterov=self.nesterov
            )
            # Cast to bf16 BEFORE the transpose so the two all-to-alls move bf16 (not
            # fp32) -- matching the previous path's communication volume. This is
            # numerically identical: Newton-Schulz casts to bf16 anyway, and bf16
            # rounding commutes with the reconstruction (round-then-gather ==
            # gather-then-round elementwise).
            pre_ns_list.append(pre_ns.to(torch.bfloat16))

        # Pad the param axis up to world_size (mirrors batch padding); the dummy
        # entries' Newton-Schulz results are discarded on apply.
        padded = pre_ns_list + [pre_ns_list[-1]] * (world_size - actual_batch_size)

        # Step 2: stack -> [W, ...]; the shard tensor dim moves to stack_axis.
        stacked = torch.stack(padded, dim=0)
        expected = Shard(stack_axis)
        if stacked.placements[shard_mesh_dim] != expected:
            raise RuntimeError(
                f"DION2 expected stacked placement {expected} on mesh dim {shard_mesh_dim}, "
                f"got {stacked.placements} (torch.stack should shift Shard({shard_dim}) -> "
                f"Shard({stack_axis}))."
            )

        # Step 3: forward all-to-all -- shard the PARAM axis on the FSDP shard mesh dim
        # (keep any other mesh-dim placements, e.g. Replicate under HSDP). Each rank
        # then owns one whole param.
        fwd_placements = list(stacked.placements)
        fwd_placements[shard_mesh_dim] = Shard(0)
        per_matrix = stacked.redistribute(mesh, fwd_placements)
        full_p = per_matrix.to_local()[0]  # this rank's whole param

        # Step 4: Newton-Schulz on the whole param.
        ortho_p = zeropower_via_newtonschulz5(full_p, steps=self.ns_steps)

        # Step 5: backward all-to-all -- re-shard the data axis, then unstack.
        ortho_dt = DTensor.from_local(ortho_p.unsqueeze(0), mesh, fwd_placements, run_check=False)
        back = ortho_dt.redistribute(mesh, list(stacked.placements))
        back_local = back.to_local()  # [W, <local shard on shard_dim>, ...]

        # Apply the orthogonalized update to each real param's local shard.
        for i in range(actual_batch_size):
            if not active[i]:
                # No gradient this step: momentum was frozen above; skip weight
                # decay and the update entirely (matches the reference, which
                # leaves a None-grad param completely untouched).
                continue
            p = batch[i]
            local_param = p._local_tensor
            update_local = back_local[i]
            base_lr = self._base_lr_for(p)
            wd = self._wd_for(p)
            adjusted_lr = self._get_adjusted_lr(tuple(p.shape), base_lr)
            if self.master_weights:
                master = get_local_tensor_if_DTensor(self._param_to_master[id(p)])
                master.mul_(1 - base_lr * wd)
                master.add_(update_local.float() * (-adjusted_lr))
                local_param.copy_(master)
            else:
                local_param.mul_(1 - base_lr * wd)
                local_param.add_(update_local.to(local_param.dtype) * (-adjusted_lr))

    def _process_dion2_batch_single(self, batch: list[nn.Parameter], actual_batch_size: int) -> None:
        """Process batch on single device (no distribution)."""
        for i, p in enumerate(batch):
            if i >= actual_batch_size:
                continue

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

            # Newton-Schulz
            ortho = zeropower_via_newtonschulz5(submatrix, steps=self.ns_steps)

            # Get adjusted LR / wd from the owning param-group.
            base_lr = self._base_lr_for(p)
            wd = self._wd_for(p)
            adjusted_lr = self._get_adjusted_lr(p.shape, base_lr)

            if self.master_weights:
                # Update FP32 master, then write the param directly inside
                # _apply_submatrix_update_master.
                master = get_local_tensor_if_DTensor(self._param_to_master[id(p)])
                master.mul_(1 - base_lr * wd)
                self._apply_submatrix_update_master(master, param, ortho, indices, adjusted_lr, select_dim=-2)
            else:
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

    def _apply_submatrix_update_master(
        self,
        master: torch.Tensor,
        param: torch.Tensor,
        ortho: torch.Tensor,
        indices: torch.Tensor | None,
        lr: float | torch.Tensor,
        select_dim: int,
    ) -> None:
        """Apply orthogonalized update to FP32 master, then copy to BF16 param."""
        ortho = ortho.float()

        if indices is None or self.fraction == 1.0:
            # Full matrix update. lr may be a tensor (capturable), so scale via
            # multiply rather than ``alpha=`` (Tensor.add_ only accepts a Number).
            master.add_(ortho * (-lr))
        else:
            # Submatrix update at selected indices
            scaled_ortho = -lr * ortho
            if select_dim == -2 or select_dim == 0:
                master.index_add_(dim=0, index=indices, source=scaled_ortho)
            else:
                master.index_add_(dim=1, index=indices, source=scaled_ortho)
        # Write the BF16 param directly from the updated FP32 master so we do not
        # depend on LowPrecisionCallback (the OptimizersContainer hides
        # master_weights from it). Matches FusedAdam's in-kernel param write.
        param.copy_(master)

    def _step_adamw(self) -> None:
        """
        AdamW step using Transformer Engine's fused kernel.

        Iterates over param groups so each group's lr / betas / eps / weight_decay
        (set by the factory's lr_multipliers and disable_weight_decay_for_1d_params)
        is honored, then batches by dtype within the group. The per-group step
        counter lives on ``group["step"]`` (FusedAdam-style) so it is
        round-tripped by the distributed-checkpoint optimizer state dict.

        When master_weights=True and capturable=True, uses
        multi_tensor_adam_capturable_master which maintains FP32 master weights.
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

            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master, p_bf_master, p_32_master = [], [], []

            for p in group_params:
                if p.grad is None:
                    continue

                grad = get_local_tensor_if_DTensor(p.grad)
                param = get_local_tensor_if_DTensor(p)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p).float()
                    state["exp_avg_sq"] = torch.zeros_like(p).float()

                exp_avg = get_local_tensor_if_DTensor(state["exp_avg"])
                exp_avg_sq = get_local_tensor_if_DTensor(state["exp_avg_sq"])
                master = get_local_tensor_if_DTensor(self._param_to_master[id(p)]) if self.master_weights else None

                if p.dtype == torch.float16:
                    g_16.append(grad)
                    p_16.append(param)
                    m_16.append(exp_avg)
                    v_16.append(exp_avg_sq)
                    if self.master_weights:
                        p_16_master.append(master)
                elif p.dtype == torch.bfloat16:
                    g_bf.append(grad)
                    p_bf.append(param)
                    m_bf.append(exp_avg)
                    v_bf.append(exp_avg_sq)
                    if self.master_weights:
                        p_bf_master.append(master)
                elif p.dtype == torch.float32:
                    g_32.append(grad)
                    p_32.append(param)
                    m_32.append(exp_avg)
                    v_32.append(exp_avg_sq)
                    if self.master_weights:
                        p_32_master.append(master)
                else:
                    raise RuntimeError(f"Unsupported dtype {p.dtype}")

            if self.capturable:
                # The capturable-master kernel requires an inverse-scale argument;
                # bf16-only training has no grad scaler, so it is a constant one.
                kernel_inv_scale = torch.ones((1,), device=device, dtype=torch.float32)
                dtype_batches = (
                    (g_16, p_16, m_16, v_16, p_16_master),
                    (g_bf, p_bf, m_bf, v_bf, p_bf_master),
                    (g_32, p_32, m_32, v_32, p_32_master),
                )
                kernel = (
                    self._multi_tensor_adam_capturable_master
                    if self.master_weights
                    else self._multi_tensor_adam_capturable
                )
                for g_, p_, m_, v_, pm_ in dtype_batches:
                    if len(g_) == 0:
                        continue
                    tensor_lists = [g_, p_, m_, v_, pm_] if self.master_weights else [g_, p_, m_, v_]
                    te.pytorch.optimizers.multi_tensor_applier(
                        kernel,
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
                        kernel_inv_scale,
                    )
            else:
                dtype_batches = (
                    (g_16, p_16, m_16, v_16),
                    (g_bf, p_bf, m_bf, v_bf),
                    (g_32, p_32, m_32, v_32),
                )
                for g_, p_, m_, v_ in dtype_batches:
                    if len(g_) == 0:
                        continue
                    te.pytorch.optimizers.multi_tensor_applier(
                        self._multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_, p_, m_, v_],
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
        with FSDP2 resharding just like FusedAdam. Master weights are *not*
        checkpointed; they are rebuilt from the restored params on the next
        ``step()`` (see ``_maybe_init_master_weights``).
        """
        super().load_state_dict(state_dict)

        # Force master weights to be rebuilt from the (now restored) params.
        self._masters_initialized = False
        self.param_groups_master = None

        for group in self.param_groups:
            device = group["params"][0].device if group["params"] else "cuda"
            if self.capturable:
                if isinstance(group["lr"], torch.Tensor):
                    group["lr"] = group["lr"].to(device=device)
                else:
                    group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device=device)
                if group.get("step", None) is not None and not isinstance(group["step"], torch.Tensor):
                    group["step"] = torch.tensor(group["step"], dtype=torch.int32, device=device)
            for p in group["params"]:
                state = self.state[p]
                if "exp_avg" in state:
                    state["exp_avg"] = state["exp_avg"].float()
                    state["exp_avg_sq"] = state["exp_avg_sq"].float()
                if "momentum_buffer" in state:
                    state["momentum_buffer"] = state["momentum_buffer"].float()
