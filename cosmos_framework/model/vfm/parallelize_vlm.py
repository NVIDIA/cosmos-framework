# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""FSDP2 wrapping for Cosmos3 VLM ``HFModel`` instances.

Hosts the single VLM-specific ``parallelize`` entry point used by
``vlm_model.VLMModel._init_vlm``.  Lives under ``projects/cosmos3/vfm/models/``
so the FSDP wrapping concern sits next to the model class it operates on
(mirroring the layout of ``models/mot/parallelize_unified_mot.py`` for the
MoT path).

Pure parallelism plumbing — :class:`~projects.cosmos3.vfm.utils.parallelism.ParallelDims`
and its meshes — stays in ``vfm/utils/parallelism.py``.
"""

from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.parallelism import (
    PRECISION_TO_TORCH_DTYPE,
    ParallelismConfig,
)
from cosmos_framework.model.vfm.hf_model import HFModel
from cosmos_framework.utils.vfm.parallelism import ParallelDims


def parallelize(
    model: HFModel,
    parallel_dims: ParallelDims,
    parallelism_config: ParallelismConfig,
    precision: str,
) -> None:
    """Apply FSDP2 to an HFModel in-place.

    Uses torch.distributed.fsdp.fully_shard (FSDP2).  Each transformer block is
    sharded individually for fine-grained memory savings; the outer model is then
    wrapped to cover remaining parameters (embeddings, layer norms, lm_head).

    Supported architectures:
    - Language models: ``inner.model.layers`` (standard HF LLM structure)
    - Vision-language models: additionally ``inner.visual.blocks`` (Qwen3-VL)

    No-op when FSDP is not needed (single-GPU or replicate-only).

    Args:
        model:              HFModel instance (``model`` attribute must be on meta or CPU device).
        parallel_dims:      ParallelDims with meshes already built via
                            :meth:`ParallelDims.build_meshes`.
        parallelism_config: Source of FSDP master dtype (``fsdp_master_dtype``;
                            threaded to ``MixedPrecisionPolicy.reduce_dtype``).
        precision:          FSDP MixedPrecisionPolicy parameter dtype
                            (``"bfloat16"``, ``"float16"``, or ``"float32"``).
    """
    if not parallel_dims.dp_shard_enabled:
        # No shard axis: dp_shard <= 1.  FSDP2 (fully_shard) has nothing to do.
        # For replicate-only (dp_replicate > 1, dp_shard == 1), use DDP outside
        # this function.
        log.info("parallelize: dp_shard <= 1 — skipping FSDP2 wrapping")
        return

    mp_policy = MixedPrecisionPolicy(
        param_dtype=PRECISION_TO_TORCH_DTYPE[precision],
        reduce_dtype=PRECISION_TO_TORCH_DTYPE[parallelism_config.fsdp_master_dtype],
    )

    # 2-D (dp_replicate × dp_shard) mesh for HSDP, or 1-D dp_shard sub-mesh
    # for pure FSDP. In the overlay design cp does NOT fold into the FSDP
    # shard axis; cp/cfgp are handled by separate meshes.
    if parallel_dims.dp_replicate_enabled:
        fsdp_mesh = parallel_dims.dp_mesh
    else:
        fsdp_mesh = parallel_dims.dp_shard_mesh
    fsdp_kwargs = {"mesh": fsdp_mesh, "mp_policy": mp_policy}

    inner = model.model

    no_split_names = set(getattr(inner, "_no_split_modules", []))
    wrapped = 0
    for module in reversed(list(inner.modules())):
        if type(module).__name__ in no_split_names:
            fully_shard(module, **fsdp_kwargs)
            wrapped += 1
    log.info(f"Wrapped {wrapped} sub-modules.")

    # Wrap the full inner model to cover remaining parameters
    # (embed_tokens, final layer norm, lm_head, visual projector stem, etc.)
    # NOTE: FSDP-2 CPU offload (offload_policy=CPUOffloadPolicy()) was never
    # wired through to any active recipe and the path was untested; see the
    # comment in vlm_model._init_vlm meta-materialize block (search for
    # "FSDP-2 CPU offload") for how to re-enable it.
    fully_shard(inner, **fsdp_kwargs)
    log.info("parallelize: FSDP2 applied to HFModel.model")
