# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""posttrain ParallelDims — a thin adapter over cosmos3's ParallelDims so the RL infra and the
diffusion/VFM path share ONE mesh spec (``cosmos_framework.utils.generator.parallelism``) instead of
two that drift. It reuses cosmos3's mesh construction (``build_meshes``) + submesh accessors
(``dp_mesh`` / ``dp_shard_mesh`` / ``dp_shard_enabled`` / …) verbatim and adds only what the RL call
sites need:

  - ``build_device_mesh()`` — the name the RL sites use; delegates to cosmos3's ``build_meshes``.
  - ``tp`` / ``ep`` / ``pp`` factorization stubs, plus a guard that EVERY non-dp axis
    (``tp``/``ep``/``pp``/``cp``/``cfgp``) is ``1`` — the RL FSDP path (``model/parallelize.py``)
    implements only ``dp_replicate × dp_shard``, so any other axis must fail loud until that
    parallelism is actually wired (additive drop-in, spec §5). ``cp``/``cfgp`` are cosmos3-native
    axes (context / CFG parallel) the RL path doesn't use yet, hence pinned to 1 here too.
"""

from __future__ import annotations

from dataclasses import dataclass

from cosmos_framework.utils.generator.parallelism import ParallelDims as _Cosmos3ParallelDims

_UNIMPLEMENTED_AXES = ("tp", "ep", "pp", "cp", "cfgp")


@dataclass
class ParallelDims(_Cosmos3ParallelDims):
    tp: int = 1
    ep: int = 1
    pp: int = 1

    def __post_init__(self) -> None:
        # RL path is pure data-parallel (dp_replicate × dp_shard); every other axis is unimplemented
        # in model/parallelize.py, so fail loud rather than silently build an unused mesh dim. Run
        # before cosmos3's _validate so a bad axis is rejected with a clear RL-facing message.
        for name in _UNIMPLEMENTED_AXES:
            if getattr(self, name) != 1:
                raise ValueError(
                    f"ParallelDims.{name}={getattr(self, name)} is not implemented in the RL FSDP path "
                    f"(must be 1); only dp_replicate × dp_shard are wired in model/parallelize.py."
                )
        super().__post_init__()  # cosmos3: infer dp_shard/dp_replicate + validate dp × cp × cfgp

    def build_device_mesh(self, device_type: str = "cuda"):
        """The RL call-site name — delegates to cosmos3's ``build_meshes`` (which stores the meshes
        on the instance) and returns the HSDP ``dp`` mesh (``dp_replicate>1``) or the pure-FSDP
        ``dp_shard`` mesh, for signature parity with the old primitive (callers ignore the return)."""
        self.build_meshes(device_type)
        return self.dp_mesh if self.dp_replicate > 1 else self.dp_shard_mesh
