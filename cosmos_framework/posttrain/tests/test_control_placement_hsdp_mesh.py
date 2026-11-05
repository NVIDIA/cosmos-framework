# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L1 @GPU: a trainer WG must inject its model-owned HSDP mesh.

This is the one test that spans the WHOLE chain — `bind_device_groups` → `spawn_grid`
(`dist_scope="replica"`, policy R=1) → K-rank rendezvous → `run_setup` → `set_mesh` → `resolve_pdims` — in
real Ray actors on real GPUs. Every other test covers a link in isolation:
`test_workers_grid.py` checks the rendezvous, `test_algorithm_grpo_dp_multirank.py` checks FSDP
given a mesh someone else built, and the CPU tests stub the mesh entirely.

The failure it exists to catch is silent by construction. Flat FSDP and a configured 2-D HSDP mesh
over the same K ranks both build and emit plausible losses, so the test pins the factorization and
the full-DP ingress coordinate directly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class _MeshProbe:
    """A role body that records the mesh it was built under, then reports it on demand."""

    def __init__(self) -> None:
        # Constructed INSIDE setup_fn, so the injected mesh is live.
        from cosmos_framework.posttrain.config.build import _dp_coords, resolve_pdims
        from cosmos_framework.posttrain.workers.ray import current_pdims

        pdims = resolve_pdims()
        self._at_setup = None if pdims is None else (pdims.world_size, pdims.dp_replicate, pdims.dp_shard)
        self._coords = _dp_coords(pdims)
        # The INJECTED accessor specifically — resolve_pdims would also succeed via its inference
        # fallback, which is exactly how a dead accessor stayed hidden.
        injected = current_pdims()
        self._injected = None if injected is None else (injected.world_size, injected.dp_replicate, injected.dp_shard)

    def dims(self) -> dict:
        return {"at_setup": self._at_setup, "dp_coords": self._coords, "injected": self._injected}

    def dims_after_setup(self) -> tuple[int, int, int] | None:
        """Resolve the same model-owned factor outside the setup injection window."""
        from cosmos_framework.posttrain.config.build import resolve_pdims

        pdims = resolve_pdims(parallelism=(2, 2))
        return None if pdims is None else (pdims.world_size, pdims.dp_replicate, pdims.dp_shard)


@pytest.mark.L1
@pytest.mark.GPU
def test_single_policy_replica_gets_a_real_2x2_hsdp_mesh() -> None:
    """Policy placement `(1,4)` with model parallelism `(2,2)` yields a real 2-D mesh."""
    ray = pytest.importorskip("ray")
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu < 4:
        pytest.skip(f"2x2 grid needs >= 4 visible GPUs, found {n_gpu}")

    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    policy = roles.trainer("policy", replicas=1, ranks=4, model_parallelism=(2, 2))
    try:
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(policy,), setup=lambda: {"policy": _MeshProbe()}),),
                    device="gpu",
                )
            ],
            backend="ray",
            threshold=0,
        )

        got = policy.dims().result()
        assert got["at_setup"] == (4, 2, 2), (
            f"expected an HSDP 2x2 mesh over 4 cells, got {got['at_setup']}. "
            f"(4, 1, 4) means model parallelism was ignored and the world stayed flat FSDP."
        )
        # The actor's model-owned injection must serve the mesh, not an inference fallback.
        assert got["injected"] == (4, 2, 2), (
            f"current_pdims() gave {got['injected']} inside setup_fn: the model mesh injection "
            f"did not reach the actor setup"
        )
        assert got["dp_coords"][1] == 4, f"full policy dp extent {got['dp_coords'][1]} != 4"

        assert policy.dims_after_setup().result() == (4, 2, 2), (
            "post-setup explicit model factorization must still resolve to 2x2"
        )
    finally:
        ray.shutdown()
