# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""init_weight_plane forms a cross-actor side group (independent of each actor's default group)
and a broadcast crosses it. Two Ray actors, CPU/gloo."""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray")


def _bcast_probe(host):
    # runs inside the actor: broadcast this actor's rank tensor from global src=0
    import torch

    comm = host.weight_comm_local()  # test-only direct accessor added alongside weight_comm()
    t = torch.tensor([7.0]) if comm.rank == 0 else torch.tensor([0.0])
    comm.broadcast(t, src=0)
    return float(t.item())


@pytest.mark.L0
@pytest.mark.CPU
def test_init_weight_plane_two_actor_gloo():
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcwp")
    try:
        from cosmos_framework.posttrain.workers.ray import _Host

        a0, a1 = _Host.remote(), _Host.remote()
        ip, port = ray.get(a0.rendezvous_endpoint.remote())
        # driver assigns global weight-plane ranks 0..1
        f0 = a0.init_weight_plane.remote(ip, port, 0, 2, "gloo")
        f1 = a1.init_weight_plane.remote(ip, port, 1, 2, "gloo")
        assert ray.get([f0, f1]) == [0, 1]
        r0 = a0.run.remote(_bcast_probe, a0)  # helper that calls fn(host) inside the actor
        r1 = a1.run.remote(_bcast_probe, a1)
        assert ray.get([r0, r1]) == [7.0, 7.0]
    finally:
        ray.shutdown()


def _body_with_nccl_backend_setup():
    # A model-bearing body whose weight backend is a COLLECTIVE NcclMover bound at "bind" time,
    # BEFORE the weight plane exists — so its mover starts with _comm=None (mirrors set_weight_backend).
    from cosmos_framework.posttrain.comm.transfer.movers import NcclMover
    from cosmos_framework.posttrain.comm.weight_transfer import WeightSync

    class B:
        def __init__(self):
            self.model = object()
            self._weight_backend = WeightSync(NcclMover(comm=None))

        def mover_state(self):
            mv = self._weight_backend.mover
            return (mv._comm is not None, mv.recv_plan)

    return {"b": B()}


@pytest.mark.L0
@pytest.mark.CPU
def test_init_weight_plane_injects_comm_and_manifest_into_mover():
    # P1 fix: init_weight_plane must thread the live comm (and, for sharded sinks, the recv_manifest)
    # into the body's ALREADY-bound NcclMover — else stage/fetch hit _comm is None and raise.
    ray.init(ignore_reinit_error=True, num_cpus=2, include_dashboard=False, log_to_driver=False, _temp_dir="/tmp/rcwp")
    try:
        from cosmos_framework.posttrain.workers.ray import _Host

        a = _Host.remote()
        ray.get(a.run_setup.remote(_body_with_nccl_backend_setup))
        ip, port = ray.get(a.rendezvous_endpoint.remote())
        # Pass the manifest in state_dict (insertion) order — weight BEFORE bias — to prove it gets
        # re-sorted to match NcclMover.stage's sorted(obj) broadcast order.
        manifest = [("weight", (2, 2), "torch.float32"), ("bias", (2,), "torch.float32")]
        ray.get(a.init_weight_plane.remote(ip, port, 0, 1, "gloo", manifest))  # world=1 single-rank plane
        comm_set, recv_plan = ray.get(a.invoke.remote("b", "mover_state", (), {}))
        assert comm_set is True  # comm threaded into the bound mover
        assert recv_plan == [("bias", (2,), "torch.float32"), ("weight", (2, 2), "torch.float32")]  # sorted
    finally:
        ray.shutdown()
