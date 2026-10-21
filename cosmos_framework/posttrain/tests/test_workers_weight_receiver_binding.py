# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exercise the real actor binding method without a Ray cluster or engine process."""

import atexit

import pytest
import torch

from cosmos_framework.posttrain.comm.weight_transfer import WeightSync
from cosmos_framework.posttrain.roles.rollout.base import WeightReceiverBase
from cosmos_framework.posttrain.workers.ray import _HostImpl

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Receiver(WeightReceiverBase):
    def __init__(self, fail=False):
        self.model = torch.nn.Linear(2, 2)
        self.initialized = []
        self.fail = fail

    def initialize_weight_receiver(self):
        self.initialized.append(self._weight_backend)
        if self.fail:
            raise RuntimeError("receiver initialization failed")


@pytest.fixture
def host():
    value = _HostImpl()
    try:
        yield value
    finally:
        atexit.unregister(value.drain_pending_releases)
        for body in value._bodies.values():
            backend = getattr(body, "_weight_backend", None)
            if backend is not None:
                backend.close()


def test_actor_initializes_only_the_declared_receiver(host):
    selected, other = _Receiver(), _Receiver()
    host._bodies = {"selected": selected, "other": other}
    host.set_weight_backend({"kind": "copy"}, ("selected",))
    assert selected.initialized == [selected._weight_backend]
    assert other.initialized == [] and not hasattr(other, "_weight_backend")


def test_actor_binding_failure_releases_the_new_backend(host, monkeypatch):
    receiver = _Receiver(fail=True)
    host._bodies = {"receiver": receiver}
    closed = []
    original = WeightSync.close

    def closing(backend):
        closed.append(backend)
        original(backend)

    monkeypatch.setattr(WeightSync, "close", closing)
    with pytest.raises(RuntimeError, match="receiver initialization failed"):
        host.set_weight_backend({"kind": "copy"}, ("receiver",))
    assert closed == receiver.initialized
    assert receiver._weight_backend is None
    receiver.fail = False
    host.set_weight_backend({"kind": "copy"}, ("receiver",))
    assert receiver._weight_backend is not closed[0]
