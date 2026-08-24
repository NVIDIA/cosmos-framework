# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
import pickle
import threading
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.utils import distributed


def test_init_uses_gloo_backend_for_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    init_process_group = Mock()
    monkeypatch.setenv("COSMOS_DEVICE", "cpu")
    monkeypatch.setenv("TORCH_NCCL_BLOCKING_WAIT", "0")
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    monkeypatch.setattr(distributed, "INTERNAL", False)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "init_process_group", init_process_group)
    monkeypatch.setattr(distributed.pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(
        distributed,
        "Device",
        lambda _local_rank: SimpleNamespace(get_cpu_affinity=lambda: [0]),
    )
    monkeypatch.setattr(distributed.os, "sched_setaffinity", lambda _pid, _affinity: None)
    monkeypatch.setattr(distributed.torch.cuda, "set_device", lambda _local_rank: None)

    distributed.init()

    assert init_process_group.call_args.kwargs["backend"] == "gloo"


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


@pytest.mark.L0
@pytest.mark.CPU
def test_broadcast_object_list_optimized_broadcasts_only_large_tensor_leaves_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = 0
    skeletons: list[bytes] = []
    tensor_payloads: deque[torch.Tensor] = deque()

    def _get_rank() -> int:
        return rank

    def _get_backend(group: Any) -> str:
        del group
        return torch.distributed.Backend.GLOO

    def _broadcast_object_list(
        object_list: list[Any],
        *,
        src: int | None,
        group: Any,
        device: torch.device | None,
        group_src: int | None,
    ) -> None:
        del src, group, device, group_src
        if rank == 0:
            skeletons.append(pickle.dumps(object_list[0]))
        else:
            object_list[0] = pickle.loads(skeletons[0])

    def _broadcast_tensor(
        tensor: torch.Tensor,
        *,
        src: int | None,
        group: Any,
        group_src: int | None,
    ) -> None:  # tensor: [*shape]
        del src, group, group_src
        if rank == 0:
            tensor_payloads.append(tensor.clone())  # [*shape]
        else:
            tensor.copy_(tensor_payloads.popleft())  # [*shape]

    monkeypatch.setattr(distributed.dist, "get_rank", _get_rank)
    monkeypatch.setattr(distributed.dist, "get_backend", _get_backend)
    monkeypatch.setattr(distributed.dist, "broadcast_object_list", _broadcast_object_list)
    monkeypatch.setattr(distributed.dist, "broadcast", _broadcast_tensor)

    video = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)  # [2,3,4]
    frame_count = torch.tensor(197, dtype=torch.int64)  # []
    source_objects = [
        {
            "video": [[video]],
            "video_alias": video,
            "metadata": (frame_count, "mads"),
        },
        False,
    ]
    result = distributed.broadcast_object_list_optimized(
        source_objects,
        src=0,
        group=object(),
        min_tensor_bytes=16,
    )

    skeleton = pickle.loads(skeletons[0])
    assert not _contains_tensor(skeleton[0]["video"])
    assert _contains_tensor(skeleton[0]["metadata"])
    assert len(tensor_payloads) == 1

    rank = 1
    receiver_objects: list[Any] = [None, None]
    distributed.broadcast_object_list_optimized(
        receiver_objects,
        src=0,
        group=object(),
        min_tensor_bytes=16,
    )

    assert result is None
    assert receiver_objects[1] is False
    assert receiver_objects[0]["metadata"][1] == "mads"
    torch.testing.assert_close(receiver_objects[0]["video"][0][0], video)
    assert receiver_objects[0]["video_alias"] is receiver_objects[0]["video"][0][0]
    torch.testing.assert_close(receiver_objects[0]["metadata"][0], frame_count)
    torch.testing.assert_close(source_objects[0]["video"][0][0], video)
    assert source_objects[0]["video_alias"] is source_objects[0]["video"][0][0]
    assert not tensor_payloads


@pytest.mark.L0
@pytest.mark.CPU
def test_broadcast_object_list_optimized_handles_stop_signal_without_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank = 0
    skeletons: list[bytes] = []

    def _get_rank() -> int:
        return rank

    def _get_backend(group: Any) -> str:
        del group
        return torch.distributed.Backend.GLOO

    def _broadcast_object_list(
        object_list: list[Any],
        *,
        src: int | None,
        group: Any,
        device: torch.device | None,
        group_src: int | None,
    ) -> None:
        del src, group, device, group_src
        if rank == 0:
            skeletons.append(pickle.dumps(object_list[0]))
        else:
            object_list[0] = pickle.loads(skeletons[0])

    monkeypatch.setattr(distributed.dist, "get_rank", _get_rank)
    monkeypatch.setattr(distributed.dist, "get_backend", _get_backend)
    monkeypatch.setattr(distributed.dist, "broadcast_object_list", _broadcast_object_list)

    tensor_broadcast_count = 0

    def _broadcast_tensor(
        tensor: torch.Tensor,
        *,
        src: int | None,
        group: Any,
        group_src: int | None,
    ) -> None:  # tensor: [*shape]
        del tensor, src, group, group_src
        nonlocal tensor_broadcast_count
        tensor_broadcast_count += 1

    monkeypatch.setattr(distributed.dist, "broadcast", _broadcast_tensor)

    source_objects = [None, True]
    distributed.broadcast_object_list_optimized(source_objects, group=object(), min_tensor_bytes=0)

    rank = 1
    receiver_objects = [None, False]
    distributed.broadcast_object_list_optimized(receiver_objects, group=object(), min_tensor_bytes=0)

    assert source_objects == [None, True]
    assert receiver_objects == [None, True]
    assert tensor_broadcast_count == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_broadcast_object_list_optimized_defaults_to_original_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[Any], int | None, Any, torch.device | None, int | None]] = []

    def _broadcast_object_list(
        object_list: list[Any],
        *,
        src: int | None,
        group: Any,
        device: torch.device | None,
        group_src: int | None,
    ) -> None:
        calls.append((object_list, src, group, device, group_src))
        object_list[:] = ["from_source"]

    monkeypatch.setattr(distributed.dist, "broadcast_object_list", _broadcast_object_list)
    object_list: list[Any] = [None]
    group = object()
    device = torch.device("cpu")

    result = distributed.broadcast_object_list_optimized(
        object_list,
        group=group,
        device=device,
        group_src=0,
    )

    assert result is None
    assert object_list == ["from_source"]
    assert calls == [(object_list, None, group, device, 0)]


@pytest.mark.L0
@pytest.mark.CPU
def test_broadcast_object_list_optimized_delegates_non_member_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_broadcast = Mock()
    monkeypatch.setattr(distributed.dist, "broadcast_object_list", original_broadcast)
    object_list: list[Any] = ["local_value"]
    group = distributed.dist.GroupMember.NON_GROUP_MEMBER

    result = distributed.broadcast_object_list_optimized(
        object_list,
        src=0,
        group=group,
        min_tensor_bytes=0,
    )

    assert result is None
    assert object_list == ["local_value"]
    original_broadcast.assert_called_once_with(
        object_list,
        src=0,
        group=group,
        device=None,
        group_src=None,
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_broadcast_object_list_optimized_rejects_container_subclasses_before_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DictSubclass(dict):
        pass

    def _get_rank() -> int:
        return 0

    object_broadcast = Mock()
    monkeypatch.setattr(distributed.dist, "get_rank", _get_rank)
    monkeypatch.setattr(distributed.dist, "broadcast_object_list", object_broadcast)

    with pytest.raises(TypeError, match="plain dict containers"):
        distributed.broadcast_object_list_optimized(
            [DictSubclass(value="metadata")],
            src=0,
            min_tensor_bytes=0,
        )

    object_broadcast.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_broadcast_object_list_optimized_rejects_shared_containers_before_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _get_rank() -> int:
        return 0

    shared: list[Any] = []
    object_broadcast = Mock()
    monkeypatch.setattr(distributed.dist, "get_rank", _get_rank)
    monkeypatch.setattr(distributed.dist, "broadcast_object_list", object_broadcast)

    with pytest.raises(ValueError, match="without shared container references"):
        distributed.broadcast_object_list_optimized(
            [shared, shared],
            src=0,
            min_tensor_bytes=0,
        )

    object_broadcast.assert_not_called()


@pytest.mark.L0
@pytest.mark.CPU
def test_broadcast_object_list_optimized_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        distributed.broadcast_object_list_optimized([], src=0, min_tensor_bytes=-1)


@pytest.mark.L0
@pytest.mark.CPU
def test_world_comm_timeout_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(distributed.WORLD_COMM_TIMEOUT_ENV_VAR, raising=False)
    assert distributed.get_world_comm_timeout_sec() == distributed.DEFAULT_WORLD_COMM_TIMEOUT_SEC


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(("value", "expected"), [("120", 120.0), ("0", 0.0), ("  90.5  ", 90.5)])
def test_world_comm_timeout_honors_override(monkeypatch: pytest.MonkeyPatch, value: str, expected: float) -> None:
    monkeypatch.setenv(distributed.WORLD_COMM_TIMEOUT_ENV_VAR, value)
    assert distributed.get_world_comm_timeout_sec() == expected


@pytest.mark.L0
@pytest.mark.CPU
def test_world_comm_timeout_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(distributed.WORLD_COMM_TIMEOUT_ENV_VAR, "not-a-number")
    assert distributed.get_world_comm_timeout_sec() == distributed.DEFAULT_WORLD_COMM_TIMEOUT_SEC


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "Infinity", "1e400"])
def test_world_comm_timeout_rejects_non_finite(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # float() takes these, and each one disarms the watchdog: nan fails the `> 0` guard so no thread
    # is started, and inf makes the thread's Event.wait raise OverflowError.
    monkeypatch.setenv(distributed.WORLD_COMM_TIMEOUT_ENV_VAR, value)

    timeout_sec = distributed.get_world_comm_timeout_sec()

    assert timeout_sec == distributed.DEFAULT_WORLD_COMM_TIMEOUT_SEC
    assert math.isfinite(timeout_sec)
    assert timeout_sec > 0


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("value", ["-60", "-0.5"])
def test_world_comm_timeout_rejects_negative(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    # 0 is the documented way to wait indefinitely, so clamping a negative into it would read a typo
    # as a deliberate request to disarm the watchdog.
    monkeypatch.setenv(distributed.WORLD_COMM_TIMEOUT_ENV_VAR, value)

    timeout_sec = distributed.get_world_comm_timeout_sec()

    assert timeout_sec == distributed.DEFAULT_WORLD_COMM_TIMEOUT_SEC
    assert timeout_sec > 0


@pytest.mark.L0
@pytest.mark.CPU
def test_world_comm_watchdog_stays_quiet_when_build_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_codes: list[int] = []
    monkeypatch.setattr(distributed.os, "_exit", exit_codes.append)
    done = threading.Event()
    done.set()

    distributed._abort_on_world_comm_timeout(done, 30.0, rank=0, world_size=8)

    assert exit_codes == []


@pytest.mark.L0
@pytest.mark.CPU
def test_world_comm_watchdog_aborts_on_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    exit_codes: list[int] = []
    monkeypatch.setattr(distributed.os, "_exit", exit_codes.append)
    monkeypatch.setattr(distributed, "WORLD_COMM_ABORT_GRACE_SEC", 0.01)

    distributed._abort_on_world_comm_timeout(threading.Event(), 0.01, rank=3, world_size=128)

    assert exit_codes == [distributed.WORLD_COMM_TIMEOUT_EXIT_CODE]


@pytest.mark.L0
@pytest.mark.CPU
def test_world_comm_watchdog_spares_a_build_that_lands_during_the_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    # A build completing in the same instant the deadline expires must not be killed as a stall.
    exit_codes: list[int] = []
    monkeypatch.setattr(distributed.os, "_exit", exit_codes.append)
    monkeypatch.setattr(distributed, "WORLD_COMM_ABORT_GRACE_SEC", 30.0)
    done = threading.Event()
    threading.Timer(0.05, done.set).start()

    distributed._abort_on_world_comm_timeout(done, 0.01, rank=3, world_size=128)

    assert exit_codes == []


def _stub_world(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    monkeypatch.setattr(distributed.dist, "is_available", lambda: True)
    monkeypatch.setattr(distributed.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda *_args, **_kwargs: "gloo")
    monkeypatch.setattr(distributed, "get_world_size", lambda *_args, **_kwargs: world_size)
    monkeypatch.setattr(distributed, "get_rank", lambda *_args, **_kwargs: 0)


@pytest.mark.L0
@pytest.mark.CPU
def test_ensure_world_communicator_probes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_world(monkeypatch, world_size=4)
    probes: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor, *_args: Any, **_kwargs: Any) -> None:
        probes.append(tensor)
        tensor.fill_(4.0)

    monkeypatch.setattr(distributed.dist, "all_reduce", fake_all_reduce)

    distributed.ensure_world_communicator()

    assert len(probes) == 1


@pytest.mark.L0
@pytest.mark.CPU
def test_ensure_world_communicator_rejects_inconsistent_world(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_world(monkeypatch, world_size=4)

    def fake_all_reduce(tensor: torch.Tensor, *_args: Any, **_kwargs: Any) -> None:
        tensor.fill_(3.0)

    monkeypatch.setattr(distributed.dist, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="not consistent"):
        distributed.ensure_world_communicator()


@pytest.mark.L0
@pytest.mark.CPU
def test_ensure_world_communicator_skips_single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_world(monkeypatch, world_size=1)
    all_reduce = Mock()
    monkeypatch.setattr(distributed.dist, "all_reduce", all_reduce)

    distributed.ensure_world_communicator()

    all_reduce.assert_not_called()
