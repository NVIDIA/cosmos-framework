# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from cosmos_framework.utils import log, win_compat  # noqa: F401  log: imported before sys.platform is patched

posix_only = pytest.mark.skipif(win_compat.WINDOWS, reason="exercises the POSIX primitives")


def test_cosmos_attention_falls_back_to_sdpa_on_windows_only() -> None:
    with patch.object(win_compat.sys, "platform", "win32"):
        assert win_compat.attention_implementation("cosmos") == "sdpa"
        assert win_compat.attention_implementation("flash_attention_2") == "flash_attention_2"
        assert win_compat.attention_implementation("sdpa") == "sdpa"
    with patch.object(win_compat.sys, "platform", "linux"):
        assert win_compat.attention_implementation("cosmos") == "cosmos"


def test_flock_excludes_a_second_holder_until_unlocked(tmp_path: Path) -> None:
    lock_file = tmp_path / "lock"
    lock_file.write_bytes(b"\0")
    with open(lock_file, "r+b") as first, open(lock_file, "r+b") as second:
        win_compat.flock(first, win_compat.LOCK_EX)
        with pytest.raises(OSError):
            win_compat.flock(second.fileno(), win_compat.LOCK_EX | win_compat.LOCK_NB)
        win_compat.flock(first, win_compat.LOCK_UN)
        win_compat.flock(second, win_compat.LOCK_EX | win_compat.LOCK_NB)
        win_compat.flock(second, win_compat.LOCK_UN)


@posix_only
def test_link_dir_is_a_plain_symlink_off_windows(tmp_path: Path) -> None:
    target, link = tmp_path / "target", tmp_path / "link"
    target.mkdir()
    assert win_compat.link_dir(target, link) == "symlink"
    assert link.is_symlink() and link.resolve() == target.resolve()
    # A failure raises instead of degrading to a pointer file.
    with pytest.raises(FileExistsError):
        win_compat.link_dir(target, link)
    assert not (tmp_path / "link.txt").exists()


def _no_symlink_privilege(*_args, **_kwargs):
    raise OSError("A required privilege is not held by the client")


def test_link_dir_falls_back_to_a_junction_then_a_pointer_file_on_windows(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    junctions = []
    winapi = types.SimpleNamespace(CreateJunction=lambda src, dst: junctions.append((src, dst)))
    with (
        patch.object(win_compat, "WINDOWS", True),
        patch.object(win_compat.os, "symlink", _no_symlink_privilege),
        patch.dict(sys.modules, {"_winapi": winapi}),
    ):
        assert win_compat.link_dir(target, tmp_path / "a") == "junction"
        assert junctions == [(str(target), str(tmp_path / "a"))]

        winapi.CreateJunction = _no_symlink_privilege
        assert win_compat.link_dir(target, tmp_path / "b") == "pointer-file"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == f"{target}\n"


@posix_only
def test_remove_dir_link_removes_the_link_and_never_a_directory(tmp_path: Path) -> None:
    target, link = tmp_path / "target", tmp_path / "link"
    target.mkdir()
    (target / "checkpoint").write_text("keep")
    link.symlink_to(target, target_is_directory=True)

    win_compat.remove_dir_link(link)
    assert not os.path.lexists(link)
    assert (target / "checkpoint").read_text() == "keep"

    # A real directory where the link should be: raise, as os.remove always did.
    with pytest.raises(OSError):
        win_compat.remove_dir_link(target)
    assert (target / "checkpoint").read_text() == "keep"


def test_remove_dir_link_windows_fallback_is_not_recursive(tmp_path: Path) -> None:
    full, empty = tmp_path / "full", tmp_path / "empty"
    full.mkdir()
    empty.mkdir()
    (full / "checkpoint").write_text("keep")
    with patch.object(win_compat, "WINDOWS", True):
        # os.remove refuses a directory, which is how a junction can look; os.rmdir is the fallback.
        win_compat.remove_dir_link(empty)
        with pytest.raises(OSError):
            win_compat.remove_dir_link(full)
    assert not empty.exists()
    assert (full / "checkpoint").read_text() == "keep"


def _fake_collectives():
    """Stand-ins with the torch.distributed parameter names; they record what they receive."""
    received = []

    def all_reduce(tensor, op=None, group=None, async_op=False):
        received.append(tensor)
        tensor.mul_(2)

    def all_gather(tensor_list, tensor, group=None, async_op=False):
        received.append(tensor)
        for out in tensor_list:
            out.copy_(tensor)

    def gather(tensor, gather_list=None, dst=0, group=None, async_op=False):
        received.append(tensor)

    def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        received.append(input_tensor_list)

    namespace = types.SimpleNamespace(
        all_reduce=all_reduce, all_gather=all_gather, gather=gather, all_to_all=all_to_all
    )
    return namespace, received


def test_gloo_shims_stage_through_a_copy_and_write_the_result_back() -> None:
    namespace, received = _fake_collectives()
    alias = types.SimpleNamespace(all_reduce=namespace.all_reduce)
    wrapped = win_compat._shim_gloo_collectives((namespace, alias), needs_staging=lambda tensor: True)
    assert wrapped == ["all_reduce", "all_gather", "all_to_all", "gather"]
    assert alias.all_reduce is namespace.all_reduce  # one wrapper, installed on every namespace

    value = torch.tensor([1.0, 2.0])
    assert namespace.all_reduce(value) is None
    assert received[-1] is not value  # the backend saw the staged copy
    assert value.tolist() == [2.0, 4.0]  # and the result came back in place

    outputs = [torch.zeros(2), torch.zeros(2)]
    namespace.all_gather(outputs, tensor=value)
    assert received[-1] is not value
    assert [out.tolist() for out in outputs] == [[2.0, 4.0], [2.0, 4.0]]


def test_gloo_shims_see_async_op_however_it_is_passed() -> None:
    namespace, received = _fake_collectives()
    win_compat._shim_gloo_collectives((namespace,), needs_staging=lambda tensor: True)
    value = torch.ones(2)
    with pytest.raises(RuntimeError, match="async_op=True"):
        namespace.all_reduce(value, None, None, True)
    with pytest.raises(RuntimeError, match="async_op=True"):
        namespace.all_reduce(value, async_op=True)
    with pytest.raises(RuntimeError, match="async_op=True"):
        namespace.all_gather([torch.zeros(2)], value, None, True)
    assert received == [] and value.tolist() == [1.0, 1.0]


def test_gloo_shims_guard_the_collectives_they_do_not_stage() -> None:
    namespace, received = _fake_collectives()
    win_compat._shim_gloo_collectives((namespace,), needs_staging=lambda tensor: True)
    value = torch.ones(2)
    with pytest.raises(RuntimeError, match=r"torch\.distributed\.gather was called with a CUDA tensor"):
        namespace.gather(value)
    with pytest.raises(RuntimeError, match=r"torch\.distributed\.all_to_all "):
        namespace.all_to_all([torch.zeros(2)], input_tensor_list=[value])
    assert received == []


def test_gloo_shims_pass_host_tensors_straight_through() -> None:
    namespace, received = _fake_collectives()
    win_compat._shim_gloo_collectives((namespace,), needs_staging=lambda tensor: False)
    value = torch.ones(2)
    namespace.all_reduce(value, None, None, True)
    namespace.gather(value)
    assert received[0] is value and received[1] is value
    assert value.tolist() == [2.0, 2.0]


def test_gloo_shims_bind_the_real_torch_signatures() -> None:
    names = (*win_compat._GLOO_STAGED_COLLECTIVES, *win_compat._GLOO_GUARDED_COLLECTIVES)
    namespace = types.SimpleNamespace(**{name: getattr(dist, name) for name in names if hasattr(dist, name)})
    wrapped = win_compat._shim_gloo_collectives((namespace,), needs_staging=lambda tensor: True)
    assert set(win_compat._GLOO_STAGED_COLLECTIVES) <= set(wrapped)
    value = torch.ones(2)
    # Only a staged wrapper raises this message, and it does so before reaching the backend.
    with pytest.raises(RuntimeError, match="async_op=True"):
        namespace.all_reduce(value, dist.ReduceOp.SUM, None, True)
    with pytest.raises(RuntimeError, match="async_op=True"):
        namespace.broadcast(value, 0, None, True)
    with pytest.raises(RuntimeError, match="async_op=True"):
        namespace.reduce(value, 0, dist.ReduceOp.SUM, None, True)
    with pytest.raises(RuntimeError, match="async_op=True"):
        namespace.all_gather([torch.zeros(2)], value, None, True)


@posix_only
def test_gloo_shims_are_not_installed_off_windows() -> None:
    before = dist.all_reduce
    assert win_compat.install_gloo_cuda_collective_shims() is False
    assert dist.all_reduce is before
