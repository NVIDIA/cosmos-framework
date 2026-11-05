# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Move a trainer's GPU state to host memory and back.

Two mechanisms with one interface (``offload`` / ``onload`` returning MiB moved):

- ``MemorySaverOffload``: the process was started with torch_memory_saver's preload hook, so every
  CUDA allocation it ever made is tracked; ``pause`` unmaps them all (after a host backup) and
  ``resume`` maps them back at the same addresses. Nothing in the process changes device or
  identity. Used for a trainer that owns its process, such as the GRPO policy actor.
- ``HostOffload``: public torch APIs only. Networks move with ``Module.to`` (FSDP2 keeps its sharded
  parameters consistent through that path), gradients by reassigning ``param.grad``, and optimizer
  moments by replacing the entries of ``optimizer.state`` with pinned host copies. Used when the
  trainer shares its process with a rollout that must keep running.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch

MEMORY_SAVER_ENABLE_ENV = "TMS_INIT_ENABLE"


def memory_saver_active() -> bool:
    """True when this process runs under torch_memory_saver's preload hook."""
    return os.environ.get(MEMORY_SAVER_ENABLE_ENV) == "1" and importlib.util.find_spec("torch_memory_saver") is not None


def memory_saver_env() -> dict[str, str] | None:
    """Environment that starts a fresh process under the preload hook, or None when unavailable.

    ``LD_PRELOAD`` must be in place before the interpreter starts, so this is for launching actors,
    not for the current process. The hook library links ``libcudart``, which lives in the CUDA
    runtime wheel rather than on the system library path.
    """
    if importlib.util.find_spec("torch_memory_saver") is None:
        return None
    from torch_memory_saver.utils import get_binary_path_from_package

    try:
        import nvidia.cuda_runtime as cuda_runtime

        cudart_dir = os.path.join(os.path.dirname(cuda_runtime.__file__), "lib")
    except ImportError:
        cudart_dir = ""
    library_path = ":".join(p for p in (cudart_dir, os.environ.get("LD_LIBRARY_PATH", "")) if p)
    return {
        "LD_PRELOAD": str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload")),
        "LD_LIBRARY_PATH": library_path,
        MEMORY_SAVER_ENABLE_ENV: "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }


@contextmanager
def untracked_allocation() -> Iterator[None]:
    """Allocate outside the memory saver so the memory stays mapped while the process is paused.

    Needed for buffers another process reads through CUDA IPC: IPC handles cannot be exported from
    saver-managed mappings. The allocator cache is emptied first so the allocation is a fresh
    ``cudaMalloc`` and not a cached segment that was already tracked. No-op without the saver.
    """
    if not memory_saver_active():
        yield
        return
    from torch_memory_saver import torch_memory_saver

    trim_device_memory()
    with torch_memory_saver.disable():
        yield


class MemorySaverOffload:
    """Pause/resume this whole process's CUDA memory through torch_memory_saver.

    While paused nothing in the process may touch the GPU: even a scratch allocation would be
    served from an unmapped segment. The loop guarantees that by calling nothing on the trainer
    between ``offload`` and ``onload``.
    """

    def __init__(self) -> None:
        from torch_memory_saver import torch_memory_saver

        torch_memory_saver.hook_mode = "preload"
        self._saver = torch_memory_saver
        self._paused = False
        self._bytes = 0

    @property
    def offloaded(self) -> bool:
        return self._paused

    def offload(
        self,
        modules: Iterable[torch.nn.Module] = (),
        *,
        optimizer: torch.optim.Optimizer | None = None,
        grads_of: Iterable[torch.nn.Parameter] = (),
    ) -> float:
        """Pause every tracked allocation. The arguments are accepted for interface parity; the saver
        covers the whole process. Returns the MiB of live tensors at the time of the pause."""
        del modules, optimizer, grads_of
        if self._paused:
            return 0.0
        self._bytes = torch.cuda.memory_allocated()
        trim_device_memory()
        self._saver.pause()
        torch.cuda.synchronize()
        self._paused = True
        return self._bytes / 2**20

    def onload(self) -> float:
        if not self._paused:
            return 0.0
        self._saver.resume()
        torch.cuda.synchronize()
        self._paused = False
        return self._bytes / 2**20


def make_offload() -> "MemorySaverOffload | HostOffload":
    """The memory saver when this process was started under its hook, else the host mover."""
    return MemorySaverOffload() if memory_saver_active() else HostOffload()


def trim_device_memory() -> None:
    """Return every fully free allocator segment to the driver so a colocated process can use it."""
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _local_bytes(tensor: torch.Tensor) -> int:
    """Bytes this rank actually stores: the local shard for a DTensor, the tensor otherwise."""
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    return local.numel() * local.element_size()


def _pinned_host_copy(tensor: torch.Tensor) -> torch.Tensor:
    host = torch.empty_like(tensor, device="cpu", pin_memory=True)
    host.copy_(tensor, non_blocking=True)
    return host


def _optimizer_device_entries(optimizer: torch.optim.Optimizer) -> list[tuple[dict | list, str | int]]:
    """Every CUDA tensor the optimizer owns, as ``(container, key)`` so it can be replaced in place.

    Covers the per-parameter ``state`` (moments, steps), tensor-valued group options (capturable
    ``lr`` / ``step``), and fp32 master copies kept in ``param_groups_master`` by ``FusedAdam``. The
    model parameters listed in ``param_groups`` are the modules' own and are not included here.
    """
    entries: list[tuple[dict | list, str | int]] = []
    for state in optimizer.state.values():
        entries.extend((state, key) for key, value in state.items() if torch.is_tensor(value) and value.is_cuda)
    for group in optimizer.param_groups:
        entries.extend(
            (group, key) for key, value in group.items() if key != "params" and torch.is_tensor(value) and value.is_cuda
        )
    for group in getattr(optimizer, "param_groups_master", None) or []:
        masters = group.get("params", [])
        entries.extend((masters, i) for i, value in enumerate(masters) if torch.is_tensor(value) and value.is_cuda)
    return entries


class HostOffload:
    """Park networks, gradients and optimizer state on the host between GPU phases.

    ``offload`` records what moved and where it came from; ``onload`` reverses exactly that. Both
    are idempotent, and both return the MiB moved so the loop can report it.
    """

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._modules: list[torch.nn.Module] = []
        self._grad_params: list[torch.nn.Parameter] = []
        self._states: list[tuple[dict | list, str | int]] = []
        self._bytes = 0

    @property
    def offloaded(self) -> bool:
        return self._device is not None

    def host_mib(self) -> float:
        return self._bytes / 2**20

    def offload(
        self,
        modules: Iterable[torch.nn.Module] = (),
        *,
        optimizer: torch.optim.Optimizer | None = None,
        grads_of: Iterable[torch.nn.Parameter] = (),
    ) -> float:
        """Move ``modules`` (parameters, buffers and their grads), the listed grads and every tensor
        the optimizer owns to the host. Returns the MiB moved (0 when already offloaded or nothing is
        on a GPU)."""
        if self.offloaded:
            return 0.0
        modules = list(modules)
        grad_params = [p for p in grads_of if p.grad is not None and p.grad.device.type == "cuda"]
        states = _optimizer_device_entries(optimizer) if optimizer is not None else []
        device = next(
            (
                t.device
                for t in (*(p for m in modules for p in m.parameters()), *(p.grad for p in grad_params))
                if t.is_cuda
            ),
            None,
        )
        if device is None and states:
            device = states[0][0][states[0][1]].device
        if device is None:
            return 0.0

        moved = 0
        for module in modules:
            moved += sum(_local_bytes(t) for t in (*module.parameters(), *module.buffers()) if t.is_cuda)
            moved += sum(_local_bytes(p.grad) for p in module.parameters() if p.grad is not None and p.grad.is_cuda)
            module.to("cpu")
        for param in grad_params:
            moved += _local_bytes(param.grad)
            param.grad = param.grad.to("cpu")
        for state, key in states:
            moved += _local_bytes(state[key])
            state[key] = _pinned_host_copy(state[key])
        trim_device_memory()

        self._device, self._modules, self._grad_params, self._states, self._bytes = (
            device,
            modules,
            grad_params,
            states,
            moved,
        )
        return moved / 2**20

    def onload(self) -> float:
        """Move everything ``offload`` parked back to its device. Returns the MiB moved."""
        if not self.offloaded:
            return 0.0
        device = self._device
        for module in self._modules:
            module.to(device)
        for param in self._grad_params:
            param.grad = param.grad.to(device)
        for state, key in self._states:
            state[key] = state[key].to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mib = self._bytes / 2**20
        self._device, self._modules, self._grad_params, self._states, self._bytes = None, [], [], [], 0
        return mib
