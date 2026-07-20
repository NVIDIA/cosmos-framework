# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import collections
import collections.abc
import ctypes
import errno
import functools
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable, Container, Optional

import pynvml
import torch
import torch.distributed as dist
from torch.distributed import get_process_group_ranks

from cosmos_framework.utils.device import Device
from cosmos_framework.utils.flags import INTERNAL

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group
    from torch.distributed.utils import _sync_module_states, _verify_param_shape_across_processes

from cosmos_framework.utils import log

if TYPE_CHECKING:
    from cosmos_framework.utils.config import DDPConfig


def init() -> int | None:
    """Initialize distributed training."""
    if dist.is_initialized():
        return torch.cuda.current_device()

    # Set GPU affinity.
    pynvml.nvmlInit()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    try:
        device = Device(local_rank)
        os.sched_setaffinity(0, device.get_cpu_affinity())
    except pynvml.NVMLError as e:
        log.warning(f"Failed to set device affinity: {e}")
    except OSError as e:
        if e.errno != errno.EINVAL:
            raise
        log.warning(f"Failed to set device affinity: {e}")
    # Set up distributed communication. CPU checkpoint conversion needs Gloo
    # because NCCL cannot synchronize CPU-resident tokenizer or model tensors.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        torch.cuda.set_device(local_rank)
        # Get the timeout value from environment variable
        timeout_seconds = os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)
        # Convert the timeout to an integer (if it isn't already) and then to a timedelta
        timeout_timedelta = timedelta(seconds=int(timeout_seconds))
        backend = "nccl" if os.environ.get("COSMOS_DEVICE", "cuda").lower() == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", timeout=timeout_timedelta)
        log.critical(
            f"Initialized distributed training with local rank {local_rank} using {backend} with timeout {timeout_seconds}",
            rank0_only=False,
        )
    # Increase the L2 fetch granularity for faster speed.
    # For oss, we need to search for the library in site-packages.
    if INTERNAL:
        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device.
        p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
    log.info(f"Training with {get_world_size()} GPUs.")


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get world size. How many GPUs are available in this job.

    Returns:
        world_size (int): The total number of GPUs available in this job.
    """
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group)
    return world_size


def is_rank0() -> bool:
    """Check if current process is the master GPU.

    Returns:
        (bool): True if this function is called from the master GPU, else False.
    """
    return get_rank() == 0


def is_local_rank0() -> bool:
    """Check if current process is the local master GPU in the current node.

    Returns:
        (bool): True if this function is called from the local master GPU, else False.
    """
    return torch.cuda.current_device() == 0


def rank0_only(func: Callable) -> Callable:
    """Apply this function only to the master GPU.

    Example usage:
        @rank0_only
        def func(x):
            return x + 3

    Args:
        func (Callable): a function.

    Returns:
        (Callable): A function wrapper executing the function only on the master GPU.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if is_rank0():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def barrier() -> None:
    """Barrier for all GPUs."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def rank0_first(func: Callable) -> Callable:
    """run the function on rank 0 first, then on other ranks."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if is_rank0():
            result = func(*args, **kwargs)
        barrier()
        if not is_rank0():
            result = func(*args, **kwargs)
        return result

    return wrapper


def parallel_model_wrapper(config_ddp: DDPConfig, model: torch.nn.Module) -> torch.nn.Module | DistributedDataParallel:
    """Wraps the model to enable data parallalism for training across multiple GPU devices.

    Args:
        config_ddp (DDPConfig): The data parallel config.
        model (torch.nn.Module): The PyTorch module.

    Returns:
        model (torch.nn.Module | DistributedDataParallel): The data parallel model wrapper
            if distributed environment is available, otherwise return the original model.
    """
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        try:
            from megatron.core import parallel_state

            ddp_group = parallel_state.get_data_parallel_group(with_context_parallel=True)
        except Exception as e:
            log.info(e)
            log.info("parallel_state not initialized, treating all GPUs equally for DDP")
            ddp_group = None

        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config_ddp.find_unused_parameters,
            static_graph=config_ddp.static_graph,
            broadcast_buffers=config_ddp.broadcast_buffers,
            process_group=ddp_group,
        )
    return model


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """This extends torch.nn.parallel.DistributedDataParallel with .training_step().

    This borrows the concept of `forward-redirection` from Pytorch lightning. It wraps an ImaginaireModel such that
    model.training_step() would be executed when calling self.training_step(), while preserving the behavior of calling
    model() for Pytorch modules. Internally, this is a double rerouting mechanism (training_step -> forward ->
    training_step), allowing us to preserve the function names and signatures.
    """

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.show_sync_grad_static_graph_warning = True

    def training_step(self, *args, **kwargs) -> Any:
        # Cache the original model.forward() method.
        original_forward = self.module.forward

        def wrapped_training_step(*_args, **_kwargs):  # noqa: ANN202
            # Unpatch immediately before calling training_step() because itself may want to call the real forward.
            self.module.forward = original_forward
            # The actual .training_step().
            return self.module.training_step(*_args, **_kwargs)

        # Patch the original_module's forward so we can redirect the arguments back to the real method.
        self.module.forward = wrapped_training_step
        # Call self, which implicitly calls self.forward() --> model.forward(), which is now model.training_step().
        # Without calling self.forward() or model.forward() explciitly, implicit hooks are also executed.
        return self(*args, **kwargs)


@contextmanager
def ddp_sync_grad(model, enabled):
    r"""
    Context manager to enable/disable gradient synchronizations across DDP processes for DDP model.
    Modified from:
    https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html#DistributedDataParallel.no_sync
    Note that this is incompatible with static_graph=True and will be an no-op if static_graph=True.

    Within this context, gradients will be accumulated on module
    variables, which will later be synchronized in the first
    forward-backward pass exiting the context.

    .. warning::
        The forward pass should be included inside the context manager, or
        else gradients will still be synchronized.
    """
    assert isinstance(model, torch.nn.Module)
    if isinstance(model, DistributedDataParallel):
        old_require_backward_grad_sync = model.require_backward_grad_sync
        if model.static_graph and model.require_backward_grad_sync != enabled:
            if model.show_sync_grad_static_graph_warning:
                log.warning("DDP static_graph=True is incompatible with sync_grad(). Performance will be reduced.")
                model.show_sync_grad_static_graph_warning = False
        else:
            model.require_backward_grad_sync = enabled
    try:
        yield
    finally:
        if isinstance(model, DistributedDataParallel):
            model.require_backward_grad_sync = old_require_backward_grad_sync


def collate_batches(data_batches: list[dict[str, torch.Tensor]]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Aggregate the list of data batches from all devices and process the results.

    This is used for gathering validation data batches with cosmos_framework.utils.dataloader.DistributedEvalSampler.
    It will return the data/output of the entire validation set in its original index order. The sizes of data_batches
    in different ranks may differ by 1 (if dataset size is not evenly divisible), in which case a dummy sample will be
    created before calling dis.all_gather().

    Args:
        data_batches (list[dict[str, torch.Tensor]]): List of tensors or (hierarchical) dictionary where
            leaf entries are tensors.

    Returns:
        data_gather (torch.Tensor | dict[str, torch.Tensor]): tensors or (hierarchical) dictionary where
            leaf entries are concatenated tensors.
    """
    if isinstance(data_batches[0], torch.Tensor):
        # Concatenate the local data batches.
        data_concat = torch.cat(data_batches, dim=0)  # type: ignore
        # Get the largest number of local samples from all ranks to determine whether to dummy-pad on this rank.
        max_num_local_samples = torch.tensor(len(data_concat), device="cuda")
        dist.all_reduce(max_num_local_samples, op=dist.ReduceOp.MAX)
        if len(data_concat) < max_num_local_samples:
            assert len(data_concat) + 1 == max_num_local_samples
            dummy = torch.empty_like(data_concat[:1])
            data_concat = torch.cat([data_concat, dummy], dim=0)
            dummy_count = torch.tensor(1, device="cuda")
        else:
            dummy_count = torch.tensor(0, device="cuda")
        # Get all concatenated batches from all ranks and concatenate again.
        dist.all_reduce(dummy_count, op=dist.ReduceOp.SUM)
        data_concat = all_gather_tensor(data_concat.contiguous())
        data_collate = torch.stack(data_concat, dim=1).flatten(start_dim=0, end_dim=1)
        # Remove the dummy samples.
        if dummy_count > 0:
            data_collate = data_collate[:-dummy_count]
    elif isinstance(data_batches[0], collections.abc.Mapping):
        data_collate = dict()
        for key in data_batches[0].keys():
            data_collate[key] = collate_batches([data[key] for data in data_batches])  # type: ignore
    else:
        raise TypeError
    return data_collate


@torch.no_grad()
def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather the corresponding tensor from all GPU devices to a list.

    Args:
        tensor (torch.Tensor): Pytorch tensor.

    Returns:
        tensor_list (list[torch.Tensor]): A list of Pytorch tensors gathered from all GPU devices.
    """
    tensor_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def gather_object(payload: Any) -> list[Any] | None:
    """Gather the corresponding object from all GPU devices to a rank 0 hosted list.

    Args:
        payload: Any pickle-able object.

    Returns:
        payload_list (list[Any]) | None:
            Rank 0: A list of Pytorch tensors gathered from all RANK process.
            Rest : None
    """
    rank, world_size = get_rank(), get_world_size()
    payload_gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(payload, object_gather_list=payload_gathered, dst=0)
    return payload_gathered


def broadcast(tensor, src, group=None, async_op=False):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    dist.broadcast(tensor, src=src, group=group, async_op=async_op)


def dist_reduce_tensor(tensor, rank=0, reduce="mean"):
    r"""Reduce to rank 0"""
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        dist.reduce(tensor, dst=rank)
        if get_rank() == rank:
            if reduce == "mean":
                tensor /= world_size
            elif reduce == "sum":
                pass
            else:
                raise NotImplementedError
    return tensor


def sync_model_states(
    model: torch.nn.Module,
    process_group: Optional[dist.ProcessGroup] = None,
    src: int = 0,
    params_and_buffers_to_ignore: Optional[Container[str]] = None,
    broadcast_buffers: bool = True,
):
    """
    Modify based on DDP source code
    Synchronizes the parameters and buffers of a model across different processes in a distributed setting.

    This function ensures that all processes in the specified process group have the same initial parameters and
    buffers from the source rank, typically rank 0. It is useful when different processes start with different model
    states and a synchronization is required to ensure consistency across all ranks.

    Args:
        model (nn.Module): The model whose parameters and buffers are to be synchronized.
        process_group (dist.ProcessGroup, optional): The process group for communication. If None,
            the default group is used. Defaults to None.
        src (int, optional): The source rank from which parameters and buffers will be broadcasted.
            Defaults to 0.
        params_and_buffers_to_ignore (Optional[Container[str]], optional): A container of parameter and buffer
            names to exclude from synchronization. Defaults to None, which means all parameters and buffers are
            included.
        broadcast_buffers (bool, optional): Whether to broadcast buffers or not. Defaults to True.

    Side Effects:
        This function modifies the state of the model in-place to synchronize it with the source rank's model state.

    Raises:
        RuntimeError: If the shapes of parameters across processes do not match, a runtime error will be raised.

    Examples:
        >>> # downloading duplicated model weights from s3 in each rank and save network bandwidth
        >>> # useful and save our time when model weights are huge
        >>> if dist.get_rank == 0:
        >>>     model.load_state_dict(network_bound_weights_download_fn(s3_weights_path))
        >>> dist.barrir()
        >>> sync_model_states(model) # sync rank0 weights to other ranks
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    if process_group is None:
        process_group = _get_default_group()
    if not params_and_buffers_to_ignore:
        params_and_buffers_to_ignore = set()

    log.info(
        f"Synchronizing model states from rank {src} to all ranks in process group {get_process_group_ranks(process_group)}."
    )

    # Build tuple of (module, parameter) for all parameters that require grads.
    modules_and_parameters = [
        (module, parameter)
        for module_name, module in model.named_modules()
        for parameter in [
            param
            # Note that we access module.named_parameters instead of
            # parameters(module). parameters(module) is only needed in the
            # single-process multi device case, where it accesses replicated
            # parameters through _former_parameters.
            for param_name, param in module.named_parameters(recurse=False)
            if f"{module_name}.{param_name}" not in params_and_buffers_to_ignore
            # if param.requires_grad
            # and f"{module_name}.{param_name}" not in params_and_buffers_to_ignore
        ]
    ]

    # Deduplicate any parameters that might be shared across child modules.
    memo = set()
    modules_and_parameters = [
        # "p not in memo" is the deduplication check.
        # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
        (m, p)
        for m, p in modules_and_parameters
        if p not in memo and not memo.add(p)  # type: ignore[func-returns-value]
    ]

    # Build list of parameters.
    parameters = [parameter for _, parameter in modules_and_parameters]
    if len(parameters) == 0:
        return

    _verify_param_shape_across_processes(process_group, parameters)

    _sync_module_states(
        module=model,
        process_group=process_group,
        broadcast_bucket_size=int(250 * 1024 * 1024),
        src=src,
        params_and_buffers_to_ignore=params_and_buffers_to_ignore,
        broadcast_buffers=broadcast_buffers,
    )


def all_gather_object(payload: Any) -> list[Any]:
    """Gather the corresponding object from all GPU devices to all ranks."""
    world_size = get_world_size()
    payload_gathered = [None] * world_size
    dist.all_gather_object(payload_gathered, payload)
    return payload_gathered  # type: ignore[return-value]


def broadcast_object(object, *args, **kwargs):
    """Broadcast a object to all GPU."""
    if not dist.is_available() or not dist.is_initialized():
        return object
    object_list = [object]
    dist.broadcast_object_list(object_list, *args, **kwargs)
    return object_list[0]


def broadcast_object_list(object_list, *args, **kwargs):
    """Broadcast a object list to all GPU. (the list is inplace edited)"""
    if not dist.is_available() or not dist.is_initialized():
        return None
    else:
        dist.broadcast_object_list(object_list, *args, **kwargs)


@dataclass(frozen=True)
class _TensorBroadcastMetadata:
    """Small placeholder used while broadcasting a nested object containing tensors."""

    shape: tuple[int, ...]
    dtype: torch.dtype


def _extract_tensor_leaves(value: Any, min_tensor_bytes: int) -> tuple[Any, list[torch.Tensor]]:
    """Replace sufficiently large tensor leaves while preserving the surrounding containers."""
    tensor_leaves: list[torch.Tensor] = []
    seen_direct_ids: set[int] = set()

    def _require_unique(current_value: Any) -> None:
        object_id = id(current_value)
        if object_id in seen_direct_ids:
            raise ValueError(
                "Optimized object broadcast requires an acyclic tree without shared container or tensor references; "
                f"encountered {type(current_value).__name__} more than once."
            )
        seen_direct_ids.add(object_id)

    def _extract(current_value: Any) -> Any:
        if isinstance(current_value, torch.Tensor):
            tensor_bytes = current_value.numel() * current_value.element_size()
            if tensor_bytes < min_tensor_bytes:
                return current_value
            if current_value.layout != torch.strided:
                log.warning(f"Only strided tensors can be broadcast; skip layout={current_value.layout}.")
                return current_value
            if type(current_value) is not torch.Tensor:
                raise TypeError(
                    "Optimized object broadcast only supports plain torch.Tensor leaves, "
                    f"got {type(current_value).__name__}."
                )
            _require_unique(current_value)
            tensor_leaves.append(current_value)
            return _TensorBroadcastMetadata(shape=tuple(current_value.shape), dtype=current_value.dtype)
        if isinstance(current_value, dict):
            if type(current_value) is not dict:
                raise TypeError(
                    "Optimized object broadcast only supports plain dict containers, "
                    f"got {type(current_value).__name__}."
                )
            _require_unique(current_value)
            return {key: _extract(item) for key, item in current_value.items()}
        if isinstance(current_value, list):
            if type(current_value) is not list:
                raise TypeError(
                    "Optimized object broadcast only supports plain list containers, "
                    f"got {type(current_value).__name__}."
                )
            _require_unique(current_value)
            return [_extract(item) for item in current_value]
        if isinstance(current_value, tuple):
            is_namedtuple = hasattr(current_value, "_fields")
            if type(current_value) is not tuple and not is_namedtuple:
                raise TypeError(
                    "Optimized object broadcast only supports plain tuple or namedtuple containers, "
                    f"got {type(current_value).__name__}."
                )
            _require_unique(current_value)
            extracted_items = tuple(_extract(item) for item in current_value)
            # Namedtuples are tuple subclasses whose constructors expect each field as a positional argument.
            if is_namedtuple:
                return type(current_value)(*extracted_items)
            return extracted_items
        return current_value

    return _extract(value), tensor_leaves


def broadcast_object_list_optimized(
    object_list: list[Any],
    src: int | None = None,
    group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
    group_src: int | None = None,
    *,
    min_tensor_bytes: int = sys.maxsize,
) -> None:
    """Broadcast objects in place, sending sufficiently large tensor leaves directly.

    This is a signature-compatible alternative to :func:`torch.distributed.broadcast_object_list`.
    The source broadcasts an object-list skeleton first. Tensor leaves at least
    ``min_tensor_bytes`` bytes large are replaced by metadata, broadcast
    directly, and inserted back into the skeleton in traversal order. Smaller
    tensors remain in the object payload. NCCL groups place directly broadcast
    tensors on ``device`` or the current CUDA device; other backends use CPU
    tensors. The default ``sys.maxsize`` threshold calls the original PyTorch
    collective directly.

    Direct tensor extraction requires an acyclic tree of plain ``dict``, ``list``,
    and ``tuple`` containers; namedtuples are also supported. The source raises
    before entering the collective when it encounters a container subclass,
    a tensor subclass selected for direct broadcast, a shared container/tensor
    reference that would be rebuilt, or a cycle. Use the default threshold for
    arbitrary picklable object graphs.

    Args:
        object_list: List of objects to broadcast. Updated in place on every participating rank.
        src: Global source rank. Mutually exclusive with ``group_src``. Defaults to rank 0 when both are omitted.
        group: Process group whose ranks receive the value.
        device: Device used by the object collective and by direct NCCL tensor broadcasts.
        group_src: Source rank relative to ``group``. Mutually exclusive with ``src``.
        min_tensor_bytes: Minimum tensor payload size to broadcast directly.
    """
    if min_tensor_bytes < 0:
        raise ValueError(f"min_tensor_bytes must be non-negative, got {min_tensor_bytes}.")
    if min_tensor_bytes == sys.maxsize or group == dist.GroupMember.NON_GROUP_MEMBER:
        dist.broadcast_object_list(
            object_list,
            src=src,
            group=group,
            device=device,
            group_src=group_src,
        )
        return
    if src is None and group_src is None:
        src = 0
    elif src is not None and group_src is not None:
        raise ValueError("src and group_src cannot both be specified.")

    is_source = dist.get_rank() == src if src is not None else dist.get_rank(group) == group_src
    if is_source:
        skeleton, tensor_leaves = _extract_tensor_leaves(object_list, min_tensor_bytes)
    else:
        skeleton, tensor_leaves = None, []

    skeleton_box = [skeleton]
    dist.broadcast_object_list(
        skeleton_box,
        src=src,
        group=group,
        device=device,
        group_src=group_src,
    )
    skeleton = skeleton_box[0]

    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL:
        collective_device = device or torch.device("cuda", torch.cuda.current_device())
    else:
        collective_device = torch.device("cpu")
    tensor_index = 0

    def _rebuild(current_value: Any) -> Any:
        nonlocal tensor_index
        if isinstance(current_value, _TensorBroadcastMetadata):
            if is_source:
                source_tensor = tensor_leaves[tensor_index]  # [*shape]
                if (
                    source_tensor.device.type == "cuda"
                    and collective_device.type == "cuda"
                    and source_tensor.device != collective_device
                ):
                    log.warning(
                        "Direct tensor broadcast moves a tensor between CUDA devices: "
                        f"index={tensor_index}, source_device={source_tensor.device}, "
                        f"collective_device={collective_device}, src={src}",
                        rank0_only=False,
                    )
                tensor = source_tensor.to(device=collective_device)  # [*shape]
                tensor = tensor.contiguous()  # [*shape]
            else:
                tensor = torch.empty(  # [*shape]
                    current_value.shape,
                    dtype=current_value.dtype,
                    device=collective_device,
                )
            tensor_index += 1
            dist.broadcast(tensor, src=src, group=group, group_src=group_src)
            return tensor
        if isinstance(current_value, dict):
            return {key: _rebuild(item) for key, item in current_value.items()}
        if isinstance(current_value, list):
            return [_rebuild(item) for item in current_value]
        if isinstance(current_value, tuple):
            rebuilt_items = tuple(_rebuild(item) for item in current_value)
            # Namedtuples are tuple subclasses whose constructors expect each field as a positional argument.
            if hasattr(current_value, "_fields"):
                return type(current_value)(*rebuilt_items)
            return rebuilt_items
        return current_value

    rebuilt_object_list = _rebuild(skeleton)
    if is_source and tensor_index != len(tensor_leaves):
        raise RuntimeError(f"Broadcast rebuilt {tensor_index} tensors but source provided {len(tensor_leaves)}.")
    if not isinstance(rebuilt_object_list, list):
        raise RuntimeError(f"Broadcast object-list skeleton must be a list, got {type(rebuilt_object_list).__name__}.")
    object_list[:] = rebuilt_object_list


def destroy_process_group():
    if not dist.is_available() or not dist.is_initialized():
        return
    dist.destroy_process_group()
