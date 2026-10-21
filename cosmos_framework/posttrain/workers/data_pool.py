# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Standalone Ray RDT/NIXL pools for large posttrain sample fields.

The prototype keeps the control plane small: samples contain inline metadata and
``PoolSlice`` descriptors.  Physical pages remain in producer-owned pools until
the driver releases the complete synchronous step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import ray
import ray.experimental as ray_experimental
import torch
from pynvml import nvmlDeviceGetHandleByUUID, nvmlDeviceGetMemoryInfo, nvmlInit
from ray._private.internal_api import free as free_ray_objects

from cosmos_framework.posttrain.utils.profile import ProfileRecord
from cosmos_framework.posttrain.workers.nixl import NixlTransport

PoolKind = Literal["cuda", "cpu"]

PAGE_BYTES = 32 * 1024 * 1024
# Every field starts on this byte boundary inside its page. Both consumer paths reinterpret the
# packed bytes with ``Tensor.view(dtype)``, which requires the storage offset to be a multiple of
# the element size; 256 covers every torch dtype and matches CUDA allocation alignment.
FIELD_ALIGN_BYTES = 256


def _require_nixl_api() -> None:
    """Fail clearly unless this Ray runtime exposes the required RDT/NIXL APIs."""
    required = (
        "deregister_nixl_memory",
        "register_nixl_memory",
        "set_target_for_ref",
        "wait_tensor_freed",
    )
    missing = [name for name in required if not hasattr(ray_experimental, name)]
    if missing:
        raise RuntimeError(
            "ray_nixl_pool requires the RDT-enabled Ray build with ray.experimental APIs "
            f"{missing}; current Ray is {getattr(ray, '__version__', 'unknown')}"
        )


def _deregister_nixl_memory(tensor: torch.Tensor) -> None:
    """Drop an explicit NIXL registration."""
    ray_experimental.deregister_nixl_memory(tensor)


def actor_cuda_accessible() -> bool:
    """Return whether this process may use CUDA for pool storage.

    ``torch.cuda.is_available()`` alone is the wrong signal inside Ray: a worker with no assigned
    GPU inherits the raylet's ``CUDA_VISIBLE_DEVICES`` and sees every device on the node. A
    CPU-placed actor must therefore also hold a Ray GPU assignment before it creates a CUDA context
    or allocates a CUDA tier, otherwise it silently lands on GPU 0 beside the actor that owns it.
    Outside Ray, CUDA availability is the only signal.
    """
    if not torch.cuda.is_available():
        return False
    if not ray.is_initialized():
        return True
    return bool(ray.get_gpu_ids())


def tensor_as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous 1-D uint8 view of ``tensor``'s elements.

    Flattening before the dtype reinterpretation is what makes 0-D tensors work: ``view(uint8)``
    rejects a 0-D input, while a 1-element 1-D input reinterprets to ``element_size`` bytes.
    """
    if tensor.layout != torch.strided:
        raise ValueError(f"pooled tensor must use strided layout, got {tensor.layout}")
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8)  # [tensor_nbytes]


@dataclass
class PoolSlice:
    """One large sample field inside a producer-owned pool extent."""

    owner_id: str
    lease_id: str
    pool_kind: PoolKind
    page_id: int
    generation: int
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    page_nbytes: int
    page_ref: Any = None
    oversize: bool = False

    def cache_key(self) -> tuple[str, str, PoolKind, int, int]:
        """Return the immutable identity of the page this slice belongs to."""
        return self.owner_id, self.lease_id, self.pool_kind, self.page_id, self.generation

    def tensor_dtype(self) -> torch.dtype:
        """Resolve the serialized Torch dtype used by this field."""
        name = self.dtype.removeprefix("torch.")
        dtype = getattr(torch, name, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported pooled tensor dtype {self.dtype!r}")
        return dtype


@dataclass(frozen=True)
class PoolField:
    """Synthetic field staged into a pool-backed prototype sample."""

    pool_kind: PoolKind
    nbytes: int
    value: int
    shape: tuple[int, ...] = ()
    dtype: str = "uint8"


@dataclass(frozen=True)
class PoolExtent:
    """One contiguous physical range reserved from a fixed-size pool."""

    page_id: int
    generation: int
    offset: int
    nbytes: int
    page_nbytes: int
    oversize: bool


@dataclass(frozen=True)
class PagePacking:
    """Placement decision for one field."""

    offset: int
    reserve_units: int
    oversize: bool


def plan_page_packing(nbytes: int, page_bytes: int, normal_used: int | None) -> PagePacking:
    """Plan one field without reserving physical pages.

    Normal fields are placed at the next ``FIELD_ALIGN_BYTES`` boundary after the bytes already
    used on the shared page; a field that fits only unaligned spills to a fresh page instead.
    """
    if nbytes < 1 or page_bytes < 1:
        raise ValueError("nbytes and page_bytes must be positive")
    if normal_used is not None and not 0 <= normal_used <= page_bytes:
        raise ValueError(f"normal_used must be in [0, {page_bytes}], got {normal_used}")
    if nbytes > page_bytes:
        return PagePacking(offset=0, reserve_units=math.ceil(nbytes / page_bytes), oversize=True)
    if normal_used is None:
        return PagePacking(offset=0, reserve_units=1, oversize=False)
    offset = -(-normal_used // FIELD_ALIGN_BYTES) * FIELD_ALIGN_BYTES
    if offset + nbytes > page_bytes:
        return PagePacking(offset=0, reserve_units=1, oversize=False)
    return PagePacking(offset=offset, reserve_units=0, oversize=False)


@dataclass
class _LeaseAllocation:
    """Mutable allocation state for one published step."""

    units: set[int] = field(default_factory=set)
    normal_page: int | None = None
    normal_used: int = 0
    extents: list[PoolExtent] = field(default_factory=list)


class PageAllocator:
    """Page allocator with packed normal fields and contiguous oversize extents.

    It is deliberately tensor- and Ray-free so the allocation/lifecycle rules can
    be tested on CPU.  An extent remains owned by its lease until ``release``.
    """

    def __init__(self, capacity_bytes: int, page_bytes: int = PAGE_BYTES):
        if capacity_bytes < page_bytes or capacity_bytes % page_bytes:
            raise ValueError("capacity_bytes must be a positive multiple of page_bytes")
        if page_bytes < 1 or page_bytes % FIELD_ALIGN_BYTES:
            raise ValueError(f"page_bytes must be a positive multiple of {FIELD_ALIGN_BYTES}, got {page_bytes}")
        self.capacity_bytes: int = capacity_bytes
        self.page_bytes: int = page_bytes
        self.page_count: int = capacity_bytes // page_bytes
        self._available: list[bool] = [True] * self.page_count
        self._generation: list[int] = [0] * self.page_count
        self._leases: dict[str, _LeaseAllocation] = {}

    def allocate(self, lease_id: str, nbytes: int) -> PoolExtent:
        """Reserve storage for one field under ``lease_id``.

        Normal fields share one page until full.  Oversize fields receive a
        contiguous, dedicated extent rounded to whole pages.
        """
        if not lease_id:
            raise ValueError("lease_id must be non-empty")
        if not 0 < nbytes <= self.capacity_bytes:
            raise ValueError(f"nbytes must be in [1, {self.capacity_bytes}], got {nbytes}")
        state = self._leases.setdefault(lease_id, _LeaseAllocation())
        normal_used = state.normal_used if state.normal_page is not None else None
        packing = plan_page_packing(nbytes, self.page_bytes, normal_used)
        if packing.oversize:
            return self._allocate_oversize(state, nbytes, packing.reserve_units)
        if packing.reserve_units:
            state.normal_page = self._reserve_run(state, packing.reserve_units)
        page_id = state.normal_page
        if page_id is None:
            raise RuntimeError("normal packing has no reserved page")
        offset = page_id * self.page_bytes + packing.offset
        state.normal_used = packing.offset + nbytes
        extent = PoolExtent(
            page_id=page_id,
            generation=self._generation[page_id],
            offset=offset,
            nbytes=nbytes,
            page_nbytes=self.page_bytes,
            oversize=False,
        )
        state.extents.append(extent)
        return extent

    def extents(self, lease_id: str) -> tuple[PoolExtent, ...]:
        """Return every extent owned by ``lease_id`` in staging order."""
        try:
            return tuple(self._leases[lease_id].extents)
        except KeyError as error:
            raise KeyError(f"unknown lease {lease_id!r}") from error

    def has_lease(self, lease_id: str) -> bool:
        """Return whether this pool owns storage for ``lease_id``."""
        return lease_id in self._leases

    def release(self, lease_id: str) -> None:
        """Return all storage owned by a completed synchronous step."""
        try:
            state = self._leases.pop(lease_id)
        except KeyError as error:
            raise KeyError(f"unknown or already released lease {lease_id!r}") from error
        for unit in state.units:
            self._available[unit] = True

    def _allocate_oversize(self, state: _LeaseAllocation, nbytes: int, units: int) -> PoolExtent:
        page_id = self._reserve_run(state, units)
        extent = PoolExtent(
            page_id=page_id,
            generation=self._generation[page_id],
            offset=page_id * self.page_bytes,
            nbytes=nbytes,
            page_nbytes=units * self.page_bytes,
            oversize=True,
        )
        state.extents.append(extent)
        return extent

    def _reserve_run(self, state: _LeaseAllocation, units: int) -> int:
        for start in range(self.page_count - units + 1):
            run = range(start, start + units)
            if all(self._available[index] for index in run):
                for index in run:
                    self._available[index] = False
                    state.units.add(index)
                self._generation[start] += 1
                return start
        raise MemoryError(
            f"pool capacity {self.capacity_bytes} has no contiguous {units * self.page_bytes}-byte extent"
        )


def iter_pool_slices(payload: Any) -> Iterable[PoolSlice]:
    """Yield pool descriptors in a trajectory-shaped payload without importing schema."""
    if isinstance(payload, PoolSlice):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from iter_pool_slices(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            yield from iter_pool_slices(value)
    elif hasattr(payload, "__dataclass_fields__"):
        for name in payload.__dataclass_fields__:
            yield from iter_pool_slices(getattr(payload, name))


def has_pool_slices(payload: Any) -> bool:
    """Return whether a payload owns any pooled field references."""
    return next(iter_pool_slices(payload), None) is not None


def clear_pool_slice_refs(payload: Any) -> None:
    """Clear every nested source-page reference."""
    for slice_ in iter_pool_slices(payload):
        slice_.page_ref = None


def release_pool_step_barrier(payload: Any, owner_actors: dict[str, Any]) -> int:
    """Clear driver refs and release every live producer lease at a step barrier."""
    slices = list(iter_pool_slices(payload))
    unknown = sorted({slice_.owner_id for slice_ in slices if slice_.owner_id not in owner_actors})
    if unknown:
        raise KeyError(f"no producer actor registered for owner ids {unknown}")
    clear_pool_slice_refs(payload)
    released = ray.get([owner_actors[owner_id].release_all_pool_leases.remote() for owner_id in sorted(owner_actors)])
    return sum(len(lease_ids) for lease_ids in released)


class ProducerDataPool:
    """Actor-local source pools and batch-level publication lifecycle.

    Construct and use this class inside a Ray actor.
    """

    def __init__(
        self,
        owner_id: str,
        *,
        cuda_capacity_bytes: int = 1024 * 1024 * 1024,
        cpu_capacity_bytes: int = 512 * 1024 * 1024,
        page_bytes: int = PAGE_BYTES,
    ) -> None:
        if not owner_id:
            raise ValueError("owner_id must be non-empty")
        self.owner_id: str = owner_id
        self.page_bytes: int = page_bytes
        self._allocators: dict[PoolKind, PageAllocator] = {
            "cuda": PageAllocator(cuda_capacity_bytes, page_bytes),
            "cpu": PageAllocator(cpu_capacity_bytes, page_bytes),
        }
        self._backing: dict[PoolKind, Any] = {}
        self._cuda_keepalive: Any = None
        self._views: dict[tuple[PoolKind, int, int], Any] = {}
        self._lease_slices: dict[str, list[PoolSlice]] = {}
        self._lease_pages: dict[str, dict[tuple[PoolKind, int, int, int], Any]] = {}
        self._published: dict[str, list[Any]] = {}
        self._warmed: bool = False
        self._cuda_accessible: bool = False

    def warmup(self) -> None:
        """Allocate CPU storage, plus CUDA storage when Ray assigned this actor a GPU."""
        if self._warmed:
            return
        _require_nixl_api()
        self._cuda_accessible = actor_cuda_accessible()
        if self._cuda_accessible:
            transport = NixlTransport.get()
            transport.warmup()
            cuda_capacity = self._allocators["cuda"].capacity_bytes
            self._backing["cuda"], self._cuda_keepalive = transport.new_buffer(cuda_capacity)  # [cuda_pool_bytes]
        cpu_capacity = self._allocators["cpu"].capacity_bytes
        self._backing["cpu"] = torch.empty(
            cpu_capacity, dtype=torch.uint8, pin_memory=self._cuda_accessible
        )  # [cpu_pool_bytes]
        ray_experimental.register_nixl_memory(self._backing["cpu"])
        self._warmed = True

    def stage_sample(
        self,
        lease_id: str,
        metadata: dict[str, Any],
        fields: dict[str, PoolField],
    ) -> dict[str, Any]:
        """Stage one metadata-plus-slice sample under the current step lease."""
        if not self._warmed:
            raise RuntimeError("warmup() must run before stage_sample()")
        if lease_id in self._published:
            raise RuntimeError(f"lease {lease_id!r} is already published")
        sample = dict(metadata)
        slices = self._lease_slices.setdefault(lease_id, [])
        pages = self._lease_pages.setdefault(lease_id, {})
        for name, pool_field in fields.items():
            if pool_field.pool_kind == "cuda" and not self._cuda_accessible:
                raise RuntimeError("cannot stage a CUDA pool field in an actor without CUDA access")
            if not 0 <= pool_field.value <= 255:
                raise ValueError(f"field {name!r} value must be in [0, 255], got {pool_field.value}")
            extent = self._allocators[pool_field.pool_kind].allocate(lease_id, pool_field.nbytes)
            page = self._page_view(pool_field.pool_kind, extent)
            local_offset = extent.offset - extent.page_id * self.page_bytes
            page[local_offset : local_offset + pool_field.nbytes].fill_(pool_field.value)  # [field_bytes]
            slice_ = PoolSlice(
                owner_id=self.owner_id,
                lease_id=lease_id,
                pool_kind=pool_field.pool_kind,
                page_id=extent.page_id,
                generation=extent.generation,
                offset=local_offset,
                nbytes=pool_field.nbytes,
                shape=pool_field.shape or (pool_field.nbytes,),
                dtype=pool_field.dtype,
                page_nbytes=extent.page_nbytes,
                oversize=extent.oversize,
            )
            sample[name] = slice_
            slices.append(slice_)
            pages[(pool_field.pool_kind, extent.page_id, extent.generation, extent.page_nbytes)] = page
        return sample

    def stage_tensor(self, lease_id: str, tensor: torch.Tensor) -> PoolSlice:
        """Copy one tensor into its persistent source pool and return its descriptor."""
        if not self._warmed:
            raise RuntimeError("warmup() must run before stage_tensor()")
        if lease_id in self._published:
            raise RuntimeError(f"lease {lease_id!r} is already published")
        pool_kind: PoolKind = "cuda" if tensor.is_cuda else "cpu"
        if pool_kind == "cuda" and not self._cuda_accessible:
            raise RuntimeError("cannot stage a CUDA tensor in an actor without CUDA access")
        source = tensor_as_bytes(tensor)  # [tensor_nbytes]
        extent = self._allocators[pool_kind].allocate(lease_id, source.numel())
        page = self._page_view(pool_kind, extent)
        local_offset = extent.offset - extent.page_id * self.page_bytes
        page[local_offset : local_offset + source.numel()].copy_(source)  # [tensor_nbytes]
        slice_ = PoolSlice(
            owner_id=self.owner_id,
            lease_id=lease_id,
            pool_kind=pool_kind,
            page_id=extent.page_id,
            generation=extent.generation,
            offset=local_offset,
            nbytes=source.numel(),
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            page_nbytes=extent.page_nbytes,
            oversize=extent.oversize,
        )
        self._lease_slices.setdefault(lease_id, []).append(slice_)
        self._lease_pages.setdefault(lease_id, {})[
            (pool_kind, extent.page_id, extent.generation, extent.page_nbytes)
        ] = page
        return slice_

    def view_tensor(self, slice_: PoolSlice) -> torch.Tensor:
        """Return a read-only consumer view of this actor's published source page."""
        if slice_.owner_id != self.owner_id:
            raise ValueError(f"PoolSlice owner {slice_.owner_id!r} is not {self.owner_id!r}")
        if slice_.lease_id not in self._published:
            raise KeyError(f"PoolSlice lease {slice_.lease_id!r} is not published")
        key = (slice_.pool_kind, slice_.page_id, slice_.generation, slice_.page_nbytes)
        try:
            page = self._lease_pages[slice_.lease_id][key]
        except KeyError as error:
            raise KeyError(f"PoolSlice page {key!r} is not owned by lease {slice_.lease_id!r}") from error
        bytes_ = page[slice_.offset : slice_.offset + slice_.nbytes]
        return bytes_.view(slice_.tensor_dtype()).reshape(slice_.shape)

    def publish_step(self, lease_id: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Publish every used page as its own RDT ObjectRef and attach the refs to the sample slices.

        One ``ray.put`` per page keeps fetches page-granular. Each CUDA put performs its own device
        synchronize inside Ray; on the idle producer device that costs tens of microseconds per page
        against ~1.5 ms for the put itself, so no batched publication API is needed.
        """
        if lease_id in self._published:
            raise RuntimeError(f"lease {lease_id!r} is already published")
        try:
            slices = self._lease_slices[lease_id]
            pages = self._lease_pages[lease_id]
        except KeyError as error:
            raise KeyError(f"lease {lease_id!r} has no staged fields") from error
        keys = list(pages)
        profile = ProfileRecord("data_pool_publish")
        with profile.phase("publish"):
            published = [ray.put([pages[key]], _tensor_transport="nixl") for key in keys]
        refs = dict(zip(keys, published, strict=True))
        for slice_ in slices:
            key = (slice_.pool_kind, slice_.page_id, slice_.generation, slice_.page_nbytes)
            slice_.page_ref = refs[key]
        self._published[lease_id] = list(refs.values())
        if profile.enabled:
            cuda_pages = sum(key[0] == "cuda" for key in keys)
            profile.emit(
                owner=self.owner_id,
                pages=len(keys),
                cuda_pages=cuda_pages,
                cpu_pages=len(keys) - cuda_pages,
            )
        return samples

    def release_step(self, lease_id: str) -> None:
        """Release one completed batch after the driver step barrier."""
        try:
            refs = self._published.pop(lease_id)
            pages = self._lease_pages.pop(lease_id)
            slices = self._lease_slices.pop(lease_id)
        except KeyError as error:
            raise KeyError(f"unknown or already released lease {lease_id!r}") from error
        for slice_ in slices:
            slice_.page_ref = None
        free_ray_objects(list(refs))
        refs.clear()
        for page in pages.values():
            ray_experimental.wait_tensor_freed(page, timeout=60)
        for allocator in self._allocators.values():
            if allocator.has_lease(lease_id):
                allocator.release(lease_id)

    def release_all_leases(self) -> tuple[str, ...]:
        """Release every published lease at a synchronous step barrier."""
        lease_ids = tuple(self._published)
        for lease_id in lease_ids:
            self.release_step(lease_id)
        return lease_ids

    def close(self) -> None:
        """Deregister persistent backing after all step leases have been released."""
        if self._published:
            raise RuntimeError(f"cannot close pool with live leases {sorted(self._published)}")
        if not self._warmed:
            return
        _deregister_nixl_memory(self._backing["cpu"])
        cuda_backing = self._backing.get("cuda")
        if cuda_backing is not None:
            NixlTransport.get().release_buffer(cuda_backing, self._cuda_keepalive)
        self._backing.clear()
        self._views.clear()
        self._cuda_keepalive = None
        self._cuda_accessible = False
        self._warmed = False

    def _page_view(self, pool_kind: PoolKind, extent: PoolExtent) -> Any:
        key = (pool_kind, extent.page_id, extent.page_nbytes)
        view = self._views.get(key)
        if view is None:
            start = extent.page_id * self.page_bytes
            view = self._backing[pool_kind][start : start + extent.page_nbytes]  # [page_bytes]
            self._views[key] = view
        return view


class ConsumerDataPool:
    """Actor-local persistent target pages with a bounded page-level LRU."""

    def __init__(
        self,
        *,
        cuda_cache_bytes: int = 512 * 1024 * 1024,
        cpu_cache_bytes: int = 512 * 1024 * 1024,
        page_bytes: int = PAGE_BYTES,
    ) -> None:
        if cuda_cache_bytes % page_bytes or cpu_cache_bytes % page_bytes:
            raise ValueError("cache capacities must be multiples of page_bytes")
        self.page_bytes: int = page_bytes
        self._slot_count: dict[PoolKind, int] = {
            "cuda": cuda_cache_bytes // page_bytes,
            "cpu": cpu_cache_bytes // page_bytes,
        }
        self._slots: dict[PoolKind, list[Any]] = {"cuda": [], "cpu": []}
        self._cache: dict[PoolKind, dict[tuple[str, str, PoolKind, int, int], int]] = {"cuda": {}, "cpu": {}}
        self._slot_keys: dict[PoolKind, list[tuple[str, str, PoolKind, int, int] | None]] = {"cuda": [], "cpu": []}
        self._last_used: dict[PoolKind, list[int]] = {"cuda": [], "cpu": []}
        self._clock: int = 0
        self._warmed: bool = False
        self._cuda_accessible: bool = False

    def warmup(self) -> None:
        """Allocate CPU targets, plus CUDA targets when Ray assigned this actor a GPU."""
        if self._warmed:
            return
        _require_nixl_api()
        self._cuda_accessible = actor_cuda_accessible()
        pool_kinds: tuple[PoolKind, ...] = ("cuda", "cpu") if self._cuda_accessible else ("cpu",)
        for pool_kind in pool_kinds:
            slots = [
                torch.empty(
                    self.page_bytes,
                    dtype=torch.uint8,
                    device="cuda" if pool_kind == "cuda" else "cpu",
                    pin_memory=pool_kind == "cpu" and self._cuda_accessible,
                )  # [page_bytes]
                for _ in range(self._slot_count[pool_kind])
            ]
            for slot in slots:
                ray_experimental.register_nixl_memory(slot)
            self._slots[pool_kind] = slots
            self._slot_keys[pool_kind] = [None] * len(slots)
            self._last_used[pool_kind] = [0] * len(slots)
        self._warmed = True

    def fetch(self, slice_: PoolSlice) -> tuple[Any, bool]:
        """Fetch one slice's page on an LRU miss and return an owned copy of the field bytes.

        Receive slots are registered landing zones only. A slot may be evicted and rewritten
        by the next miss while the caller still holds the result, so the result never aliases
        slot storage. The copy also starts at storage offset zero, which keeps dtype
        reinterpretation valid regardless of the field's byte offset inside the page.
        """
        if slice_.page_ref is None:
            raise ValueError("PoolSlice has no page_ref; it was released or never published")
        if not self._warmed:
            raise RuntimeError("warmup() must run before fetch()")
        if slice_.pool_kind == "cuda" and not self._cuda_accessible:
            raise RuntimeError("cannot materialize a CUDA pool slice in an actor without CUDA access")
        key = slice_.cache_key()
        if slice_.oversize:
            return self._fetch_oversize(slice_), False
        slot, hit = self._slot_for(slice_.pool_kind, key)
        page = self._slots[slice_.pool_kind][slot]
        if not hit:
            self._fetch_into(slice_, page)
        self._clock += 1
        self._last_used[slice_.pool_kind][slot] = self._clock
        return page[slice_.offset : slice_.offset + slice_.nbytes].clone(), hit  # [field_bytes]

    def fetch_tensor(self, slice_: PoolSlice) -> tuple[torch.Tensor, bool]:
        """Fetch a descriptor and rebuild the original shape and dtype on the owned copy."""
        bytes_, hit = self.fetch(slice_)
        tensor = bytes_.view(slice_.tensor_dtype()).reshape(slice_.shape)  # [slice_shape]
        return tensor, hit

    def close(self) -> None:
        """Deregister all persistent receive targets."""
        for slots in self._slots.values():
            for slot in slots:
                _deregister_nixl_memory(slot)
        self._slots = {"cuda": [], "cpu": []}
        self._cache = {"cuda": {}, "cpu": {}}
        self._slot_keys = {"cuda": [], "cpu": []}
        self._last_used = {"cuda": [], "cpu": []}
        self._cuda_accessible = False
        self._warmed = False

    def _slot_for(self, pool_kind: PoolKind, key: tuple[str, str, PoolKind, int, int]) -> tuple[int, bool]:
        cached = self._cache[pool_kind].get(key)
        if cached is not None:
            return cached, True
        keys = self._slot_keys[pool_kind]
        try:
            slot = keys.index(None)
        except ValueError:
            slot = min(range(len(keys)), key=self._last_used[pool_kind].__getitem__)
            old_key = keys[slot]
            del self._cache[pool_kind][old_key]
        keys[slot] = key
        self._cache[pool_kind][key] = slot
        return slot, False

    def _fetch_oversize(self, slice_: PoolSlice) -> Any:
        """Receive one oversize extent into a temporary registered target and return the field copy."""
        page = torch.empty(
            slice_.page_nbytes,
            dtype=torch.uint8,
            device="cuda" if slice_.pool_kind == "cuda" else "cpu",
            pin_memory=slice_.pool_kind == "cpu" and self._cuda_accessible,
        )  # [oversize_page_bytes]
        ray_experimental.register_nixl_memory(page)
        try:
            self._fetch_into(slice_, page)
            return page[slice_.offset : slice_.offset + slice_.nbytes].clone()  # [field_bytes]
        finally:
            _deregister_nixl_memory(page)

    @staticmethod
    def _fetch_into(slice_: PoolSlice, target: Any) -> None:
        ray_experimental.set_target_for_ref(slice_.page_ref, [target])
        ray.get(slice_.page_ref)


def cuda_memory_used_bytes() -> int:
    """Return NVML device memory for this actor's Ray-visible CUDA device."""
    torch.cuda.init()
    uuid = str(torch.cuda.get_device_properties(0).uuid).encode()
    nvmlInit()
    return int(nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByUUID(uuid)).used)
