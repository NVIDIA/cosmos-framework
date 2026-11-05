# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit operator-declared fabric domains for node-pinned GPU placement.

Ray node identities prove placement, not fabric connectivity. The operator must obtain domain
labels from the allocation's topology inventory; hostnames and rank numbers never imply NVLink.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GpuTopologyNode:
    """One complete physical node in domain-major policy rank order."""

    domain: str
    node_id: str
    address: str
    gpus: int
    node_resource: str
    resource_per_gpu: float


def validate_gpu_node_domains(value: Mapping[str, str]) -> None:
    """Reject malformed declarations before querying Ray or reserving resources."""
    if not isinstance(value, Mapping):
        raise TypeError("gpu_node_domains must map Ray node addresses or hostnames to fabric domain labels")
    for node, domain in value.items():
        if any(not isinstance(item, str) or not item or item.strip() != item for item in (node, domain)):
            raise ValueError("gpu_node_domains keys and domain labels must be nonempty strings without edge whitespace")


class _PendingGpuNodes(ValueError):
    """Declared workers may still be joining the allocation's Ray cluster."""


def plan_gpu_topology(
    nodes: list[dict[str, Any]],
    node_domains: Mapping[str, str],
    *,
    shard_degree: int,
    expected_gpus: int,
    cpus_per_gpu: int,
    rollout_degree: int = 1,
) -> tuple[GpuTopologyNode, ...]:
    """Resolve a complete allocation into contiguous, domain-local FSDP shard groups.

    Node aliases must exactly match Ray's NodeManagerAddress, NodeManagerHostname or NodeName.
    All GPU nodes must be declared once and expose equal integer GPU capacities. Each domain's
    capacity must be divisible by the shard degree, so row-major HSDP groups never cross domains.
    """
    validate_gpu_node_domains(node_domains)
    if not node_domains:
        raise ValueError("an explicit GPU topology requires a nonempty gpu_node_domains mapping")
    for name, value in (
        ("shard_degree", shard_degree),
        ("expected_gpus", expected_gpus),
        ("cpus_per_gpu", cpus_per_gpu),
        ("rollout_degree", rollout_degree),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    live = [node for node in nodes if node.get("Alive") and node.get("Resources", {}).get("GPU", 0)]
    aliases: dict[str, list[dict[str, Any]]] = {}
    for node in live:
        for alias in {node.get(key) for key in ("NodeManagerAddress", "NodeManagerHostname", "NodeName")}:
            if isinstance(alias, str) and alias:
                aliases.setdefault(alias, []).append(node)
    resolved: dict[str, tuple[dict[str, Any], str]] = {}
    missing: list[str] = []
    for alias, domain in node_domains.items():
        matches = aliases.get(alias, [])
        if not matches:
            missing.append(alias)
            continue
        if len(matches) != 1:
            raise ValueError(f"gpu_node_domains alias {alias!r} matches multiple live Ray GPU nodes")
        node = matches[0]
        node_id = str(node["NodeID"])
        if node_id in resolved:
            raise ValueError(f"gpu_node_domains declares Ray node {node_id!r} more than once through different aliases")
        resolved[node_id] = (node, domain)
    undeclared = [str(node["NodeManagerAddress"]) for node in live if str(node["NodeID"]) not in resolved]
    if undeclared:
        raise ValueError(f"gpu_node_domains omits live GPU nodes {sorted(undeclared)}; declare the complete allocation")
    if missing:
        raise _PendingGpuNodes(f"declared GPU nodes are not registered in Ray: {sorted(missing)}")
    planned: list[GpuTopologyNode] = []
    domain_gpus: dict[str, int] = {}
    for node, domain in resolved.values():
        resources = node["Resources"]
        capacity = float(resources["GPU"])
        if not math.isfinite(capacity) or capacity < 1 or not capacity.is_integer():
            raise ValueError(f"Ray node {node['NodeID']!r} has invalid physical GPU capacity {capacity!r}")
        gpus = int(capacity)
        if gpus % rollout_degree:
            raise ValueError(
                f"Ray node {node['NodeID']!r} has {gpus} GPUs, not divisible by rollout TP degree {rollout_degree}; "
                "explicit topology placement requires node-local rollout engines"
            )
        if float(resources.get("CPU", 0)) < cpus_per_gpu * gpus:
            raise ValueError(f"Ray node {node['NodeID']!r} needs {cpus_per_gpu * gpus} CPUs for its {gpus} GPU slots")
        address = str(node["NodeManagerAddress"])
        node_resource = f"node:{address}"
        resource_capacity = float(resources.get(node_resource, 0))
        if not math.isfinite(resource_capacity) or resource_capacity <= 0:
            raise ValueError(f"Ray node {node['NodeID']!r} lacks its node-affinity resource {node_resource!r}")
        # Ray rounds custom resources to 1e-4. A small request pins the entire STRICT_PACK group
        # without exhausting the node resource through per-bundle rounding.
        if resource_capacity < gpus * 0.0001:
            raise ValueError(f"Ray node-affinity resource {node_resource!r} cannot pin {gpus} GPU bundles")
        planned.append(GpuTopologyNode(domain, str(node["NodeID"]), address, gpus, node_resource, 0.0001))
        domain_gpus[domain] = domain_gpus.get(domain, 0) + gpus
    if len({node.gpus for node in planned}) != 1:
        raise ValueError("gpu_node_domains currently requires equal physical GPU counts per node")
    total_gpus = sum(node.gpus for node in planned)
    if total_gpus != expected_gpus:
        raise ValueError(
            f"declared topology has {total_gpus} GPUs but the shared GPU DeviceGroup needs {expected_gpus}"
        )
    for domain, capacity in domain_gpus.items():
        if capacity % shard_degree:
            raise ValueError(
                f"fabric domain {domain!r} has {capacity} GPUs, not divisible by policy shard degree {shard_degree}"
            )
    return tuple(sorted(planned, key=lambda node: (node.domain, node.address, node.node_id)))


def wait_for_gpu_topology(
    read_nodes: Callable[[], list[dict[str, Any]]],
    node_domains: Mapping[str, str],
    *,
    shard_degree: int,
    expected_gpus: int,
    cpus_per_gpu: int,
    rollout_degree: int = 1,
    timeout_s: float = 300.0,
) -> tuple[GpuTopologyNode, ...]:
    """Wait only for declared workers to register, before allocating any placement groups."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(f"topology registration timeout must be positive and finite, got {timeout_s!r}")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return plan_gpu_topology(
                read_nodes(),
                node_domains,
                shard_degree=shard_degree,
                expected_gpus=expected_gpus,
                cpus_per_gpu=cpus_per_gpu,
                rollout_degree=rollout_degree,
            )
        except _PendingGpuNodes as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"GPU topology was not ready within {timeout_s:g}s: {exc}") from exc
            time.sleep(min(1.0, remaining))
