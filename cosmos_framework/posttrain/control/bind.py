# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GPU-less AOT bind check: validate a Manifest's wiring and per-edge fields before any GPU/cluster.

`bind_check` is pure and side-effect-free (torch/ray-free): it returns a `CheckReport` of errors plus
non-failing runtime obligations, collecting every problem rather than raising on the first. Distinct
from `bind_device_groups`, which needs a live Ray cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

from cosmos_framework.posttrain.comm.transfer.movers import transport_capability_ok
from cosmos_framework.posttrain.control.capability import CapabilityDescriptor, get_capability
from cosmos_framework.posttrain.control.manifest import Manifest, Plane, role_by_name


@dataclass(frozen=True)
class CheckError:
    # 'wiring' | 'field' | 'transport' (placement/temperature arrive with recipes whose config axes
    # make them falsifiable)
    category: str
    where: str
    message: str


@dataclass(frozen=True)
class CheckObligation:
    """A runtime obligation `bind_check` cannot decide ahead of time — reported, never a failure."""

    category: str
    where: str
    message: str


@dataclass(frozen=True)
class CheckReport:
    ok: bool
    errors: tuple[CheckError, ...]  # tuple, not list: `frozen` must not be defeated by a mutable field
    obligations: tuple[CheckObligation, ...] = ()

    def format(self) -> str:
        lines: list[str] = []
        if self.errors:
            lines.append(f"bind_check: {len(self.errors)} error(s)")
            by_category: dict[str, list[CheckError]] = {}
            for err in self.errors:
                by_category.setdefault(err.category, []).append(err)
            for category in sorted(by_category):
                lines.append(f"  [{category}]")
                for err in by_category[category]:
                    lines.append(f"    - {err.where}: {err.message}")
        else:
            lines.append("bind_check: ok")
        if self.obligations:
            lines.append("runtime obligations (not failures):")
            for ob in self.obligations:
                lines.append(f"  - [{ob.category}] {ob.where}: {ob.message}")
        return "\n".join(lines)

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ValueError(self.format())


def bind_check(manifest: Manifest) -> CheckReport:
    """Validate a Manifest GPU-lessly, collecting all errors (never raising on the first)."""
    errors: list[CheckError] = []
    obligations: list[CheckObligation] = []

    # Resolve each role's descriptor once; an unknown impl_key is a wiring error.
    descriptors: dict[str, CapabilityDescriptor] = {}
    for role in manifest.roles:
        try:
            descriptors[role.name] = get_capability(role.impl_key)
        except KeyError as exc:
            errors.append(CheckError("wiring", role.name, str(exc)))

    # WIRING: every edge names existing roles, and its plane is supported by both endpoints.
    for edge in manifest.edges:
        where = f"{edge.src}->{edge.dst}"
        if role_by_name(manifest, edge.src) is None:
            errors.append(CheckError("wiring", where, f"edge src {edge.src!r} names no role"))
        if role_by_name(manifest, edge.dst) is None:
            errors.append(CheckError("wiring", where, f"edge dst {edge.dst!r} names no role"))
        for endpoint in (edge.src, edge.dst):
            desc = descriptors.get(endpoint)
            if desc is not None and edge.plane not in desc.planes:
                errors.append(
                    CheckError("wiring", where, f"role {endpoint!r} does not support plane {edge.plane.value!r}")
                )

    # FIELD: on each DATA edge, every required field the consumer needs must be produced by the
    # producer with a matching kind.
    for edge in manifest.edges:
        if edge.plane is not Plane.DATA:
            continue
        src_desc = descriptors.get(edge.src)
        dst_desc = descriptors.get(edge.dst)
        if src_desc is None or dst_desc is None:
            continue
        produced = {fs.name: fs.kind for fs in src_desc.produces}
        where = f"{edge.src}->{edge.dst}"
        for need in dst_desc.consumes:
            if not need.required:
                continue
            if need.name not in produced:
                message = f"consumer needs {need.name!r} ({need.kind}); producer emits {sorted(produced)}"
                errors.append(CheckError("field", where, f"{message}; {need.remedy}" if need.remedy else message))
            elif produced[need.name] != need.kind:
                errors.append(
                    CheckError(
                        "field",
                        where,
                        f"field {need.name!r} kind: producer {produced[need.name]!r} vs consumer {need.kind!r}",
                    )
                )

    # TRANSPORT: a tagged edge is validated by the runtime mover selector's own predicate; untagged
    # edges surface as obligations, not silence.
    t_errors, t_obligations = _check_transport(manifest)
    errors.extend(t_errors)
    obligations.extend(t_obligations)

    return CheckReport(ok=not errors, errors=tuple(errors), obligations=tuple(obligations))


def _check_transport(manifest: Manifest) -> tuple[list[CheckError], list[CheckObligation]]:
    """Per-edge AOT transport check, delegated to `transport_capability_ok` (the runtime mover
    selector's own predicate — one source of truth); a mismatch is a `transport` error. The
    `relationship` alone decides movability, so an edge carrying one is evaluated even without a
    `transport_capability`: a missing capability correctly fails the tensors_only movers instead of
    hiding as an obligation. Only a relationship-less WEIGHT edge is genuinely undecidable here; data
    edges are unconditionally movable, so they get no obligation either."""
    errors: list[CheckError] = []
    obligations: list[CheckObligation] = []
    for edge in manifest.edges:
        where = f"{edge.src}->{edge.dst}"
        if edge.relationship is None:
            if edge.plane is not Plane.DATA:
                obligations.append(
                    CheckObligation(
                        "transport", where, "edge untagged: transport is checked only at runtime mover construction"
                    )
                )
            continue
        ok, reason = transport_capability_ok(edge.relationship, edge.plane.value, edge.transport_capability)
        if not ok:
            errors.append(CheckError("transport", where, reason))
    return errors, obligations
