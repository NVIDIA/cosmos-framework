# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Fixed-schema metrics emitted by run-bound reward providers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from cosmos_framework.posttrain.schema.metrics import MetricContribution, MetricReport, merge_metric_reports

_SAMPLE_COUNT_DIAGNOSTIC = "sample_count"
_SAFE_LOCAL_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class ProviderDiagnosticRecord:
    """One provider evaluation's typed diagnostics and immutable provenance."""

    provider_name: str
    provider_revision: str
    diagnostics: Mapping[str, MetricContribution]

    def __post_init__(self) -> None:
        for field_name in ("provider_name", "provider_revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"reward diagnostic {field_name} must be a non-empty string, got {value!r}")
        if _SAFE_LOCAL_NAME.fullmatch(self.provider_name) is None:
            raise ValueError(
                f"reward diagnostic provider_name must match '[a-z][a-z0-9_]*', got {self.provider_name!r}"
            )
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError(f"reward provider diagnostics must be a mapping, got {type(self.diagnostics).__name__}")
        normalized: dict[str, MetricContribution] = {}
        for name, contribution in self.diagnostics.items():
            if not isinstance(name, str) or _SAFE_LOCAL_NAME.fullmatch(name) is None:
                raise ValueError(f"reward diagnostic names must match '[a-z][a-z0-9_]*', got {name!r}")
            if name == _SAMPLE_COUNT_DIAGNOSTIC:
                raise ValueError(f"reward diagnostic name {_SAMPLE_COUNT_DIAGNOSTIC!r} is reserved for the framework")
            if not isinstance(contribution, MetricContribution):
                raise TypeError(f"reward diagnostic {name!r} must be a MetricContribution, got {contribution!r}")
            normalized[name] = contribution
        object.__setattr__(self, "diagnostics", MappingProxyType(normalized))


@dataclass(frozen=True)
class ProviderMetricReport:
    """One serializable provider-only report without a locally complete reward-group assumption."""

    metrics: Mapping[str, MetricContribution]
    provenance: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        # This report crosses the reward-actor RPC boundary. MappingProxyType is immutable but is
        # not pickleable by the standard Ray serialization path, so retain a normalized plain dict.
        object.__setattr__(self, "metrics", merge_metric_reports([self.metrics]))
        normalized: dict[str, str] = {}
        for item in self.provenance:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or any(not isinstance(value, str) or not value for value in item)
            ):
                raise ValueError(
                    f"reward provider provenance must contain non-empty (provider, revision) strings, got {item!r}"
                )
            provider_name, provider_revision = item
            previous = normalized.get(provider_name)
            if previous is not None and previous != provider_revision:
                raise ValueError(
                    f"reward provider {provider_name!r} changed revision from {previous!r} to {provider_revision!r}"
                )
            normalized[provider_name] = provider_revision
        object.__setattr__(self, "provenance", tuple(sorted(normalized.items())))


def _diagnostic_metric_report(record: ProviderDiagnosticRecord, namespace: str) -> MetricReport:
    """Prefix one provider-local contribution mapping into the public metric schema."""
    prefix = f"{namespace}/reward_provider/{record.provider_name}"
    report = {f"{prefix}/{_SAMPLE_COUNT_DIAGNOSTIC}": MetricContribution.sum(1)}
    report.update({f"{prefix}/{name}": contribution for name, contribution in record.diagnostics.items()})
    return report


def provider_diagnostics_report(
    records: Sequence[ProviderDiagnosticRecord],
    *,
    namespace: str,
) -> tuple[MetricReport, tuple[tuple[str, str], ...]]:
    """Aggregate provider-local diagnostics and retain provider/revision provenance."""
    if namespace not in ("train", "val"):
        raise ValueError(f"reward diagnostic namespace must be 'train' or 'val', got {namespace!r}")
    if not records:
        return {}, ()
    revisions: dict[str, str] = {}
    for record in records:
        previous = revisions.get(record.provider_name)
        if previous is not None and previous != record.provider_revision:
            raise ValueError(
                f"reward provider {record.provider_name!r} changed revision "
                f"from {previous!r} to {record.provider_revision!r}"
            )
        revisions[record.provider_name] = record.provider_revision
    metrics = merge_metric_reports([_diagnostic_metric_report(record, namespace) for record in records])
    provenance = tuple(sorted(revisions.items()))
    return metrics, provenance
