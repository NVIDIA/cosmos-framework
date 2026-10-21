# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Applied-update state program for the diffusion trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from cosmos_framework.posttrain.model.protocol import EMAFlowMatchingPolicy, FlowMatchingPolicy

PolicyStateName = Literal["net", "net_old", "net_ref", "net_ema"]
PolicySemanticRole = Literal["policy", "old", "reference", "ema"]
PolicyCheckpointComponent = Literal["model", "old", "ref"]
PolicyStatePersistence = Literal["model", "moving_old", "moving_reference", "copy_only", "offload_only"]


@dataclass(frozen=True)
class AppliedUpdateBoundary:
    """The two clocks naming one successfully applied optimizer update."""

    pre_commit_index: int
    committed_step: int

    def __post_init__(self) -> None:
        if type(self.pre_commit_index) is not int or self.pre_commit_index < 0:
            raise ValueError(f"pre_commit_index must be a non-negative integer, got {self.pre_commit_index!r}")
        if type(self.committed_step) is not int or self.committed_step != self.pre_commit_index + 1:
            raise ValueError(
                "committed_step must be exactly one past pre_commit_index, got "
                f"pre_commit_index={self.pre_commit_index!r}, committed_step={self.committed_step!r}"
            )


@dataclass(frozen=True)
class AppliedUpdateEvents:
    """State transitions performed after the optimizer clock commits."""

    old_policy_refreshed: bool
    reference_refreshed: bool


@dataclass(frozen=True)
class PolicyStateDescriptor:
    """One configured policy-state role and its checkpoint semantics."""

    name: PolicyStateName
    semantic_role: PolicySemanticRole
    module: torch.nn.Module
    checkpoint_component: PolicyCheckpointComponent
    persistence: PolicyStatePersistence


class DiffusionPolicyStateProgram:
    """Trainer-owned update clocks and network projections for diffusion policy state.

    Model adapters own the copy/update operations. This program owns when those
    operations occur and which roles participate in each checkpoint operation.
    """

    def __init__(
        self,
        model: FlowMatchingPolicy,
        *,
        update_ema_after_iter: bool,
        update_old_every_iter: int,
        update_ref_every_iter: int,
    ) -> None:
        self.model = model
        self.update_ema_after_iter = update_ema_after_iter
        self.update_old_every_iter = update_old_every_iter
        self.update_ref_every_iter = update_ref_every_iter
        self.ema_policy = model if isinstance(model, EMAFlowMatchingPolicy) else None
        self.ema_net = self.ema_policy.net_ema if self.ema_policy is not None else None
        descriptors = [
            PolicyStateDescriptor(
                name="net",
                semantic_role="policy",
                module=model.net,
                checkpoint_component="model",
                persistence="model",
            )
        ]
        old = getattr(model, "net_old", None)
        if old is not None:
            descriptors.append(
                PolicyStateDescriptor(
                    name="net_old",
                    semantic_role="old",
                    module=old,
                    checkpoint_component="old",
                    persistence="moving_old" if update_old_every_iter > 0 else "copy_only",
                )
            )
        reference = getattr(model, "net_ref", None)
        if reference is not None:
            descriptors.append(
                PolicyStateDescriptor(
                    name="net_ref",
                    semantic_role="reference",
                    module=reference,
                    checkpoint_component="ref",
                    persistence="moving_reference" if update_ref_every_iter > 0 else "copy_only",
                )
            )
        exposed_ema = getattr(model, "net_ema", None)
        if isinstance(exposed_ema, torch.nn.Module):
            descriptors.append(
                PolicyStateDescriptor(
                    name="net_ema",
                    semantic_role="ema",
                    module=exposed_ema,
                    checkpoint_component="model",
                    persistence="model" if update_ema_after_iter else "offload_only",
                )
            )
        self._descriptors: tuple[PolicyStateDescriptor, ...] = tuple(descriptors)

    def validate_ema(self) -> None:
        """Validate the optional maintained-EMA capability."""

        if self.update_ema_after_iter and self.ema_policy is None:
            raise ValueError(
                "update_ema_after_iter=True requires the policy adapter to implement "
                "EMAFlowMatchingPolicy (net_ema and update_ema())"
            )
        if self.update_ema_after_iter and self.ema_net is None:
            raise ValueError("update_ema_after_iter=True requires EMAFlowMatchingPolicy.net_ema to be present")

    def validate_moving_roles(self) -> None:
        """Validate cadence and adapter support for moving old/reference roles."""

        if self.update_old_every_iter < 0:
            raise ValueError(f"update_old_every_iter must be non-negative, got {self.update_old_every_iter}")
        if self.update_old_every_iter > 0 and getattr(self.model, "net_old", None) is None:
            raise ValueError("update_old_every_iter>0 requires the policy adapter to expose net_old")
        if self.update_old_every_iter > 0 and not callable(getattr(self.model, "refresh_old", None)):
            raise ValueError("update_old_every_iter>0 requires the policy adapter to implement refresh_old()")
        if self.update_ref_every_iter < 0:
            raise ValueError(f"update_ref_every_iter must be non-negative, got {self.update_ref_every_iter}")
        if self.update_ref_every_iter > 0 and not callable(getattr(self.model, "refresh_ref", None)):
            raise ValueError("update_ref_every_iter>0 requires the policy adapter to implement refresh_ref()")

    def initialize(self) -> None:
        """Establish moving behavior/reference snapshots before optimizer step zero."""

        if self.update_old_every_iter > 0:
            self.refresh_old()
        if self.update_ref_every_iter > 0:
            self.refresh_ref()

    def before_commit(self, boundary: AppliedUpdateBoundary) -> None:
        """Apply pre-commit state transitions for one successful optimizer update."""

        if not self.update_ema_after_iter:
            return
        if self.ema_policy is None:
            raise RuntimeError("EMA updates are enabled without a validated EMA capability")
        self.ema_policy.update_ema(boundary.pre_commit_index)

    def after_commit(self, boundary: AppliedUpdateBoundary) -> AppliedUpdateEvents:
        """Refresh cadence-driven snapshots using the committed optimizer-step clock."""

        old_policy_refreshed = (
            self.update_old_every_iter > 0 and boundary.committed_step % self.update_old_every_iter == 0
        )
        if old_policy_refreshed:
            self.refresh_old()
        reference_refreshed = (
            self.update_ref_every_iter > 0 and boundary.committed_step % self.update_ref_every_iter == 0
        )
        if reference_refreshed:
            self.refresh_ref()
        return AppliedUpdateEvents(
            old_policy_refreshed=old_policy_refreshed,
            reference_refreshed=reference_refreshed,
        )

    def descriptors(self) -> tuple[PolicyStateDescriptor, ...]:
        """Return the immutable, canonical policy-state inventory."""

        if self.update_ema_after_iter and self.ema_net is None:
            raise RuntimeError("EMA updates are enabled without a validated EMA network")
        return self._descriptors

    def networks(self) -> dict[PolicyStateName, torch.nn.Module]:
        """Project the typed inventory to the mapping expected by state-dict APIs."""

        return {descriptor.name: descriptor.module for descriptor in self.descriptors()}

    def checkpoint_descriptors(
        self,
        *,
        model_selected: bool,
        training_state_selected: bool,
        materialize_old: bool,
    ) -> tuple[PolicyStateDescriptor, ...]:
        """Project logical checkpoint selections onto typed policy-state roles."""

        selected: list[PolicyStateDescriptor] = []
        for descriptor in self.descriptors():
            if model_selected and descriptor.persistence == "model":
                selected.append(descriptor)
            elif training_state_selected and descriptor.persistence == "moving_old" and materialize_old:
                selected.append(descriptor)
            elif training_state_selected and descriptor.persistence == "moving_reference":
                selected.append(descriptor)
        return tuple(selected)

    def exact_resume_descriptors(self, *, old_policy_is_derived: bool) -> tuple[PolicyStateDescriptor, ...]:
        """Return typed roles physically required for exact resume at this boundary."""

        return tuple(
            descriptor
            for descriptor in self.descriptors()
            if descriptor.persistence not in ("copy_only", "offload_only")
            and not (descriptor.persistence == "moving_old" and old_policy_is_derived)
        )

    def copy_descriptors(self) -> tuple[PolicyStateDescriptor, ...]:
        """Return every configured policy role initialized from model-only state."""

        return tuple(descriptor for descriptor in self.descriptors() if descriptor.persistence != "offload_only")

    def offload_descriptors(self) -> tuple[PolicyStateDescriptor, ...]:
        """Return update-side snapshots parked while the policy performs rollout."""

        return tuple(descriptor for descriptor in self.descriptors() if descriptor.semantic_role != "policy")

    def refresh_old(self) -> None:
        """Drive the adapter-owned moving-old copy."""

        refresh = getattr(self.model, "refresh_old", None)
        if not callable(refresh):
            raise NotImplementedError("model does not support an old policy (update_old_every_iter>0)")
        refresh()

    def refresh_ref(self) -> None:
        """Drive the adapter-owned moving-reference copy."""

        refresh = getattr(self.model, "refresh_ref", None)
        if not callable(refresh):
            raise NotImplementedError("model does not support a moving reference (update_ref_every_iter>0)")
        refresh()
