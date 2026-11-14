# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in FP64 numerator accumulation over owned native FSDP gradient shards.

Registered native flow SFT builders declare FP64 reduction before registration. Installation
follows final optimizer construction and precedes any backward. Public FSDP
setters own mutations; private fields are read only under exact Torch source
admission. The native model, autograd, optimizer and checkpoint transactions
remain owned by the existing training workflow.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

PROFILE = "fp64_owned_shard_window_v1"
TORCH_SOURCE_SHA256 = {
    "_fully_shard.py": "3e1299e618dcaa589c3a70c8df51d26fc416dd34af1d37c2b8a5d30d0aa9d7bf",
    "_fsdp_collectives.py": "f8527a45d7d8ae029e8b2ce78b9fd6fc909b95e4b3cf1be7cfa9ba3da722be31",
    "_fsdp_param.py": "27160f15d74b42d4f97e8356f644a7ea74a146b43baffa4af669a5b5587f3a91",
    "_fsdp_param_group.py": "0f81daf147e30914f9f96bd5e654c3bc3df7a5f4a0db37819ed76d3f00858cb6",
}


class NonFiniteGradientWindowError(RuntimeError):
    """Converged numerical rejection before any optimizer mutation.

    The trainer maps this to its existing skip/fail-stop policy. Ownership and
    reducer failures deliberately retain their ordinary fail-closed exceptions.
    """


def local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def integer_count(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
        or value >= 2**53
    ):
        raise ValueError("global valid count must be a nonnegative exactly represented integer")
    return int(value)


def count_from_native_dp_mean(dp_mean, world_size):
    if type(world_size) is not int or world_size < 1:
        raise ValueError("invalid data-parallel world size")
    if isinstance(dp_mean, bool) or not isinstance(dp_mean, (int, float)) or not math.isfinite(dp_mean) or dp_mean < 0:
        raise ValueError("invalid native DP-mean count")
    return integer_count(dp_mean * world_size)


@dataclass(frozen=True)
class Owner:
    name: str
    parameter: torch.nn.Parameter
    global_shape: tuple
    shard_count: int
    replica_count: int
    aliases: tuple = ()

    @property
    def local_shape(self):
        return tuple(local(self.parameter).shape)

    @property
    def padded_numel(self):
        return math.ceil(self.global_shape[0] / self.shard_count) * math.prod(self.global_shape[1:])


class OwnedGradientWindow:
    """One update window, with explicit per-microbatch and optimizer boundaries.

    Lifecycle: begin_window -> (begin_micro, native backward, after_backward)*N ->
    finalize -> optional clip -> ordinary optimizer/scheduler -> finish_step.
    abort clears all policy buffers/gradients after a converged failed backward.
    Checkpoint state is available only at the committed optimizer boundary.
    """

    def __init__(
        self,
        groups,
        *,
        microbatches,
        world_size,
        converge=None,
        before_micro=None,
        before_backward=None,
        count_sum=None,
        count_consensus=None,
    ):
        if type(microbatches) is not int or microbatches < 1:
            raise ValueError("invalid microbatch count")
        if type(world_size) is not int or world_size < 1:
            raise ValueError("invalid world size")
        self.groups = {name: tuple(owners) for name, owners in groups.items()}
        if not self.groups or any(not isinstance(name, str) or not owners for name, owners in self.groups.items()):
            raise ValueError("empty owner group")
        self.owners = tuple(owner for owners in self.groups.values() for owner in owners)
        if len({id(o.parameter) for o in self.owners}) != len(self.owners):
            raise ValueError("one parameter cannot belong to two owning FSDP groups")
        if len({o.name for o in self.owners}) != len(self.owners):
            raise ValueError("duplicate canonical owner")
        alias_owners = {}
        for owner in self.owners:
            if type(owner.shard_count) is not int or type(owner.replica_count) is not int:
                raise ValueError("integer topology required")
            for name in (owner.name, *owner.aliases):
                if (
                    not isinstance(name, str)
                    or not name
                    or (name in alias_owners and alias_owners[name] != id(owner.parameter))
                ):
                    raise ValueError("conflicting parameter alias")
                alias_owners[name] = id(owner.parameter)
            p = owner.parameter
            if (
                not p.requires_grad
                or p.dtype != torch.float32
                or not owner.global_shape
                or math.prod(owner.global_shape) <= 0
            ):
                raise ValueError("owned FP64 gradient policy requires positive-shape FP32 trainable owners")
            if owner.shard_count * owner.replica_count != world_size or min(owner.shard_count, owner.replica_count) < 1:
                raise ValueError("owner topology does not cover data-parallel ranks")
            if not local(p).is_contiguous() or local(p).numel() > owner.padded_numel:
                raise ValueError("unsupported local shard layout")
            if p.grad is not None:
                raise ValueError("install before gradient accumulation")
        self.microbatches = microbatches
        self.world_size = world_size
        self.converge = converge or (lambda fn, context: fn())
        self.before_micro = before_micro or (lambda: None)
        self.before_backward = before_backward or (lambda: None)
        self.count_sum = count_sum
        self.count_consensus = count_consensus or (lambda count: None)
        self.phase = "idle"
        self.completed_updates = 0
        self.clipped = False
        self.buffers = {}
        self.pending = {}
        self.templates = {}
        self.index = 0
        self.count = 0
        self._handles = []
        self.forward_prepared = False
        self.failure = None

    def identity(self):
        return {
            "profile": PROFILE,
            "integration": "public_sft_v1",
            "native_forward": "unchanged",
            "native_derivatives": "unchanged_fp32",
            "collective": "fp64_global_sum",
            "accumulation": "fp64_by_unique_owner",
            "normalization": "once_global_valid_count_before_fp32_cast",
            "clip": "fp64_global_unique_owner_norm_then_fp32_coefficient",
            "denominator_owner": "native_valid_mask_int64_global_sum"
            if self.count_sum is not None
            else "explicit_global_integer_fixture",
            "torch_reducer_source_sha256": dict(TORCH_SOURCE_SHA256),
            "world_size": self.world_size,
            "microbatches": self.microbatches,
            "groups": {
                g: [
                    {
                        "name": o.name,
                        "aliases": list(o.aliases),
                        "shape": list(o.global_shape),
                        "shards": o.shard_count,
                        "replicas": o.replica_count,
                    }
                    for o in owners
                ]
                for g, owners in self.groups.items()
            },
        }

    def begin_window(self):
        if self.failure is not None:
            raise RuntimeError("gradient policy is invalidated; restore a complete checkpoint") from self.failure
        if (
            self.phase != "idle"
            or self.buffers
            or self.pending
            or self.forward_prepared
            or any(o.parameter.grad is not None for o in self.owners)
        ):
            raise RuntimeError("begin a fresh, zero-gradient optimizer window")
        self.phase = "open"
        self.index = 0
        self.count = 0

    def prepare_forward(self):
        """Check sharded ownership before a native forward leaves its root unsharded.

        Native objectives discover the valid-mask count during forward. Prepare only
        ownership here; begin_native_micro must still admit the actual count before
        backward. Direct callers that know the count may keep begin_micro first.
        """

        def prepare():
            if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
                raise RuntimeError("owned FP64 gradient policy does not support autocast")
            if self.phase != "open" or self.index >= self.microbatches or self.pending or self.forward_prepared:
                raise RuntimeError("invalid native forward preparation")
            if any(o.parameter.grad is not None for o in self.owners):
                raise RuntimeError("unconsumed parameter gradient")
            self.before_micro()

        self.converge(prepare, "fp64 window before native forward")
        self.forward_prepared = True

    def begin_micro(self, global_valid_count):
        def prepare():
            count = integer_count(global_valid_count)
            if self.phase != "open" or self.index >= self.microbatches or self.pending:
                raise RuntimeError("invalid microbatch lifecycle")
            if any(o.parameter.grad is not None for o in self.owners):
                raise RuntimeError("unconsumed parameter gradient")
            if not self.forward_prepared:
                self.before_micro()
            # The forward may leave parameters unsharded, so recheck reducer
            # flags/hooks here without re-enumerating the temporary parameters.
            self.before_backward()
            return count

        count = self.converge(prepare, "fp64 window before native backward")
        self.count_consensus(count)
        self.current_count = count
        self.phase = "micro"
        self.forward_prepared = False

    def begin_native_micro(self, local_valid_count):
        if self.count_sum is None:
            raise RuntimeError("native count collective is not installed")

        def prepare():
            count = integer_count(local_valid_count)
            integer_count(count * self.world_size)  # bound the collective integer SUM before dispatch
            return count

        count = self.converge(prepare, "native valid-mask count preparation")
        total = integer_count(self.count_sum(count))
        self.begin_micro(total)
        return total

    def hook(self, group):
        if group not in self.groups:
            raise ValueError("unknown group")

        def capture(output):
            if self.phase != "micro" or group in self.pending:
                raise RuntimeError("unexpected or duplicate group reduction")
            if output.ndim != 1 or output.dtype != torch.float64 or output.requires_grad:
                raise RuntimeError("requires native FP64 flattened SUM output")
            saved = output.detach().clone()
            event = None
            output.zero_()
            if output.device.type == "cuda":
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(output.device))
            self.pending[group] = (saved, event)

        return capture

    def _wait(self, pair):
        tensor, event = pair
        if event is not None:
            stream = torch.cuda.current_stream(tensor.device)
            stream.wait_event(event)
            tensor.record_stream(stream)  # keep hook-stream allocation alive through queued owner copies/additions
        return tensor

    def after_backward(self):
        def ingest():
            if self.phase != "micro":
                raise RuntimeError("no active backward")
            finite = True
            for group, owners in self.groups.items():
                active = [o for o in owners if o.parameter.grad is not None]
                if not active:
                    if group in self.pending:
                        raise RuntimeError("reduction without a public gradient owner")
                    continue
                if group not in self.pending:
                    raise RuntimeError("gradient bypassed the admitted FP64 group hook")
                captured = self._wait(self.pending[group])
                if captured.numel() != sum(o.padded_numel for o in active):
                    raise RuntimeError("active packed owner layout differs from admitted FSDP order")
                captured_finite = bool(torch.isfinite(captured).all())
                finite = finite and captured_finite
                if captured_finite and self.current_count == 0 and bool(torch.count_nonzero(captured)):
                    raise RuntimeError("zero-weight padding has a nonzero gradient")
                offset = 0
                for owner in active:
                    grad = owner.parameter.grad
                    raw = local(grad)
                    if (
                        grad.dtype != torch.float32
                        or tuple(raw.shape) != owner.local_shape
                        or not raw.is_contiguous()
                        or bool(torch.count_nonzero(raw))
                    ):
                        raise RuntimeError("FSDP gradient placeholder/ownership changed")
                    if hasattr(owner.parameter, "placements") and (
                        grad.placements != owner.parameter.placements or grad.device_mesh != owner.parameter.device_mesh
                    ):
                        raise RuntimeError("gradient DTensor ownership differs")
                    piece = captured[offset : offset + owner.padded_numel][: raw.numel()].view(raw.shape)
                    if owner.name not in self.buffers:
                        self.buffers[owner.name] = torch.zeros_like(raw, dtype=torch.float64)
                        self.templates[owner.name] = grad.detach().clone()
                    self.buffers[owner.name].add_(piece)
                    offset += owner.padded_numel
                    owner.parameter.grad = None
                if offset != captured.numel():
                    raise RuntimeError("packed gradient remainder")
            self.count += self.current_count
            self.index += 1
            self.pending.clear()
            self.phase = "open"
            return finite

        try:
            finite = self.converge(ingest, "fp64 gradient-window shard ownership")
            self._require_numerically_finite(finite, "native derivative")
        except BaseException:
            self.abort()
            raise

    def finalize(self):
        def materialize():
            if self.phase != "open" or self.index != self.microbatches or self.pending or self.forward_prepared:
                raise RuntimeError("incomplete optimizer window")
            if self.count <= 0:
                raise RuntimeError("empty global objective window")
            if not self.buffers:
                raise RuntimeError("native objective produced no owned gradients")
            # Backward has restored sharded parameters. Re-enumerate final
            # optimizer/model ownership before publishing any update gradients.
            self.before_micro()
            finite = True
            for owner in self.owners:
                if owner.name not in self.buffers:
                    continue  # never-used parameters stay None, as native AdamW expects
                value = (self.buffers[owner.name] / self.count).to(torch.float32)
                finite = finite and bool(torch.isfinite(value).all())
                gradient = self.templates[owner.name]
                local(gradient).copy_(value)
                owner.parameter.grad = gradient
            self.phase = "ready"
            return finite

        try:
            finite = self.converge(materialize, "fp64 gradient-window final normalization")
            self._require_numerically_finite(finite, "normalized gradient")
        except BaseException:
            self.abort()
            raise

    def _require_numerically_finite(self, local_finite, context):
        from cosmos_framework.posttrain.roles.trainer.finite import all_ranks_tensor_true

        flag = torch.tensor(local_finite, dtype=torch.bool, device=local(self.owners[0].parameter).device)
        if not all_ranks_tensor_true(flag):
            raise NonFiniteGradientWindowError(f"nonfinite {context} on at least one rank")

    def clip(self, max_norm, *, reduce_sum=None):
        def prepare():
            if self.clipped:
                raise RuntimeError("clip coefficient may be applied only once")
            if (
                self.phase != "ready"
                or isinstance(max_norm, bool)
                or not isinstance(max_norm, (int, float))
                or not math.isfinite(max_norm)
                or max_norm <= 0
            ):
                raise ValueError("clip requires ready gradients and positive finite norm")
            exemplar = local(self.owners[0].parameter)
            squared = torch.zeros((), dtype=torch.float64, device=exemplar.device)
            for owner in self.owners:
                if owner.parameter.grad is not None:
                    gradient = local(owner.parameter.grad)
                    if gradient.shape != local(owner.parameter).shape or gradient.dtype != torch.float32:
                        raise RuntimeError("owned gradient changed before clipping")
                    squared.add_(gradient.double().square().sum() / owner.replica_count)
            return squared

        squared = self.converge(prepare, "owned gradient clipping preparation")
        if self.world_size > 1:
            if reduce_sum is None:
                raise RuntimeError("explicit global norm SUM is required")
            squared = reduce_sum(squared)
        if squared.ndim or squared.dtype != torch.float64 or not bool(torch.isfinite(squared)) or squared < 0:
            raise RuntimeError("invalid global squared norm")
        norm = squared.sqrt()
        coefficient = (max_norm / (norm + 1e-6)).clamp(max=1).to(torch.float32)
        for owner in self.owners:
            if owner.parameter.grad is not None:
                local(owner.parameter.grad).mul_(coefficient)
        self.clipped = True
        return {"norm": norm, "coefficient": coefficient, "profile": PROFILE}

    def finish_step(self):
        if self.phase != "ready":
            raise RuntimeError("no finalized optimizer update")
        self.completed_updates += 1
        self.abort()

    def abort(self):
        for pair in self.pending.values():
            self._wait(pair)
        for owner in self.owners:
            owner.parameter.grad = None
        self.pending.clear()
        self.buffers.clear()
        self.templates.clear()
        self.phase = "idle"
        self.index = 0
        self.count = 0
        self.clipped = False
        self.forward_prepared = False

    def state_dict(self):
        if self.failure is not None:
            raise RuntimeError("invalidated gradient policy cannot be checkpointed") from self.failure
        if self.phase != "idle" or self.buffers or self.pending or self.forward_prepared:
            raise RuntimeError("only committed update-boundary state is resumable")
        return {
            "identity": copy.deepcopy(self.identity()),
            "completed_updates": self.completed_updates,
            "phase": "idle",
        }

    def validate_checkpoint_state(self, state, optimizer_step):
        def same_identity(value):
            # Python equality accepts True == 1 and 4.0 == 4. Checkpoints must
            # preserve the actual typed policy contract, not only numeric value.
            try:
                return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(
                    self.identity(), sort_keys=True, allow_nan=False
                )
            except (TypeError, ValueError):
                return False

        if (
            not isinstance(state, dict)
            or set(state) != {"identity", "completed_updates", "phase"}
            or not same_identity(state["identity"])
            or state["phase"] != "idle"
        ):
            raise ValueError(
                "missing or incompatible gradient-policy checkpoint identity/state; use exact original policy or an explicit warm start"
            )
        if (
            type(optimizer_step) is not int
            or type(state["completed_updates"]) is not int
            or state["completed_updates"] != optimizer_step
            or optimizer_step < 0
        ):
            raise ValueError("gradient-policy checkpoint clock differs from optimizer clock")

    def load_state_dict(self, state):
        self.validate_checkpoint_state(state, state.get("completed_updates") if isinstance(state, dict) else None)
        self.abort()
        self.failure = None
        self.completed_updates = state["completed_updates"]

    def invalidate(self, cause):
        self.failure = cause
        self.abort()

    def prepare_update(self, accumulation):
        def prepare():
            if self.phase == "idle":
                if accumulation != 0:
                    raise RuntimeError("trainer accumulation has no gradient window")
                self.begin_window()
            if self.index != accumulation:
                raise RuntimeError("trainer/gradient-window accumulation differs")

        self.converge(prepare, "owned gradient update clock")
        self.prepare_forward()

    def backward(self, loss, metrics):
        from cosmos_framework.posttrain.algorithm.primitives.objective import OBJECTIVE_WEIGHT_SUM

        def prepare():
            if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
                raise RuntimeError("owned FP64 gradient policy does not support autocast")
            local_count = integer_count(metrics["flow_bc_masked_elements"])
            denom = metrics[OBJECTIVE_WEIGHT_SUM]
            count_from_native_dp_mean(denom, self.world_size)
            return local_count, denom

        local_count, denom = self.converge(prepare, "owned gradient objective preparation")
        total = self.begin_native_micro(local_count)

        def validate_count():
            if total != count_from_native_dp_mean(denom, self.world_size):
                raise RuntimeError("native objective divisor differs from global valid mask count")

        self.converge(validate_count, "owned gradient native objective count")
        (loss * denom).backward()
        self.after_backward()
        if self.index == self.microbatches:
            self.finalize()

    def validate_normalization(self, denom):
        def validate():
            if self.phase != "ready" or self.count != count_from_native_dp_mean(denom, self.world_size):
                raise RuntimeError("gradient window count differs from complete native objective divisor")

        self.converge(validate, "owned gradient final normalization")

    def clip_global(self, max_norm):
        import torch.distributed as dist

        def reduce_sum(value):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            return value

        return self.clip(max_norm, reduce_sum=reduce_sum)


def install_fsdp(model, optimizer, *, microbatches):
    """Admit exact native owner order, then mutate only documented FSDP setters.

    The native flow builder must already use reduce_dtype=float64. The trainer records
    identity() in its existing checkpoint contract. Private reducer layout reads
    are deliberately gated to the inspected Torch2.10 implementation.
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FSDPModule

    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    if not dist.is_initialized():
        raise RuntimeError("owned FP64 gradient policy installation requires initialized native FSDP")

    def prepare():
        if type(optimizer) is not torch.optim.AdamW or optimizer.defaults.get("differentiable", False):
            raise RuntimeError("owned FP64 gradient policy requires ordinary non-differentiable torch AdamW")
        source_root = Path(inspect.getsourcefile(FSDPModule.set_all_reduce_hook)).parent
        for name, digest in TORCH_SOURCE_SHA256.items():
            if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("unreviewed FSDP source " + name)
        aliases = {}
        for name, param in model.named_parameters(remove_duplicate=False):
            if param.requires_grad:
                aliases.setdefault(id(param), (param, []))[1].append(name)
        optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
        if len({id(p) for p in optimizer_params}) != len(optimizer_params) or set(map(id, optimizer_params)) != set(
            aliases
        ):
            raise RuntimeError("optimizer membership differs from unique trainable owners")
        world = dist.get_world_size()
        groups = {}
        modules = {}
        if world != 4:
            raise RuntimeError("owned FP64 policy requires four data-parallel ranks")
        for group_name, module in model.named_modules():
            if not isinstance(module, FSDPModule):
                continue
            group = module._get_fsdp_state()._fsdp_param_group  # read-only, source-pinned admission
            if group is None:
                continue
            if (
                group._all_reduce_hook is not None
                or group._all_reduce_hook_stream is not None
                or group.gradient_divide_factor is not None
            ):
                raise RuntimeError("existing reduction policy conflict")
            if not group.reduce_grads or not group.all_reduce_grads:
                raise RuntimeError("install only on a synchronous native reducer")
            if getattr(group, "_partial_reduce_output", None) is not None:
                raise RuntimeError("install only before accumulation begins")
            owners = []
            for slot in group.fsdp_params:  # read only exact original packed order; never rewrite reducer ownership
                param = slot.sharded_param
                if not param.requires_grad:
                    continue
                if slot.mp_policy.reduce_dtype != torch.float64:
                    raise RuntimeError("builder must declare FP64 reduction before registration")
                if slot.mp_policy.param_dtype not in (None, torch.float32):
                    raise RuntimeError("owned FP64 gradient policy requires FP32 native trainable leaf derivatives")
                if slot.offload_to_cpu:
                    raise RuntimeError("CPU offload is not admitted for the owned-shard policy")
                if getattr(slot, "unsharded_accumulated_grad", None) is not None:
                    raise RuntimeError("pending unsharded accumulation is not admitted")
                if id(param) not in aliases:
                    raise RuntimeError("FSDP owner is absent from model parameter registry")
                mesh = param.device_mesh
                placements = param.placements
                from torch.distributed.tensor import Replicate, Shard

                shard_axes = [i for i, p in enumerate(placements) if isinstance(p, Shard) and p.dim == 0]
                if len(shard_axes) != 1 or any(
                    not isinstance(p, (Shard, Replicate)) or isinstance(p, Shard) and p.dim != 0 for p in placements
                ):
                    raise RuntimeError("owned FP64 gradient policy supports DP dim0 shards/replicas only")
                if sorted(mesh.mesh.flatten().tolist()) != list(range(world)):
                    raise RuntimeError("DP mesh must own exactly this process group")
                shards = mesh.size(shard_axes[0])
                replicas = world // shards
                if (shards, replicas) not in ((4, 1), (2, 2)) or (mesh.ndim, shard_axes[0]) not in ((1, 0), (2, 1)):
                    raise RuntimeError("owned FP64 policy requires FSDP4 or HSDP2x2")
                names = aliases[id(param)][1]
                owners.append(Owner(names[0], param, tuple(param.shape), shards, replicas, tuple(names)))
            if owners:
                groups[group_name or "<root>"] = owners
                modules[group_name or "<root>"] = module
        if {id(o.parameter) for owners in groups.values() for o in owners} != set(aliases):
            raise RuntimeError("unowned trainable parameter")
        # These read-only checks are repeated before every native backward. No-sync,
        # CPU offload, foreign hooks, owner replacement or policy mutation must not
        # silently bypass capture and produce an apparently empty gradient window.
        hooks = {}
        initial_local_shapes = {id(o.parameter): o.local_shape for owners in groups.values() for o in owners}

        def validate_policy():
            current = [p for group in optimizer.param_groups for p in group["params"]]
            if len(current) != len(aliases) or {id(p) for p in current} != set(aliases):
                raise RuntimeError("optimizer ownership changed after installation")
            registry = {}
            for name, param in model.named_parameters(remove_duplicate=False):
                if param.requires_grad:
                    registry.setdefault(id(param), []).append(name)
            if registry != {key: value[1] for key, value in aliases.items()}:
                raise RuntimeError("trainable owner/alias registry changed after installation")
            for owner in window.owners:
                if (
                    owner.parameter.dtype != torch.float32
                    or tuple(owner.parameter.shape) != owner.global_shape
                    or owner.local_shape != initial_local_shapes[id(owner.parameter)]
                ):
                    raise RuntimeError("trainable owner shape/dtype changed")
            validate_reducers()

        def validate_reducers():
            current = [p for group in optimizer.param_groups for p in group["params"]]
            if len(current) != len(aliases) or {id(p) for p in current} != set(aliases):
                raise RuntimeError("optimizer ownership changed after installation")
            for name, module in modules.items():
                group = module._get_fsdp_state()._fsdp_param_group
                if not group.reduce_grads or not group.all_reduce_grads:
                    raise RuntimeError(
                        "owned FP64 gradient policy requires every microbatch gradient SUM; no-sync is unsupported"
                    )
                if (
                    group.gradient_divide_factor != 1.0
                    or group._all_reduce_hook is not hooks[name]
                    or group._all_reduce_hook_stream is not None
                ):
                    raise RuntimeError("FSDP reduction policy changed")
                if getattr(group, "_partial_reduce_output", None) is not None:
                    raise RuntimeError("pending FSDP reduction accumulation is unsupported")
                active = [slot for slot in group.fsdp_params if slot.sharded_param.requires_grad]
                if [id(slot.sharded_param) for slot in active] != [id(o.parameter) for o in groups[name]]:
                    raise RuntimeError("FSDP packed owners changed")
                if any(
                    slot.offload_to_cpu
                    or slot.mp_policy.reduce_dtype != torch.float64
                    or slot.mp_policy.param_dtype not in (None, torch.float32)
                    or getattr(slot, "unsharded_accumulated_grad", None) is not None
                    for slot in active
                ):
                    raise RuntimeError("FSDP precision/offload policy changed")

        from cosmos_framework.posttrain.utils.dist import collective_device

        def count_sum(count):
            value = torch.tensor([count], dtype=torch.int64, device=collective_device())
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            return int(value.item())

        def count_consensus(count):
            lower = torch.tensor([count], dtype=torch.int64, device=collective_device())
            upper = lower.clone()
            dist.all_reduce(lower, op=dist.ReduceOp.MIN)
            dist.all_reduce(upper, op=dist.ReduceOp.MAX)
            if int(lower.item()) != int(upper.item()):
                raise RuntimeError("global denominator differs across ranks")

        window = OwnedGradientWindow(
            groups,
            microbatches=microbatches,
            world_size=world,
            converge=run_rank_converged,
            before_micro=validate_policy,
            before_backward=validate_reducers,
            count_sum=count_sum,
            count_consensus=count_consensus,
        )
        return window, modules, hooks, validate_policy

    window, modules, hooks, validate_policy = run_rank_converged(prepare, "FP64 gradient policy installation admission")

    def commit():
        for name, module in modules.items():
            hooks[name] = window.hook(name)
            module.set_gradient_divide_factor(1.0)
            module.set_all_reduce_hook(hooks[name])
        window._handles = list(modules.values())
        validate_policy()

    run_rank_converged(commit, "FP64 gradient policy public hook registration")
    return window
