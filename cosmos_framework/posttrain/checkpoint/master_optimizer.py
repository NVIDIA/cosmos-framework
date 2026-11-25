# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Rank-owned optimizer payloads inside the existing atomic DCP transaction.

Each rank's optimizer state (masters, uint8 moments, block scales, counters) is one
opaque uint8 tensor leaf: a SHA-256 digest followed by ``torch.save`` bytes. DCP writes
tensor leaves natively, so it never pickles or unpickles this payload; the bytes are
decoded only by ``torch.load(weights_only=True)`` with the minimal allow-list in
``checkpoint/safe_dcp.py``. (The component's DCP ``.metadata`` file is read by the
storage reader exactly as for every other component.) Each rank key is unique, so DCP
never interprets the payload as a parameter shard. This is deliberately same-topology
recovery, not topology-changing conversion. Shared by posttrain trainers and the
generator ``OptimizersContainer``.
"""

from __future__ import annotations

import hashlib
import io
import pickle
from collections.abc import Callable, Mapping
from copy import deepcopy
from traceback import clear_frames

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.utils._pytree import tree_map

from cosmos_framework.posttrain.utils.dist import run_rank_converged
from cosmos_framework.utils.optimizer_master import local_tensor

# v4: tensor leaves only (v3 stored a pickled tuple, which a default DCP planner unpickles).
_FORMAT = "cosmos_rank_master_v4"
_DIGEST_BYTES = 32


def _rank_key(rank: int) -> str:
    return f"rank_{rank}"


def _encode_rank_state(state) -> torch.Tensor:
    """``[sha256(body) | torch.save(state)]`` as one uint8 tensor, with no copy beyond the serialization.

    Host peak: ``state`` (the host snapshot) and these bytes coexist until ``torch.save`` returns,
    about twice this rank's optimizer state. ``torch.save`` needs the whole object graph, and
    serializing entry by entry would change the payload format, so that peak is accepted.
    """
    buffer = io.BytesIO()
    buffer.write(bytes(_DIGEST_BYTES))
    torch.save(state, buffer)
    view = buffer.getbuffer()
    view[:_DIGEST_BYTES] = hashlib.sha256(view[_DIGEST_BYTES:]).digest()
    # frombuffer keeps the exported view (and its BytesIO) alive. Nothing in this serialized
    # snapshot aliases live parameters, moments or managed buffers.
    return torch.frombuffer(view, dtype=torch.uint8)


def _decode_rank_state(tensor):
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.uint8
        or tensor.ndim != 1
        or not tensor.is_contiguous()
        or tensor.numel() <= _DIGEST_BYTES
    ):
        raise ValueError("invalid master optimizer byte tensor schema/length")
    content = memoryview(tensor.numpy())
    if hashlib.sha256(content[_DIGEST_BYTES:]).digest() != bytes(content[:_DIGEST_BYTES]):
        raise ValueError("master optimizer payload checksum mismatch")
    # Tensors, Python containers/scalars, plus the minimal NumPy types safe_dcp allows for
    # checkpoint values (generator group lr/decay can be NumPy scalars). No other globals.
    from cosmos_framework.checkpoint.safe_dcp import _BYTES_SAFE_GLOBALS

    try:
        with torch.serialization.safe_globals(_BYTES_SAFE_GLOBALS):
            return torch.load(io.BytesIO(content[_DIGEST_BYTES:]), map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(f"refusing to load master optimizer payload: {error}") from error


def optimizer_payload(optimizer):
    """This rank's DCP optimizer component: a replicated world-size tensor and one rank-unique payload."""
    identity = optimizer.identity
    return {
        _FORMAT: {
            "world_size": torch.tensor([identity["world_size"]], dtype=torch.int64),
            _rank_key(identity["rank"]): _encode_rank_state(optimizer.state_dict()),
        }
    }


def tensor_shapes(metadata) -> dict[str, tuple[int, ...]]:
    """Logical tensor shapes from DCP ``Metadata`` (keys as ``load_template`` expects)."""
    return {key: tuple(entry.size) for key, entry in metadata.state_dict_metadata.items() if hasattr(entry, "size")}


def load_template(optimizer, shapes: Mapping[str, tuple[int, ...] | None]):
    """Allocation-only DCP load target for this rank's payload, or None if the checkpoint has none.

    Decided from the checkpoint's global DCP metadata alone, so every rank reaches the same
    outcome before any collective: None (no payload at all) on every rank, or a topology
    mismatch raised on every rank when the payload was written by a different world size.
    """
    prefix = f"{_FORMAT}."
    saved = {key[len(prefix) :] for key in shapes if key.startswith(prefix)}
    if not saved:
        return None
    world = optimizer.identity["world_size"]
    expected = {"world_size", *(_rank_key(rank) for rank in range(world))}
    if saved != expected or shapes.get(f"{prefix}world_size") != (1,):
        written = len(saved - {"world_size"})
        raise ValueError(
            f"master optimizer checkpoint format/topology mismatch: written by {written} ranks, resuming on "
            f"{world}; exact resume requires the same topology"
        )
    # Check every rank's entry, not only this rank's own, so every rank takes the same decision.
    sizes = [shapes[f"{prefix}{_rank_key(rank)}"] for rank in range(world)]
    if any(size is None or len(size) != 1 for size in sizes):
        raise ValueError("invalid master optimizer byte tensor schema/length")
    key = _rank_key(optimizer.identity["rank"])
    size = sizes[optimizer.identity["rank"]]
    return {_FORMAT: {"world_size": torch.zeros(1, dtype=torch.int64), key: torch.empty(size, dtype=torch.uint8)}}


def stage_payload(optimizer, loaded):
    """Decode and validate this rank's payload on the host; nothing live changes."""
    payload = loaded.get(_FORMAT) if isinstance(loaded, Mapping) else None
    key = _rank_key(optimizer.identity["rank"])
    if not isinstance(payload, Mapping) or set(payload) != {"world_size", key}:
        raise ValueError("master optimizer checkpoint format mismatch")
    world = payload["world_size"]
    if not isinstance(world, torch.Tensor) or world.dtype != torch.int64 or world.numel() != 1:
        raise ValueError("invalid master optimizer world size")
    if int(world.item()) != optimizer.identity["world_size"]:
        raise ValueError("master optimizer checkpoint format/topology mismatch")
    return optimizer.stage_state_dict(_decode_rank_state(payload[key]))


def require_masters_match(optimizer, saved, model_value: Callable[[str, torch.Tensor], torch.Tensor]):
    """Every saved master must round to the model value the same checkpoint restores for it.

    ``model_value(name, live_parameter)`` returns that value; names are the canonical
    state-dict names (activation-checkpoint and compile segments removed).
    """
    for name, master, live in zip(optimizer.names, saved["masters"], optimizer.live):
        value = local_tensor(model_value(name, live)).detach()
        if not torch.equal(value.cpu(), master.detach().to(dtype=live.dtype, device="cpu")):
            raise ValueError(f"checkpoint model/master values disagree for {name}")


def restore_into_live_model(optimizer, loaded, context: str) -> None:
    """Restore after the model was loaded in place (generator DCP): validate, commit, roll back on failure."""

    def stage():
        saved = stage_payload(optimizer, loaded)
        require_masters_match(optimizer, saved, lambda _, live: live)
        return saved

    _commit_with_rollback(optimizer, run_rank_converged(stage, f"{context} staging"), context)


def _rollback_point(optimizer, context: str):
    """Host rollback point, or None for a fresh optimizer (rolling back re-seeds it).

    ``state_dict`` may refuse on one rank (masters that disagree with a model changed under its
    moments), so the refusal is converged before any rank enters the next collective.
    """
    return run_rank_converged(
        lambda: optimizer.state_dict() if optimizer.backend.state else None, f"{context} rollback snapshot"
    )


def _commit_with_rollback(optimizer, saved, context: str) -> None:
    before = _rollback_point(optimizer, context)
    try:
        run_rank_converged(lambda: optimizer.commit_state_dict(saved), f"{context} commit")
    except BaseException:
        run_rank_converged(lambda: _rollback(optimizer, before), f"{context} rollback")
        raise


def _rollback(optimizer, before) -> None:
    """Undo a commit: the previous host snapshot, or a fresh optimizer re-seeded from the model."""
    if before is None:
        optimizer.reset_state()
    else:
        optimizer.commit_state_dict(before)


def staged_optimizer(manager, path, optimizer):
    # The inventory is rank-identical, so the presence/topology decision below is taken by every
    # rank before load_dcp_component enters its collectives.
    target = load_template(optimizer, manager.dcp_tensor_inventory(path, "optim"))
    try:
        if target is None or not manager.load_dcp_component(path, "optim", target):
            raise FileNotFoundError(f"master optimizer checkpoint is missing its rank-local {_FORMAT} component")
        return run_rank_converged(lambda: stage_payload(optimizer, target), "master optimizer restore staging")
    finally:
        if target is not None:
            target.clear()


def restore(trainer, path):
    """A retained failure must not retain replacement moments or model snapshots."""
    try:
        return _restore(trainer, path)
    except BaseException as error:
        pending, seen = [error], set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            clear_frames(current.__traceback__)
            pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
        raise


def _stage_restore(trainer, path):
    """Read a complete replacement without mutating any live training state."""
    replacement = before_model = model = None
    try:
        manager, optimizer, net = trainer.checkpoint_manager, trainer.opt, trainer._net()
        trainer._require_checkpoint_boundary()
        control = manager.load_component(path, "trainer")
        if control is None:
            raise ValueError("master optimizer resume requires trainer control state")
        run_rank_converged(lambda: trainer._validate_objective_identity(control), "master restore objective")
        replacement = staged_optimizer(manager, path, optimizer)
        options = StateDictOptions(cpu_offload=True)
        before_model = run_rank_converged(
            lambda: get_model_state_dict(net, options=options), "master restore model snapshot"
        )
        # cpu_offload already returned independent host copies of device state. Only a CPU model's
        # state aliases live storage; clone just those, so the rollback snapshot is independent.
        live = {
            local_tensor(v).untyped_storage().data_ptr()
            for v in net.state_dict().values()
            if isinstance(v, torch.Tensor)
        }
        before_model = tree_map(
            lambda v: (
                v.detach().clone()
                if isinstance(v, torch.Tensor) and local_tensor(v).untyped_storage().data_ptr() in live
                else v
            ),
            before_model,
        )
        # The load target is fully overwritten (strict DCP load / consolidated fill), so it needs
        # matching allocations, not a value copy of the snapshot.
        model = tree_map(lambda v: torch.empty_like(v) if isinstance(v, torch.Tensor) else deepcopy(v), before_model)
        if manager.has_dcp_component(path, "model"):
            if not manager.load_dcp_component(path, "model", model):
                raise FileNotFoundError("master optimizer resume requires model state")
        else:
            from cosmos_framework.posttrain.checkpoint.consolidated import fill_sharded_state_from_consolidated

            saved = manager.load_component(path, "model")
            if saved is None:
                raise FileNotFoundError("master optimizer resume requires model state")
            run_rank_converged(
                lambda: fill_sharded_state_from_consolidated(model, saved["net"], context="master model restore"),
                "master consolidated model staging",
            )

        # ``model`` is keyed like ``get_model_state_dict``; optimizer names use the same canonical form.
        run_rank_converged(
            lambda: require_masters_match(optimizer, replacement, lambda name, _: model[name]),
            "master/model snapshot consistency",
        )
        return control, replacement, before_model, model
    finally:
        replacement = before_model = model = None


def _restore(trainer, path):
    """Validate one stable generation before mutation; roll back application failures."""
    replacement = before_model = model = before_optimizer = None
    try:
        manager, optimizer, net = trainer.checkpoint_manager, trainer.opt, trainer._net()
        # The lifecycle also binds provenance around trainer+cursor loading. That
        # outer check alone is too late for transactional restore: a mutation
        # during reads would be detected after the new model/moments are live.
        (control, replacement, before_model, model), _ = manager.load_with_stable_provenance(
            path, lambda: _stage_restore(trainer, path)
        )
        before_control = trainer._trainer_state()
        before_validation = deepcopy(trainer.validation_state)
        optimizer.sync_groups()
        # A fresh optimizer (the usual resume) needs no host rollback point: rolling back re-seeds it.
        before_optimizer = _rollback_point(optimizer, "master restore")
        try:
            # Validate the control program first. Its mutations are rolled back if any
            # rank rejects it; model and optimizer still have their original values.
            resumed = trainer._restore_trainer_state(control)
            run_rank_converged(lambda: set_model_state_dict(net, model), "master restore model commit")
            run_rank_converged(lambda: optimizer.commit_state_dict(replacement), "master restore optimizer commit")
        except BaseException:
            run_rank_converged(lambda: set_model_state_dict(net, before_model), "master restore model rollback")
            run_rank_converged(lambda: _rollback(optimizer, before_optimizer), "master restore optimizer rollback")
            trainer._restore_trainer_state(before_control)
            trainer.validation_state = before_validation
            raise
        trainer._resumed_from_checkpoint = True
        return resumed
    finally:
        # These cells are captured by collective callbacks. Clearing frames alone
        # cannot release their contents while a retained traceback owns a callback.
        replacement = before_model = model = before_optimizer = None
