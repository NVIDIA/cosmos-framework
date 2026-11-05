# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reject synchronous call chains that re-enter an actor blocked on a peer."""

from __future__ import annotations

from collections.abc import Sequence

from cosmos_framework.posttrain.workers.worker_group import WorkerGroup


def validate_peer_chain(chain: Sequence[WorkerGroup]) -> None:
    """Validate an already-bound ordinary peer chain before issuing its first call.

    Consecutive roles in one actor call each other directly. After a remote hop that actor's
    single invocation thread is blocked, so a later hop back into it cannot complete. Actor
    identities and per-cell routes come from the live backend; role names are only diagnostics.
    Local bindings are synchronous in one process. Asynchronous and weight-fanout calls are
    different contracts and are not represented by this chain.
    """
    if len(chain) < 2:
        raise ValueError("a synchronous peer chain requires at least two bound roles")
    backends = [role._backend for role in chain]
    if all(backend is None and role.body is not None for role, backend in zip(chain, backends, strict=True)):
        return
    from cosmos_framework.posttrain.workers.grid import GridBackend

    if any(not isinstance(backend, GridBackend) for backend in backends):
        raise ValueError("a synchronous peer chain requires local bodies or bound GridBackend roles")

    shapes = [backend.wg_shapes.get(role.name, backend.shape) for role, backend in zip(chain, backends, strict=True)]
    routes = [
        backend.peer_cells(peer, shape)
        for backend, peer, shape in zip(backends[:-1], chain[1:], shapes[:-1], strict=True)
    ]
    waits: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for replica in range(shapes[0][0]):
        for rank in range(shapes[0][1]):
            cell = (replica, rank)
            active_actor = backends[0].actors[cell]._actor_id.hex()
            entered = {active_actor}
            path = [f"{chain[0].name}{cell}"]
            labels.setdefault(active_actor, path[-1])
            for role, backend, route in zip(chain[1:], backends[1:], routes, strict=True):
                cell = route[cell]
                actor = backend.actors[cell]._actor_id.hex()
                path.append(f"{role.name}{cell}")
                labels.setdefault(actor, path[-1])
                if actor != active_actor and actor in entered:
                    raise ValueError("synchronous peer chain re-enters a blocked actor: " + " -> ".join(path))
                if actor != active_actor:
                    waits.setdefault(active_actor, set()).add(actor)
                entered.add(actor)
                active_actor = actor

    # Every root cell can be active concurrently: custom alignments may create a cycle
    # across replicas even when no individual root's path repeats an actor.
    complete: set[str] = set()
    stack: list[str] = []

    def visit(actor: str) -> None:
        if actor in stack:
            cycle = stack[stack.index(actor) :] + [actor]
            raise ValueError(
                "synchronous peer chain has an actor wait cycle across replicas: "
                + " -> ".join(labels[cell] for cell in cycle)
            )
        if actor in complete:
            return
        stack.append(actor)
        for target in waits.get(actor, ()):
            visit(target)
        stack.pop()
        complete.add(actor)

    for actor in waits:
        visit(actor)
