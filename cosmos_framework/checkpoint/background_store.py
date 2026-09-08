# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Rendezvous for the background checkpoint-saving process group.

Asynchronous DCP checkpointing spawns a background process per rank that needs its own
process group, separate from the training group. That group needs a port, and deriving one
arithmetically from ``MASTER_PORT`` does not work: the port is left unheld for the whole life
of the job, and ``MASTER_PORT`` frequently lands inside the kernel's ephemeral range, so any
outbound connection the job makes -- S3, NCCL, wandb -- can be handed it before the first
checkpoint. The background process then dies with EADDRINUSE, taking checkpointing with it.

The approach here never releases the port. The parent on rank 0 binds and listens on an
ephemeral port, hands the live socket to its background process, and that process's TCPStore
adopts the fd instead of binding one of its own, so the port is continuously held from
reservation through use. The port number is broadcast over the training group so every other
rank's background process knows where to connect.

Usage, in the parent::

    listen_socket, port = None, [0]
    if dist.get_rank() == 0:
        listen_socket, port[0] = reserve_background_store_socket()
    dist.broadcast_object_list(port, src=0)
    ctx.Process(target=..., args=(..., listen_socket, port[0]))

and in the spawned background process::

    store = build_background_store(listen_socket, port)
    dist.init_process_group(backend=..., store=store, rank=rank, world_size=world_size)
"""

import os
import socket
from datetime import timedelta
from typing import Optional, Tuple

from torch.distributed import TCPStore

# Every rank's background process has to spawn, import torch, and initialize CUDA before it
# joins this store, so allow well beyond the TCPStore default of 300s at large rank counts.
DEFAULT_BACKGROUND_STORE_TIMEOUT = timedelta(minutes=30)


def reserve_background_store_socket() -> Tuple[socket.socket, int]:
    """Reserve the port the background checkpoint process group's TCPStore will serve on.

    Call on rank 0 only, in the parent process. The returned socket must be passed to that
    rank's background process and handed to :func:`build_background_store`, which adopts it.
    Keep it open in the parent until then -- closing it reopens the race this exists to avoid.

    The ``listen`` call is what makes the reservation exclusive, not the ``bind``: on Linux a
    bound-but-unlistening socket does not stop another SO_REUSEADDR socket from binding the
    same port. TCPStore accepts an already-listening fd, so claiming LISTEN here is safe.

    Returns:
        The listening socket, and the port it is bound to.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", 0))
    sock.listen(socket.SOMAXCONN)
    return sock, sock.getsockname()[1]


def build_background_store(
    listen_socket: Optional[socket.socket],
    port: int,
    timeout: timedelta = DEFAULT_BACKGROUND_STORE_TIMEOUT,
) -> TCPStore:
    """Build the TCPStore for the background checkpoint process group.

    Call inside the spawned background process. Rank 0 adopts ``listen_socket`` via
    ``master_listen_fd`` rather than letting TCPStore bind a port of its own; every other rank
    builds a client store pointed at ``MASTER_ADDR:port``. Pass the result to
    ``init_process_group(store=...)`` -- the store is independent of the collective backend, so
    it works for gloo and nccl alike.

    Args:
        listen_socket: The socket from :func:`reserve_background_store_socket` on rank 0, and
            ``None`` on every other rank.
        port: The reserved port, as broadcast by the parent.
        timeout: How long the store waits for the other ranks to join.

    Returns:
        A TCPStore, serving on rank 0 and connecting as a client elsewhere.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if (listen_socket is not None) != (rank == 0):
        raise ValueError(f"listen_socket must be provided on rank 0 and only rank 0, got rank={rank}")

    # detach() hands the fd over without closing it, so ownership passes cleanly to TCPStore and
    # the socket object cannot close it a second time when it is garbage collected.
    master_listen_fd = listen_socket.detach() if listen_socket is not None else None
    return TCPStore(
        host_name=os.environ["MASTER_ADDR"],
        port=port,
        world_size=world_size,
        is_master=rank == 0,
        timeout=timeout,
        master_listen_fd=master_listen_fd,
    )
