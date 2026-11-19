#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage local files/dirs to a remote Lustre path over SSH/SCP for SLURM Cosmos jobs."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _ssh_base() -> list[str]:
    user = os.environ["SLURM_USER"]
    host = os.environ["SLURM_HOSTNAME"].split(",")[0]
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
    ]
    key = os.environ.get("SSH_KEY_PATH")
    if key:
        cmd += ["-i", str(Path(key).expanduser()), "-o", "IdentitiesOnly=yes"]
    return cmd, f"{user}@{host}"


def remote_ls(path: str) -> None:
    ssh, target = _ssh_base()
    r = subprocess.run(
        ssh + [target, f"ls -la {shlex.quote(path)} 2>&1 | head -40"], capture_output=True, text=True, timeout=30
    )
    print(r.stdout or r.stderr)


def remote_mkdir(path: str) -> None:
    ssh, target = _ssh_base()
    r = subprocess.run(ssh + [target, f"mkdir -p {shlex.quote(path)}"], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print(f"mkdir failed: {r.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Created: {path}")


def scp_upload(local: str, remote_path: str) -> None:
    _, target = _ssh_base()
    key = os.environ.get("SSH_KEY_PATH")
    scp_cmd = ["scp", "-r", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes"]
    if key:
        scp_cmd += ["-i", str(Path(key).expanduser())]
    scp_cmd += [local, f"{target}:{remote_path}"]
    print(f"Uploading {local} → {remote_path} ...")
    r = subprocess.run(scp_cmd, timeout=600)
    if r.returncode != 0:
        print(f"scp failed with rc={r.returncode}", file=sys.stderr)
        sys.exit(1)
    print("Done.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ls").add_argument("path")
    mk = sub.add_parser("mkdir")
    mk.add_argument("path")
    up = sub.add_parser("upload")
    up.add_argument("local")
    up.add_argument("remote")
    args = p.parse_args()
    if args.cmd == "ls":
        remote_ls(args.path)
    elif args.cmd == "mkdir":
        remote_mkdir(args.path)
    elif args.cmd == "upload":
        scp_upload(args.local, args.remote)


if __name__ == "__main__":
    main()
