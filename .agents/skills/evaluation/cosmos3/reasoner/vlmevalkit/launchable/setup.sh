#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
command -v docker >/dev/null
command -v python3 >/dev/null
docker compose version
docker info >/dev/null
export COSMOS_COMMIT
COSMOS_COMMIT=$(git rev-parse HEAD)
: "${LAUNCHABLE_STATE_DIR:?Set an absolute state directory outside the Framework checkout}"
export MVBENCH_DATA_DIR="${MVBENCH_DATA_DIR:-$LAUNCHABLE_STATE_DIR/mvbench}"
export LAUNCHABLE_STATE_DIR
python3 - "$LAUNCHABLE_STATE_DIR" "$MVBENCH_DATA_DIR" "$(git rev-parse --show-toplevel)" <<'PY'
import os
from pathlib import Path
import sys

checkout = Path(sys.argv[3]).resolve()
state, data = (Path(value) for value in sys.argv[1:3])
for path in (state, data):
    if not path.is_absolute() or path.resolve().is_relative_to(checkout):
        raise SystemExit("State and MVBench data must use absolute paths outside the Framework checkout")
if any(data.resolve().is_relative_to((state / name).resolve())
       or (state / name).resolve().is_relative_to(data.resolve()) for name in ("cache", "results")):
    raise SystemExit("MVBench data must be separate from writable cache/results mounts")
for path in (state / "cache", state / "results", data):
    path.mkdir(parents=True, exist_ok=True)
    if not os.access(path, os.W_OK):
        raise SystemExit(f"Runtime directory must be writable: {path}")
PY
export EVAL_SCORE_ONLY=0
for argument in "$@"; do
    if [[ "$argument" == "--score-only" ]]; then EVAL_SCORE_ONLY=1; fi
done
# Persist notebook authentication and its non-secret bind address, never model/judge credentials.
export JUPYTER_BIND_ADDRESS
JUPYTER_BIND_ADDRESS=$(python3 - "$@" <<'PY'
import ipaddress
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

if set(sys.argv[1:]) - {"--brev", "--score-only"}:
    raise SystemExit("Usage: bash setup.sh [--brev] [--score-only]")
path = Path(".env")
lines = path.read_text().splitlines() if path.exists() else [
    "JUPYTER_TOKEN=" + secrets.token_hex(32)
]
stored = dict(line.split("=", 1) for line in lines if line.startswith("JUPYTER_BIND_ADDRESS="))
address = os.environ.get("JUPYTER_BIND_ADDRESS") or stored.get("JUPYTER_BIND_ADDRESS", "127.0.0.1")
if "--brev" in sys.argv[1:] and not os.environ.get("JUPYTER_BIND_ADDRESS"):
    try:
        interfaces = json.loads(subprocess.check_output(
            ["ip", "-j", "-4", "addr", "show", "dev", "wt0"], timeout=5))
        addresses = [item["local"] for interface in interfaces for item in interface["addr_info"]
                     if item["family"] == "inet" and item.get("scope") == "global"]
        if len(addresses) != 1:
            raise ValueError("expected exactly one global wt0 IPv4 address")
        address = addresses[0]
        if ipaddress.IPv4Address(address) not in ipaddress.IPv4Network("100.64.0.0/10"):
            raise ValueError("wt0 is not in the private Brev mesh range")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        raise SystemExit(f"Brev private notebook binding unavailable: {error}") from error
allowed = ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10")
if not any(ipaddress.IPv4Address(address) in ipaddress.IPv4Network(network) for network in allowed):
    raise SystemExit("JUPYTER_BIND_ADDRESS must be a loopback or private IPv4 address, never a public/wildcard address")
lines = [line for line in lines if not line.startswith("JUPYTER_BIND_ADDRESS=")]
path.write_text("\n".join(lines + ["JUPYTER_BIND_ADDRESS=" + address]) + "\n")
print(address)
PY
)
chmod 600 .env
docker compose config --quiet
docker compose build mvbench-eval
if [[ "$EVAL_SCORE_ONLY" == "0" && -z "${COSMOS_API_BASE:-}" ]]; then
    nvidia-smi
    export COSMOS_MODEL_REVISION="${COSMOS_MODEL_REVISION:-e59a53c25979a090fa8706c9acc0c254a6e89b92}"
    docker compose --profile local pull cosmos-endpoint
    docker compose --profile local up -d
else
    # Existing-prediction scoring and BYOM never start a local model/GPU.
    export COSMOS_MODEL_REVISION="${COSMOS_MODEL_REVISION:-unknown}"
    docker compose up -d mvbench-eval
fi
printf 'Jupyter is starting on %s:8888. Open its authenticated secure link or SSH tunnel.\n' "${JUPYTER_BIND_ADDRESS:-127.0.0.1}"
printf '%s\n' \
    'Retrieve the notebook login token with: docker compose exec mvbench-eval jupyter server list' \
    'Open Evaluate_Cosmos3.ipynb to download public MVBench data, prepare, infer, and score.' \
    'Full evaluation also requires manually acquired authorized NTU RGB+D media; see the notebook.'
