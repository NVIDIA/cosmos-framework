#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Usage: UV=/path/to/uv bash install.sh /new/prefix /path/to/cosmos_framework.whl
# Install inside a CUDA 12.8-capable Linux x86_64 container; never mutate its base Python.
set -Eeuo pipefail
if [ "$#" -ne 2 ]; then
    echo "usage: $0 /new/prefix /path/to/cosmos_framework.whl" >&2
    exit 2
fi
test "$(uname -m)" = x86_64
test "$(uname -s)" = Linux
FLOW_PREFIX=$(realpath -m "$1")
FLOW_WHEEL=$(realpath "$2")
FLOW_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
UV=${UV:-uv}
test -f "$FLOW_WHEEL"
test ! -e "$FLOW_PREFIX" || { echo "refusing existing runtime: $FLOW_PREFIX" >&2; exit 1; }
mkdir -p "$FLOW_PREFIX"
export UV_PYTHON_INSTALL_DIR="$FLOW_PREFIX/python" UV_LINK_MODE=copy NO_ALBUMENTATIONS_UPDATE=1
"$UV" --no-config python install 3.12.13
"$UV" --no-config venv "$FLOW_PREFIX/venv" --python 3.12.13
FLOW_PYTHON="$FLOW_PREFIX/venv/bin/python"
# Resolve no new versions. The lock is the complete installed closure from the
# tested runtime. Separate indexes identify the exact CUDA and CPU wheel flavors.
"$FLOW_PYTHON" - "$FLOW_SOURCE/requirements.lock" "$FLOW_PREFIX/pypi.lock" <<'PY'
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text().splitlines()
Path(sys.argv[2]).write_text("\n".join(x for x in lines if not x.startswith(("torch==", "torchvision==", "torchcodec=="))) + "\n")
PY
"$UV" --no-config pip install --python "$FLOW_PYTHON" --no-deps -r "$FLOW_PREFIX/pypi.lock"
"$UV" --no-config pip install --python "$FLOW_PYTHON" --no-deps \
    torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
"$UV" --no-config pip install --python "$FLOW_PYTHON" --no-deps \
    torchcodec==0.10.0 --index-url https://download.pytorch.org/whl/cpu
for pair in \
    NVIDIA/Isaac-GR00T@51d4c89f72fda44cbf77285c6a8114b52676b8a1 \
    NVlabs/alpamayo@11a0e01c13a5622377c45ee37d653351453ec43b; do
    FLOW_REPOSITORY=${pair%@*}
    FLOW_REVISION=${pair#*@}
    FLOW_CHECKOUT="$FLOW_PREFIX/sources/${FLOW_REPOSITORY##*/}"
    git clone --filter=blob:none --no-checkout "https://github.com/$FLOW_REPOSITORY.git" "$FLOW_CHECKOUT"
    git -C "$FLOW_CHECKOUT" checkout --detach "$FLOW_REVISION"
    test "$(git -C "$FLOW_CHECKOUT" rev-parse HEAD)" = "$FLOW_REVISION"
    "$UV" --no-config pip install --python "$FLOW_PYTHON" --no-deps "$FLOW_CHECKOUT"
done
"$UV" --no-config pip install --python "$FLOW_PYTHON" --no-deps "$FLOW_WHEEL"
"$UV" --no-config pip freeze --python "$FLOW_PYTHON" > "$FLOW_PREFIX/installed.lock"
"$FLOW_PYTHON" - "$FLOW_SOURCE/requirements.lock" "$FLOW_WHEEL" "$FLOW_PREFIX/runtime.json" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

from cosmos_framework.posttrain.model.alpamayo.policy import external_identity as alpamayo_identity
from cosmos_framework.posttrain.model.gr00t.sft_policy import external_identity as gr00t_identity
from cosmos_framework.posttrain.model.pi05.modeling import Pi05Model

lock, wheel, output = map(Path, sys.argv[1:])
for line in lock.read_text().splitlines():
    if line and not line.startswith("#"):
        name, version = line.split("==", 1)
        assert metadata.version(name) == version, (name, metadata.version(name), version)
record = {
    "format": "cosmos-flow-sft-runtime-v1",
    "python": sys.version,
    "framework_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
    "requirements_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
    "alpamayo": alpamayo_identity(),
    "gr00t": gr00t_identity(),
    "pi05_import": Pi05Model.__name__,
}
output.write_text(json.dumps(record, indent=2) + "\n")
PY
echo "Runtime created: $FLOW_PREFIX/venv/bin/python; provenance: $FLOW_PREFIX/runtime.json"
