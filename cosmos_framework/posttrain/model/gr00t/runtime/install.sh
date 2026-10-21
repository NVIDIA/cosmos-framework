#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Run in a fresh nvcr.io/nvidia/pytorch:25.09-py3 aarch64 container.
# /build contains immutable source.tar, uv and the build's writable dependency cache.
set -euo pipefail
test "$(uname -m)" = aarch64
export TMPDIR=/build/tmp UV_CACHE_DIR=/build/uv-cache
export UV_PYTHON_INSTALL_DIR=/opt/posttrain/python
export UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" /opt/posttrain/bin /opt/posttrain/vendor /workspace/cosmos-framework
install /build/uv /opt/posttrain/bin/uv
UV=/opt/posttrain/bin/uv
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libgl1 libegl1 libgles2 ffmpeg git build-essential libglib2.0-0
tar -xf /build/source.tar -C /workspace/cosmos-framework
PT=/workspace/cosmos-framework/cosmos_framework/posttrain
export PYTHONPATH=/workspace/cosmos-framework
"$UV" --no-config python install 3.12.13
export UV_PROJECT_ENVIRONMENT=/opt/venv/posttrain
"$UV" sync --project "$PT" --frozen --python 3.12.13 --no-default-groups \
    --group gr00t-cu130-aarch64 --group nixl-cu130-aarch64 --group dev \
    --no-install-package vllm --no-install-package nixl --no-install-package nixl-cu13
MODEL=/opt/venv/posttrain/bin/python
# Match the repository's fixed i4 image without changing model dependency pins.
"$UV" --no-config pip install --python "$MODEL" --no-deps \
    --index-url https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple \
    one-logger==1.5.5 toml==0.10.2
# The same pinned wheel and hash used by Dockerfile.runtime's cu130/arm64 branch.
VLLM_WHEEL=/build/vllm-0.23.0-cp38-abi3-manylinux_2_28_aarch64.whl
curl --fail --location --retry 2 \
    https://wheels.vllm.ai/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm-0.23.0-cp38-abi3-manylinux_2_28_aarch64.whl \
    --output "$VLLM_WHEEL"
printf '%s  %s\n' b597c1c71e8732d751233942675c4d74f623743ac9262a505b256ce2ec97fe05 "$VLLM_WHEEL" | sha256sum --check
"$UV" --no-config pip install --python "$MODEL" --no-deps "$VLLM_WHEEL" nixl==1.3.2 nixl-cu13==1.3.2
"$MODEL" "$PT/docker/normalize_nixl_metadata.py" \
    /opt/venv/posttrain/lib/python3.12/site-packages/nixl-1.3.2.dist-info nixl-cu13
ISAAC_REV=$(python3 -c 'from cosmos_framework.posttrain.embodied.sims.libero.protocol import LIBERO_RUNTIME; print(LIBERO_RUNTIME.isaac_gr00t)')
"$UV" --no-config pip install --python "$MODEL" --no-deps \
    "git+https://github.com/NVIDIA/Isaac-GR00T@$ISAAC_REV" \
    git+https://github.com/vllm-project/vllm-omni@2d0b1d8a29ff686698dbf0a123ca15be20fb0a8f

# The tested RLinf package bytes stay isolated from model-side NumPy/CUDA constraints.
# The release archive includes the known contourpy/NumPy metadata conflict; verify it explicitly.
"$UV" --no-config venv --python 3.12.13 /opt/venv/libero
SIM=/opt/venv/libero/bin/python
sha256sum --check /build/sim-site.tar.sha256
tar -xf /build/sim-site.tar -C /opt/venv/libero/lib/python3.12/site-packages
"$MODEL" "$PT/model/gr00t/runtime/verify_sim.py"
ln -s /resources/libero/assets /opt/venv/libero/lib/python3.12/site-packages/libero/libero/assets
git clone --filter=blob:none --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git /opt/posttrain/vendor/Isaac-GR00T
git -C /opt/posttrain/vendor/Isaac-GR00T checkout --detach "$ISAAC_REV"
"$SIM" - <<'PY'
import json
import sysconfig
from pathlib import Path

from cosmos_framework.posttrain.embodied.sims.libero.protocol import LIBERO_SUITES, fixed_manifest

# This path is internal to the image and exposes only the native simulator wrapper.
Path(sysconfig.get_path("purelib"), "isaac_reference.pth").write_text("/opt/posttrain/vendor/Isaac-GR00T\n")
root = Path("/opt/posttrain/libero")
(root / "config").mkdir(parents=True)
(root / "manifests").mkdir()
assets = Path("/resources/libero")
(root / "config/config.yaml").write_text("\n".join([
    f"benchmark_root: {assets}", f"bddl_files: {assets / 'bddl_files'}",
    f"init_states: {assets / 'init_files'}", f"assets: {assets / 'assets'}",
    "datasets: /resources/datasets", "",
]))
(root / "egl-nvidia.json").write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}))
for suite in LIBERO_SUITES:
    for split in ("train", "eval"):
        fixed_manifest(split, suite, "legacy_100").write(root / "manifests" / f"{suite}-{split}.json")
PY
install -m 644 "$PT/model/gr00t/runtime/env.sh" /opt/posttrain/env.sh
"$UV" --no-config pip freeze --python "$MODEL" > /opt/posttrain/model-packages.txt
"$UV" --no-config pip freeze --python "$SIM" > /opt/posttrain/sim-packages.txt
cp /build/source-manifest.json /opt/posttrain/source-manifest.json
cp /build/build-input.json /opt/posttrain/build-input.json
echo "GR00T runtime filesystem assembled"
