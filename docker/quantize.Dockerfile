# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

# Extend a locally built Cosmos Framework image. Quantization has a separate
# dependency overlay, loaded only by cosmos-reasoner-quantize.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY docker/requirements-quantize.txt /opt/cosmos/requirements-quantize.txt
# The Framework base already supplies CUDA/PyTorch, datasets, accelerate, PyAV,
# and the other shared prerequisites. Never resolve their replacements here.
# uv is shipped in that base; pip need not be installed in its managed venv.
RUN uv pip install --no-deps --target /opt/quantize_deps -r /opt/cosmos/requirements-quantize.txt && \
    LD_LIBRARY_PATH='' python -c 'from cosmos_framework.scripts.quantize_reasoner import _load_quantization_dependencies; _load_quantization_dependencies()' && \
    chmod -R a+rX /opt/quantize_deps
