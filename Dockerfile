# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Dockerfile using uv environment.

ARG CUDA_VERSION=13.0.2
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04
FROM ${BASE_IMAGE}

# Set the DEBIAN_FRONTEND environment variable to avoid interactive prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        git \
        git-lfs \
        tree \
        wget

# Install uv: https://docs.astral.sh/uv/getting-started/installation/
# https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
COPY --from=ghcr.io/astral-sh/uv:0.10.8 /uv /uvx /usr/local/bin/
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
# Cache python downloads
ENV UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

# Install just: https://just.systems/man/en/pre-built-binaries.html
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin --tag 1.46.0

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Install python
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv python install

# Install into virtual environment
RUN echo "$CUDA_VERSION" | sed -E 's/^([0-9]+)\.([0-9]+).*/cu\1\2/' > /root/.cuda-name
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=packages,target=packages \
    uv sync --locked --no-install-project --no-editable --all-extras --group=$(cat /root/.cuda-name)
ENV PATH="/workspace/.venv/bin:$PATH"

# Triton bundled ptxas doesn't support latest GPU architectures
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]

CMD ["/bin/bash"]
