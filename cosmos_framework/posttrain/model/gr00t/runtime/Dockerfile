# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Context: the release staging directory, containing source.tar, uv and build-input.json.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}
COPY source.tar source-manifest.json build-input.json uv install.sh /build/
COPY sim-site.tar sim-site.tar.sha256 sim-files.json sim-packages.json /build/
RUN bash /build/install.sh
WORKDIR /workspace/cosmos-framework
CMD ["/bin/bash"]
