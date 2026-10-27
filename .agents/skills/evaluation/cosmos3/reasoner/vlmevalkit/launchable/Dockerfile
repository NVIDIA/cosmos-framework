# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg git libgl1 libglib2.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/vlmevalkit
COPY requirements.txt setup.py setup.cfg README-vlmevalkit.md ./
# Evaluation is API-based; keep its PyTorch stack off the serving GPU.
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt einops accelerate 'setuptools<81' evaluate jupyterlab==4.4.10
COPY . .
RUN pip install --no-cache-dir --no-deps -e . \
    && python -c 'import vlmeval' \
    && pip freeze > /opt/evaluation-packages.txt
ARG COSMOS_COMMIT=unknown
ENV COSMOS_COMMIT=${COSMOS_COMMIT} PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface \
    AWS_CONFIG_FILE=/dev/null AWS_SHARED_CREDENTIALS_FILE=/dev/null \
    AWS_EC2_METADATA_DISABLED=true SCORING_ASSETS_DIR=/scoring-assets LMUData=/cache/lmudata
WORKDIR /opt/vlmevalkit/launchable
CMD ["python", "jupyter_config.py"]
