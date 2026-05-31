# Docker

The end-to-end Docker setup — base image, build command, `docker run` invocation, and NVIDIA Container Toolkit configuration — is documented in [docs/setup.md → Docker Container](../docs/setup.md#docker-container).

This directory contains:

- `ci.Dockerfile` — minimal CI image (CUDA base + uv + just + Python 3.13 + pre-commit); not intended for local training or inference.
- `nightly.Dockerfile` — full runtime image built on the NGC PyTorch base (`nvcr.io/nvidia/pytorch:${BASE_VERSION}-py3`) with all dependencies installed via `uv pip install -r pyproject.toml --all-extras`. Use this for nightly local builds when the recommended base image's pinned PyTorch version is too old.
- `entrypoint.sh` — container entrypoint; runs `uv pip install --no-deps -e .` before exec'ing the container command.
