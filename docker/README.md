# Docker

The end-to-end Docker setup — base image, build command, `docker run` invocation, and NVIDIA Container Toolkit configuration — is documented in [docs/setup.md → Docker Container](../docs/setup.md#docker-container).

This directory contains:

- `ci.Dockerfile` — minimal CI image (CUDA base + uv + just + Python 3.13 + pre-commit); not intended for local training or inference.
- `nightly.Dockerfile` — full runtime image built on the NGC PyTorch base (`nvcr.io/nvidia/pytorch:${BASE_VERSION}-py3`) with all dependencies installed via `uv pip install -r pyproject.toml --all-extras`. Use this for nightly local builds when the recommended base image's pinned PyTorch version is too old.
- `entrypoint.sh` — container entrypoint; executes the command without mutating the packaged Python environment.

The root `Dockerfile` writes source hashes and runtime versions to
`/opt/cosmos/image-provenance.json` and `/opt/cosmos/source-manifest.sha256`.
A default build has unverified source identity. For a reproducibility image,
supply `SOURCE_COMMIT`, `SOURCE_TREE`, `BUILD_TIMESTAMP`, and `SOURCE_DIRTY=0`
as build arguments only after verifying a clean checkout. Partial metadata or
a dirty source claim fails the build. `REQUIRE_SOURCE_PROVENANCE=1` also makes
missing metadata fatal, so release wrappers can enforce this contract explicitly.
