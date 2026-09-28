---
name: cosmos3-setup
description: >
  Guide users through Cosmos3 installation, environment setup, checkpoint downloading,
  and verification. Use when the user asks "how do I install cosmos3", "how do I set up
  the environment", "how do I download checkpoints", "how do I use Docker", or any
  question about getting the package running for the first time. Also use for
  managed workflow platform selection, credential preflight, and GPU-host checks.
---

# Cosmos3 Setup

## When to use this skill

- Use when a user wants to install Cosmos3 or set up a development environment
- Use when a user asks about system requirements, CUDA versions, or GPU compatibility
- Use when a user needs to download model checkpoints or configure HuggingFace auth
- Use when a user wants to run Cosmos3 inside a Docker container or NGC container
- For errors during setup, hand off to the **cosmos3-env-troubleshoot** skill

## Path convention

All paths below are relative to the cosmos3 package root (`../../../` from this skill file). All `uv run` / `python` commands should also be run from there.

## Where to find answers

The canonical setup reference is `docs/setup.md`. The README (`README.md` § Setup) has the shortest quickstart.

| User question                                          | Go to                                                                |
| ------------------------------------------------------ | -------------------------------------------------------------------- |
| What are the system requirements?                      | `docs/setup.md` § System Requirements                                |
| How do I install with uv? (sync, pip venv, pip system) | `docs/setup.md` § Virtual Environment                                |
| How do I install with Docker?                          | `docs/setup.md` § Docker Container                                   |
| Custom torch/CUDA versions or attention backends?      | `docs/setup.md` § Advanced                                           |
| Which CUDA version? (cu130 vs cu128)                   | `docs/setup.md` § CUDA Variants, `docs/faq.md` § Which CUDA version? |
| How do I download checkpoints?                         | `docs/setup.md` § Downloading Base Checkpoints                       |
| NGC container issues?                                  | `docs/setup.md` § PyTorch Import Issue                               |
| Any installation error                                 | `../cosmos3-env-troubleshoot/SKILL.md`                               |

## Setup steps at a glance

### Managed workflow execution

Native installation still follows the steps below. For a managed reasoner,
Embed, Predict, or annotation action, shared helpers live in
`tools/cosmos_workflows`; no Skill Bank plugin or separate skill registry is needed.
From the repository root, `source tools/cosmos_workflows/env.sh` sets
`COSMOS_WORKFLOWS_ROOT`. Re-source it in each shell call that needs it; commands
inside the following references are relative to that helper root unless stated
otherwise. Read only the references needed for the selected action.

- [Launch review and job lifecycle](../../../tools/cosmos_workflows/execution/guide.md):
  before managed submission, resolve the model/backend/action, select the
  platform, preflight, review the concrete run, and obtain launch approval.
  An explicit Framework request remains Framework. Planning is not approval
  for downloads, image builds/pulls, GPU jobs, or host changes.
- Platform procedures: [Docker](../../../tools/cosmos_workflows/execution/platforms/docker/guide.md),
  [SLURM](../../../tools/cosmos_workflows/execution/platforms/slurm/guide.md),
  [Kubernetes](../../../tools/cosmos_workflows/execution/platforms/kubernetes/guide.md),
  [Brev](../../../tools/cosmos_workflows/execution/platforms/brev/guide.md), and
  [virtualenv](../../../tools/cosmos_workflows/execution/platforms/virtualenv/guide.md).
  Use the requested supported platform. If several fit and none was selected,
  ask once; a single-platform workflow already determines the choice.
- [Execution artifacts](../../../tools/cosmos_workflows/execution/artifacts/guide.md):
  nested specs, fixed status vocabulary, and job records bound before submit.
- [Data staging](../../../tools/cosmos_workflows/execution/data-io/guide.md):
  only when compute cannot already read the inputs; pre-positioned mounts need
  no download. Outputs use the job record's `results_dir`.
- [GPU-host checks](../../../tools/cosmos_workflows/execution/gpu-host/guide.md):
  only for host preparation or a missing prerequisite. Inspect first, preserve
  model-specific minimums, and obtain approval before installation.

Check credential presence only. Never print token values, use `printenv` to
inspect secrets, or read credential-file contents. Use session variables or a
user-approved env file sourced in the consuming shell. Missing registry access
does not authorize substituting a different model image.

### Native installation

1. **Clone** the repository and `cd` into the project root (the directory containing `pyproject.toml`)
2. **System deps**: `sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libgl1 libglib2.0-0 libx11-dev tree wget`
3. **Install uv**: `curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh; export PATH="$HOME/.local/bin:$PATH"`. Verify `uv --version` matches `tool.uv.required-version`; update an existing installation with `uv self update 0.11.29`.
4. **Install package**: `uv sync --locked --all-extras --group=cu130-train && source .venv/bin/activate` (use `cu128-train` on older drivers without a supported CUDA 13 compatibility setup; the inference-only `cu130` / `cu128` groups omit the training extras)
5. **Checkpoints**: auto-downloaded during inference; requires HuggingFace auth (see docs)
6. **Verify**: follow `docs/setup.md` § Verify the GPU and media runtime. Check CUDA initialization and actual video decoding, not just package imports.

## Things not obvious from the docs

- **NGC container caveat**: inherited libraries can conflict with PyTorch, but clearing `LD_LIBRARY_PATH` can also remove required CUDA compatibility libraries. Follow the setup guide's conditional troubleshooting instead of clearing it automatically.
- **CUDA compatibility**: compare the host driver with the selected CUDA build's requirements. `nvidia-smi` reports driver capability, not the installed toolkit; its CUDA version need not equal `torch.version.cuda`.
- **`HF_HOME`**: controls where checkpoints are cached (default: `~/.cache/huggingface`). Set this if disk space is tight or you want a shared cache.
- **Conflicting env vars**: stale `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` env vars can silently override CLI auth. Check presence only, for example `[ -n "${HF_TOKEN:-}" ] && echo HF_TOKEN_SET`; never print their values.
- **AutoML is a separate source install**: `uv sync --all-extras` does not install
  the core engine. For AutoML setup, follow `docs/automl.md#installation-from-source`
  and the `cosmos-automl` skill: ask whether to use `NVIDIA-TAO/tao-automl`'s `main`
  branch or a user-selected development branch before a new installation. Keep
  the recorded source revision for existing experiments; do not upgrade on resume.

## Related skills

| Skill                                  | When to use                                    |
| -------------------------------------- | ---------------------------------------------- |
| `../cosmos3-inference/SKILL.md`        | Running inference after setup is complete      |
| `../cosmos3-codebase-nav/SKILL.md`     | Finding files, parameters, and configs in code |
| `../cosmos3-env-troubleshoot/SKILL.md` | Debugging environment and runtime errors       |
