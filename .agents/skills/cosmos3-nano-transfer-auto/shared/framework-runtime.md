# Environment and model assets

Run from a Cosmos Framework checkout containing `docs/training_transfer_auto.md`,
`docs/training_sim_auto.md` and `cosmos_framework/scripts/train_experiment.py`.
Follow its `docs/setup.md` for the container, system packages and GPU driver.
The commands use the repository's Python 3.13 training environment. Install the
required `uv` version (`0.11.29`) and system packages from the setup guide first.
Choose CUDA 13 on a compatible driver, or the CUDA 12.8 group on older drivers
without a supported forward-compatibility setup.

```bash
uv sync --locked --python 3.13 --group cu130-train --extra train --extra guardrail
source .venv/bin/activate
export COSMOS_INTERNAL=0
```

The training extra is also needed by imported inference utilities. The separate
guardrail extra supplies `nltk` and the other dependencies used by default
inference; `--extra train` alone does not install them. Keep the checkout's
lockfile: it already specifies FlashAttention-4. Do not layer an
unrelated attention/CUTLASS upgrade on top and assume numerical parity.
Verify PyTorch, TorchCodec/FFmpeg, CUDA and attention imports. Runtime compatibility
does not establish memory capacity for a particular training or inference request.
Also run the setup guide's GPU-allocation and video-decoding checks. Preserve
required CUDA compatibility library paths; do not reset `LD_LIBRARY_PATH` merely
because the environment is a container.

Training needs a multi-GPU allocation, working NCCL, and dataset/checkpoint paths
accessible from every rank. Set `IMAGINAIRE_OUTPUT_ROOT` to your training-output
directory and export it so worker processes can read it. Set and export
`HF_HOME` and `UV_CACHE_DIR` for your caches if needed. Keep secrets
in your normal credential mechanism, never in manifests.

## Training memory

Size training separately from inference. Training needs gradients, optimizer
state and activations for backward computation in addition to model weights;
distillation stages can also hold teacher and fake-score networks. Memory needs
depend on the training phase, camera count, resolution, clip length and packed
token budget, not GPU count alone.

Context parallelism distributes sequence computation; FSDP shards model and
optimizer state. Increasing CP does not replace FSDP state sharding. Activation
checkpointing trades extra computation for lower activation memory, while CPU
activation offload trades GPU memory for host memory and transfer overhead.
Choose the topology and memory settings through the training guide, then check
forward, backward, checkpoint save and reload at the intended workload. Passing
a short training check does not establish memory fit for longer or larger batches.

## Checkpoint memory

GPU memory, host RAM and `/dev/shm` capacity are separate constraints. Inspect
`free -h` and `df -h /dev/shm` inside the training container. Asynchronous DCP
saves stage model and optimizer state in pinned shared CPU memory on each rank;
enough VRAM does not ensure saving will fit.

For short train/save/resume checks use
`checkpoint.dcp_async_mode_enabled=false`. This avoids the background pinned
shared-memory staging path, but still requires host RAM and normal dataloader
shared memory. Before enabling async saves for a longer run, provision host RAM
and container shared memory for all ranks' checkpoint state, then complete a
save and reload. Use the setup guide's Docker IPC options where permitted.
Do not treat a directory appearing on disk as a completed checkpoint.

## Model snapshots

| Workflow                                  | Model repository                    | Local variable     |
| ----------------------------------------- | ----------------------------------- | ------------------ |
| Curriculum base                           | `nvidia/Cosmos3-Nano`               | `NANO_HF`          |
| Transfer inference, fine-tuning, Sim base | `nvidia/Cosmos3-Nano-Transfer-Auto` | `TRANSFER_AUTO_HF` |
| Released Sim inference                    | `nvidia/Cosmos3-Nano-Sim-Auto`      | `SIM_AUTO_HF`      |

Use an authorized complete snapshot. Obtain its immutable revision from the model
distribution and set the corresponding `*_REVISION` variable before downloading;
do not invent a revision or treat `main` as immutable. For example:

```bash
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --revision "${TRANSFER_AUTO_REVISION:?set the chosen snapshot revision}" \
  --local-dir "${TRANSFER_AUTO_HF:?set a local snapshot directory}"
```

Use the same pattern for Nano or Sim-Auto when needed. Accept applicable licenses
and authenticate through `hf auth login` or `HF_TOKEN` without printing the token.
A repository ID is not a local checkpoint directory. Keep config, weights and
tokenizers from the same snapshot.

## Training components

- Convert main-model HF weights to DCP using the post-training skill. This is
  weights-only initialization, not restoration of optimizer history.
- Use the matched RGB VAE: the recipes expect a Wan2.2 VAE `.pth`, distributed
  in `Wan-AI/Wan2.2-TI2V-5B`.
- Keep `lidar_vae/config.json` and `diffusion_pytorch_model.safetensors` from the
  matching Auto snapshot together. Do not convert this tokenizer to DCP.
- Use the matching local text tokenizer directory, normally `text_tokenizer/`
  in the snapshot, when building training data.

The VAE is a supporting component, not an alternative model starting point.
Inspect the actual snapshot layout and bind paths explicitly. A changed LiDAR
sensor/projection needs more than a same-shaped tokenizer input.

## Execution modes

Set `COSMOS_TRAINING=1` for training/config resolution and `COSMOS_TRAINING=0`
for inference. Use `cosmos_framework.scripts.train_experiment` for Auto training;
generic single-video SFT examples do not implement these sensor recipes.
Use `cosmos_framework.scripts.inference` for both Auto inference paths.

Retain the Framework revision, lockfile and snapshot revision with results so
you can reproduce the environment when comparing outputs or resuming training.
