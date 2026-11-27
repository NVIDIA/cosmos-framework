# Environment and model assets

> **Skill note (not in the source):** the environment setup, model snapshots and execution modes sections of this page moved to the [Auto inference skill's environment and model assets](../../cosmos3-inference-simulation-auto/references/framework-runtime.md); the setup guide named below is introduced there. The paragraph below is the training-only close of the environment setup.

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
