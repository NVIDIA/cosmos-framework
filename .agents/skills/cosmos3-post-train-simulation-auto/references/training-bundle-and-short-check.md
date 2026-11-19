# One daytime scene for both Auto models

> **Skill note (not in the source):** the other sections of this page (get the scene, files and meanings, caption styles, inference requests, own data and verification) moved to the [Auto inference skill's reference scene](../../cosmos3-inference-simulation-auto/references/reference-scene.md); `AUTO_SAMPLE_ROOT` and the edited index copy named below are set there.

## Prepare the training bundle

For the original paragraph captions, use the supplied index below. For timestamped
captions, keep `AUTO_INDEX` pointing to the edited copy described above.

```bash
export AUTO_INDEX="${AUTO_SAMPLE_ROOT:?set the local scene directory}/training-index.jsonl"
```

Run [Framework's native bundle builder](../SKILL.md#prepare-a-local-training-bundle)
with the seven camera keys and your matched text tokenizer. This produces
`manifest.json`, `samples.json`, `assets/` and `resources/tokenizer/` in
a new directory. Both models' configuration bindings select this same bundle
format. No sample-specific dataset class or new adapter is used.

The files have different jobs:

| File                                       | Reader                       | Purpose                                               |
| ------------------------------------------ | ---------------------------- | ----------------------------------------------------- |
| `input-*-auto.jsonl`                       | Native inference entry point | Controls, captions, initial RGB and sampling settings |
| `training-index.jsonl`                     | Native `auto_bundle` builder | Pairs the existing real targets and controls          |
| Generated `manifest.json` / `samples.json` | Native training loader       | Packaged members, tokenizer and transform settings    |

## Short training check

This sample has **720p videos and 300 source frames**. The short check below
uses 21 aligned RGB frames and the native rig sampler's camera subsets. It checks
data loading, parameter updates and checkpoint saving; evaluate model quality
after full training on your corpus. The full curriculum needs both resolution
strata and recordings long enough for its 400-source-frame windows.

For a bounded check of **Transfer-Auto direct fine-tuning** on one four-H200
node, start with CP=2 and FSDP=2. CP divides the sequence work; FSDP shards model
and optimizer state. CP=4/FSDP=1 leaves that state unsharded and is not an
equivalent memory-saving layout. These degrees describe overlapping meshes,
not two dimensions that must always multiply to the GPU count.

> **Skill note (not in the source):** measured on four H100 NVL 95 GB: at CP=2 and FSDP=2 this check runs out of GPU memory in the forward pass of the first update on all four processes (`torch.OutOfMemoryError` with 85.4 GiB allocated by PyTorch per GPU; no loss printed, no checkpoint written). With FSDP=4 and nothing else changed it passes: losses 0.5182 and 0.5027, a synchronous save of 117 s, 283 s for the whole launch, and a peak of about 95,000 MiB of 95,830 MiB per GPU in `nvidia-smi`, which includes memory the allocator holds in reserve. The block below therefore selects FSDP=4 by itself when the smallest GPU reports less than 140,000 MiB and keeps FSDP=2 otherwise. GPUs with less memory than these are not measured.

After constructing the fine-tuning `settings` array, append:

```bash
sample_stream=dataloader_train.dataloaders.joint_720
settings+=(
  '~dataloader_train.dataloaders.joint_480'
  "${sample_stream}.ratio=1.0"
  "${sample_stream}.dataloader.dataset.transform_overrides.num_video_frames=24"
  "${sample_stream}.dataloader.dataset.transform_overrides.joint_aligned_num_video_frames=21"
  trainer.max_iter=2
  checkpoint.save_iter=2
  checkpoint.dcp_async_mode_enabled=false
  model_parallel.context_parallel_size=2
  model.config.parallelism.context_parallel_shard_degree=2
  model.config.parallelism.data_parallel_shard_degree=2
)
# Skill note (not in the source): FSDP=4 when the smallest GPU reports less than 140000 MiB; appended last, it replaces the 2 above.
if [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)" -lt 140000 ]; then settings+=(model.config.parallelism.data_parallel_shard_degree=4); fi
python -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" --dryrun -- "${settings[@]}"
torchrun --standalone --nproc-per-node=4 \
  -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" -- "${settings[@]}"
```

This is a short 720p check with the bundle's camera-subset sampler, not a
guarantee for the full-length or fixed-seven-camera workload. Validate peak GPU
memory on your allocation. The save is synchronous; see
[checkpoint memory](runtime.md#checkpoint-memory) before enabling
asynchronous saves. Require finite losses, a completed checkpoint, and a
successful same-run reload before treating the check as passed.

> **Skill note (not in the source):** the same-run reload is the launch command once more, with the same `job.name`, `IMAGINAIRE_OUTPUT_ROOT` and `settings`. The run then finds its own `checkpoints/latest_checkpoint.txt` and restores that checkpoint with its training state in place of the converted weights; the saved iteration already equals `trainer.max_iter`, so it ends without another update or save. Measured on four H100 NVL 95 GB with FSDP=4: 66 s, a peak of about 51,600 MiB per GPU. The three lines that the `grep` below prints are the evidence of a successful reload.

```bash
torchrun --standalone --nproc-per-node=4 -m cosmos_framework.scripts.train_experiment "${entry[@]}" -- "${settings[@]}" \
  > "${IMAGINAIRE_OUTPUT_ROOT:?}/${AUTO_RUN_NAME:?}.reload.log" 2>&1
grep -E 'Resuming ckpt .*iter_000000002 \(same-job, local\)|Loaded checkpoint from .*iter_000000002 \(same-job, local\) in iteration 2|Done with training' \
  "$IMAGINAIRE_OUTPUT_ROOT/$AUTO_RUN_NAME.reload.log"
```

For **Sim-Auto Stage 1 TF**, first apply the
[bundle camera-map overrides](sim-auto-data.md#preserve-your-bundles-camera-mappings).
Append these sample overrides **after** that array in the
[stage launch command](sim-auto-runtime.md):

```bash
sample_stream=dataloader_train.dataloaders.cam_lidar_joint_720p_separate_view_text
sample_overrides=(
  '~dataloader_train.dataloaders.cam_lidar_joint_480p_separate_view_text'
  "${sample_stream}.ratio=1.0"
  "${sample_stream}.dataloader.dataset.transform_overrides.num_video_frames=24"
  "${sample_stream}.dataloader.dataset.transform_overrides.joint_aligned_num_video_frames=21"
  trainer.max_iter=2
  checkpoint.save_iter=2
  checkpoint.dcp_async_mode_enabled=false
)
settings+=("${sample_overrides[@]}")
```

> **Skill note (not in the source):** for a short first run, the order that resolved and trained is: the stage selection, the bundle camera-map overrides, the [smoke settings](sim-auto-smoke.md#prepare-a-reduced-source), then the block above as the last addition to `settings`, so that its 720p stream replaces the 480p stream that the smoke loop selects. On one node with four GPUs the launcher variables are the ones in the block below: with the first address of `hostname -I` as `MASTER_ADDR` the launcher did not reach the rendezvous endpoint from inside a container (`RendezvousConnectionError` after 92 s, before any training process started), with `127.0.0.1` it did. Measured on four H100 NVL 95 GB at CP4/FSDP4: the two-update launch with its synchronous save took 345 s (losses 0.9585 and 0.7980), the resume to iteration 3 took 294 s (loss 0.7485), and both peaked at about 90,000 MiB of 95,830 MiB per GPU in `nvidia-smi`.

```bash
export NNODES=1 GPUS_PER_NODE=4 NODE_RANK=0 MASTER_ADDR=127.0.0.1 CONTEXT_PARALLEL=4 FSDP_SHARDS=4
```

Apply this after constructing `settings` and before the first dry run.
These are reduced data settings, not changes to the published training recipe.
For Sim-Auto allocation and all three stages, follow the
[bounded train/save/resume guide](sim-auto-smoke.md).
Do not overwrite the original full-run config.
A two-update smoke does not establish convergence, resume correctness or quality.
Use additional, longer recordings for the full curriculum and keep validation
drives separate from training drives.
