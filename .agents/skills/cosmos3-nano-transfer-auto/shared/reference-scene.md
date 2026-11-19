# One daytime scene for both Auto models

> **Skill note (not in the source):** the other sections of this page (get the scene, files and meanings, caption styles, inference requests, own data and verification) moved to the [Auto inference skill's reference scene](../../cosmos3-inference-simulation-auto/references/reference-scene.md); `AUTO_SAMPLE_ROOT` and the edited index copy named below are set there.

## Prepare the training bundle

For the original paragraph captions, use the supplied index below. For timestamped
captions, keep `AUTO_INDEX` pointing to the edited copy described above.

```bash
export AUTO_INDEX="${AUTO_SAMPLE_ROOT:?set the local scene directory}/training-index.jsonl"
```

Run [Framework's native bundle builder](../post-training/SKILL.md#prepare-a-local-training-bundle)
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
python -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" --dryrun -- "${settings[@]}"
torchrun --standalone --nproc-per-node=4 \
  -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" -- "${settings[@]}"
```

This is a short 720p check with the bundle's camera-subset sampler, not a
guarantee for the full-length or fixed-seven-camera workload. Validate peak GPU
memory on your allocation. The save is synchronous; see
[checkpoint memory](framework-runtime.md#checkpoint-memory) before enabling
asynchronous saves. Require finite losses, a completed checkpoint, and a
successful same-run reload before treating the check as passed.

For **Sim-Auto Stage 1 TF**, first apply the
[bundle camera-map overrides](../../cosmos3-nano-sim-auto/shared/data-contract.md#preserve-your-bundles-camera-mappings).
Append these sample overrides **after** that array in the
[stage launch command](../../cosmos3-nano-sim-auto/post-training/references/runtime.md):

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

Apply this after constructing `settings` and before the first dry run.
These are reduced data settings, not changes to the published training recipe.
For Sim-Auto allocation and all three stages, follow the
[bounded train/save/resume guide](../../cosmos3-nano-sim-auto/post-training/references/smoke.md).
Do not overwrite the original full-run config.
A two-update smoke does not establish convergence, resume correctness or quality.
Use additional, longer recordings for the full curriculum and keep validation
drives separate from training drives.
