# One daytime scene for both Auto models

Use the same seven-camera scene for Transfer-Auto and Sim-Auto. It contains real
RGB/LiDAR targets, shared WSM v4.0.1 controls and camera-local captions. Model
weights, VAEs and the text tokenizer are separate assets.

The scene and original `captions/` files come from the released
[NVIDIA PhysicalAI Autonomous Vehicles dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).

## Get the scene

Download the [daytime sample](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto/tree/b8dcc2f70421e31db86d1b81d194c791d65eaad2/assets/sample_data)
from the Transfer-Auto **model repository**. Both models use these same assets:

- Repository: `nvidia/Cosmos3-Nano-Transfer-Auto`
- Repository type: `model`
- Subdirectory: `assets/sample_data`
- Immutable sample revision: `b8dcc2f70421e31db86d1b81d194c791d65eaad2`

With the Hugging Face CLI available in your Framework environment, choose a local
download directory outside the repository. If repository access requires it,
accept the model's access terms and authenticate with `hf auth login` first.

```bash
export AUTO_SAMPLE_DOWNLOAD_DIR=/path/to/your/auto-example
export AUTO_SAMPLE_REVISION=b8dcc2f70421e31db86d1b81d194c791d65eaad2
hf download nvidia/Cosmos3-Nano-Transfer-Auto \
  --repo-type model --revision "$AUTO_SAMPLE_REVISION" \
  --include 'assets/sample_data/*' \
  --local-dir "$AUTO_SAMPLE_DOWNLOAD_DIR"
export AUTO_SAMPLE_ROOT="$AUTO_SAMPLE_DOWNLOAD_DIR/assets/sample_data"
(cd "$AUTO_SAMPLE_ROOT" && sha256sum -c SHA256SUMS)
```

Keep the relative paths intact. This downloads only the sample, not model
weights or tokenizers. The revision above pins the **data**; obtain compatible
model snapshots separately using [environment and model assets](framework-runtime.md).
The checksum list covers 36 payload files; the directory also includes
`SHA256SUMS` and `FILES.tsv`.

## Files and meanings

```text
sample_data/
  README.md
  training-index.jsonl
  input-transfer-auto.jsonl
  input-sim-auto.jsonl
  SHA256SUMS
  FILES.tsv
  daytime_scene/
    rgb/<camera_key>.mp4       # seven real target videos
    wsm/<camera_key>.mp4       # seven shared control videos
    captions/<camera_key>.json # seven paragraph-style captions
    captions/negative_prompt.txt # per-view negative prompt
    captions_with_timestamp/
      <camera_key>.json        # seven timestamped alternatives
      README.md
    lidar/control.tar         # HD-map range controls
    lidar/target.tar          # recorded LiDAR targets
```

The camera keys are front-wide, cross-right, rear-right, rear-tele, rear-left,
cross-left and front-tele, using their full
[physical names and learned IDs](../post-training/references/data-contract.md#camera-identities).
Files are not renamed to c0/c1. The two inference requests explicitly map each
view using `camera_key`.

RGB and WSM contain 300 frames at 30 FPS, 1280×720. Captions use
`caption_structured.chunk_0_300` with exclusive bounds [0,300).
Each LiDAR archive has 100 sweeps at camera-clock IDs 0,3,...,297;
archive member clip keys are `daytime_scene`.
The controls are called **WSM v4.0.1** in these guides; the renderer option
remains `version=v4`.

The example contains already-rendered controls, not raw calibration, timestamp
sidecars or a scenario that can be rerendered. Preserve those source records when
preparing your own data so you can verify calibration and synchronization.

## Choose a caption style

The sample provides two caption variants for the same scene:

- `captions/`: a single descriptive paragraph per camera, which the model also
  supports.
- `captions_with_timestamp/`: timestamped descriptions of how each camera's view
  evolves. This matches the format used in our training data and has empirically
  worked better for generation. Prefer this format for post-training to reduce
  the caption-format gap between your data and the model's training distribution.

Both use the same JSON schema and camera filenames. The timestamped variant has
multiple paragraphs inside one `caption_structured.chunk_0_300.caption` string, with
outer frame bounds `[0,300)`. Text intervals such as `[0.0s-3.2s]:` use the shared
clip-local clock; paragraph boundaries may differ across views. These timestamps
do not create separate loader or generation chunks.

The supplied requests and training index still select `captions/`. To use the
timestamped variant, make a copy of the relevant JSONL **beside the original**
and change `daytime_scene/captions/` to
`daytime_scene/captions_with_timestamp/` in all seven caption paths:

- **Inference:** update each `multiview.views[].caption_path` and set
  `AUTO_REQUEST` to the copy. Keep `load_caption_from_data=true` and
  `first_caption_paragraph_only=false` under `multiview` to retain all paragraphs.
- **Training:** update the `assets.caption_<camera_key>` paths, set `AUTO_INDEX`
  to the copy, and build a new bundle. Existing packaged bundles do not change.
  Preserve the selected recipe's full-caption or opening-paragraph policy.

Keep media, camera keys and other settings unchanged. Review all views together
for consistent object appearance, physical motion and event timing, following
the [inference checks](../inference/references/input-preparation.md#4-delegate-caption-authoring)
or [target-grounded training guidance](../post-training/references/data-contract.md#training-captions).

## Run inference

Follow the matching inference skill. For the original paragraph captions, bind
the supplied request below; if you selected timestamped captions above, keep
`AUTO_REQUEST` pointing to your edited copy instead.

```bash
# Transfer-Auto:
export AUTO_REQUEST="${AUTO_SAMPLE_ROOT:?set the local scene directory}/input-transfer-auto.jsonl"
# Or Sim-Auto:
export AUTO_REQUEST="${AUTO_SAMPLE_ROOT:?set the local scene directory}/input-sim-auto.jsonl"
```

[Transfer-Auto inference](../inference/SKILL.md#run) reads guidance 3 and shift 10
from the request, and sets 720p, 35 steps and the bundled per-view negative prompt
through the CLI, matching the native notebook. The downloaded JSONL is unchanged.
[Sim-Auto inference](../../cosmos3-nano-sim-auto/inference/SKILL.md)
uses guidance 1 and the student's exported four-step schedule. Both requests
select seven cameras, first-frame RGB conditioning and HD-map LiDAR controls;
neither supplies an initial measured LiDAR sweep.

The requests use one caption interval covering all 300 source frames.
`use_first_chunk_only: true` selects that whole interval. Expected output is
297 camera frames at 30 FPS and 99 LiDAR sweeps at 10 Hz.
Do not add `joint_generation` to the JSONL: Framework selects joint output from
the camera/WSM and LiDAR inputs. Paths resolve relative to the JSONL file.

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

## Use your own data and verify results

Keep confidential source files outside the repository. Follow the
[training-data contract](../post-training/references/data-contract.md)
to produce equivalent paired assets and index rows; do not manufacture targets
from generated outputs or label them with fictional weather.

1. Verify the supplied checksums after downloading.
2. Inspect synchronized RGB/WSM overlays, physical camera mapping and caption spans.
3. Check decoded training data: control/target roles, camera IDs, time selection,
   range units, intensity and validity.
4. Run both models and inspect every generated camera and numeric LiDAR output.
   Record the model revision, sample revision and resolved request.

One scene illustrates the interface; use a larger, diverse dataset for training
and separate held-out recordings to evaluate generalization.
