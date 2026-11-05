# Transfer-Auto post-training

This is the bidirectional RGB/LiDAR curriculum, not Sim-Auto's causal
TF → TF-dCM → SF-DMD workflow. Start the curriculum from Cosmos3-Nano.
Starting from released Transfer-Auto instead is a fresh fine-tune using phase 4.

Use [Framework setup](setup.md). Training needs the `train` extra, the CUDA group
matching the allocated GPU driver, and working native TorchCodec/FFmpeg decoding.
Run all commands from the Framework repository root. The workflow below uses
your local paired recordings, controls and captions.

## Prepare checkpoints

Download a complete, authorized `nvidia/Cosmos3-Nano` snapshot at an immutable
revision. Convert the main model with:

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path "$NANO_HF" \
  -o "$NANO_DCP"
```

Set `COSMOS_TRANSFER_AUTO_NANO_DCP` to the resulting iteration-root directory
containing `model/.metadata`. This is model-only initialization, not a resumable
training checkpoint. The first phase initializes only new LiDAR projections and
the physical sensor embedding table, and begins a new EMA history.

Download the matching `lidar_vae/` from an authorized Transfer-Auto HF snapshot
with `hf download --include 'lidar_vae/*'`. Keep its config and safetensors together.
Set `COSMOS_TRANSFER_AUTO_LIDAR_VAE` to the local safetensors file. The tokenizer
itself is not converted to DCP. Set `COSMOS_TRANSFER_AUTO_VIDEO_VAE` to the matching
Wan2.2 VAE `.pth` file, available from `Wan-AI/Wan2.2-TI2V-5B`.

## Prepare local inputs

The user supplies synchronized real RGB, rendered WSM, target-grounded per-view
captions, and measured/HD-map LiDAR ranges. The loader does not produce WSM or
invent measured targets. Camera keys retain the physical IDs in
[`camera_attributes.py`](../cosmos_framework/data/generator/multiview/camera_attributes.py).
RGB and WSM use the same temporal selection and resize/center crop. LiDAR uses
the matched 128×1808 projection, metric range, normalized measured intensity,
and validity; HD-map controls use unit intensity on occupied rays.

For a new Auto bundle, create a JSONL index with one scene interval per row:

```json
{"id":"drive1-clip1","stratum":"480","assets":{"video_camera_front_wide_120fov":"rgb/front.mp4","wsm_camera_front_wide_120fov":"wsm/front.mp4","caption_camera_front_wide_120fov":"captions/front.json","lidar/lidar_rangemap":"ranges/real.tar","lidar/hdmap_rangemap":"ranges/hdmap.tar"}}
```

Paths are relative to the index. Add the same three camera members for every
selected view. Supply the prepared range-video format accepted by
[`load_prepared_lidar_video_3ch`](../cosmos_framework/model/generator/tokenizers/lidar/preprocessing.py),
not arbitrary images reshaped to the same dimensions. These are sparse sweep
archives containing the source-frame-keyed row/column/range NPZ members and,
for measured targets, intensity members; they are not single dense NPZ files.
Existing encoded bundles
can be used directly. The front-wide caption metadata is also needed for
LiDAR-only samples even if that camera is not a selected RGB view.

A camera caption file contains `caption_structured`: a JSON object or serialized
JSON object of chunks. Each chunk has `start_frame`, exclusive `end_frame`, and a
`caption` describing the recorded target interval. Keep the same clock across
cameras and range sweep IDs. Use disjoint drives/scenes for train and validation.

```bash
python -m cosmos_framework.data.generator.local_simulation.auto_bundle \
  --index "$AUTO_INDEX" \
  --tokenizer "$LOCAL_QWEN_TOKENIZER" \
  --output "$AUTO_BUNDLE" \
  --camera-keys camera_front_wide_120fov
```

The tokenizer directory is the complete local Qwen tokenizer/processor asset
directory matching the model. For the Nano HF export above, use
`LOCAL_QWEN_TOKENIZER="$NANO_HF/text_tokenizer"`. Keep each index `id` matched
to the clip key embedded in its LiDAR archives. The builder creates a new directory only. Its
completion marker establishes packaging, not sensor alignment or training
compatibility. Decode an actual sample/batch before training. Optional `--hardlink`
avoids a media copy on the same filesystem but requires immutable input files.
Joint phases retain the bundle's rig-selection policy. With all eleven reference
cameras, the builder uses the reference 480p 60% seven-view / 30% eleven-view /
10% other-rig mix and the 720p 90% six-of-seven / 10% other-rig mix. With fewer
cameras it removes unavailable rigs and renormalizes, which is a documented data
adaptation, not exact rig-exposure reproduction.

Set `COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE` to the paired camera/range Auto bundle.
All four phases use this source; no separate general-video or driving-prediction
dataset is required. All phases use selective activation checkpointing; joint
phases also save only marked operations.

## Resolve and run each phase

The explicit registration module provides these selectors:

| Selector               | Task and logical-sample mixture                                                  | Default stop      | State from predecessor                                |
| ---------------------- | -------------------------------------------------------------------------------- | ----------------- | ----------------------------------------------------- |
| `transfer_auto_phase1` | Separate sensors: 30% single-camera RGB, up to 397 frames / 70% LiDAR, 21 sweeps | 70k               | Nano weights only, fresh optimizer/schedule/EMA       |
| `transfer_auto_phase2` | Short joint multiview, 61 RGB frames / 21 sweeps                                 | 10k new           | Model/EMA/Adam retained, counter/scheduler/data reset |
| `transfer_auto_phase3` | Long joint multiview, up to 397 RGB frames / 133 sweeps                          | Joint counter 22k | Retain model/EMA/Adam/counter/scheduler, restart data |
| `transfer_auto_phase4` | Continue long joint training with the same attention and dropout policy          | Joint counter 50k | Retain model/EMA/Adam/counter/scheduler, restart data |

Phase 1 requests 400 RGB frames, trimmed to the tokenizer-compatible maximum of
397; its separate 21-sweep LiDAR samples do not need to match the RGB duration.
All joint phases use deduplicated maskless attention with a preceding 0.4-second
cross-view window, including the current time. Phase 1 uses single-view/two-way
attention. Phases 3 and 4 both drop the complete LiDAR target and control with
10% probability; phase 2 does not drop LiDAR.

Caption dropout is 10% sample-wide in every phase. Joint phases additionally
drop each camera's description independently with 10% probability, preserving
the camera headers. Sample-wide dropout blanks every view's caption together;
the task system prompt remains. RGB views use the rig/subset sampling described
above, not independent whole-RGB modality dropout. View-completion, control
dropout and LiDAR-prefix dropout are disabled. T2V/I2V conditioning is 50/50.

Joint streams use 60/40 resolution exposure. The reference context/FSDP degree is 32.
Adapt topology using both `model_parallel.context_parallel_size` and
`model.config.parallelism.context_parallel_shard_degree`, plus the FSDP shard
degree. Check global batch and memory on the actual hardware.

Resolve the selected configuration before allocating training GPUs:

```bash
python -m cosmos_framework.scripts.train_experiment \
  --config cosmos_framework/configs/simulation/config.py \
  --experiment-module cosmos_framework.configs.simulation.experiment.auto.transfer_auto \
  --dryrun \
  -- \
  experiment=transfer_auto_phase1
```

Use the same entry point for training under the allocation's launcher:

Set `NNODES`, `GPUS_PER_NODE`, `NODE_RANK`, and `MASTER_ADDR` from the allocation.
Choose `CONTEXT_PARALLEL` and `FSDP_SHARDS` compatible with that world size and the
model. The reference uses 32 for both. Keep all three topology settings consistent.

```bash
torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$GPUS_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:29500" \
  -m cosmos_framework.scripts.train_experiment \
  --config cosmos_framework/configs/simulation/config.py \
  --experiment-module cosmos_framework.configs.simulation.experiment.auto.transfer_auto \
  -- \
  experiment=transfer_auto_phase1 \
  model_parallel.context_parallel_size="$CONTEXT_PARALLEL" \
  model.config.parallelism.context_parallel_shard_degree="$CONTEXT_PARALLEL" \
  model.config.parallelism.data_parallel_shard_degree="$FSDP_SHARDS"
```

The launcher must provide the stated resources and a reachable rendezvous.
Choose a fresh `IMAGINAIRE_OUTPUT_ROOT` and `job.name` for each new phase. After a
complete phase-1 save, set `COSMOS_TRANSFER_AUTO_PHASE1_CHECKPOINT` to that exact
output and select phase 2. Then bind `COSMOS_TRANSFER_AUTO_PHASE2_CHECKPOINT`
for phase 3, and `COSMOS_TRANSFER_AUTO_PHASE3_CHECKPOINT` for phase 4.
These variables name complete iteration-root checkpoints, not model-only files.

## Shortened smoke and recovery

For a concrete single-node starting point, use the
[reference-scene short check](../.agents/skills/cosmos3-nano-transfer-auto/shared/reference-scene.md#short-training-check):
four H200 GPUs, CP=2/FSDP=2, 21 aligned RGB frames and synchronous checkpointing.
CP=4/FSDP=1 does not shard model/optimizer state. These short-check settings do
not establish memory requirements for the full curriculum.
See [checkpoint memory](../.agents/skills/cosmos3-nano-transfer-auto/shared/framework-runtime.md#checkpoint-memory)
for host RAM and `/dev/shm` requirements before enabling async saves.

Shorten **native** `trainer.max_iter` and `checkpoint.save_iter` overrides. Keep
the canonical scheduler cycles and attention policies. A shorter input/view
selection tests execution, not the original maximum horizon or rig exposure.
Check every task, finite modality losses, intended parameter updates, completed
checkpoint saves, and actual state restoration.
For a bounded smoke, align save/handoff boundaries to complete CP data windows
(`iteration * grad_accum_iter` divisible by the context degree). This exercises
full-window restoration without claiming bit-exact replay of discarded partial windows.

`trainer.max_iter` is an absolute endpoint. A same-phase checkpoint at iteration
N takes one more update with `trainer.max_iter=N+1`. Reuse its job identity and
output root so native latest-checkpoint resume restores all state. Phase-transition
resets do not apply again to a same-job resume.

When shortening phase 1, set `model.config.ema.iteration_shift` to its **actual
completed update count** in all joint phases. The default is 70,000. Phase 2
resets the trainer/scheduler counter; phases 3 and 4 retain that joint counter
and EMA age offset. Loading Adam while resetting a schedule applies that new schedule's
initial LR before the first update, without discarding its moments.

For a fresh direct fine-tune from released Transfer-Auto, use
`transfer_auto_phase4`, override `checkpoint.load_path`
with the converted release model weights, set `checkpoint.load_training_state=False`,
`checkpoint.keys_not_to_resume=[]` and `model.config.ema.iteration_shift=0`, and
choose a fresh job identity, LR, schedule and duration. For model-only DCP without
EMA, use `checkpoint.keys_to_skip_loading=[net_ema.]`; retain strict regular-weight
loading and do not skip the trained LiDAR heads or sensor embedding table. Begin
fresh optimizer/scheduler/counter/EMA history, not a full-state resume. HF exports
do not contain optimizer or data-cursor state. Export the chosen regular/EMA
result with matched config and tokenizers only after a successful train/save/reload
check. A curriculum smoke does not separately validate release-HF direct fine-tuning.

## Export

Export a phase checkpoint to Hugging Face format with
`cosmos_framework.scripts.export_model`, passing the phase experiment
(`--config-file cosmos_framework/configs/simulation/config.py --experiment transfer_auto_phase2`,
for example) and its checkpoint. The export is self-contained for inference:

- **LiDAR VAE (optional).** With `--bundle-lidar-vae`, the VAE named by
  `COSMOS_TRANSFER_AUTO_LIDAR_VAE` is copied into `lidar_vae/`
  (`diffusion_pytorch_model.safetensors` plus `config.json`), the same layout as
  the released Transfer-Auto snapshot, and the exported `config.json` points
  `lidar_tokenizer.vae_path` at it. The loader binds the bundled file, so the
  training-time VAE path does not need to exist at inference time. Without the
  flag, the export keeps the training-time location.
- **Per-view caption layout.** Phases 2–4 tokenize one caption per camera.
  `export_model` derives this from the training dataloaders and records
  `per_view_captions: true` and
  `per_view_caption_task: "av_joint_camera_lidar_transfer"` in the exported
  `config.json`; phase 1 records `per_view_captions: false`. Inference
  reads these keys to pack prompts the way the weights were trained. For a
  config without dataloaders, set them with `--per-view-captions on|off` and
  `--per-view-caption-task`.

`convert_model_to_diffusers` carries both keys into the Diffusers layout; pass
`--carry-lidar-vae` to also copy a bundled `lidar_vae/`.
