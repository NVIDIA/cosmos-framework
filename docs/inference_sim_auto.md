# Sim Auto inference

Cosmos3-Nano-Sim-Auto is the causal, self-forcing-distilled variant of
[Transfer Auto](inference_auto.md). From per-camera WSM control videos and
per-camera captions, it generates synchronized multiview camera video and,
optionally, LiDAR. It generates chunk by chunk with a rolling key/value cache,
using a fixed four-step sampler.

The released model is optimized for the reference camera rig; a custom rig can
produce degraded results even with valid camera IDs. The shared
[rig guide](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-preparation.md#choose-reference-rig-or-custom-rig-generation)
explains reference-rig WSM retargeting and custom-rig trials. Retargeted controls
need matching initial RGB. For adaptation, [post-train Transfer-Auto](../.agents/skills/cosmos3-nano-transfer-auto/post-training/SKILL.md)
on your rig's paired data, then [distill it into Sim-Auto](../.agents/skills/cosmos3-nano-sim-auto/post-training/SKILL.md).

For a custom rig, use [Camera Rig Explorer](../.agents/skills/cookbooks/cosmos3/simulation/auto/wsm/docs/camera-rig/guide.md) and the
[shared mapping and caption workflow](../.agents/skills/cosmos3-nano-sim-auto/inference/SKILL.md#start-here).
Keep guidance and control guidance at 1 for the released student. Its learned
four-step schedule replaces Transfer-Auto's sampling recipe; see
[why the settings differ](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#sim-auto-uses-a-distilled-schedule).

Both **480p (832×480) and 720p (1280×720)** are supported. Prefer
`aspect_ratio: "16,9"` to best match training; see [size and aspect-ratio guidance](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#resolution-and-aspect-ratio).
Sim-Auto supports long rolling generation; qualitative testing has shown
multi-minute rollouts without significant visible quality loss. See [duration guidance](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#generation-length-and-long-rollouts)
for 10-/30-second requests, full-horizon controls and long-rollout quality checks.

Use aligned 30-FPS cameras for the reference recipe; current joint Sim-Auto
requires 30-FPS camera timing with its 10-Hz LiDAR. See the
[FPS guide](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#source-timing-model-fps-and-playback-fps)
before experimenting with lower-rate inputs. Playback changes do not correct
the model's interpretation of motion.

Follow [framework setup](setup.md) and the environment section of the
[Transfer Auto recipe](inference_auto.md#environment-and-checkpoint) first.
Run the commands below from the framework repository root with
`COSMOS_TRAINING=0` and `COSMOS_INTERNAL=0`.

Rolling replay runs its same-camera attention through the FlashAttention-4
kernels. The `cu130-train` group installed there includes FlashAttention-4, so
no separate installation is needed.

On SM120 GPUs such as RTX 6000D, replay inference uses the same native
FlashAttention-4 gapped-varlen path as the other supported architectures.
The pinned FlashAttention-4 4.0.0b30 release includes the SM80/SM120 invalid-tile
fix required by this workload. Upgrade the environment through the setup command
above; admitting SM120 with the older 4.0.0b24 pin can cause illegal memory access.
SM120 replay backward is not qualified and is explicitly rejected; SM120 support
is inference only.

## Checkpoint

Download the published snapshot and pass its local directory. A bare Hugging
Face repository ID is not a checkpoint directory.

```bash
hf download nvidia/Cosmos3-Nano-Sim-Auto \
  --local-dir checkpoints/Cosmos3-Nano-Sim-Auto
export SIM_AUTO_CHECKPOINT=checkpoints/Cosmos3-Nano-Sim-Auto
```

The snapshot is in the Diffusers layout (`transformer/`, `vae/`,
`text_tokenizer/`) and bundles the LiDAR VAE under `lidar_vae/`, which is used
in place of a registry download.

An exported student directory (`config.json`, `checkpoint.json` and
`model-*.safetensors`) also works. Either form must record the rolling-history
settings in `config.json`: `rolling_kv_cache_chunks`, `rolling_kv_sink_chunks`
and `rolling_prompt_policy: "segmented"`. A checkpoint without them is refused
before the model loads, and the error names the export overrides that add
them. A snapshot without `lidar_vae/` resolves the LiDAR VAE through the
registry to the Transfer Auto repository.

## Inputs

Follow [Download the shared example](inference_auto.md#download-the-shared-example)
to download and checksum the pinned sample and set `AUTO_SAMPLE_ROOT`.
Transfer-Auto and Sim-Auto intentionally share these assets, hosted in the
Transfer-Auto repository. Authenticate with read access to both model repositories
if required.

Use `$AUTO_SAMPLE_ROOT/input-sim-auto.jsonl` directly. It requests joint camera
and LiDAR generation with seven named cameras, first-frame RGB conditioning,
720p, 30 FPS, 297 frames, seed 0 and guidance 1. The HD-map control is
`daytime_scene/lidar/control.tar`; the RGB, WSM and caption filenames use their
physical camera names. All paths resolve relative to the JSONL directory.

This checkpoint tokenizes one caption per camera. Keep
`multiview.load_caption_from_data: true` and a `caption_path` on every camera,
and keep the supported physical camera keys, because the model uses rig-view
embeddings. A request that does not load per-camera caption files is refused
rather than served with one merged prompt.

`num_frames` is the generation horizon. The caption spans must start at
frame 0 and cover it.

Set `seed` explicitly. Without one, the seed is derived from the sample name
and output directory, so it changes with the output path.

## Run

With context parallelism across four GPUs:

```bash
torchrun --standalone --nproc-per-node=4 -m cosmos_framework.scripts.inference \
  -i "$AUTO_SAMPLE_ROOT/input-sim-auto.jsonl" \
  -o outputs/sim_auto \
  --checkpoint-path "$SIM_AUTO_CHECKPOINT" \
  --cp-size 4 --cfgp-size 1 \
  --no-use-torch-compile --no-diffusion-cache
```

For RGB-only generation, copy the request alongside the original JSONL and
remove its `lidar` object. Use a separate output directory to avoid colliding
with the joint run's `daytime_scene` outputs.

Rolling inference runs eagerly with serial classifier-free guidance. For this
checkpoint, the CLI therefore switches off `--use-torch-compile`,
`--use-cuda-graphs` and `--diffusion-cache`, sets `--cfgp-size 1`, and sets
`--cp-size` to the world size when it is unset. Each change is logged as a
warning. It refuses `--kv-cache-inference-size`, `--attention-sink-size` and
`--use-separate-pipeline-vision-decode-gpu`, because rolling history is
configured by the checkpoint.

## Outputs

This sample writes to `<output>/daytime_scene/`, using the request's `name`:

- `vision_view<NN>_<camera_key>.mp4`: one generated video per camera.
- `control_view<NN>_<camera_key>.mp4`: the WSM control each camera followed.
- `lidar.safetensors` (joint only): `range_m`, `intensity` and `frame_ids`.
  LiDAR runs at 10 Hz against the 30 FPS camera clock, so 297 frames give
  99 sweeps with `frame_ids` stepping by 3.
- `lidar.pt` (joint only): the decoded LiDAR tensor.
- `caption_chunks.json` and `rolling_prompt.json`: the caption spans used and
  the segmented prompt schedule.
- `sample_args.json` and `sample_outputs.json`.
- `latents.pt`: only when the sample sets `save_latents: true`.

The shared example produces seven 1280 × 720 videos with 297 frames at 30 FPS,
and finite `range_m` and `intensity` arrays of shape `[99,128,1800]`.

Collage the camera videos with
`python -m cosmos_framework.scripts.multiview_collage <output> --tile-width 1280`.
Add `--visualize_control_input` to show the controls beside them.

For repeatable comparisons, keep the model and data revisions, seed, GPU layout
and locked software environment fixed. Preserve the resolved request alongside
the outputs.
