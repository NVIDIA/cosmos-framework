# Cosmos3-Nano-Transfer-Auto

The Auto use case generates a **synchronized multi-camera driving clip together with its LiDAR
range map**, from one World Scenario Model (WSM) control video per camera plus an HD-map control —
a structured representation of the scene as lane lines, road surface, and vehicle bounding boxes.
One rollout produces every camera on the vehicle and the LiDAR sweep stream at once, so a
perception stack can be tested the way it ships rather than one sensor at a time.

This page is the inference recipe for the published
[`nvidia/Cosmos3-Nano-Transfer-Auto`](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto)
artifact. For what the Simulation family is, the environment setup every page shares, and the
shared inference and validation notes, start at [Simulation cookbooks](../README.md). For the other
use cases see [Cosmos3-Nano-Sim-Bimanual](../bimanual/bimanual.md),
[Cosmos3-Nano-Sim-Auto](run_sim-auto_with_cosmos_framework.ipynb) (the autoregressive student of this
model), and [Cosmos3-Nano-Sim-Transfer](../transfer/transfer.md).

## Supported backends

This cookbook supports two runtimes:

- **Cosmos Framework** for export, native PyTorch inference.
- **vLLM-Omni** for an OpenAI-compatible video server. The runnable
  [Multiview Simulation notebook](run_auto_with_vllm_omni.ipynb) uses the published WSM controls and the synchronous video endpoint, followed by a joint WSM/RGB/LiDAR example
  that retrieves both generated RGB and numeric LiDAR through asynchronous jobs.

These examples consume the same fixed camera order and WSM-per-view contract. The commands,
request fields, and environment are runtime-specific.

## Cosmos Framework

Use the [native Transfer-Auto notebook](run_auto_with_cosmos_framework.ipynb) or
[native Sim-Auto notebook](run_sim-auto_with_cosmos_framework.ipynb) with the shared HF sample.
Both default to seven cameras, 297 frames, 30 FPS and 720p, and offer RGB-only,
joint RGB/LiDAR, or WSM-only T2V. The same sample feeds the CLI below.

The notebooks default to `captions_with_timestamp/`; set `CAPTION_FOLDER="captions"`
to use the original dense paragraphs. Timestamped captions better match training
and empirically improve inference. Keep shared object appearance and motion
consistent across views. The downloaded JSONL uses the original paragraph paths;
the notebook selects either variant without modifying the source sample.

To render your own controls, [Generate WSM](generate_wsm.ipynb) requires your raw
RDS-HQ scene, calibration and captions; the published sample has rendered controls only.
Sim-Auto needs the FlashAttention-4 environment described in its inference guide.

### Prerequisites

Beyond the [shared Simulation prerequisites and environment
setup](../README.md#prerequisites):

- The CLI example below uses four GPUs with context parallelism for seven-view, 720p generation.
- Authenticate with read access to the model repository if required.

### Get the checkpoint

`--checkpoint-path` takes a local directory or a registered checkpoint name. There is no registry
entry for this model, so a bare repository id fails with `Checkpoint directory does not exist`.
Download the snapshot first:

```bash
hf download nvidia/Cosmos3-Nano-Transfer-Auto --local-dir ./Cosmos3-Nano-Transfer-Auto
```

### Prepare the input

Download and verify the [shared daytime sample](../../../../../../docs/inference_auto.md#download-the-shared-example)
and set `AUTO_SAMPLE_ROOT` as documented there. Its `input-transfer-auto.jsonl` is a native
request for seven-camera joint RGB/LiDAR generation, with first-frame RGB conditioning, 720p,
30 FPS and 297 output frames. `input-sim-auto.jsonl` uses the same media with the causal model.
Keep the complete downloaded folder together; no manifest conversion is required.

The shape, if you are writing your own: each camera view supplies its own control video in
`multiview.views`, the HD map goes in `lidar.control_path`, and `wsm` stays empty. There is no
top-level `vision_path` or `control_path` for a multiview sample.

```json
{
  "model_mode": "text2video",
  "prompt": "Generate a multi-camera driving video and a LiDAR range map that follow the per-camera control videos and the HD-map control.",
  "wsm": {},
  "lidar": { "control_path": "daytime_scene/lidar/control.tar" },
  "multiview": {
    "load_caption_from_data": true,
    "views": [
      { "camera_key": "camera_front_wide_120fov", "control_path": "daytime_scene/wsm/camera_front_wide_120fov.mp4",
        "caption_path": "daytime_scene/captions_with_timestamp/camera_front_wide_120fov.json" }
    ]
  }
}
```

For **image2video**, set `"model_mode": "image2video"`, add
`multiview.condition_video_as_image`, and give *every* view a `vision_path` — the first RGB frame
for that camera.

Notes that will save you a failed run:

- `multiview.load_caption_from_data` and a `caption_path` on **every** camera are required. This
  checkpoint tokenizes one caption per camera and refuses to generate without one caption file per
  camera. Each camera's caption supplies its text conditioning and must cover the generation horizon.
- `wsm: {}` is required. It is the transfer hint; without it the sample is rejected with
  *"Multiview transfer inference requires an explicit transfer hint such as wsm={}"*.
- Every view must supply `control_path`, or none may — a partial set is rejected. The same
  all-or-none rule applies to `vision_path`.
- Each view's `control_path`, `caption_path` and `vision_path` must describe the physical camera
  identified by its `camera_key`. The published sample uses that camera name in each filename.
- `camera_key` values must be unique. They become the output filenames, and their order in the
  list is the view order — the control filenames do not have to match it.
- Do **not** set a top-level `vision_path` or `view_point` — both are rejected for multiview.
- `prompt_upsampling` and `autoregressive` are rejected for multiview as well.

### Run inference

```bash
source .venv/bin/activate

torchrun --standalone --nproc-per-node=4 \
  -m cosmos_framework.scripts.inference \
  -i "$AUTO_SAMPLE_ROOT/input-transfer-auto.jsonl" \
  -o outputs/cam_lidar \
  --checkpoint-path ./Cosmos3-Nano-Transfer-Auto \
  --cp-size 4 --cfgp-size 1 \
  --no-use-torch-compile --no-diffusion-cache
```

The downloaded request explicitly supplies 35 denoising steps, guidance 3, shift 10,
control guidance 1 and seed 0. It reads the downloaded local assets rather than fetching
individual media files during inference. Keep guardrails at their default settings.
Use the full [Framework environment setup](../../../../../../docs/inference_auto.md#environment-and-checkpoint)
so the required runtime and guardrail dependencies are installed.

#### On rank count

Keep rank count and parallelism settings fixed when comparing generated results.
For performance comparisons, vary the topology while holding the request and model
fixed. `dp_shard_size` defaults to the world size; camera count is independent of
the parallelism degree. See the [parallelism options](../../../../../../docs/inference.md#parallelism-arguments).

### Output

One video per camera, plus a single LiDAR file:

```
<output>/<sample_name>/
  vision_view00_camera_cross_left_120fov.mp4
  vision_view01_camera_front_wide_120fov.mp4
  ...
  vision_view06_camera_rear_right_70fov.mp4
  lidar.safetensors
  sample_args.json    sample_outputs.json
```

Videos are h264, 1280×720, 30 fps. `lidar.safetensors` holds `range_m`, `intensity` and
`frame_ids`. LiDAR runs at 10 Hz against the 30 fps camera clock, so a 297-frame generation
produces 99 sweeps with `frame_ids` stepping by 3.

A joint run writes **no** `control_view*` files. The re-encoded controls come from the RGB-only
multiview branch; the joint branch saves the generated views and the range map only. Do not read
their absence as a partial run.

### Validate the rollout

Run the [shared validation checklist](../README.md#validate-a-rollout) first, then check the three
things specific to a joint multiview rollout:

1. Every requested camera produced a `vision_view*` file, and each decodes to the full frame count.
   A partial set of views is a failure even when `status: success`.
2. `lidar.safetensors` has `round(frames × 10/30)` sweeps, `frame_ids` stepping by 3, and finite
   `range_m` within the sensor's range. A sweep count that does not match the video length means
   the two streams were generated against different clocks.
3. The same object appears consistently across overlapping views, and in the LiDAR return — a
   vehicle in the front-wide view should appear in cross-left or cross-right from the
   corresponding angle, and as a return at the matching bearing. This is the property joint
   multi-sensor generation exists for, and single-sensor inspection will not catch its absence.

For this model, "follows the control" means vehicles track their WSM boxes, lane geometry matches,
and the range map agrees with the HD map it was given.

For numerical comparisons, compare raw tensors and repeat each setting to measure
run-to-run variation.

## vLLM-Omni

vLLM-Omni serves the exported model through the OpenAI-compatible Videos
API using the `vllm/vllm-omni:cosmos3` Docker image. The examples use maskless Multiview
attention, joint LiDAR encoding and decoding, and asynchronous numeric output, with
`nvidia/Cosmos3-Nano-Transfer-Auto` revision `9a51f9b119f7ab7628f3a064562e2de26a113200`.
The examples' inputs come from the shared daytime sample in the same model repository,
`assets/sample_data` at revision `ab50792f104af5304bce8b4cb7c27c096b33e63f`: one scene recorded by
seven synchronized cameras, each with an RGB video, a WSM control, and a caption, all 1280×720 with
300 frames at 30 FPS. The notebook downloads them into the Hugging Face cache and verifies them
against the sample's `SHA256SUMS`; set `AUTO_SAMPLE_ROOT` to use a local copy. The
[request files](assets/daytime_scene/README.md) live in `assets/daytime_scene/`.

The [notebook](run_auto_with_vllm_omni.ipynb) follows the same steps as the other Simulation
vLLM-Omni notebooks: prerequisites, the server, client setup, then the **WSM-only example** and
the **joint RGB and LiDAR example**. One server serves both examples. To run LiDAR only, complete
the first three steps and go to step 5; it prepares its own request and outputs and uses no
WSM-example state. The client helpers in `helpers/auto_helpers.py` and `helpers/daytime_sample.py`
and the notebook need no GPU libraries; the notebook environment needs `httpx`, `pillow`, `numpy`,
`attrs`, `safetensors`, `matplotlib`, `ipython`, `huggingface_hub`, FFmpeg, and FFprobe, and a Hugging Face
token with read access to the model repository for the sample download.

Install FFmpeg on the notebook client host; its package also includes FFprobe. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On macOS with Homebrew, use `brew install ffmpeg`. Verify both commands are available:

```bash
ffmpeg -version
ffprobe -version
```

Both notebook examples load the already-exported `nvidia/Cosmos3-Nano-Transfer-Auto` model artifact. The notebook does
not export a training checkpoint. Its repository must contain:

- `model_index.json` with `"_class_name": "Cosmos3MultiviewPipeline"`.
- `transformer/config.json` with `"backbone_type": "cosmos3_multiview"` and a `multiview`
  object declaring `attention_scope: "decomposed"`, the exact eleven-camera order shown in
  the [supported camera mapping](../../../../cosmos3-nano-transfer-auto/inference/references/input-preparation.md#preserve-camera-identity), and `backend: "maskless"`.
- The normal Cosmos3-Nano transformer, tokenizer, VAE, scheduler, and text-encoder artifacts.

Use the `transformer/config.json` delivered with the exported weights. It carries
the attention settings and trained camera mapping; keep it with the matching
checkpoint instead of constructing a configuration by hand.

The vLLM-Omni server identifies each camera by its `camera_key` and looks up its trained
camera ID in the checkpoint's `multiview.rig_view_embedding.camera_ids`, so a request may list any
subset of the eleven cameras; the camera-major output follows the request order. The daytime
sample uses camera IDs 0 to 6 from that mapping, and the notebook's requests list
them in that order; `prepare_request` requires distinct cameras in canonical order. The WSM
`text2video` example supplies `camera_key`, `control_path`, and `prompt` for each view, without
`vision_path`. The [joint example](#joint-rgb-and-lidar-generation) adds RGB first-frame conditions
and numeric LiDAR. Both examples require a nonempty plain-text caption in every view's `prompt`
field; the request files name each camera's caption file, and the notebook sends its text. A
top-level prompt alone does not satisfy checkpoints with separate per-camera text tokenization.

### Pull the server image

Use a Linux server with an NVIDIA GPU, Docker, and the NVIDIA Container Toolkit.
Export `HF_TOKEN` with read access to the model, then pull the same image used by
all Simulation vLLM-Omni notebooks:

```bash
docker pull vllm/vllm-omni:cosmos3
```

The container reuses `${HF_HOME:-$HOME/.cache/huggingface}` for model downloads.
These recipes select GPU 0 and the sample's seven cameras in canonical order.

### Serve with maskless attention

The released checkpoint uses maskless attention; no attention-backend environment override is
needed.

The synchronous Videos API and the GPU workers each default to a 600-second timeout per request. A
201-frame Multiview request may take longer, depending on the GPU and compilation state. Set both
timeouts **before** starting vLLM-Omni; `1800` seconds is a reasonable starting point, but choose a
value above the measured request time on your hardware. Start the container and leave it
running on port 8000; both notebook examples use it:

```bash
docker run --rm --gpus '"device=0"' --ipc=host \
  -e HF_TOKEN \
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
  -e VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=1800 \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 127.0.0.1:8000:8000 \
  -w /app/vllm-omni \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano-Transfer-Auto \
    --revision 9a51f9b119f7ab7628f3a064562e2de26a113200 \
    --omni --model-class-name Cosmos3MultiviewPipeline \
    --diffusion-compile-granularity regional --no-diffusion-compile-dynamic \
    --no-guardrails \
    --host 0.0.0.0 --port 8000 --init-timeout 1800
```

The two diffusion compilation arguments are **required for Multiview serving**. They compile the
fixed-shape regional graph and prevent prompt-dependent dynamic graphs from repeatedly compiling
or falling back to an eager attention score matrix that does not fit this sequence length.
`--no-guardrails` disables the Cosmos3 text and video safety guardrails for this example server.
The server binds `0.0.0.0` inside the container; Docker publishes port 8000 on host
loopback. For a remote notebook, use `ssh -L 8000:127.0.0.1:8000 <server-host>`, or publish
with `-p 8000:8000` and set `VLLM_BASE_URL`. For two GPUs, change Docker's selection to
`--gpus all` and add `--cfg-parallel-size 2`; for more, also set `--ulysses-degree`.

When startup finishes, `curl http://127.0.0.1:8000/v1/models` must return the served model before
you submit generation requests. The complete client is in
[`run_auto_with_vllm_omni.ipynb`](run_auto_with_vllm_omni.ipynb).

### Send the multipart request

`POST /v1/videos/sync` accepts one control per camera as repeated multipart `input_references`
parts. In the JSON-encoded `extra_params`, each ordered view replaces its local `control_path`
with a zero-based `control_reference_index` pointing at the corresponding upload. Each view
also retains its plain-text `prompt`:

```python
prompt = "A city intersection with lane markings and surrounding traffic."
extra_params = {
    "wsm": {},
    "multiview": {
        "views": [
            {"camera_key": camera_key, "control_reference_index": index, "prompt": prompt}
            for index, camera_key in enumerate(SAMPLE_CAMERAS)
        ]
    },
}
files = [
    ("input_references", (path.name, path.open("rb"), "video/mp4"))
    for path in control_paths
]
response = requests.post(
    "http://localhost:8000/v1/videos/sync",
    data={
        "prompt": prompt,
        "size": "832x480",
        "fps": "30",
        "num_frames": "201",
        "num_inference_steps": "35",
        "guidance_scale": "3.0",
        "flow_shift": "10.0",
        "seed": "0",
        "extra_params": json.dumps(extra_params),
    },
    files=files,
    timeout=3600,
)
response.raise_for_status()
```

Every upload must be referenced exactly once. The WSM-only sample sends no `vision_path`,
`vision_reference_index`, singular `input_reference`, or server-local media path. A bearer token
is optional; set `VLLM_API_KEY` for the notebook when the server requires one.

The synchronous response is one camera-major MP4: 201 frames for the first requested camera, then
201 for the second, through the seventh. The notebook saves the original response under
`outputs/vllm_omni/auto/wsm/`, validates its full decode, splits it into seven labeled camera
videos, and shows their frames and compact previews. Generated outputs are ignored by the simulation-local
`.gitignore`.

### Joint RGB and LiDAR generation

The second half of the same notebook uses the
[joint request](assets/daytime_scene/transfer_auto_joint.json) on the same sample. It adds each
camera's RGB recording and the published `daytime_scene/lidar/control.tar` archive.
The joint preparation cell converts it to a local `prepared/lidar_control.safetensors`
file in the output directory. No Git LFS download is needed. See the
[format notes](assets/daytime_scene/README.md) for the range-map conversion.
The request uses 201 frames at 30 camera FPS, 67 LiDAR sweeps at 10 Hz,
480p landscape (832×480), seed 0, 35 steps, guidance 3, flow shift 10, and control
guidance 1. WSM clips are trimmed to the first 201 frames without resizing. RGB videos are
uploaded unchanged with `multiview.condition_video_as_image=true`; only their first decoded
frames condition generation.

Use the same `nvidia/Cosmos3-Nano-Transfer-Auto` model and server as the WSM-only example. The
notebook's `VLLM_BASE_URL`, `VLLM_API_KEY`, `COSMOS3_OUTPUT_DIR`, and `AUTO_SAMPLE_ROOT` settings
apply to both examples; this one writes to `outputs/vllm_omni/auto/lidar/` and polls its job for up
to an hour.

The joint request sends 15 repeated `input_references` uploads, interleaving WSM and RGB for
each ordered camera. Camera index `i` receives `control_reference_index=2*i` and
`vision_reference_index=2*i+1`; `extra_params.lidar.control_reference_index=14` refers to the
numeric file. Each view carries its camera's plain-text caption as `prompt`. The Videos API
requires a nonempty top-level prompt even with per-camera captions; the request copies the front
wide camera's caption. The notebook adds the inference skill’s recommended `per_view_negative_prompt` to
`extra_params` for both Transfer-Auto examples. Current vLLM-Omni
rejects the legacy `negative_prompt` form field with HTTP 400. `extra_params` retains `wsm: {}`,
`multiview.condition_video_as_image: true`, and `lidar.return_output: true`.

The numeric input contains one contiguous FP32 tensor `frames[3,100,128,1800]`, with channels
range in metres, intensity in [0,1], and validity in [0,1]. An HD map records no return
intensity, so occupied rays carry unit intensity, as in Cosmos Framework's reader. The 201-frame
request uses its first 67 sweeps (6.7 seconds), which the notebook validates sweep by sweep; the
server truncates trailing sweeps and applies padding and normalization. Do not upload a LiDAR
preview image in place of this tensor. The API permits 24 files, each at most 512 MiB; the
request totals about 330 MiB. Configure proxy body limits and server temporary disk capacity for
the uploaded controls, RGB conditions, and 264 MiB numeric tensor.

Submit to **`POST /v1/videos`**, poll `GET /v1/videos/{id}`, then download `/content` and
`/lidar`. The synchronous endpoint rejects `return_output=true`. Save `job.json`, the original
camera-major MP4, `multiview_camera_major.lidar.safetensors`, and its `.lidar.json` descriptor.
Polling timeout leaves the job available at its printed URL; rerunning submission creates a
new job. Failed HTTP responses are saved to `response_error.txt`.

The notebook validates `7 × 201 = 1407` camera-major frames, splits seven labeled camera
videos, and checks generated numeric shape `[3,67,128,1800]`, FP32 dtype, 10 Hz FPS, and embedded
metadata against the job descriptor. It displays the camera grid and saves
`sample_outputs.json`. Generated LiDAR remains in the downloaded safetensors file with its
metadata sidecar for further analysis.

### vLLM-Omni troubleshooting

- **`missing views[0].prompt`:** supply a nonempty `prompt` in every camera view, even for
  WSM-only input. Rerun the example's **Prepare the inputs** cell to rebuild `wsm_form`
  before submitting again; rerunning only the submission cell retains the old payload.
- **The server does not become ready:** confirm that `vllm/vllm-omni:cosmos3` was pulled, the model
  revision, exported class names, and `backbone_type`; then inspect the startup log before changing request code.
- **Compilation fails or exhausts memory:** confirm both required diffusion compilation flags are
  present exactly as shown in the server command.
- **`POST /v1/videos/sync` times out:** restart the server with
  `VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800`, or a larger hardware-appropriate value. This is a
  server-side import-time setting, so changing it after startup has no effect.
- **HTTP 400 before generation:** check that the camera keys are distinct and in canonical order,
  and that upload indexes are unique integers. WSM-only has one control per camera and no vision
  references; joint I2V has one control and one RGB reference per camera, and one numeric LiDAR
  reference. Numeric output requires the async API.
- **An input is missing or a Git LFS pointer:** rerun the verified sample download and joint preparation.
- **The sample download fails or a checksum does not match:** confirm read access to
  `nvidia/Cosmos3-Nano-Transfer-Auto` with `hf auth whoami`, then rerun **Fetch the sample**. For
  a local copy, point `AUTO_SAMPLE_ROOT` at its `sample_data` folder and keep its structure.
- **Joint generation fails after admission:** inspect `job.json` and the server log for the
  complete joint checkpoint/VAE, per-view captions, or numeric-value errors.
- **Media decode failure:** decode every source control locally and compare rational FPS, frame
  count, duration, pixel format, and codec. Native perspective and fisheye resolutions may differ;
  the notebook checks synchronization independently of geometry, then takes the first 201 frames
  from each control without resizing before uploading.
- **The response cannot be split:** the response must contain exactly
  `cameras * frames_per_view` camera-major frames. Preserve the server-returned stream before debugging the split.

## Related

- [Simulation cookbooks](../README.md) — shared setup, agent skills, and validation.
- [Cosmos3-Nano-Sim-Bimanual](../bimanual/bimanual.md) — action-conditioned robotic simulation.
- [Cosmos3-Nano-Sim-Auto](run_sim-auto_with_cosmos_framework.ipynb) — the causal, autoregressive
  student of this model, with a rolling cache and four-step sampling, run from this page's inputs.
- [Cosmos3-Nano-Sim-Transfer](../transfer/transfer.md) — caption-guided video generation from one
  spatial control.
- [Cosmos3 cookbook environment setup](../../README.md)
