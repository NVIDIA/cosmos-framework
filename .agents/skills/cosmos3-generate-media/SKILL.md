---
name: cosmos3-generate-media
description: >
  Generate images, videos, and audio-synchronized videos with the Cosmos3
  audiovisual generator models (Edge / Nano / Super / Super-4Step) through any
  of the seven supported serving paths: diffusers in-process, cosmos_framework
  torchrun, vLLM-Omni, SGLang, TensorRT-LLM, distilled TensorRT-LLM, or NIM.
  Use when the user asks "generate a video from a prompt", "run text2image /
  text2video / image2video with Cosmos3", "serve a Cosmos3 generator", or asks
  which Cosmos3 generation backend fits their GPU. Self-contained: sample
  prompts, reference images and negative prompts ship in this skill's assets/.
---

# Cosmos3 Audiovisual Media Generation

One skill for the whole audiovisual generation family: seven serving-path
variants that share the same model family, prompt format and bundled assets.

## Read this first

Before the first command of any variant — including a single-image smoke:

1. Read the variant's reference file (linked under its heading, with the
   sections the smoke path needs) in full, plus § 2.3–2.4, § 3.3, § 4.5, § 6
   and § 7 of the shared [`references/payloads.md`](references/payloads.md).
   The variant sections below only summarise the notebooks; the reference files
   are the notebooks migrated 1:1 (all but the Jupyter preview/viewer cells; Variant 7: the NIM cookbook pages).
2. Check the GPU with `nvidia-smi`: the CUDA major version picks the torch
   backend / dependency group, the memory picks the model (table below).
3. After a failure, go back to the reference section that covers the failing
   step before improvising a fix. Do not look for the original notebooks; the
   reference files replace them.

## When to use this skill

- Use to produce an image (t2i), video (t2v), image-conditioned video (i2v),
  video-to-video (v2v), or audio-synchronized video (t2av/i2av) from Cosmos3
  generator checkpoints
- Use to choose a serving backend (in-process vs server) and a model size
  that fits the available GPU
- For Reasoner (image/video understanding) inference this is the wrong skill
- For environment setup problems, hand off to `../cosmos3-env-troubleshoot/`
  (CUDA / import / server failures; TensorRT-LLM installation is in Variant 5)

## Bundled assets

Everything the t2i, t2v, i2v and audio use cases need ships in `assets/`
(structured-JSON prompts in the Cosmos3 prompt schema plus reference media; v2v
also needs a reference video that is not vendored, payloads.md § 3.4). Paths are
relative to this skill directory:

| Asset                | Path                                                                                                                                            | Used by          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| T2I prompt           | `assets/prompts/text2image/robot_draping.json`                                                                                                  | all t2i cases    |
| T2V prompt           | `assets/prompts/text2video/robot_kitchen.json`                                                                                                  | t2v (no audio)   |
| T2V+audio prompt     | `assets/prompts/text2video/robot_pouring_water_audio.json`                                                                                      | t2av             |
| I2V prompt           | `assets/prompts/image2video/car_driving.json`                                                                                                   | i2v (Nano/Super) |
| I2V prompt (Edge)    | `assets/prompts/image2video/car_driving_edge.json`                                                                                              | i2v on Edge      |
| I2V prompt (+audio)  | `assets/prompts/image2video/coastal_road_audio.json`                                                                                            | i2av             |
| I2V prompt (robot)   | `assets/prompts/image2video/humanoid_robot.json`                                                                                                | i2v              |
| I2V reference images | `assets/images/image2video/car_driving.jpg`, `assets/images/image2video/coastal_road_audio.jpg`, `assets/images/image2video/humanoid_robot.jpg` | i2v/i2av         |
| Negative prompts     | `assets/negative_prompts/text2video/neg_prompt.json`, `assets/negative_prompts/image2video/neg_prompt.json`                                     | video cases      |

Rules of the six `create_payload` notebooks (payloads.md § 3.3, § 7, § 9.2; the
NIM uses inline payloads): the prompt is the asset's JSON serialised compactly
(`json.dumps(asset, ensure_ascii=True, separators=(",", ":"))`); for `text2image`
the framework, diffusers, vLLM-Omni and SGLang payloads carry `"negative_prompt": ""`
(TensorRT-LLM leaves the key out) and the framework, diffusers and distilled
payloads set `num_frames = 1` (TensorRT-LLM drops `num_frames` / `fps`; vLLM-Omni
and SGLang send no frame fields); video modes load `assets/negative_prompts/<mode>/neg_prompt.json`
(video2video reuses the image2video file); the request size comes from the
`resolution` / `aspect_ratio` table, not from the asset's own `resolution`.

## Model sizes vs GPU

| Checkpoint                     | Size, as the sources state it                       | GPUs, as the notebooks serve it                                                                                                                             |
| ------------------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Edge`          | "compact 4B Nemotron-dense checkpoint" (trt-llm.md) | "fits on a single GPU"; 480p-native (832x480; 640x640 for TensorRT-LLM text-to-image)                                                                       |
| `nvidia/Cosmos3-Nano`          | "8B Nano" (setup guide, NIM table)                  | single GPU in every notebook                                                                                                                                |
| `nvidia/Cosmos3-Super`         | "the larger 64B model" (vllm-omni.md, sglang.md)    | four GPUs in every server notebook (`--tensor-parallel-size 4`, `--num-gpus 4`, the TensorRT-LLM 4-GPU config); framework: more GPUs via `--nproc-per-node` |
| `nvidia/Cosmos3-Super-*-4Step` | "the 32B checkpoint" (cosmos-framework.md)          | TensorRT-LLM: "Both checkpoints serve on a single GPU", one server at a time; framework: four GPUs with `--dp-shard-size=4`                                 |

All generator checkpoints are on Hugging Face; the notebooks require model access
there and `uvx hf@latest auth login` or `HF_TOKEN`. The safety checker
`nvidia/Cosmos-1.0-Guardrail` is a separately gated repository — for smoke runs
disable it with the backend's switch below.

## Disabling the guardrail, per backend

| Backend          | How to disable (README / notebook wording; payloads.md § 2.3)                                                                                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| diffusers        | `enable_safety_checker=False` (the notebook's `get_pipe` passes `safety_checker=None, enable_safety_checker=True` to keep it enabled)                                                                                                                                                    |
| cosmos_framework | `--no-guardrails`                                                                                                                                                                                                                                                                        |
| vLLM-Omni        | `--no-guardrails` on `vllm serve` (server-wide), or `guardrails: false` in `extra_params` / `extra_args` (per request). The server loads the gated guardrail at startup; without Hugging Face access to it the server exits before serving, and the per-request switch does not fix that |
| SGLang           | no switch documented in the notebook or README; the notebook's helpers send `guardrails: True` in `extra_params` (sglang.md § 6.2)                                                                                                                                                       |
| TensorRT-LLM     | `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` before starting the server, or `use_guardrails` set to `False` in the request `extra_params`                                                                                                                                                       |
| NIM              | `NIM_ENABLE_TEXT_GUARDRAILS=0 NIM_ENABLE_VIDEO_GUARDRAILS=0 NIM_ENABLE_SIGLIP_GUARDRAILS=0` (2.1.0 README; nim/operations.md: disabling the video guardrail also disables SigLIP)                                                                                                        |

## Variant 1 — diffusers (in-process, no server)

Reference: [`references/diffusers.md`](references/diffusers.md) — read it before
the first command (Quickstart; § 7 `get_pipe` / `run_diffusers_payload`; use-case table).

The notebook's venv, with the torch backend matching the driver's CUDA major
version (13.x → `cu130`, 12.x → `cu128`; check `nvidia-smi`):

```bash
export COSMOS3_DIFFUSERS_VENV="${COSMOS3_DIFFUSERS_VENV:-$PWD/.venv-cosmos3-diffusers}"
uv venv "$COSMOS3_DIFFUSERS_VENV" --python 3.13 --seed --managed-python --allow-existing
source "$COSMOS3_DIFFUSERS_VENV/bin/activate"
uv pip install --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg \
  torch torchvision transformers
```

> **Headless servers** (notebook § 1): if you see an error like `libxcb.so.1: cannot open
> shared object file` when importing or running the pipeline, install the required
> system libraries: `apt-get install -y libxcb1 libgl1 libglib2.0-0`. Setup guide
> (TensorRT-LLM section): `pip uninstall -y opencv-python && pip install opencv-python-headless==5.0.0.93`.

Text-to-image as the notebook runs it (`get_pipe` + the `text2image` branch of
`run_diffusers_payload`; `FIXED_SAMPLING`: 35 steps, guidance 6.0, shift 10.0, 720 × 1280,
seed 1234; `get_pipe` loads with `enable_safety_checker=True` — `False` is the § 1 switch; output path per § 4.2):

```python
import json, os, torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16,
                                           safety_checker=None, enable_safety_checker=False)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=10.0)
pipe.to("cuda")
prompt = json.dumps(json.load(open("assets/prompts/text2image/robot_draping.json")),
                    ensure_ascii=True, separators=(",", ":"))
result = pipe(prompt=prompt, negative_prompt="", num_frames=1, height=720, width=1280,
              num_inference_steps=35, guidance_scale=6.0,
              add_resolution_template=False, add_duration_template=False,
              generator=torch.Generator(device="cuda").manual_seed(1234))
os.makedirs("outputs/diffusers/t2i", exist_ok=True)   # output_dir.mkdir(parents=True, exist_ok=True)
result.video[0].save("outputs/diffusers/t2i/t2i.png")
```

README: "To run **Cosmos3-Edge** instead, load `nvidia/Cosmos3-Edge` and use its
single-GPU, no-audio settings: `height=480`, `width=832`, `num_frames=121`,
`num_inference_steps=20`, `guidance_scale=5.0`, and `flow_shift=8.0`" (`num_frames=1`
for text2image). Video modes pass the text2video prompt asset, `negative_prompt=json.dumps(<neg asset>)`,
`num_frames=189` (Nano/Super), `fps=24`, and `export_to_video(result.video, "<path>.mp4", fps=24)`.

## Variant 2 — cosmos_framework (torchrun, in-repo)

Reference: [`references/cosmos-framework.md`](references/cosmos-framework.md) —
read it before the first command (§ 2 Configure Paths and Environment, § 4
Install, the use-case table's `--checkpoint-path` column; payloads.md § 9.2).

Requires a framework checkout set up per `../cosmos3-setup/`. The inference input
is a full payload JSON — the prompt assets are NOT payloads by themselves; wrap
them as the notebook's `create_payload` does: absolute asset paths (`asset_path()`),
written under `<OUT>/pytorch/payloads/<use_case>/` (§ 4.1–4.2) before `cd "$COSMOS3_REPO"`:

```python
import json, os
from pathlib import Path
SKILL = Path.cwd().resolve()                               # run from this skill directory (the one with assets/)
OUT = Path(os.environ.get("OUT", "outputs")).resolve()     # payloads.md § 4.1
compact = lambda p: json.dumps(json.load(open(p)), ensure_ascii=True, separators=(",", ":"))
# create_payload("t2i_edge", backend="pytorch"): FIXED_SAMPLING + text2image rules + "resolution": "480" (Nano `t2i`: "720").
# Video cases: "text2video", the text2video prompt asset, negative_prompt = compact(<neg asset>), 189 frames (`t2v_edge_noaudio`: "480", 121).
payload = {"model_mode": "text2image", "name": "t2i_edge",
           "prompt": compact(SKILL / "assets/prompts/text2image/robot_draping.json"),
           "negative_prompt": "", "enable_sound": False, "num_steps": 35,
           "guidance": 6.0, "shift": 10.0, "fps": 24, "num_frames": 1,
           "resolution": "480", "aspect_ratio": "16,9", "seed": 0}
payload_path = OUT / "pytorch" / "payloads" / payload["name"] / f"{payload['name']}.json"
payload_path.parent.mkdir(parents=True, exist_ok=True)
payload_path.write_text(json.dumps(payload, indent=2) + "\n")
```

```bash
# Environment, as in the notebook's "Configure Paths and Environment" cell: run from the
# framework checkout root; the dependency group follows the driver's CUDA version
# (cu130-train for CUDA 13, cu128-train for CUDA 12.8); UV_PROJECT_ENVIRONMENT is
# respected when set, otherwise the checkout's .venv is used.
export OUT="${OUT:-$PWD/outputs}"                                   # still in the skill directory
export COSMOS3_REPO="${COSMOS3_REPO:-/path/to/cosmos-framework}"
cd "$COSMOS3_REPO"
export COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
export COSMOS3_UV_ENV="$UV_PROJECT_ENVIRONMENT"
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --group="$COSMOS3_UV_GROUP"
[ -x "$COSMOS3_UV_ENV/bin/python" ] || { echo "uv sync completed, but $COSMOS3_UV_ENV/bin/python is missing; check UV_PROJECT_ENVIRONMENT"; exit 1; }
LD_LIBRARY_PATH= "$COSMOS3_UV_ENV/bin/torchrun" --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference --parallelism-preset=throughput \
  --no-guardrails --checkpoint-path "Cosmos3-Edge" --seed=0 \
  -i "$OUT/pytorch/payloads/t2i_edge/t2i_edge.json" -o "$OUT/pytorch/t2i_edge"
```

`--checkpoint-path` takes the short names the notebook and README pass —
`Cosmos3-Edge`, `Cosmos3-Nano`, `Cosmos3-Super`, `Cosmos3-Super-Text2Image-4Step`,
`Cosmos3-Super-Image2Video-4Step` — or a local checkpoint directory (the notebook's
Nano FP8 example). `view_run` finds the result under the `-o` directory as `*.mp4`
(ignoring `*_preview.mp4` / `*_browser.mp4`) or `*.png` / `*.jpg` / `*.jpeg`. Edge runs
single-GPU (`--nproc-per-node=1`) but supports only 256p/480p, has no audio modules
(keep `"enable_sound": false`), and uses `"num_frames": 121` at `"fps": 24`. The 4-Step
students run on four GPUs with `--dp-shard-size=4` here; Super: "use more GPUs via `--nproc-per-node`".

## Variant 3 — vLLM-Omni (Docker server + OpenAI-style client)

Reference: [`references/vllm-omni.md`](references/vllm-omni.md) — read it before
the first command (§ 1 Prerequisites, § 2 Start the Server, § 6 request helpers).

The notebook's Nano launch (its `t2i` case runs against `Cosmos3-Nano`; the § 2
"Docker Image: Cosmos3-Edge" block differs only in the model name), with
`--no-guardrails` (§ 1: "To disable guardrails server-wide, pass `--no-guardrails`
to `vllm serve`"); the launch mounts the Hugging Face cache and the working directory at its host path and passes
`HF_TOKEN` through (export it — with guardrails enabled § 1 says "also set and pass `HF_TOKEN` to the container").

```bash
docker run --runtime nvidia --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -e HF_TOKEN \
  -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
  --omni \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --allowed-local-media-path / \
  --port 8000 \
  --init-timeout 1800 \
  --no-guardrails
```

Setup guide: "Cosmos3 checkpoints can exceed the default server init timeout —
always pass `--init-timeout 1800`"; vLLM-Omni prints `Application startup complete.`
when the API is ready; confirm it is serving with `curl http://localhost:8000/v1/models`.
Text-to-image uses `/v1/images/generations` (video modes: multipart `/v1/videos/sync`).
The notebook's `build_vllm_image_body` for `t2i` (`FIXED_SAMPLING`), with `guardrails` set
to `false` (the helper sends `true`; § 1 names the per-request switch), posted by `post_image` as
`curl -sS --fail-with-body -X POST <url> -H "Content-Type: application/json" -d '<compact JSON>'`, then `base64.b64decode(response["data"][0]["b64_json"])` → PNG:

```json
{"prompt": "<compact JSON of assets/prompts/text2image/robot_draping.json>",
 "size": "1280x720", "n": 1, "num_inference_steps": 35, "guidance_scale": 6.0,
 "flow_shift": 10.0, "seed": 0,
 "extra_args": {"use_resolution_template": false, "guardrails": false}}
```

## Variant 4 — SGLang (Docker server)

Reference: [`references/sglang.md`](references/sglang.md) — read it before the
first command (§ 2 Start the Server, § 6.2 SGLang request helpers, use-case table).

The notebook's "Set up HF Cache" and "Docker Image: Cosmos3-Edge" blocks ("Run
the SGLang server before running the request cells"):

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"     # SGLang container is run under root by default

docker run --runtime nvidia --gpus all \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Edge \
  --host 0.0.0.0
```

Keep `--host 0.0.0.0` and `-p 30000:30000` exactly as in the notebook's Docker serve
blocks (`COSMOS3_SGLANG_BASE_URL` defaults to `http://localhost:30000`). Text-to-image
uses `/v1/images/generations`; video uses the asynchronous `/v1/videos` route (POST, poll
`/v1/videos/<id>`, download `/v1/videos/<id>/content`). The notebook's `build_sglang_image_body`
for `t2i_edge` — `create_payload` sets `resolution "480"` (→ `832x480`) and removes `guidance` /
`shift` for Edge models, so those keys are sent as `null` — posted and decoded like Variant 3:

```json
{"prompt": "<compact JSON of assets/prompts/text2image/robot_draping.json>",
 "negative_prompt": "", "size": "832x480", "n": 1, "num_inference_steps": 35,
 "guidance_scale": null, "flow_shift": null, "seed": 0, "max_sequence_length": 4096,
 "response_format": "b64_json",
 "extra_params": {"use_resolution_template": false, "guardrails": true}}
```

## Variant 5 — TensorRT-LLM serve

Reference: [`references/trt-llm.md`](references/trt-llm.md) — read it before the
first command (§ 2 Start the Server, § 6 request helpers, "Cosmos3-Edge
Examples", Quickstart); install: the setup guide's "TensorRT-LLM Generator" section.

Setup guide: "Use a TensorRT-LLM build that includes the Cosmos3 VisualGen backend
and the modality required by your notebook. Install TensorRT-LLM following its
upstream documentation." Building from source is "the right path when you need a
checkout that contains a recent Cosmos3 VisualGen change before it is available in your installed package or release image":

```bash
apt-get update && apt-get -y install git git-lfs && git lfs install
git clone https://github.com/NVIDIA/TensorRT-LLM.git && cd TensorRT-LLM
git checkout bca6761ab84fbcd58fc7f914eade7de48b32e35e && git submodule update --init --recursive && git lfs pull
# Pick a devel tag from the upstream build-from-source guide or NGC, then inside
#   docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
#     --volume "$PWD":"$PWD" --workdir "$PWD" nvcr.io/nvidia/tensorrt-llm/devel:<tag>
apt-get update && apt-get -y install ffmpeg
python3 scripts/build_wheel.py --use_ccache --skip_building_wheel --linking_install_binary
pip install -e .          # Python-only changes: TRTLLM_USE_PRECOMPILED=1 pip install -e .
pip install cosmos_guardrail==0.3.0   # unless you explicitly disable guardrails before starting the server
```

Run the serve commands from inside the TensorRT-LLM checkout — the directory that
contains `examples/` — with TensorRT-LLM and `ffmpeg` installed in the active
environment. Notebook: "to disable [guardrails], set `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1`
before starting the server or set `use_guardrails` to `False` in the request `extra_params`":

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"
test -d "$TRTLLM_ROOT/examples/visual_gen/configs" \
  || echo "TRTLLM_ROOT=$TRTLLM_ROOT does not look like a TensorRT-LLM checkout"
export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1
trtllm-serve nvidia/Cosmos3-Edge --port 8000              # Edge: single GPU, no config override
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" --port 8000
# Super: torchrun --nproc_per_node=4 -m tensorrt_llm.commands.serve nvidia/Cosmos3-Super \
#   --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-super-4gpu.yaml" --port 8000
```

"TensorRT-LLM exposes `/health` when the server is ready": the notebook's
`check_trtllm_server` polls `GET <server root>/health` until 200 (up to 1800 s, every
10 s). Requests carry `Authorization: Bearer $COSMOS3_TRTLLM_API_KEY` (default
`tensorrt_llm`); the base URL is `COSMOS3_TRTLLM_BASE_URL` (`http://localhost:8000`).
Text-to-image is a native image request (README, "Edge: Text to Image"): it posts to
`/v1/images/generations` with `output_type="image"` in `extra_params` — "that flag is
what selects the image path; without it the server runs video mode, and video mode
defaults the negative prompt to Cosmos3's video negative prompt" — with `format: "png"`,
`response_format: "b64_json"` and no frame or frame-rate fields; decode `data[0].b64_json`.
Edge t2i runs at its native 640x640, guidance 4.0, 50 steps (`EDGE_T2I_SAMPLING`);
Nano/Super use `FIXED_SAMPLING` (1280x720, 35 steps, guidance 6.0); `max_sequence_length`
4096. The notebook's `build_trtllm_image_body` for `t2i_edge`, with `use_guardrails` set
to `false` (`TRTLLM_EXTRA_PARAMS` sends `true`; § 1 names the per-request switch):

```json
{"prompt": "<compact JSON of assets/prompts/text2image/robot_draping.json>",
 "size": "640x640", "format": "png", "response_format": "b64_json",
 "max_sequence_length": 4096, "seed": 0, "num_inference_steps": 50, "guidance_scale": 4.0,
 "extra_params": {"use_resolution_template": false, "use_duration_template": false,
                  "use_system_prompt": false, "use_guardrails": false, "enable_audio": false,
                  "output_type": "image"}}
```

## Variant 6 — distilled TensorRT-LLM (4-step Super students)

Reference: [`references/distilled-trt-llm.md`](references/distilled-trt-llm.md)
— read it before the first command (§ 1 Prerequisites, § 2 Start the Server,
§ 6 request helpers; TensorRT-LLM install as in Variant 5).

Serves `nvidia/Cosmos3-Super-Text2Image-4Step` / `nvidia/Cosmos3-Super-Image2Video-4Step`.
Notebook: "Both checkpoints serve on a single GPU", "run one server at a time on a
single GPU and point the matching environment variable at it". For smoke, only the
t2i case (single frame) is sensible. The notebook's server block (check the config path first — `trtllm-serve` validates `--visual_gen_args` only after it has started):

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"
test -f "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-t2i-1gpu.yaml" \
  || echo "TRTLLM_ROOT=$TRTLLM_ROOT does not look like a TensorRT-LLM checkout"
export TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1
trtllm-serve nvidia/Cosmos3-Super-Text2Image-4Step \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-t2i-1gpu.yaml" \
  --port 8000       # warms the deployed 1024x1024 single-frame shape
```

"TensorRT-LLM exposes `/health` when the server is ready. On startup it logs the
schedule it read from the checkpoint": `Distilled Cosmos3 checkpoint: fixed 4-step schedule [...]`.
`t2i_distilled` is a one-frame video request on `POST /v1/videos/generations` (`num_frames=1`,
`seconds=1`, `fps=8`; the notebook's only route helper is `video_api_url`), sent by `post_video`
with `Accept: video/mp4, video/x-msvideo, application/octet-stream` and the Bearer header of
Variant 5. "Leave `num_inference_steps` and `guidance_scale` out of the request" (the server
reads the four steps and guidance 1.0 from the checkpoint; other values are rejected with a
400) and "leave `use_system_prompt` unset". The response is the video bytes — MP4, or AVI when
the server has no `ffmpeg`. `build_trtllm_video_body` for `t2i_distilled`, with `use_guardrails` set to `false` (`TRTLLM_EXTRA_PARAMS` sends `true`):

```json
{"prompt": "<compact JSON of assets/prompts/text2image/robot_draping.json>",
 "size": "1024x1024", "seconds": 1, "fps": 8, "num_frames": 1,
 "max_sequence_length": 4096, "seed": 0,
 "extra_params": {"use_resolution_template": false, "use_duration_template": false,
                  "use_guardrails": false, "enable_audio": false}}
```

## Variant 7 — NIM (packaged container)

Reference: the Cosmos3 NIM cookbook in this checkout, [`../cookbooks/cosmos3/nim/`](../cookbooks/cosmos3/nim/README.md) —
[deployment.md](../cookbooks/cosmos3/nim/deployment.md) (How Selection Works, Launch Generator, Wait for Readiness, Verify the
Selection), [generation.md](../cookbooks/cosmos3/nim/generation.md) (Image-to-Video; Choose Resolution, Frames, and FPS),
[api-reference.md](../cookbooks/cosmos3/nim/api-reference.md), [operations.md](../cookbooks/cosmos3/nim/operations.md) — and its own
skill [`cosmos3-nim-user`](../cookbooks/cosmos3/nim/.agents/skills/cosmos3-nim-user/SKILL.md); read deployment.md and generation.md
before the first command. Upstream replaced `run_with_nim.ipynb` with this cookbook and its example scripts (cosmos-framework #296).

NIMs use NGC auth, not Hugging Face. One runtime per container (`NIM_MODEL_TYPE=generator`); `NIM_MODEL_VARIANT` selects the model
(`nano` in the README launch; `super` and the specialist variants: deployment.md "Select a Generator Model"), `NIM_PERF_PROFILE` the
objective (`latency` / `throughput`); add `-e NIM_PRECISION=fp8` only to pin precision. The README's launch, with `CONTAINER_NAME`,
`NIM_PORT` and `NIM_IMAGE` as this skill's parameters for the README's literals (authenticate Docker once; the cache is pre-created):

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin   # NGC_API_KEY must be exported: the launch passes it by name
export CONTAINER_NAME="${CONTAINER_NAME:-cosmos3-generator}"; export NIM_PORT="${NIM_PORT:-8000}"
export NIM_IMAGE="${NIM_IMAGE:-nvcr.io/nim/nvidia/cosmos3:2.1.0}"
export NIM_MODEL_VARIANT="${NIM_MODEL_VARIANT:-nano}"; export NIM_PERF_PROFILE="${NIM_PERF_PROFILE:-latency}"
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
mkdir -p "$LOCAL_NIM_CACHE"; chmod -R 777 "$LOCAL_NIM_CACHE" 2>/dev/null || true
docker run -d --name "$CONTAINER_NAME" \
  --gpus '"device=0"' --shm-size 16g --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
  -p "${NIM_PORT}:8000" \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator -e NIM_MODEL_VARIANT="$NIM_MODEL_VARIANT" -e NIM_PERF_PROFILE="$NIM_PERF_PROFILE" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
# Readiness (deployment.md): "A cold start (download, materialization, model load, warmup) takes much longer than HTTP startup. Send inference only after readiness succeeds."
until curl -fsS "http://127.0.0.1:${NIM_PORT}/v1/health/ready" >/dev/null; do sleep 10; done   # follow the start-up with: docker logs -f "$CONTAINER_NAME"
curl -fsS "http://127.0.0.1:${NIM_PORT}/v1/metadata" | python3 -m json.tool   # "Confirm that `model_type` and `inference_endpoint` are `generator` and `/v1/infer`"
```

Request (api-reference.md): one JSON object to `POST /v1/infer`; "Every request must set `model_mode`" and the server "rejects
unknown fields". Image-to-Video is `model_mode` `image2video` plus `input_reference` as a data URL (generation.md). The skill's
request from the bundled assets (compact-JSON `prompt` / `negative_prompt` as for the other backends, payloads.md § 9.1) is the
request of the cookbook's `../cookbooks/cosmos3/nim/examples/i2v.py` at smoke size (`"256"` / 25 frames are a skill choice; cookbook
defaults `720` / 189), run from this skill directory (the one with `assets/`). "Requests are synchronous and can stay quiet until
done … do not retry a quiet connection" (generation.md):

```python
import base64, json, os, urllib.request
from pathlib import Path
compact = lambda p: json.dumps(json.load(open(p)), ensure_ascii=True, separators=(",", ":"))
body = {"model_mode": "image2video",
        "prompt": compact("assets/prompts/image2video/car_driving.json"),
        "negative_prompt": compact("assets/negative_prompts/image2video/neg_prompt.json"),
        "input_reference": "data:image/jpeg;base64," + base64.b64encode(Path("assets/images/image2video/car_driving.jpg").read_bytes()).decode(),
        "resolution": "256", "num_frames": 25, "fps": 24.0, "num_inference_steps": 35, "guidance_scale": 6.0, "flow_shift": 10.0, "seed": 0}  # i2v.py's fields; 256 / 25 = smoke size
req = urllib.request.Request(f"http://127.0.0.1:{os.environ.get('NIM_PORT', '8000')}/v1/infer", json.dumps(body).encode(), {"Content-Type": "application/json"})
out = Path(os.environ.get("OUT", "outputs")) / "cosmos3_generator_nim" / "cosmos3_generator_nim_i2v.mp4"; out.parent.mkdir(parents=True, exist_ok=True)
with urllib.request.urlopen(req, timeout=1800) as resp:
    out.write_bytes(base64.b64decode(json.load(resp)["b64_video"])); print(f"HTTP {resp.status} -> {out} {out.stat().st_size} B")
```

**Scope (README):** "The Cosmos3 NIM serves text-to-image, text-to-video, image-to-video, and video-to-video, plus Action and
Transfer. It does not generate audio." `b64_video` decodes to an MP4 container (generation.md "Output and Playback"). When done, the
cookbook's cleanup: `docker rm -f "$CONTAINER_NAME"` (deployment.md; a stopped container still holds the name).

## Reference files

The full cookbook content for a backend (every model size, text-to-video, image/video
conditioning, audio, multi-GPU, the complete request contracts and helper code) lives in
`references/`, migrated 1:1 from the notebooks — read the matching file and
`references/payloads.md` before the first command (see the top). Do not look for the notebooks; these files replace them
(Variant 7: the NIM cookbook in `../cookbooks/cosmos3/nim/` replaced that notebook upstream — see its row).

| File                                                              | Backend                                                                                                                                                |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `references/payloads.md`                                          | Shared: assets, prompt serialisation, sampling defaults, use-case matrix                                                                               |
| `references/diffusers.md`                                         | Variant 1 — diffusers                                                                                                                                  |
| `references/cosmos-framework.md`                                  | Variant 2 — cosmos_framework                                                                                                                           |
| `references/vllm-omni.md`                                         | Variant 3 — vLLM-Omni                                                                                                                                  |
| `references/sglang.md`                                            | Variant 4 — SGLang                                                                                                                                     |
| `references/trt-llm.md`                                           | Variant 5 — TensorRT-LLM                                                                                                                               |
| `references/distilled-trt-llm.md`                                 | Variant 6 — distilled 4-step students                                                                                                                  |
| [`../cookbooks/cosmos3/nim/`](../cookbooks/cosmos3/nim/README.md) | Variant 7 — NIM: the Cosmos3 NIM cookbook (deployment, generation, api-reference, operations) and its `cosmos3-nim-user` skill; not part of this skill |

Origin: the `cookbooks/cosmos3/generator/audiovisual` notebooks of the public Cosmos
repository, pinned at <https://github.com/NVIDIA/cosmos/tree/6e2c69afafd6444c1ca14a021aba93fcc6b7996a/cookbooks/cosmos3/generator/audiovisual>
(Variant 7: the `cosmos3/nim/` cookbook pages of this checkout); superseded by this skill.

## Things not obvious from the docs

- **Guardrail gating**: default server configs load the access-gated
  `nvidia/Cosmos-1.0-Guardrail`; smoke runs should disable guardrails (see
  flags above) rather than acquire access.
- **Server-mode variants are client/server**: vLLM/SGLang/TRT-LLM/NIM need
  the server started, health-polled, and torn down by whoever automates them.
- **Checkpoint cache**: the notebooks `export HF_HOME=/path/to/large/huggingface/cache`
  (setup guide: "Nano downloads plus CUDA dependencies can take tens of GiB"); the
  vLLM-Omni container sees it only through the `-v "${HF_HOME}:/root/.cache/huggingface"`
  mount, SGLang mounts its own `SGLANG_HF_CACHE`.
- **`LD_LIBRARY_PATH=` (empty)** is prefixed to every `torchrun` run cell of the
  cosmos_framework notebook (`CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" LD_LIBRARY_PATH= "$COSMOS3_UV_ENV/bin/torchrun" ...`).

## Related skills

| Skill                                                               | When to use                           |
| ------------------------------------------------------------------- | ------------------------------------- |
| `../cosmos3-setup/SKILL.md`                                         | Framework environment for Variant 2   |
| `../cosmos3-inference/SKILL.md`                                     | Framework-native inference reference  |
| `../cosmos3-env-troubleshoot/SKILL.md`                              | CUDA/import/server failures           |
| `../cosmos3-skill-authoring/SKILL.md`                               | Editing this skill's eval cases       |
| `../cookbooks/cosmos3/nim/.agents/skills/cosmos3-nim-user/SKILL.md` | Operating the Cosmos3 NIM (Variant 7) |
