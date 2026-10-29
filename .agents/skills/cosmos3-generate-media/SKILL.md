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
   and § 7 of the shared [`references/payloads.md`](references/payloads.md), plus
   [`references/skill-notes.md`](references/skill-notes.md) (the cross-variant skill notes).
   The variant sections below only summarise the notebooks; the reference files
   are the notebooks migrated 1:1 (all but the Jupyter preview/viewer cells; Variant 7: the NIM cookbook pages).
2. Check the GPU with `nvidia-smi`: the CUDA major version picks the torch
   backend / dependency group, the memory picks the model (table below).
3. uv: this skill directory usually sits inside a cosmos-framework checkout whose
   `pyproject.toml` pins `[tool.uv] required-version = "==0.11.29"`, and uv applies
   that pin from any cwd under the checkout. Variant 1 (standalone diffusers venv)
   → pass `--no-config`; Variant 2 (`uv sync` in the checkout) → `uv` must be
   exactly 0.11.29 (payloads.md § 2.4; `../cosmos3-setup/SKILL.md` step 3).
   Variants 1, 5 and 6 (standalone venvs) keep the stock uv (0.13+) and pass
   `--no-config`; do NOT install uv 0.11.29 for them (why: `references/skill-notes.md` § 6).
4. After a failure, go back to the reference section that covers the failing
   step before improvising a fix. Do not look for the original notebooks; the
   reference files replace them.
5. Verify what you produced before reporting it (see Verify the output below):
   ffprobe or the venv's `av` for video, Pillow for images; record the facts in
   `result_summary.md`.

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

Payload rules of the six `create_payload` notebooks (serialisation, `negative_prompt`, `num_frames`, request size) and the NIM's prompt form: `references/skill-notes.md` § 1.

## Model sizes vs GPU

| Checkpoint                     | Size, as the sources state it                                                                                                                 | GPUs, as the notebooks serve it                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Edge`          | "compact 4B Nemotron-dense checkpoint" (trt-llm.md)                                                                                           | "fits on a single GPU"; 480p-native (832x480; 640x640 for TensorRT-LLM text-to-image)                                                                                                                                                                                                                                                                                                                                                              |
| `nvidia/Cosmos3-Nano`          | "8B Nano" (retired Generator NIM 1.0.0 table of the setup guide @ 6e2c69af)                                                                   | single GPU in every notebook                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `nvidia/Cosmos3-Super`         | "the larger 64B model" (vllm-omni.md, sglang.md)                                                                                              | four GPUs in every server notebook (`--tensor-parallel-size 4`, `--num-gpus 4`, the TensorRT-LLM 4-GPU config); framework: more GPUs via `--nproc-per-node`                                                                                                                                                                                                                                                                                        |
| `nvidia/Cosmos3-Super-*-4Step` | "the 32B checkpoint" (cosmos-framework.md; the vLLM/SGLang notebooks say 64B); measured: 131.4 GB = 122.4 GiB of bf16 safetensors per student | TensorRT-LLM: "Both checkpoints serve on a single GPU" — only one with ≥ ~125 GiB free (H200/B200-class); on H100 80 GB / NVL 94 GB (any GPU under ~125 GiB) the documented launch OOMs at load and needs `cpu_offload_config` with ≥ ~128 GiB host RAM, or — Text2Image-4Step only — the patched `fp8` branch (Variant 6); vLLM/setup guide: `--tensor-parallel-size 4 --enable-layerwise-offload`; framework: four GPUs with `--dp-shard-size=4` |

Measured footprints per backend (eval runs 2026-10-21/23, one H100 80 GB; "8B Nano" caveat): `references/skill-notes.md` § 2.

All generator checkpoints are on Hugging Face; the notebooks require model access
there and `uvx hf@latest auth login` or `HF_TOKEN`. The safety checker
`nvidia/Cosmos-1.0-Guardrail` is a separately gated repository — for smoke runs
disable it with the backend's switch (table: `references/skill-notes.md` § 3; every variant block below carries its switch).

## Variant 1 — diffusers (in-process, no server)

Reference: [`references/diffusers.md`](references/diffusers.md) — read it before
the first command (Quickstart; § 7 `get_pipe` / `run_diffusers_payload`; use-case table).

The notebook's venv, with the torch backend matching the driver's CUDA major
version (13.x → `cu130`, 12.x → `cu128`; check `nvidia-smi`), plus the setup guide's
headless OpenCV swap:

```bash
export COSMOS3_DIFFUSERS_VENV="${COSMOS3_DIFFUSERS_VENV:-$PWD/.venv-cosmos3-diffusers}"
# This skill sits inside the cosmos-framework checkout, whose pyproject.toml pins uv ==0.11.29 for the
# framework venv; --no-config (or UV_NO_CONFIG=1) keeps that pin, and the framework's other [tool.uv]
# settings, away from this standalone venv. UV_* environment variables still apply. Running the two
# commands from a cwd outside the checkout (with an absolute venv path) has the same effect.
uv venv "$COSMOS3_DIFFUSERS_VENV" --python 3.13 --seed --managed-python --allow-existing --no-config
source "$COSMOS3_DIFFUSERS_VENV/bin/activate"
# The export and the activation above exist only in the shell that ran them. Run this block as ONE shell call; a line
# re-run in a separate call needs the export line again. Every uv/python line below therefore names the venv itself
# (--python "$COSMOS3_DIFFUSERS_VENV", "$COSMOS3_DIFFUSERS_VENV/bin/python"): with the variable set it targets this venv
# whether or not it is activated, with the variable unset it stops ("No virtual environment found") instead of
# installing elsewhere (eval run 2026-10-26: `uv pip install` run in a later call without the export/activation went
# into the evaluator's own venv, and the import check then ran against that interpreter).
# Cosmos3OmniPipeline imports cosmos_guardrail -> cv2 at import time, also with enable_safety_checker=False;
# cosmos_guardrail pulls the GUI opencv-python wheel (needs libxcb/libGL). The headless swap (setup guide,
# TensorRT-LLM section) is part of this install step — one compound command, so the import check below cannot
# run before it (eval run 2026-10-24: an agent ran the check before the separate swap lines -> libxcb ImportError).
uv pip install --no-config --python "$COSMOS3_DIFFUSERS_VENV" --torch-backend=cu130 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  accelerate av cosmos_guardrail huggingface_hub imageio imageio-ffmpeg \
  torch torchvision transformers \
  && uv pip uninstall --no-config --python "$COSMOS3_DIFFUSERS_VENV" opencv-python \
  && uv pip install --no-config --python "$COSMOS3_DIFFUSERS_VENV" opencv-python-headless==5.0.0.93
"$COSMOS3_DIFFUSERS_VENV/bin/python" -c "import cv2; from diffusers import Cosmos3OmniPipeline; print('pipeline import OK')"   # must pass before any checkpoint download
```

Notes: `references/diffusers.md` § 8.1 (headless servers — the apt fallback). Text-to-image: run § 8.2 of
`references/diffusers.md` (the Nano snippet) as ONE file with `"$COSMOS3_DIFFUSERS_VENV/bin/python"`, verbatim, from this
skill directory (the one with `assets/`); do not compose the pipeline call from the prose.

Edge (`t2i_edge`, `t2v_edge_noaudio`): run § 8.4 of `references/diffusers.md` the same way — its setup block plus only the block of the requested mode, verbatim; § 8.3 has the README's
Edge settings and the video-mode rules (`num_frames=189`, `fps=24`, `export_to_video`).

## Variant 2 — cosmos_framework (torchrun, in-repo)

Reference: [`references/cosmos-framework.md`](references/cosmos-framework.md) —
read it before the first command (§ 2 Configure Paths and Environment, § 4
Install, the use-case table's `--checkpoint-path` column; payloads.md § 9.2).

Requires a framework checkout set up per `../cosmos3-setup/` (its step 3 installs uv
0.11.29). The inference input
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
# The framework pins [tool.uv] required-version (==0.11.29 today); any other uv aborts `uv sync`.
# `uv self update` only works for a standalone-installer uv, so install the pinned version side-by-side.
REQ_UV="$(sed -n 's/^required-version = "==\(.*\)"/\1/p' pyproject.toml)"; REQ_UV="${REQ_UV:-0.11.29}"
if [ "$(uv --version 2>/dev/null | awk '{print $2}')" != "$REQ_UV" ]; then
  curl -LsSf "https://astral.sh/uv/$REQ_UV/install.sh" | env UV_INSTALL_DIR="$HOME/.local/uv-$REQ_UV" INSTALLER_NO_MODIFY_PATH=1 sh
  export PATH="$HOME/.local/uv-$REQ_UV:$PATH"
  # this PATH change is for `uv sync` in the checkout only — do not reuse this uv for Variants 1/5/6
fi
GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --group="$COSMOS3_UV_GROUP"
[ -x "$COSMOS3_UV_ENV/bin/python" ] || { echo "uv sync completed, but $COSMOS3_UV_ENV/bin/python is missing; check UV_PROJECT_ENVIRONMENT"; exit 1; }
LD_LIBRARY_PATH= "$COSMOS3_UV_ENV/bin/torchrun" --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference --parallelism-preset=throughput \
  --no-guardrails --checkpoint-path "Cosmos3-Edge" --seed=0 \
  -i "$OUT/pytorch/payloads/t2i_edge/t2i_edge.json" -o "$OUT/pytorch/t2i_edge"
```

Notes: `references/cosmos-framework.md` § 8.1 (install-free `uv sync`), § 8.2 (`--checkpoint-path` names, where `view_run` finds
the output, Edge / Super / 4-Step run settings).

## Variant 3 — vLLM-Omni (Docker server + OpenAI-style client)

Reference: [`references/vllm-omni.md`](references/vllm-omni.md) — read it before
the first command (§ 1 Prerequisites, § 2 Start the Server, § 6 request helpers).

The notebook's Nano launch (its `t2i` case runs against `Cosmos3-Nano`; the § 2
"Docker Image: Cosmos3-Edge" block differs only in the model name), with
`--no-guardrails` (§ 1: "To disable guardrails server-wide, pass `--no-guardrails`
to `vllm serve`"); the launch mounts the Hugging Face cache and the working directory at its host path and passes
`HF_TOKEN` through (export it — with guardrails enabled § 1 says "also set and pass `HF_TOKEN` to the container";
why the token matters: `references/vllm-omni.md` § 7.1).

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
Headless: see the pattern in `references/skill-notes.md` § 4 (`-d --name cosmos3-vllm`,
poll `/v1/models`, `docker stop && docker rm`).
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
the SGLang server before running the request cells"), plus two `-e` lines that are not in
the notebook (eval-verified against `lmsysorg/sglang:dev`): `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1`,
because the server loads the gated guardrail at startup and otherwise exits with
`GatedRepoError 401`; and `HF_TOKEN`, because anonymous Hub calls are rate-limited (HTTP 429
→ incomplete-checkpoint `ValueError` on a cold cache):

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"     # SGLang container is run under root by default

docker run --runtime nvidia --gpus all \
  -e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
  -e HF_TOKEN \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$(pwd):/workspace" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Edge \
  --host 0.0.0.0
```

Readiness (not in
the notebook; measured on 1x H100 80 GB, eval runs 2026-10-21/23): poll
`GET http://localhost:30000/health` until HTTP 200 `{"status":"ok"}` — it returns 503 after
`Application startup complete.` while the warm-up runs and turns 200 after
`The server is fired up and ready to roll!` (figures: `references/sglang.md` § 7.1). Headless: `-d --name cosmos3-sglang`,
`docker stop && docker rm` (pattern in `references/skill-notes.md` § 4). Text-to-image
uses `/v1/images/generations`; video uses the asynchronous `/v1/videos` route (POST, poll
`/v1/videos/<id>`, download `/v1/videos/<id>/content`). The notebook's `build_sglang_image_body`
for `t2i_edge` — `create_payload` sets `resolution "480"` (→ `832x480`) and removes `guidance` /
`shift` for Edge models, so those keys are sent as `null` — posted and decoded like Variant 3,
with `guardrails` set to `false` (the helper sends `true`):

```json
{"prompt": "<compact JSON of assets/prompts/text2image/robot_draping.json>",
 "negative_prompt": "", "size": "832x480", "n": 1, "num_inference_steps": 35,
 "guidance_scale": null, "flow_shift": null, "seed": 0, "max_sequence_length": 4096,
 "response_format": "b64_json",
 "extra_params": {"use_resolution_template": false, "guardrails": false}}
```

## Variant 5 — TensorRT-LLM serve

Reference: [`references/trt-llm.md`](references/trt-llm.md) — read it before the
first command (§ 2 Start the Server, § 6 request helpers, "Cosmos3-Edge
Examples", Quickstart); install: Route A below (prebuilt wheel) or the setup guide's
"TensorRT-LLM Generator" section (source build).

**Route A (default: bare GPU host, no Docker)** — the `tensorrt_llm==1.3.0rc27` wheel (§ 7.1 of `references/trt-llm.md`: why
that pin): run § 7.2 of `references/trt-llm.md` as ONE shell call, verbatim (save it as a script: its preflights `exit 1`); do not
compose an install command from the prose. § 7.3: why the pin is fetched as a SHA and the checkout is YAML source only.

**Route B (source build in the devel container)** — the setup guide's path: run § 7.4 of `references/trt-llm.md` as ONE shell
call inside the devel container, verbatim. § 7.5: why the `TRTLLM_USE_PRECOMPILED=1` shortcut must not be used on this
checkout. § 7.6: only if you keep guardrails on (not the smoke path).

Serve: run the Edge or Nano line of § 7.7 of `references/trt-llm.md`, verbatim, from the Route A venv — in the same shell call, first repeat the
`TRTLLM_VENV` export, the `source "$TRTLLM_VENV/bin/activate"` and the `CUDA_HOME` export of § 7.2 (a new shell call has none of them; Route B's
devel container has the install and `CUDA_HOME` already); § 7.7 itself sets only `TRTLLM_ROOT` and `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1`. Do not
compose a serve command from the prose. § 7.8: the explicit `trtllm-serve serve` form, the Nano YAML fits 80 GB (memory is set by the `--visual_gen_args` YAML; `compilation_config` for smaller GPUs); headless launch / teardown: `references/skill-notes.md` § 4.

"TensorRT-LLM exposes `/health` when the server is ready": the notebook's
`check_trtllm_server` polls `GET <server root>/health` until 200 (up to 1800 s, every
10 s). Image-route facts (rc26 diagnosis, `output_type`, sampling per model): `references/trt-llm.md` § 7.9.
Requests carry `Authorization: Bearer $COSMOS3_TRTLLM_API_KEY` (default
`tensorrt_llm`); the base URL is `COSMOS3_TRTLLM_BASE_URL` (`http://localhost:8000`).
Nano/Super t2i: the same body as below with `"size": "1280x720", "num_inference_steps": 35, "guidance_scale": 6.0`
(`FIXED_SAMPLING`, payloads.md § 6.1; `num_frames` / `fps` stay omitted) — verified 1280x720
PNG in 6.3 s (eval run). The notebook's `build_trtllm_image_body` for `t2i_edge`, with `use_guardrails` set
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
single GPU and point the matching environment variable at it" (a single GPU that holds
~125 GiB; the bf16 weights alone are 122.4 GiB — under that, the launch script (§ 7.2 of the reference) picks the
route by host RAM). For smoke, only the t2i case (single frame) is sensible.

Run § 7.2 of `references/distilled-trt-llm.md` as ONE shell call, verbatim — save it as `variant6-serve.sh` and run
`bash variant6-serve.sh`; do not compose a launch command from the prose. § 7.1: why this script, not the notebook's launch, is
the documented first attempt on every host (two gates: GPU memory, host RAM available); § 7.3: the `plain` route = the notebook's
command; § 7.4: measured (offload and `fp8` routes); § 7.5: still unverified — not routes.

"TensorRT-LLM exposes `/health` when the server is ready. On startup it logs the
schedule it read from the checkpoint": `Distilled Cosmos3 checkpoint: fixed 4-step schedule [...]`.
`t2i_distilled` is a one-frame video request on `POST /v1/videos/generations` (`num_frames=1`,
`seconds=1`, `fps=8`; the notebook's only route helper is `video_api_url`), sent by `post_video`
with `Accept: video/mp4, video/x-msvideo, application/octet-stream` and the Bearer header of
Variant 5. "Leave `num_inference_steps` and `guidance_scale` out of the request" (the server
reads the four steps and guidance 1.0 from the checkpoint; other values are rejected with a
400) and "leave `use_system_prompt` unset". The response is the video bytes — MP4, or AVI when
the server has no `ffmpeg`. It is a one-frame video, not a PNG: for a still image extract the
frame from the path `run_trtllm_payload` returns (`<OUT>/trt_llm_distilled/t2i_distilled/t2i_distilled.mp4`, or `.avi`):
`ffmpeg -nostdin -y -i <OUT>/trt_llm_distilled/t2i_distilled/t2i_distilled.mp4 -frames:v 1 <OUT>/t2i_4step.png`
(1024x1024; ffmpeg is already the § 1 server prerequisite). If the video route answers HTTP 500 `Video generation failed`: `references/distilled-trt-llm.md` § 7.6 (documented image-route fallback, needs rc27).
`build_trtllm_video_body` for `t2i_distilled`, with `use_guardrails` set to `false` (`TRTLLM_EXTRA_PARAMS` sends `true`):

```json
{"prompt": "<compact JSON of assets/prompts/text2image/robot_draping.json>",
 "size": "1024x1024", "seconds": 1, "fps": 8, "num_frames": 1,
 "max_sequence_length": 4096, "seed": 0,
 "extra_params": {"use_resolution_template": false, "use_duration_template": false,
                  "use_guardrails": false, "enable_audio": false}}
```

Headless launch / teardown: the launch IS the § 7.2 script (it already does `setsid`, the
`/health` poll and records `serve.pid`); the teardown is the `kill -INT` → `-TERM` → `-KILL` +
leftover sequence of the pattern in `references/skill-notes.md` § 4, with `PID=$(cat serve.pid)`.
Never start a second server while one is still loading or running on the same GPU — stop it
(teardown: `references/skill-notes.md` § 4) and wait for `nvidia-smi --query-gpu=memory.used` to read 0 MiB first; the launch
script refuses to start otherwise. Notes: `references/distilled-trt-llm.md` § 7.7 (launch mechanism, teardown escalation, leftovers).

## Variant 7 — NIM (packaged container)

Reference: the Cosmos3 NIM cookbook in this checkout, [`../cookbooks/cosmos3/nim/`](../cookbooks/cosmos3/nim/README.md) —
[deployment.md](../cookbooks/cosmos3/nim/deployment.md) (How Selection Works, profile preflight, Launch Generator, Wait for
Readiness, Verify the Selection), [generation.md](../cookbooks/cosmos3/nim/generation.md) (Image-to-Video; Choose Resolution,
Frames, and FPS), [api-reference.md](../cookbooks/cosmos3/nim/api-reference.md), [operations.md](../cookbooks/cosmos3/nim/operations.md) —
and its own skill [`cosmos3-nim-user`](../cookbooks/cosmos3/nim/.agents/skills/cosmos3-nim-user/SKILL.md); read deployment.md and
generation.md before the first command. Upstream replaced `run_with_nim.ipynb` with this cookbook and its example scripts (`../cookbooks/cosmos3/nim/examples/`)
(cosmos-framework #296), so this skill keeps only the README quickstart, the request it sends from the bundled assets, and its own additions (marked). The
checkout's copy is the 2.0.0 edition (it pins `cosmos3:2.0.0`; quotes below are the pinned 2.1.0 wording) — flags, selectors, switches
and the `/v1/infer` contract are identical in both; keep `NIM_IMAGE` at `cosmos3:2.1.0`.

NIMs use NGC auth, not Hugging Face. One runtime per container (`NIM_MODEL_TYPE=generator`); explicit selection, as the cookbook
documents it, is the selector triple `NIM_MODEL_VARIANT` (`nano` — the support matrix's recommendation on H100) / `NIM_PRECISION`
(unset: "Generator prefers FP8 when compatible") / `NIM_PERF_PROFILE=latency`, confirmed by the pre-download profile preflight
(prints the chosen profile) and by `/v1/metadata` after readiness; pin `NIM_MODEL_PROFILE` only when relaunching after a failed
first launch. Skill additions to the README's launch: the cache preflight (the mount must be writable by the NIM user, uid 1000;
why: skill-notes.md § 4), the three guardrail `-e` lines (payloads.md § 1; NIM defaults are `true`), the `NIM_MODEL_PROFILE`
pass-through, the `CONTAINER_NAME` / `NIM_PORT` / `NIM_IMAGE` parameters, and the bounded readiness loop:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin   # NGC_API_KEY must be exported: the launch passes it by name
export CONTAINER_NAME="${CONTAINER_NAME:-cosmos3-generator}"; export NIM_PORT="${NIM_PORT:-8000}"   # name or port "already in use" = a previous container: docker logs it, then docker rm -f it or change these
export NIM_IMAGE="${NIM_IMAGE:-nvcr.io/nim/nvidia/cosmos3:2.1.0}"; docker pull "$NIM_IMAGE"
export NIM_MODEL_VARIANT="${NIM_MODEL_VARIANT:-nano}"; export NIM_PERF_PROFILE="${NIM_PERF_PROFILE:-latency}"
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim}"
mkdir -p "$LOCAL_NIM_CACHE"   # deployment.md "Prepare the Model Cache"; its chmod 777 step is replaced by the writability check below — a shared cache is not modified
# Pre-download profile preflight (deployment.md): the launch's selectors and GPU exposure, no NGC_API_KEY; prints the selected profile
docker run --rm --gpus '"device=0"' -e NIM_MODEL_TYPE=generator -e NIM_MODEL_VARIANT="$NIM_MODEL_VARIANT" -e NIM_PERF_PROFILE="$NIM_PERF_PROFILE" \
  --entrypoint /bin/bash "$NIM_IMAGE" -lc 'set -e; /opt/nim/.venv/bin/python -m profile_selection.startup --output /tmp/cosmos3-preflight.env; printf "Profile preflight passed.\n"; cat /tmp/cosmos3-preflight.env'
# Cache preflight (skill addition): the NIM runs as uid 1000 and the daemon resolves the mount — test write access first; repair ownership only for an empty cache this run created, or with NIM_CACHE_REPAIR=1 (explicit opt-in) — a cache that holds anything else is never modified silently
cache_ok() { docker run --rm --entrypoint /bin/sh -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" "$NIM_IMAGE" -c 'touch /opt/nim/.cache/.w && rm /opt/nim/.cache/.w' 2>/dev/null; }
cache_ok || { { [ "${NIM_CACHE_REPAIR:-0}" = 1 ] || [ -z "$(ls -A "$LOCAL_NIM_CACHE")" ]; } && docker run --rm --user 0 --entrypoint /bin/sh -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" "$NIM_IMAGE" -c 'chown -R 1000:1000 /opt/nim/.cache && chmod -R u+rwX /opt/nim/.cache'; cache_ok || { echo "LOCAL_NIM_CACHE=$LOCAL_NIM_CACHE is not writable by the NIM user (uid 1000): use a directory of your own or set NIM_CACHE_REPAIR=1 to let the NIM image take ownership of it"; exit 1; }; }
docker run -d --name "$CONTAINER_NAME" \
  --gpus '"device=0"' --shm-size 16g --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
  -p "${NIM_PORT}:8000" \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator -e NIM_MODEL_VARIANT="$NIM_MODEL_VARIANT" -e NIM_PERF_PROFILE="$NIM_PERF_PROFILE" \
  -e NIM_ENABLE_TEXT_GUARDRAILS="${NIM_ENABLE_TEXT_GUARDRAILS:-0}" -e NIM_ENABLE_VIDEO_GUARDRAILS="${NIM_ENABLE_VIDEO_GUARDRAILS:-0}" -e NIM_ENABLE_SIGLIP_GUARDRAILS="${NIM_ENABLE_SIGLIP_GUARDRAILS:-0}" \
  ${NIM_MODEL_PROFILE:+-e NIM_MODEL_PROFILE="$NIM_MODEL_PROFILE"} \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
# Readiness (deployment.md: "Send inference only after readiness succeeds"); skill note: container-exit check, 40-min deadline (the eval cell budget is 60); record in result_summary.md: image tag, selected profile, time-to-ready, request latency
START=$(date +%s); until curl -fsS "http://127.0.0.1:${NIM_PORT}/v1/health/ready" >/dev/null; do docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" | grep -q true || { docker logs --tail 160 "$CONTAINER_NAME"; echo "container exited"; break; }; [ $(( $(date +%s) - START )) -lt 2400 ] || { docker logs --tail 160 "$CONTAINER_NAME"; echo "not ready after 40 min"; break; }; sleep 10; done; echo "time-to-ready $(( $(date +%s) - START )) s"
curl -fsS "http://127.0.0.1:${NIM_PORT}/v1/metadata" | python3 -m json.tool   # "Confirm that `model_type` and `inference_endpoint` are `generator` and `/v1/infer`"; record the selected profile; `/v1/manifest` lists the image's profiles
```

Request (api-reference.md): one JSON object to `POST /v1/infer`; "Every request must set `model_mode`" and the server "rejects
unknown fields" (HTTP 422, strict JSON types). Image-to-Video is `model_mode` `image2video` plus `input_reference` as a data URL
(generation.md). The skill's request from the bundled assets (compact-JSON `prompt` / `negative_prompt` as for the other backends,
payloads.md § 9.1) is the request of the cookbook's `../cookbooks/cosmos3/nim/examples/i2v.py` at smoke size (`"256"` / 25 frames are a skill choice; cookbook
defaults `720` / 189). Run it from this skill directory (the one with `assets/`) as ONE script call with a long timeout — the first
request may include compilation (`NIM_ENABLE_TORCH_COMPILE` default `true`, configuration.md); do not retry a quiet connection (generation.md):

```python
import base64, json, os, time, urllib.request
from pathlib import Path
compact = lambda p: json.dumps(json.load(open(p)), ensure_ascii=True, separators=(",", ":"))
body = {"model_mode": "image2video",
        "prompt": compact("assets/prompts/image2video/car_driving.json"),
        "negative_prompt": compact("assets/negative_prompts/image2video/neg_prompt.json"),
        "input_reference": "data:image/jpeg;base64," + base64.b64encode(Path("assets/images/image2video/car_driving.jpg").read_bytes()).decode(),
        "resolution": "256", "num_frames": 25, "fps": 24.0, "num_inference_steps": 35, "guidance_scale": 6.0, "flow_shift": 10.0, "seed": 0}  # i2v.py's fields; 256 / 25 = smoke size
req = urllib.request.Request(f"http://127.0.0.1:{os.environ.get('NIM_PORT', '8000')}/v1/infer", json.dumps(body).encode(), {"Content-Type": "application/json"})
out = Path(os.environ.get("OUT", "outputs")) / "cosmos3_generator_nim" / "cosmos3_generator_nim_i2v.mp4"; out.parent.mkdir(parents=True, exist_ok=True)  # if the task names the file (e.g. outputs/i2v_nim.mp4), write or copy the verified MP4 there too
t0 = time.monotonic()
with urllib.request.urlopen(req, timeout=1800) as resp:
    out.write_bytes(base64.b64decode(json.load(resp)["b64_video"])); print(f"HTTP {resp.status} model_mode={body['model_mode']} {body['resolution']}/{body['num_frames']}f latency {time.monotonic()-t0:.1f} s -> {out} {out.stat().st_size} B")
```

**Scope (README):** "The Cosmos3 NIM serves text-to-image, text-to-video, image-to-video, and video-to-video, plus Action and
Transfer. It does not generate audio." Output is VP9-in-MP4 (`b64_video`) — check it on the host, never inside the container
(skill-notes.md § 4–5). When done (skill addition — the cookbook's `docker rm -f cosmos3-generator` discards the logs):
`docker stop "$CONTAINER_NAME"; mkdir -p "${OUT:-outputs}/cosmos3_generator_nim"; docker logs "$CONTAINER_NAME" > "${OUT:-outputs}/cosmos3_generator_nim/container.log" 2>&1; docker rm "$CONTAINER_NAME"`.

## Verify the output

Check what you produced before reporting it (skill addition — the notebooks only
previewed outputs inline in Jupyter):

- Video (`.mp4`): `ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration -of default=nw=1 <file>`.
  `ffprobe` is on neither a bare host nor in the venv (imageio-ffmpeg ships only an `ffmpeg`
  binary): `apt-get update && apt-get install -y ffmpeg` (root), or root-free with the `av`
  package the Variant 1 venv already installs (`pip install av` elsewhere):
  `python -c "import av,sys; c=av.open(sys.argv[1]); s=c.streams.video[0]; n=sum(1 for _ in c.decode(s)); print(s.codec_context.name, s.width, s.height, n, float(s.average_rate))" <file>`.
- Image (`.png` / `.jpg`): `python -c "import sys; from PIL import Image; p=sys.argv[1]; Image.open(p).verify(); print(Image.open(p).size)" <file>`.
  Pillow is in the diffusers/framework venvs (via torchvision) but not in a bare host
  python (`pip install pillow`); `file` is absent on minimal hosts too.
- Expected smoke facts (measured, eval runs 2026-10-21/23): `references/skill-notes.md` § 5 — record them in `result_summary.md`.

## Reference files

The full cookbook content for a backend (every model size, text-to-video, image/video
conditioning, audio, multi-GPU, the complete request contracts and helper code) lives in
`references/`, migrated 1:1 from the notebooks — read the matching file and
`references/payloads.md` before the first command (see the top). Do not look for the notebooks; these files replace them
(Variant 7: the NIM cookbook in `../cookbooks/cosmos3/nim/` replaced that notebook upstream — see its row).

| File                                                              | Backend                                                                                                                                                |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `references/skill-notes.md`                                       | Shared skill notes: payload rules, measured footprints, guardrail switches, Things not obvious from the docs, smoke facts, uv version detail           |
| `references/payloads.md`                                          | Shared: assets, prompt serialisation, sampling defaults, use-case matrix                                                                               |
| `references/diffusers.md`                                         | Variant 1 — diffusers                                                                                                                                  |
| `references/cosmos-framework.md`                                  | Variant 2 — cosmos_framework                                                                                                                           |
| `references/vllm-omni.md`                                         | Variant 3 — vLLM-Omni                                                                                                                                  |
| `references/sglang.md`                                            | Variant 4 — SGLang                                                                                                                                     |
| `references/trt-llm.md`                                           | Variant 5 — TensorRT-LLM                                                                                                                               |
| `references/distilled-trt-llm.md`                                 | Variant 6 — distilled 4-step students                                                                                                                  |
| [`../cookbooks/cosmos3/nim/`](../cookbooks/cosmos3/nim/README.md) | Variant 7 — NIM: the Cosmos3 NIM cookbook (deployment, generation, api-reference, operations) and its `cosmos3-nim-user` skill; not part of this skill |

Origin: the `.agents/skills/cookbooks/cosmos3/generator/audiovisual` notebooks and README of the public cosmos-framework
repository (Variant 7: the `cosmos3/nim/` cookbook pages); superseded by this skill.

## Related skills

| Skill                                                               | When to use                           |
| ------------------------------------------------------------------- | ------------------------------------- |
| `../cosmos3-setup/SKILL.md`                                         | Framework environment for Variant 2   |
| `../cosmos3-inference/SKILL.md`                                     | Framework-native inference reference  |
| `../cosmos3-env-troubleshoot/SKILL.md`                              | CUDA/import/server failures           |
| `../cosmos3-skill-authoring/SKILL.md`                               | Editing this skill's eval cases       |
| `../cookbooks/cosmos3/nim/.agents/skills/cosmos3-nim-user/SKILL.md` | Operating the Cosmos3 NIM (Variant 7) |
