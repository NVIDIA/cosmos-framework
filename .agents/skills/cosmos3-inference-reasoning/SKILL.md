---
name: cosmos3-inference-reasoning
description: >
  Run the Cosmos3 Reasoner (vision-language reasoning over images and video: captioning, VQA,
  temporal localization, 2D grounding, describe-anything, embodied next-action and robot
  planning, action-CoT trajectories, physical plausibility, situation understanding) with the
  Edge / Nano / Super checkpoints through any of the six cookbook backends: Cosmos Framework
  (in-repo), vLLM server, SGLang server, TensorRT-LLM server, Transformers in-process, or
  TensorRT-Edge-LLM. Use when the user asks "caption this image / describe this video with
  Cosmos3", "run the Cosmos3 Reasoner", "serve Cosmos3-Edge or Cosmos3-Nano for visual
  reasoning or VQA", "return a bounding box, a gripper trajectory or the next action with
  Cosmos3", "which Reasoner backend fits my GPU", or how to prompt the Reasoner (sampling
  parameters, the <think> format, 0-1000 coordinates). Do not use it to generate images, video
  or audio (cosmos3-inference-generate-media), for Action / robot-policy inference, to operate the
  Cosmos3 NIM container, or to quantize a checkpoint yourself (cosmos3-quantization).
  The cookbook's sample image and clips ship in assets/.
---

# Cosmos3 Visual Reasoning (Reasoner inference)

Six backend variants (the six cookbook notebooks) sharing one model family, one prompt format (the prompt guide) and the
bundled sample media. README: "Run the Cosmos3 Reasoner (vision-language reasoning over images and video) across multiple inference backends."

## Read this first

Before the first command of any variant — including a single image-caption smoke:

1. Read the variant's reference file in full (linked under its heading, with the sections the smoke path needs) and
   [`references/prompt-guide.md`](references/prompt-guide.md) before composing any prompt (sampling parameters, system
   prompt, media-before-text order, the `<think>` format, 0–1000 coordinates, task examples). The variant sections below
   only summarise the notebooks; the reference files replace the notebooks and the cookbook README — do not look for them.
2. Check the GPU with `nvidia-smi` before choosing a tier (table below). README: "If you need to save GPU memory for
   reasoner-only inference, use vLLM, TensorRT-LLM, Transformers, or NIM. Reach for the Cosmos Framework backend only
   when you also need generation."
3. Working directory and assets. The notebooks resolve their media from `COSMOS_REASONER_ASSETS = <cosmos root>/cookbooks/cosmos3/reasoner/assets`,
   found by walking up from the notebook directory. Skill note (not in the source): this skill directory replaces the
   notebook directory as the working directory — run every block from here and set, once per shell call,
   `export COSMOS_ROOT="$PWD" COSMOS3_MEDIA_ROOT="$PWD" COSMOS_REASONER_ASSETS="$PWD/assets"`; in the Python cells
   `COSMOS_REASONER_ASSETS = Path.cwd() / "assets"` replaces the notebooks' `find_repo_root` / `find_framework_root`
   block, and their `asset_path(name)` / `asset_url(name)` helpers then work unchanged. The vendored copies live under
   `assets/` at the notebooks' relative paths (12 of 15 files; Bundled assets below covers the other three).
4. Blocks run in separate shell calls: `export`, `source …/activate` and `cd` do not carry over. Repeat the exports of
   item 3 (and a variant's own exports) at the top of every call, or run a variant's setup, launch and request as one
   script.
5. uv (skill note, not in the source): a framework checkout's `pyproject.toml` pins `[tool.uv] required-version = "==0.11.29"`
   and "uv applies that pin from any cwd under the checkout" (`../cosmos3-inference-generate-media/SKILL.md`, Read this first item 3), so
   Variant 1 (`uv sync`) and Variant 5 (`uv venv` / `uv pip` from this directory inside a checkout) need exactly that version
   — transformers.md § 1: "use `uv 0.11.29`, matching the version pinned by Cosmos Framework". Before the first `uv` command,
   install it as `../cosmos3-setup/SKILL.md` step 3 does, `curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh; export PATH="$HOME/.local/bin:$PATH"`;
   every later shell call that runs `uv` starts with `export PATH="$HOME/.local/bin:$PATH"` (item 4), and `uv --version` must print 0.11.29.
6. Hugging Face access: the notebooks authenticate with `hf auth login` / `uvx hf@latest auth login` or
   `export HF_TOKEN=<your_token>`; "The Cosmos3-Super Reasoner checkpoint is gated on Hugging Face." Skill note (not in
   the source): export `HF_TOKEN` before a Docker launch — the TensorRT-LLM and FP8 launches carry `-e HF_TOKEN`, the bf16
   vLLM and SGLang launches (quoted verbatim below) do not, and anonymous downloads into an empty cache are rate-limited.
7. Verify what you produced before reporting it (Verify the output below) and record the facts in `result_summary.md`;
   after a failure, go back to the reference section that covers the failing step before improvising a fix.

## When to use this skill

- Use to run image or video understanding with a Cosmos3 checkpoint: captioning, VQA, temporal localization, 2D
  grounding, describe-anything, embodied next-action, robot planning, action-CoT trajectories, physical plausibility,
  situation understanding (prompts and expected answer shapes: `references/prompt-guide.md`)
- Use to choose a Reasoner backend (in-process vs server; Docker vs venv) and a tier (Edge / Nano / Super) for the GPU
- For image, video or audio generation this is the wrong skill: `../cosmos3-inference-generate-media/SKILL.md`
- For the Cosmos3 NIM container: `../cosmos3-inference-nim/SKILL.md`; the README's "Run with NIM" quickstart is kept in
  `references/reasoner-readme.md`, the full guide is [`../cookbooks/cosmos3/nim/reasoning.md`](../cookbooks/cosmos3/nim/reasoning.md)
- For environment problems (CUDA / import / Docker failures), hand off to `../cosmos3-env-troubleshoot/SKILL.md`

## Bundled assets

Vendored copies of the cookbook's `assets/` (same file names; paths relative to this skill directory); the notebooks
name them through `asset_path("<file>")`:

| Asset                                  | Used by (notebook sections)                                                                                    |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `assets/robot_153.jpg`                 | Image Caption (every backend); framework `nano_image` / `image_caption_detail`; TensorRT-Edge-LLM `input.json` |
| `assets/video_caption.mp4`             | Video Caption (vLLM, SGLang, TensorRT-LLM); Transformers § 8 Video Reasoning                                   |
| `assets/robotics_next_action.mp4`      | Embodied Reasoning — Robotics Next Action                                                                      |
| `assets/robot_planning.png`            | Robot Planning; Action CoT second trajectory (`trajectory_flower`)                                             |
| `assets/assisted_task_next_action.mp4` | Embodied Reasoning — Assisted Task Next Action                                                                 |
| `assets/common_sense_reasoning.mp4`    | Common Sense Reasoning                                                                                         |
| `assets/grounding_2d.png`              | 2D Grounding (`ground_load_bbox`)                                                                              |
| `assets/describe_anything.png`         | Describe Anything (`describe_marked_subjects`)                                                                 |
| `assets/action_cot_trajectory.png`     | Action CoT — Trajectory Coordinates (`trajectory_bowl`)                                                        |
| `assets/action_cot_driving_scene.mp4`  | Action CoT — Driving Scene                                                                                     |
| `assets/physical_plausibility.mp4`     | Physical Plausibility Analysis                                                                                 |
| `assets/situation_understanding.mp4`   | Situation Understanding                                                                                        |

Skill note (not in the source): three clips are NOT vendored (git-LFS objects above the vendoring size):
`drive_scene_next_action.mp4` (3.27 MiB; Drive Scene Next Action), `temporal_localization_1.mp4` (14.0 MiB; Temporal
Localization) and `temporal_localization_2.mp4` (3.5 MiB; Event Timeline, Timestamp Query, Interval Question). They stay
in the cookbook's [`../cookbooks/cosmos3/reasoner/assets/`](../cookbooks/cosmos3/reasoner/assets/) as LFS pointers:
`git lfs pull --include=".agents/skills/cookbooks/cosmos3/reasoner/assets/<file>"` from the checkout root (git-lfs installed), or download the LFS object from the public repository at the pinned commit,
`https://media.githubusercontent.com/media/NVIDIA/cosmos-framework/bbd4a67c3cf54cc6699464ad1259f3a0f6537da9/.agents/skills/cookbooks/cosmos3/reasoner/assets/<file>`,
then copy it into this skill's `assets/` under its own name, where `asset_path("<file>")` expects it.

## Model sizes vs GPU

Only what the sources state (quoted); nothing here was measured by this skill. README: "The Cosmos Framework backend
loads the full omni weights (Reasoner plus the generator: video VAE, audio tokenizer, and diffusion expert). It
therefore uses the most memory — even for text-only reasoner inference. `model_mode=reasoner` only changes the execution
path; it does not unload the generator weights." Skill note (not in the source; framework main bbd4a67): when every sample in the batch is a reasoner sample, `cosmos_framework.scripts.inference` applies `reasoner_only_overrides()` (vision tokenizer, vision/action/sound generation heads and the generation tower are not built and their tensors are never read) and logs "Reasoner-only inputs detected; generation-side modules will not be loaded"; eval 2026-11-08: Cosmos3-Nano image reasoning peaked at ~19 GiB on one H100. The quoted gap applies to a batch that also generates.

| Checkpoint             | What the sources say                                                                                                                                                                                                                                                                                                                                                               | GPUs, as the notebooks run it                                                                                                                                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nvidia/Cosmos3-Edge`  | "the smaller tier and runs on a single GPU" (vllm.md, sglang.md); "Edge is smaller" than the "tens of GiB" Nano/Super downloads (transformers.md § 6); Transformers: `AutoModelForImageTextToText`, "do not load Edge with the Omni class"                                                                                                                                         | one GPU in every notebook (`--gpus '"device=0"'`, `--tp-size 1`); TensorRT-LLM needs `release:1.4.0rc2` ("`v1.4.0rc1` and `release:1.3.0rc22` do not include `Cosmos3EdgeForConditionalGeneration`"); TensorRT-Edge-LLM is Edge-only |
| `nvidia/Cosmos3-Nano`  | README: "on a single GPU the Cosmos3-Nano Reasoner needs roughly ~34 GB with the Cosmos Framework backend versus ~16–17 GB with vLLM / TensorRT-LLM / Transformers / NIM — about half"; "Nano downloads can use tens of GiB in the Hugging Face cache"; skill note: ~19 GiB peak measured for a reasoner-only Nano image request on one H100 (eval 2026-11-08, see the note above) | one GPU: framework `--parallelism-preset=latency`, TensorRT-LLM `release:1.3.0rc22` (`--gpus '"device=0"'`), Transformers `device_map="auto"`                                                                                        |
| `nvidia/Cosmos3-Super` | README: "Larger tiers (e.g. Cosmos3-Super) use more in absolute terms, but the same full-omni-vs-reasoner-only gap applies"; "gated on Hugging Face"; framework: "needs a large-memory GPU (or multiple GPUs)"                                                                                                                                                                     | four GPUs in every server notebook (`--tensor-parallel-size 4`, `--tp-size 4`, `--tensor_parallel_size 4`); Transformers `device_map="auto"` "shards Super across multiple GPUs"                                                     |
| FP8 checkpoints        | vllm.md "Quantized Checkpoints": `nvidia/Cosmos3-Edge-FP8` on `vllm/vllm-openai:nightly`, `nvidia/Cosmos3-Super-FP8` on `vllm/vllm-openai:v0.31.0`; cosmos-framework.md § 7a: `nvidia/Cosmos3-Nano` at revision `fp8`; transformers.md § 3a / § 6: the Nano/Super `fp8` revision through ModelOpt ("Edge has none")                                                                | launched like the bf16 tier of the same backend (FP8 lines per variant below); to quantize your own checkpoint: `../cosmos3-quantization/SKILL.md`                                                                                   |

## Variant 1 — Cosmos Framework (in-repo `cosmos_framework.scripts.inference`)

Reference: [`references/cosmos-framework.md`](references/cosmos-framework.md) — read §§ 1–8 (Prerequisites, Configure
Paths, Clone or Reuse, Install, Verify GPU, Create Reasoner Input Files, Run Nano Text / Image Inference) before the
first command; README quickstart: `references/reasoner-readme.md` ("Run with Cosmos Framework"). "Models: `Cosmos3-Nano`
(default), `Cosmos3-Edge`, and `Cosmos3-Super`. All run the same Reasoner entrypoint; only `--checkpoint-path` differs."
Images only: "Cosmos Framework Reasoner currently treats `vision_path` as a PIL image input, so video Reasoner examples
should be run with" Variant 2. Requires a framework checkout (`../cosmos3-setup/SKILL.md`; setup guide
[`../cookbooks/cosmos3/README.md#cosmos-framework`](../cookbooks/cosmos3/README.md#cosmos-framework)). The notebook's § 2
cell sets the paths in `os.environ` for its bash cells; skill note (not in the notebook): in a terminal export the same
variables yourself, from this skill directory, in every shell call that runs a block below:

```bash
# Skill note (not in the notebook): the notebook's "Configure Paths" cell as exports; run from this skill directory
export COSMOS3_REPO="${COSMOS3_REPO:-/path/to/cosmos-framework}" COSMOS3_GIT_URL="${COSMOS3_GIT_URL:-https://github.com/NVIDIA/cosmos-framework.git}"   # an existing checkout, or the § 3 cell clones COSMOS3_GIT_URL there
export COSMOS3_UV_GROUP="${COSMOS3_UV_GROUP:-cu130-train}"          # CUDA 13 driver; cu128-train for a CUDA 12.x driver (§ 4 table)
export COSMOS_REASONER_ASSETS="$PWD/assets" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" COSMOS3_MASTER_ADDR=127.0.0.1
export COSMOS3_OUTPUT_ROOT="${COSMOS3_OUTPUT_ROOT:-$COSMOS3_REPO/outputs/cookbooks/cosmos3/reasoner/nano}"
export COSMOS3_INPUT_DIR="$COSMOS3_OUTPUT_ROOT/inputs"
export COSMOS3_NANO_TEXT_MASTER_PORT=29501 COSMOS3_NANO_IMAGE_MASTER_PORT=29502 COSMOS3_CAPABILITY_MASTER_PORT=29503   # the skill's own fixed values; the notebook picks free ports (free_local_port()), the README quickstart uses 29501
```

Install (§ 4 cell, verbatim; "Creates `.venv` inside `packages/cosmos3`", "May take several minutes"; uv 0.11.29 first —
Read this first item 5):

```bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

cd "$COSMOS3_REPO"
uv sync --all-extras --group="$COSMOS3_UV_GROUP"
```

Then run the § 5 bash cell (GPU check) as one shell call, and the first § 6 Python cell (the input writer; the second § 6 cell
is a Jupyter display helper) as one file with `"$COSMOS3_REPO/.venv/bin/python"` — it writes
`nano_text.json`, `nano_image.json` and the six capability inputs, each with `"enable_sound": False` ("The current
shipped Reasoner examples fail without `enable_sound=false`") and an absolute `vision_path` under
`COSMOS_REASONER_ASSETS`. Nano image inference (§ 8, verbatim):

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
COSMOS_TRAINING=false CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_IMAGE_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_INPUT_DIR/nano_image.json" \
  -o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_nano_image" \
  --checkpoint-path Cosmos3-Nano \
  --seed=0 \
  --benchmark
```

"Expected output file:" `<COSMOS3_OUTPUT_ROOT>/cosmos_framework_nano_image/nano_image/reasoner_text.txt` (`benchmark.json`
next to it). Capability prompts (§§ 9–14: detailed caption, robot planning, 2D grounding, describe-anything, two
trajectories): the same command with the `capabilities/<name>.json` input, `-o "$COSMOS3_OUTPUT_ROOT/cosmos_framework_<name>"`
and `MASTER_PORT="$COSMOS3_CAPABILITY_MASTER_PORT"`. Edge (§§ 15–16): "only `--checkpoint-path` changes; the input JSON
(`nano_image.json`) is identical". Super (§§ 17–18): same again; "Super needs a large-memory GPU (or multiple GPUs)".

FP8 (§ 7a "Optional: Run Nano FP8 Text Inference"): "FP8 checkpoints are published in the same Hugging Face
repositories as the standard checkpoints, under the `fp8` revision" — `FP8_MODEL_ID="nvidia/Cosmos3-Nano"`,
`FP8_REVISION="fp8"`, `uvx hf@latest download "$FP8_MODEL_ID" --revision "$FP8_REVISION" --local-dir "$COSMOS3_REPO/checkpoints/Cosmos3-Nano-FP8"`,
then `.venv/bin/torchrun --nproc-per-node="$FP8_NUM_GPUS" … -m cosmos_framework.scripts.inference --parallelism-preset=throughput --dp-shard-size="$FP8_NUM_GPUS" --dp-replicate-size=1 --cp-size=1 --cfgp-size=1 --no-use-torch-compile … --checkpoint-path "$FP8_CHECKPOINT"`
("Set `FP8_NUM_GPUS=1` to run without FSDP"); full block in § 7a. To quantize your own: `../cosmos3-quantization/SKILL.md`.

## Variant 2 — vLLM (Docker server + OpenAI client)

Reference: [`references/vllm.md`](references/vllm.md) — read § 1 Setup, § 2 Launch a vLLM server (Edge; "Quantized
Checkpoints"; "Wait for the server to be ready"), § 3 Query the model and § 4 Shut down before the first command.
"Serving runs entirely in Docker"; image `vllm/vllm-openai:cosmos3` ("~19 GB on first run"). Client dependencies — the
notebook's `%pip` cell as a shell command (skill note: `%pip` is Jupyter syntax): `pip install -q "openai>=1.0" pillow "huggingface_hub[cli]"`; then `docker pull vllm/vllm-openai:cosmos3`.

Edge launch (§ 2, verbatim; the exports of Read this first item 3 must precede it in the same call — `COSMOS_ROOT` is
mounted at its own path and `COSMOS3_MEDIA_ROOT` is the `--allowed-local-media-path`, so `file://` URLs under this skill
directory resolve inside the container; "Both launch cells mount `~/.cache/huggingface` into the container so downloaded weights are reused"; the port is published "on the loopback interface (`127.0.0.1`) only"):

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_MEDIA_ROOT:?run the setup cell first}"
CONTAINER=cosmos3-reasoner-vllm

# Remove any existing server first.
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus '"device=0"' \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$COSMOS_ROOT:$COSMOS_ROOT" \
  -p 127.0.0.1:8001:8000 --ipc=host \
  vllm/vllm-openai:cosmos3 \
  nvidia/Cosmos3-Edge \
    --async-scheduling \
    --allowed-local-media-path "$COSMOS3_MEDIA_ROOT" \
    --port 8000
echo "Cosmos3-Edge starting in container '$CONTAINER' — run the readiness cell next."
```

Readiness: run the § 2 "Wait for the server to be ready" block verbatim — it streams `docker logs -f`, polls
`http://127.0.0.1:${VLLM_PORT:-8001}/health` every 2 s for up to 1800 polls ("allowing up to 30 minutes on the first run while the weights download and CUDA graphs compile"), exits 0 on "vLLM server is ready." and 1 on "Container exited
early". Super (§ 2): `nvidia/Cosmos3-Super --tensor-parallel-size 4 --mm-encoder-tp-mode data` on `--gpus all`.

Request — the README's OpenAI-client quickstart (`references/reasoner-readme.md` "Run with TensorRT-LLM"), verbatim in its
port-**8001** form; the "Run with vLLM" form is the same code at port 8000 and says "point the client at `http://localhost:8001/v1`"
for the notebook launch. Skill note (not in the source): first run `docker exec "$CONTAINER" test -r "$COSMOS_REASONER_ASSETS/robot_153.jpg" || docker cp "$COSMOS_REASONER_ASSETS" "$CONTAINER:$COSMOS_ROOT/"`
(`CONTAINER` = cosmos3-reasoner-vllm / -sglang / -trtllm): a daemon that does not share this filesystem (socket-shared or remote) mounts `COSMOS_ROOT` empty, and the `file://` request then fails 500/400 "No such file or directory":

```python
from pathlib import Path
import openai

image_path = Path("assets/robot_153.jpg").resolve()
image_url = image_path.as_uri()

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8001/v1")

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Caption the image in detail."},
            ],
        }
    ],
    max_tokens=4096,
    seed=0,
)

print(response.choices[0].message.content)
```

Video requests (§ 3): the content block is `{"type": "video_url", "video_url": {"url": video_url}}` with
`extra_body={"media_io_kwargs": {"video": {"num_frames": -1, "fps": 4}}}`; the Action CoT cells add
`temperature=0.6, top_p=0.95, presence_penalty=0.0` and `extra_body={"top_k": 20, "repetition_penalty": 1.0}` (prompt
guide: "With reasoning"). Shut down (§ 4): `docker rm -f cosmos3-reasoner-vllm` (the notebook's `!docker …` cell without the IPython `!`).

FP8 (§ 2 "Quantized Checkpoints"): "Standalone vLLM uses ModelOpt FP8 for reasoner inference". Edge:
`snapshot_download("nvidia/Cosmos3-Edge-FP8", local_dir=str(FP8_MODEL_DIR))` (skipped when `FP8_MODEL_DIR/hf_quant_config.json` exists — "set `FP8_MODEL_DIR` to the local `edge-fp8` or `super-fp8` output" of your own quantization), then the § 2
FP8 launch: image `vllm/vllm-openai:nightly`, model argument `"$FP8_MODEL_DIR"` with `--served-model-name nvidia/Cosmos3-Edge --async-scheduling --trust-request-mm-kwargs` ("lets each request override media settings such as `num_frames` and `fps`"),
`-e HF_TOKEN`, the directory mounted read-only; Super FP8: `nvidia/Cosmos3-Super-FP8` on `vllm/vllm-openai:v0.31.0` with `--tensor-parallel-size 4`. "For locally generated outputs, set `FP8_SOURCE_DIR` to the source checkpoint directory … so
exported symlinks remain readable"; "Keep `HF_HOME` consistent with quantization." To quantize your own: `../cosmos3-quantization/SKILL.md`.

## Variant 3 — SGLang (Docker server + OpenAI client)

Reference: [`references/sglang.md`](references/sglang.md) — read § 1 Setup, § 2 Launch a SGLang server (Edge; "Wait
for the server to be ready"), § 3 Query the model and § 4 Shut down before the first command. Image `lmsysorg/sglang:dev`
("~14 GB; first run only"); the same client dependencies and exports as Variant 2; no FP8 section. Edge launch (§ 2, verbatim):

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_MEDIA_ROOT:?run the setup cell first}"
CONTAINER=cosmos3-reasoner-sglang

# setup hf cache dir
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"

# Remove any existing server first.
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus '"device=0"' \
  -e HF_HOME=/root/.cache/huggingface \
  -v "$SGLANG_HF_CACHE:/root/.cache/huggingface" \
  -v "$COSMOS_ROOT:$COSMOS_ROOT" \
  -p 8001:8000 --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
    --model-type llm \
    --model-path nvidia/Cosmos3-Edge \
    --tp-size 1 \
    --enable-multimodal \
    --host 0.0.0.0 \
    --port 8000
echo "Cosmos3-Edge starting in container '$CONTAINER' — run the readiness cell next."
```

Readiness: the § 2 "Wait for the server to be ready" block, verbatim (container `cosmos3-reasoner-sglang`, `SGLANG_PORT`
default 8001, the same 2 s × 1800 poll of `/health`). Requests: § 3 — the same OpenAI-client cells as Variant 2 against
`http://localhost:8001/v1` (the Variant 2 request block, already at port 8001, is the image-caption smoke). Super:
`--model-path nvidia/Cosmos3-Super --tp-size 4` on `--gpus all`. Shut down (§ 4): `docker rm -f cosmos3-reasoner-sglang`.

## Variant 4 — TensorRT-LLM (Docker server + OpenAI client)

Reference: [`references/tensorrt-llm.md`](references/tensorrt-llm.md) — read § 1 Setup, § 2 Launch a TensorRT-LLM
server (Nano / Super / Edge; "Wait for the server to be ready"), § 3 Query the model and § 4 Shut down before the first
command; setup guide [`../cookbooks/cosmos3/README.md#tensorrt-llm-reasoner`](../cookbooks/cosmos3/README.md#tensorrt-llm-reasoner).
"**Cosmos3-Nano** and **Cosmos3-Super** use the prebuilt `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` Docker image"
(the § 1 pull cell, "~20 GB on first run"); "**Cosmos3-Edge** requires TensorRT-LLM >= `v1.4.0rc2` and uses
`nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2`" (its § 2 cell pulls it; source-build fallback at `39a44fab`). Client
dependencies: `pip install -q "openai>=1.0" pillow huggingface_hub`; no FP8 section. Nano launch (§ 2, verbatim, after the
exports of Read this first item 3; `COSMOS3_TRTLLM_IMAGE` is `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc22` from § 1):

```bash
set -euo pipefail
: "${COSMOS_ROOT:?run the setup cell first}"
: "${COSMOS3_TRTLLM_IMAGE:?run the setup cell first}"

CONTAINER=cosmos3-reasoner-trtllm

# Remove any existing notebook server first.
docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus '"device=0"' \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$COSMOS_ROOT:$COSMOS_ROOT" \
  -p 8001:8000 --ipc=host \
  "$COSMOS3_TRTLLM_IMAGE" \
  bash -lc 'python -m pip install --no-cache-dir opencv-python-headless && \
    exec trtllm-serve nvidia/Cosmos3-Nano \
      --host 0.0.0.0 --port 8000 \
      --max_num_tokens 32768'

echo "Cosmos3-Nano starting in container '$CONTAINER' — run the readiness cell next."
```

Readiness: the § 2 "Wait for the server to be ready" block, verbatim (container `cosmos3-reasoner-trtllm`,
`http://127.0.0.1:8001/health`, 2 s × 1800; it also follows a host Edge process through `/tmp/cosmos3-edge-reasoner-trtllm.{log,pid}`).
Requests: § 3 — the same OpenAI-client cells against `http://localhost:8001/v1` (the Variant 2 request block is this
backend's README quickstart, verbatim); the video cells add `"mm_processor_kwargs": {"do_sample_frames": False}` to
`extra_body` next to `media_io_kwargs`. Edge: the § 2 Edge cell (`"$COSMOS3_EDGE_TRTLLM_IMAGE"`, `trtllm-serve nvidia/Cosmos3-Edge`);
Super: `--tensor_parallel_size 4` on `--gpus all`. Shut down: the § 4 block (`docker rm -f cosmos3-reasoner-trtllm`, plus the Edge PID-file cleanup).

## Variant 5 — Transformers (in-process venv, no server)

Reference: [`references/transformers.md`](references/transformers.md) — read § 1 Prerequisites (headless
`libxcb1 libgl1 libglib2.0-0`; "use `uv 0.11.29`" — Read this first item 5), § 2 Configure, § 3 Install (and § 3a FP8),
§ 6 Load the Reasoner, § 7 Image Reasoning and § 8 Video Reasoning before the first command; README quickstart:
`references/reasoner-readme.md` ("Run with Transformers"). "The integration loads only the Reasoner tower and returns text
for text, image, and video understanding." Skill note (not in the notebook): the § 4 kernel switch and § 5 kernel check are
Jupyter-only — run the Python cells as one file with `"$COSMOS3_TRANSFORMERS_VENV/bin/python"`. Install — the § 3 cell,
verbatim, with the § 2 cell's exports prepended (skill note) so the block is one self-contained shell call (item 4):

```bash
# Skill note (not in the notebook): the § 2 cell's exports (HF_HOME left at its default), the venv inside this skill directory
export COSMOS3_TRANSFORMERS_VENV="${COSMOS3_TRANSFORMERS_VENV:-$PWD/.venv-cosmos3-transformers}"
export COSMOS3_TORCH_BACKEND="${COSMOS3_TORCH_BACKEND:-auto}"   # or cu130 / cu128 to pin an explicit CUDA wheel
export UV_LINK_MODE="${UV_LINK_MODE:-copy}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# the notebook's § 3 cell, verbatim, from here:
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv venv "$COSMOS3_TRANSFORMERS_VENV" --python 3.13 --seed --managed-python --allow-existing
source "$COSMOS3_TRANSFORMERS_VENV/bin/activate"

uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" \
  accelerate \
  av \
  ipykernel \
  pillow \
  "safetensors>=0.8.0" \
  torch \
  "torchvision==0.25.0" \
  "transformers @ git+https://github.com/huggingface/transformers.git"

"$COSMOS3_TRANSFORMERS_VENV/bin/python" -m ipykernel install --user \
  --name cosmos3-transformers \
  --display-name "Cosmos3 Transformers (Python 3.13)"

echo
echo "Installed dependencies into: $COSMOS3_TRANSFORMERS_VENV"
echo "Next: switch this notebook kernel to: Cosmos3 Transformers (Python 3.13)"
```

Run: the README quickstart ("Run **Cosmos3-Nano** Reasoner inference in process") as ONE file with
`"$COSMOS3_TRANSFORMERS_VENV/bin/python"` from this skill directory — it loads `nvidia/Cosmos3-Nano` with
`Cosmos3OmniForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")`, builds the chat
input from `{"type": "image", "path": str(image_path)}` + `"Caption the image in detail."` with `processor.apply_chat_template(...)`
and generates with `model.generate(**inputs, do_sample=False, max_new_tokens=512)`; video: a `{"type": "video", "path": …}`
block and `fps=2` (the notebook's § 6 `run_reasoner` wraps the same calls). "To run **Cosmos3-Super**, change `model_id` to
`nvidia/Cosmos3-Super`." "**Edge** uses `AutoModelForImageTextToText` (`Cosmos3EdgeForConditionalGeneration`) with
`nvidia/Cosmos3-Edge` — do not load Edge with the Omni class" (§ 6 if/else).

FP8 (§ 3a + § 6): `uv pip install "nvidia-modelopt==0.44.0" requests` into the venv ("`requests` is installed explicitly
because ModelOpt `0.44.0` imports it but does not declare it as a dependency"), then `USE_MODELOPT_FP8 = True` in the § 6
cell: it imports `modelopt.torch.opt` and `modelopt.torch.quantization.backends.fp8_per_tensor_gemm`, calls
`enable_huggingface_checkpointing()` and loads `snapshot_download(model_id, revision="fp8")` — "this is not runtime
quantization" (Nano/Super only: "Edge has none"). To quantize your own: `../cosmos3-quantization/SKILL.md`.

## Variant 6 — TensorRT-Edge-LLM (ONNX export → engines → C++ `llm_inference`)

Reference: [`references/trt-edge-llm.md`](references/trt-edge-llm.md) — read all of it (§ 1 Setup paths, § 2 Export the
Reasoner on CPU, § 3 Build the LLM and visual engines, § 4 Write an image reasoning request, § 5 Run reasoning) before
the first command; setup guide [`../cookbooks/cosmos3/README.md#tensorrt-edge-llm`](../cookbooks/cosmos3/README.md#tensorrt-edge-llm)
(TensorRT-Edge-LLM **v0.11.0**; "Python package on the host; C++ runtime on the device"). Notebook: "Complete the
upstream Installation first (Python export on an x86 host; C++ runtime on Jetson Thor/Orin, DRIVE, DGX Spark, or an x86
developer build)"; "It is **not** the datacenter TensorRT-LLM OpenAI-compatible server path." Skill note (not in the
notebook): on a host without the compiled C++ runtime (the `build` tree of the TensorRT-Edge-LLM checkout), § 2 (CPU
export) and § 4 (request file) are the runnable part; §§ 3 and 5 need the built runtime and TensorRT on the target.
The § 1 cell's defaults as exports (skill note), from this skill directory: `export REASONER_ASSETS="$PWD/assets"`,
`EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/TensorRT-Edge-LLM}"`, `ONNX_DIR="${ONNX_DIR:-$HOME/tensorrt-edgellm-workspace/Cosmos3-Edge/onnx}"`,
`ENGINE_DIR="${ENGINE_DIR:-$HOME/tensorrt-edgellm-workspace/Cosmos3-Edge/engines}"`, `REASONING_CHECKPOINT="${REASONING_CHECKPOINT:-nvidia/Cosmos3-Edge}"`,
`WORK_DIR="$ENGINE_DIR/reasoning/cookbook_work"` (`mkdir -p "$WORK_DIR"`). Export (§ 2, verbatim; "Run this on the export
host (after `pip install -e .` in the TensorRT-Edge-LLM checkout). It downloads `nvidia/Cosmos3-Edge` on first use." —
Skill note (not in the source): at v0.11.0 that install must be `pip install -e ".[tools]"` in a venv of the v0.11.0 checkout (plain `-e .` lacks torch, `[export]` lacks `nvidia-modelopt`); § 2 gives the clone + venv line):

```bash
set -euo pipefail
: "${REASONING_CHECKPOINT:?run the setup cell first}"
: "${ONNX_DIR:?run the setup cell first}"

mkdir -p "$ONNX_DIR/reasoning"
tensorrt-edgellm-export \
  "$REASONING_CHECKPOINT" \
  "$ONNX_DIR/reasoning" \
  --task reasoning

echo "Exported reasoning ONNX under $ONNX_DIR/reasoning"
```

Engines (§ 3, from `$EDGELLM_ROOT`): `./build/examples/llm/llm_build --onnxDir "$ONNX_DIR/reasoning/llm" --engineDir "$ENGINE_DIR/reasoning" --maxInputLen 2048 --maxKVCacheCapacity 4096`
and `./build/examples/multimodal/visual_build --onnxDir "$ONNX_DIR/reasoning/visual" --engineDir "$ENGINE_DIR/reasoning"` ("reasoning does **not** require `-DBUILD_EXPERIMENTAL_MODELS=ON`").
Request (§ 4): the Python cell writes `$WORK_DIR/input.json` — `{"batch_size": 1, "max_generate_length": 256, "requests": [{"messages": [{"role": "user", "content": [{"type": "image", "image": "<absolute path of assets/robot_153.jpg>"}, {"type": "text", "text": "Caption the image in detail."}]}]}]}`
— and exports `INPUT_JSON` / `OUTPUT_JSON`. Run (§ 5): `./build/examples/llm/llm_inference --engineDir "$ENGINE_DIR/reasoning" --multimodalEngineDir "$ENGINE_DIR/reasoning" --inputFile "$INPUT_JSON" --outputFile "$OUTPUT_JSON"`, then the cell prints `output.json`.

## Prompting the Reasoner (read before composing a prompt)

Reference: [`references/prompt-guide.md`](references/prompt-guide.md) — "The examples below mirror the notebook prompts …
tested with `Cosmos3-Super`". Sampling parameters: without reasoning
`top_p=0.8 top_k=20 repetition_penalty=1.0 presence_penalty=1.5 temperature=0.7`; with reasoning
`top_p=0.95 top_k=20 repetition_penalty=1.0 presence_penalty=0.0 temperature=0.6`. System prompt: "You are a helpful
assistant." Reasoning on/off: "If reasoning is required, append the following instruction after the task text:"
`Answer the question using the following format:` / `<think>` / `Your reasoning.` / `</think>` /
`Write your final answer immediately after the </think> tag.` "Always place media before the user text prompt in the
payload. Use `image_url` for images and `video_url` for videos." Coordinates: "2D boxes, points, and trajectories use
normalized `0-1000` coordinates. The origin is the top-left corner; `x` increases rightward and `y` increases downward."
Boxes `[{"bbox_2d": [x1, y1, x2, y2], "label": "object name"}]`, points `[{"point_2d": [x, y], "label": "target point"}]`;
"Convert them back to pixels with `x / 1000 * width` and `y / 1000 * height` before drawing."

## Verify the output

What the notebooks check (their result cells) and what to record — the Reasoner's output is text:

- Server variants (2–4): the cells `print(response.choices[0].message.content)`; the grounding and trajectory cells
  parse the JSON that follows `</think>` (`parse_boxes`: strip ``` fences, `json.JSONDecoder().raw_decode` from the
  first `[` or `{`; `parse_points`: the `[...]` list of `{"point_2d", "label"}`) and scale `x / 1000 * W`, `y / 1000 * H`.
- Variant 1: the text is in `<out>/<name>/reasoner_text.txt`, `benchmark.json` next to it (`"average"` printed);
  `draw_boxes_if_present` / `draw_trajectory_if_present` apply the same 0–1000 rule. Variant 5: `print(output)` of
  `run_reasoner`; Variant 6: `output.json` (`print(out.read_text()[:4000])`).
- Skill note (not in the source): save the returned text to the requested `outputs/<file>.txt` unchanged (non-empty UTF-8);
  record in `result_summary.md` the checkpoint, server image and launch command (or venv / entrypoint), time-to-ready, request
  latency and the first lines of the answer; after teardown confirm with `docker ps` / `nvidia-smi` that the server and its GPU memory are gone.

## Reference files

The complete cookbook content lives in `references/`, migrated 1:1 (every notebook cell in order; outputs omitted — the
committed notebooks have none); only link targets that pointed at moved files were re-pointed (`docs/cookbook_to_skill_migration.md`).

| File                             | Content                                                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `references/reasoner-readme.md`  | The cookbook README: backend choice (memory footprint), per-backend Quickstart + Notebook walkthrough, the NIM quickstart |
| `references/prompt-guide.md`     | The Reasoner Prompt Guide: sampling parameters, system prompt, media order, coordinates, task examples with model answers |
| `references/cosmos-framework.md` | Variant 1 — `run_with_cosmos_framework.ipynb` (install, inputs, Nano / Edge / Super runs, Nano FP8, capability prompts)   |
| `references/vllm.md`             | Variant 2 — `run_with_vllm.ipynb` (Edge / Super launch, Edge / Super FP8 "Quantized Checkpoints", readiness, every query) |
| `references/sglang.md`           | Variant 3 — `run_with_sglang.ipynb`                                                                                       |
| `references/tensorrt-llm.md`     | Variant 4 — `run_with_tensorrt_llm.ipynb` (Nano / Super / Edge images, readiness, queries, shutdown)                      |
| `references/transformers.md`     | Variant 5 — `run_with_transformers.ipynb` (venv, ModelOpt FP8, load, image / video)                                       |
| `references/trt-edge-llm.md`     | Variant 6 — `run_with_trt_edge_llm.ipynb` (export, engines, request, run)                                                 |

Origin: `.agents/skills/cookbooks/cosmos3/reasoner/` of the public cosmos-framework repository (notebooks, `README.md`,
`reasoner_prompt_guide.md`), superseded by this skill; the cookbook keeps `assets/` and `finetune/`.

## Related skills

| Skill                                          | When to use                                                            |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| `../cosmos3-setup/SKILL.md`                    | Framework checkout and venv for Variant 1                              |
| `../cosmos3-env-troubleshoot/SKILL.md`         | CUDA / import / Docker / download failures                             |
| `../cosmos3-inference-generate-media/SKILL.md` | Image, video and audio generation (the Generator, not the Reasoner)    |
| `../cosmos3-quantization/SKILL.md`             | Quantizing your own FP8 checkpoint (the released FP8 routes are above) |
| `../cosmos3-inference-nim/SKILL.md`            | Operating the Cosmos3 NIM (Reasoner runtime)                           |
| `../cosmos3-skill-authoring/SKILL.md`          | Editing this skill's eval cases                                        |
