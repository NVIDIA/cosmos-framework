---
name: cosmos3-inference-nim
description: >
  Deploy, call and troubleshoot the Cosmos3 NIM (nvcr.io/nim/nvidia/cosmos3)
  as Generator or Reasoner from the public .agents/skills/cookbooks/cosmos3/nim
  cookbook: deployment, endpoint verification, example selection, hardware
  compatibility. Use when the user asks to "deploy the Cosmos3 NIM", "run the
  NIM Generator / Reasoner on my GPU", "send a /v1/infer request", "call the
  NIM's chat completions or Responses API", "which NIM model, precision or
  profile fits my GPU", or says "the NIM is not ready / returns 404 or 422".
  Do not use it to maintain the NIM documentation itself, to serve Cosmos3
  through the Hugging Face backends (diffusers, cosmos_framework, vLLM-Omni,
  SGLang, TensorRT-LLM: cosmos3-generate-media), or to quantize a checkpoint
  (cosmos3-inference-quantization).
license: OpenMDW-1.1
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 NIM customer workflow

Skill note (not in the source): this skill replaces the NIM cookbook's own agent skill `cosmos3-nim-user`, which sat in a
hidden `.agents/skills/` folder inside the cookbook that agents started at the repository root do not discover (the
repository's skill roots are `.agents/skills/` and `.claude/skills/`); that nested copy is retired with this change. The
paragraph below and sections 1–6 are that skill's text, unchanged except for link targets, which
now point into [`../cookbooks/cosmos3/nim/`](../cookbooks/cosmos3/nim/README.md), and bare page / example names, which became
links to the same files there; the Quickstart quotes the README and the
four deployment sections quote deployment.md's blocks. The pages, `../cookbooks/cosmos3/nim/AGENTS.md`, the example clients in
`../cookbooks/cosmos3/nim/examples/`, the pinned client environment (`../cookbooks/cosmos3/nim/pyproject.toml`,
`../cookbooks/cosmos3/nim/uv.lock`) stay where they are as the source of truth; nothing is converted or bundled here. Every other section is skill text in which each quotation names its page; longer notes: [`references/skill-notes.md`](references/skill-notes.md).

Work from `.agents/skills/cookbooks/cosmos3/nim` and follow [`AGENTS.md`](../cookbooks/cosmos3/nim/AGENTS.md). The public pages here
are the source of truth; the rules below say which page owns each decision.

## Read this first

Skill note (not in the source): the numbered rules below are skill additions; every quotation names its page.

1. Read the page that owns the decision before the first command — the pages are the source, this file only routes:
   [`README.md`](../cookbooks/cosmos3/nim/README.md) (task table, quickstart), then for a new deployment
   [`prerequisites.md`](../cookbooks/cosmos3/nim/prerequisites.md) → [`support-matrix.md`](../cookbooks/cosmos3/nim/support-matrix.md) →
   [`deployment.md`](../cookbooks/cosmos3/nim/deployment.md) (its blocks are quoted in the four deployment sections below); the
   task page of the request (section 4 and Reference files below); [`operations.md`](../cookbooks/cosmos3/nim/operations.md) for
   anything that fails; and [`AGENTS.md`](../cookbooks/cosmos3/nim/AGENTS.md) for the assistant rules (read-only by default;
   refer to `NGC_API_KEY` by name only; confirm `NIM_URL`, readiness and `/v1/metadata` before choosing a request). Do not
   look for a notebook: upstream replaced `run_with_nim.ipynb` with these pages and the example clients (cosmos-framework
   #296); the pages and `../cookbooks/cosmos3/nim/examples/` are the runnable path.
2. Check the GPU before choosing a model — prerequisites.md "Verify the Host":
   `nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.free --format=csv` ("The GPU query reports memory in
   MiB. Divide by 1,024 to compare with the binary-GiB floors in the Support Matrix") — and compare each device with the
   floors in the table below ("Do not add together memory from multiple GPUs to satisfy a per-device floor"). The support
   matrix's recommendation for H100-class GPUs is quoted under "Hardware floors" below ("Default to Nano, especially for video").
3. Credentials and image: the NIM authenticates with NGC, not Hugging Face — export `NGC_API_KEY` ("The variable is
   `NGC_API_KEY`, not `NGC_TOKEN`", prerequisites.md); the login, the image pin and the cache are the "Deploy the NIM" blocks
   below. `HF_TOKEN` matters only for a private Hugging Face checkpoint
   ([`bring-your-own-checkpoint.md`](../cookbooks/cosmos3/nim/bring-your-own-checkpoint.md)).
4. Where the inputs live and what the clients write: the example clients run from `.agents/skills/cookbooks/cosmos3/nim` in
   the pinned client environment (`uv sync --locked`, prerequisites.md "Initialize the Example Environment"; "add no
   dependencies", section 4 below) and read their inputs from sibling cookbook directories, not from this skill (Bundled
   assets below). In a checkout made without git-lfs the JPEG / MP4 inputs are ~130-byte pointer files, not media: before
   running a client, materialise exactly the files it reads (`git lfs pull --include=<path>`, or the blob from
   media.githubusercontent.com) — procedure and verified facts in [`references/skill-notes.md`](references/skill-notes.md) § 1.
   The clients write next to themselves ("Examples save outputs under the ignored `examples/outputs/` directory", README);
   that directory is created by the first client run and ignored by `.agents/skills/.gitignore`, so a fresh checkout has none
   — copy what a task asks for to the location it names.
5. uv: the client project has its own `[tool.uv]` table, so the framework's `[tool.uv] required-version = "==0.11.29"` pin
   does not apply inside `.agents/skills/cookbooks/cosmos3/nim` — `uv sync --locked` there resolves with the stock uv the
   prerequisites install (checked 2026-11-07 with uv 0.13.0, `--dry-run`; `references/skill-notes.md` § 4). Do not install
   uv 0.11.29 for the client environment.
6. Automation: the pages start the NIM detached (`docker run -d --name …`) and gate requests on readiness ("Send inference
   only after readiness succeeds", deployment.md "Wait for Readiness"; "A cold start (download, materialization, model load,
   warmup) takes much longer than HTTP startup"). The page's blocks are quoted in the four deployment sections below; what an
   unattended run adds around them (readiness-loop exit when the container dies, logs saved before `docker rm`, the cache-mount
   check for a container-driven NIM) is `references/skill-notes.md` § 2. Use `docker rm -f` only after saving the logs you need.
7. Verify what you produced before reporting it (Verify the output below) and record in `result_summary.md` the image tag,
   the selected profile (`/v1/metadata`, or `uv run python examples/inspect_profile.py` from the cookbook directory), time-to-ready
   and the request latency you observed — the clients print none, so run the client as `time uv run python examples/<client>.py …` and report its `real` line as client wall time ("A single run or a timeout value is not expected service latency", AGENTS.md); say
   whether each conclusion comes from the pages or from endpoint output.

## When to use this skill

- Use to deploy the Cosmos3 NIM (`nvcr.io/nim/nvidia/cosmos3:2.1.0`) as Generator or Reasoner on a GPU host, to verify an
  endpoint someone else deployed (`/v1/health/ready`, `/v1/metadata`), or to troubleshoot one: container live but never ready,
  no compatible profile, wrong-runtime HTTP 404, HTTP 422 validation or guardrail blocks, long synchronous requests
- Use to choose the model (`NIM_MODEL_VARIANT`), precision and latency/throughput objective for the hardware, and to run the
  cookbook's example clients (T2I, T2V, I2V, V2V, Action, Transfer, the Reasoner catalog, the Responses API, streaming)
- For Cosmos3 generation through diffusers, cosmos_framework, vLLM-Omni, SGLang or TensorRT-LLM use
  `../cosmos3-generate-media/SKILL.md` (its Variant 7 drives the NIM Generator from that skill's bundled prompts and images)
- Editing the NIM pages or examples is not this skill's job (AGENTS.md: "Default to read-only assistance"); host CUDA / Docker
  failures outside the NIM go to `../cosmos3-env-troubleshoot/SKILL.md` — the NIM needs no venv or CUDA toolkit on the host
  ("Do not install a host CUDA toolkit solely for the NIM", prerequisites.md)

## Bundled assets

None — nothing is vendored here; the clients read the cookbook's assets (paths relative to this skill directory; LFS = git-LFS object, item 4):

| Need                                                            | Where it lives                                                                                                                                                                                                  |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Generation prompts, negative prompts, reference image and video | `../cookbooks/cosmos3/generator/audiovisual/assets/` (JSON plain; JPEG / MP4 LFS); the same prompts and images are vendored as plain files in `../cosmos3-generate-media/assets/` for that skill's own requests |
| Reasoner images and videos                                      | `../cookbooks/cosmos3/reasoner/assets/` (all LFS; the `image_caption` case reads `../cookbooks/cosmos3/reasoner/assets/robot_153.jpg`)                                                                          |
| Action and Transfer inputs                                      | `../cookbooks/cosmos3/generator/action/assets/`, `../cookbooks/cosmos3/generator/transfer/assets/`                                                                                                              |

## Hardware floors, as the support matrix states them

Quoted from [`support-matrix.md`](../cookbooks/cosmos3/nim/support-matrix.md) ("Generator Profiles", "Reasoner Profiles";
"Memory requirements use binary GiB: VRAM floors are per device, and system-memory floors are per host") — the single-GPU rows;
the remaining rows (offload layouts, Super BF16, TP 2, NVFP4) are in `references/skill-notes.md` § 5. Nothing here is measured by this skill:

| Runtime   | `NIM_MODEL_VARIANT`  | Precision    | Layout, as the page states it                            | Minimum VRAM per device | System-memory floor | Transfer minimum VRAM per device |
| --------- | -------------------- | ------------ | -------------------------------------------------------- | ----------------------: | ------------------: | -------------------------------: |
| Generator | `nano`               | FP8          | Model offload None, guardrails Resident; 1, 2, 4, 8 GPUs |                  44 GiB |              35 GiB |                           50 GiB |
| Generator | `nano`, `nano-droid` | BF16         | Model offload None, guardrails Resident; 1, 2, 4, 8 GPUs |                  58 GiB |              40 GiB |                           64 GiB |
| Generator | `super`              | FP8          | Model offload None, guardrails Resident; 1, 2, 4, 8 GPUs |                  93 GiB |              76 GiB |                          103 GiB |
| Reasoner  | `edge`               | BF16 or FP8  | 1 GPU, TP 1                                              |                  12 GiB |               8 GiB |                                — |
| Reasoner  | `nano`               | FP8 or NVFP4 | 1 GPU, TP 1                                              |                23.1 GiB |              17 GiB |                                — |
| Reasoner  | `nano`               | BF16         | 1 GPU, TP 1                                              |                23.1 GiB |              24 GiB |                                — |
| Reasoner  | `super`              | FP8          | 1 GPU, TP 1                                              |                  67 GiB |              43 GiB |                                — |

Reasoner rows also carry a "Minimum Compute Capability" column (support-matrix.md "Reasoner Profiles"): BF16 8.0, FP8 8.9,
NVFP4 10.0 — so the merged `edge` and `nano` rows above are two profiles each, selected by the GPU's compute capability.

Precision (deployment.md "Precision Selection"): "Omit `NIM_PRECISION` normally. Generator prefers FP8 when compatible. Reasoner
prefers BF16 on compute capability 8.0 through 8.8, FP8 on 8.9 through 9.x, and NVFP4 on 10.0 or newer, falling back to another
compatible precision." Recommendation (support-matrix.md "Practical Model Recommendation"): "H100, RTX PRO 6000 Blackwell, and lower-throughput discrete GPUs: Default to Nano, especially for video."

## Quickstart (cookbook README)

Skill note (not in the source): quoted verbatim from [`README.md`](../cookbooks/cosmos3/nim/README.md) § "Get Started" › "Use
an Existing NIM Endpoint" (one link target re-pointed to the cookbook; the commands run from `.agents/skills/cookbooks/cosmos3/nim`).
The README's "Deploy the NIM Yourself" steps (prerequisites → support matrix → deployment → back here) are the four deployment sections below.

### Use an Existing NIM Endpoint

If an administrator already deployed the NIM, install and initialize the
[Client Tooling](../cookbooks/cosmos3/nim/prerequisites.md#client-tooling), then set the service URL:

```bash
cd .agents/skills/cookbooks/cosmos3/nim
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -fsS "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
```

Confirm that metadata identifies the intended runtime and primary endpoint.
Generator uses `/v1/infer`, and Reasoner uses `/v1/chat/completions`.

Run an example that matches the active runtime and model. For a general-purpose
Generator, start with an image request:

```bash
uv run python examples/t2i.py
```

For a Reasoner:

```bash
uv run python examples/reasoner.py --case image_caption
```

## Deploy the NIM (deployment.md): NGC login, image, cache

Reference: [`deployment.md`](../cookbooks/cosmos3/nim/deployment.md) — How Selection Works, Authenticate to NGC, Prepare the
Model Cache; then the runtime's section below, then Readiness and selection.

Skill note (not in the source): the bash blocks of this and the next three sections are deployment.md's, quoted verbatim in page
order (Nano Generator, Nano Reasoner). They hand `NIM_IMAGE`, `LOCAL_NIM_CACHE` and `NIM_URL` to each other through `export`
lines that do not survive separate shell calls: run them as one call, or repeat the `export` a block needs. `NGC_API_KEY` must already be exported.

"Then log in with the literal username `$oauthtoken`:"

```bash
echo "$NGC_API_KEY" \
  | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

"Pin and pull the current release image:" — "Do not replace the versioned tag with `latest` or another unvalidated image."

```bash
export NIM_IMAGE='nvcr.io/nim/nvidia/cosmos3:2.1.0'
docker pull "$NIM_IMAGE"
```

"Prepare the Model Cache" — "The container runs as UID/GID 1000 (`nvs`)." "Keep the cache between runs to avoid repeated downloads."

```bash
export LOCAL_NIM_CACHE="${LOCAL_NIM_CACHE:-$HOME/.cache/nim/cosmos3}"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+rwX "$LOCAL_NIM_CACHE"
```

Skill note (not in the source): from a container that shares the host's Docker daemon the `chmod` above does not reach the host-side directory the daemon created root-owned — before either launch run the uid-1000 `cache_ok` test and repair in [`references/skill-notes.md`](references/skill-notes.md) § 2; a launch without it dies with `ManifestDownloadError … Permission denied (os error 13)`.

## Generator (deployment.md)

Reference: [`deployment.md`](../cookbooks/cosmos3/nim/deployment.md) — Select a Generator Model, Run the Pre-Download Profile
Preflight, Launch Generator; request: the Quickstart's `uv run python examples/t2i.py` ([`generation.md`](../cookbooks/cosmos3/nim/generation.md)).

"Run the Pre-Download Profile Preflight" — "This example checks the same Nano Generator selectors used by the launch example:"

```bash
docker run --rm \
  --gpus device=0 \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_PERF_PROFILE=latency \
  --entrypoint /bin/bash \
  "$NIM_IMAGE" \
  -lc '
    set -e
    output=/tmp/cosmos3-preflight.env
    /opt/nim/.venv/bin/python -m profile_selection.startup --output "$output"
    printf "Profile preflight passed.\n"
  '
```

"Do not pass `NGC_API_KEY`, `NIM_MODEL_PATH`, or `NIM_DFLASH_MODEL_PATH` to this bundled-profile check." "Resolve any reported
failure before a cold download." Launch Generator — "This example runs general-purpose Nano with latency on host port 8000:"

```bash
docker run -d --name cosmos3-generator \
  --gpus device=0 \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=generator \
  -e NIM_MODEL_VARIANT=nano \
  -e NIM_PERF_PROFILE=latency \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

"add `-e NIM_PRECISION=fp8` only to pin precision." Then Readiness and selection; the `t2i` client itself checks `/v1/metadata` for a
`nano`, `super` or `super-t2i` profile before posting (`require_generator_profile` in [`common.py`](../cookbooks/cosmos3/nim/examples/common.py)).

## Reasoner (deployment.md)

Reference: [`deployment.md`](../cookbooks/cosmos3/nim/deployment.md) — Select a Reasoner Model, Run the Pre-Download Profile
Preflight, Launch Reasoner; request: the Quickstart's `uv run python examples/reasoner.py --case image_caption`
([`reasoning.md`](../cookbooks/cosmos3/nim/reasoning.md) "Prepare the Client and Verify Readiness", "Run the Task Catalog").

"Set `NIM_MODEL_TYPE=reasoner` and choose `NIM_MODEL_VARIANT=edge`, `nano`, or `super`. Reasoner rejects `NIM_PERF_PROFILE` at
startup. DFlash speculative decoding is on by default for Nano and Super." Preflight — "This example checks Nano Reasoner:"

```bash
docker run --rm \
  --gpus device=0 \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_VARIANT=nano \
  --entrypoint /bin/bash \
  "$NIM_IMAGE" \
  -lc '
    set -e
    output=/tmp/cosmos3-preflight.env
    /opt/nim/.venv/bin/python -m profile_selection.startup --output "$output"
    printf "Profile preflight passed.\n"
  '
```

Launch Reasoner — "Reasoner uses the same host port. Remove a running Generator first:" `docker rm -f cosmos3-generator`.

Skill note (not in the source): that `rm -f` discards the Generator's logs — save them first (`references/skill-notes.md` § 2); the § 2 `cache_ok` test and repair apply before this launch too.
The block below is the page's only Nano Reasoner launch; it stands under the heading "DGX Spark and Jetson AGX Thor" but carries
no memory-share line ("Add `-e NIM_GPU_MEMORY_UTILIZATION="$NIM_GPU_MEMORY_UTILIZATION"` only if you exported a lower measured
value"), so it is the discrete-GPU form as written; the page's "Discrete GPU Example" launches Super FP8 instead.

```bash
docker run -d --name cosmos3-reasoner \
  --gpus device=0 \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ulimit nofile=65536:65536 \
  -p 8000:8000 \
  -e NGC_API_KEY \
  -e NIM_MODEL_TYPE=reasoner \
  -e NIM_MODEL_VARIANT=nano \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  "$NIM_IMAGE"
```

Request (reasoning.md): "The NIM needs no API key on localhost, but the OpenAI client requires a non-empty placeholder such as
`not-used`." "all cases read the answer from `message.content`." The case's input image is a git-LFS object (item 4 above).

## Readiness and selection (deployment.md)

Reference: [`deployment.md`](../cookbooks/cosmos3/nim/deployment.md) — Wait for Readiness, Verify the Selection.

"Liveness means the HTTP process exists. Readiness means the selected model can serve requests:"

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -f "$NIM_URL/v1/health/live"

until curl -fsS "$NIM_URL/v1/health/ready" >/dev/null; do
  sleep 10
done
```

"A cold start (download, materialization, model load, warmup) takes much longer than HTTP startup. Send inference only after
readiness succeeds." Verify the Selection — `curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool` (also in the Quickstart)
and `curl -fsS "$NIM_URL/v1/manifest" | python3 -m json.tool`: "Metadata reports the selected model and profile. Confirm that
`model_type` and `inference_endpoint` are `generator` and `/v1/infer`, or `reasoner` and `/v1/chat/completions`, before sending requests."

Skill note (not in the source): an unattended run adds three things the page lacks — a container-exit check in the `until` loop, logs saved before `docker rm -f`, the uid-1000 `cache_ok` test before launch — as exact commands in [`references/skill-notes.md`](references/skill-notes.md) § 2.

## 1. Identify the customer's path

Ask only what routes the workflow:

1. Is there a reachable NIM endpoint, or must one be deployed?
2. Generator or Reasoner, and which task?
3. For a new deployment: GPU model and count, compute capability, per-device
   total and free VRAM, effective host or container RAM, and whether the GPU is
   discrete or unified memory (DGX Spark/GB10, Jetson AGX Thor).

## 2. Use an existing endpoint

```bash
export NIM_URL=${NIM_URL:-http://localhost:8000}
curl -fsS "$NIM_URL/v1/health/ready"
curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool
```

Metadata must show the intended runtime before you pick an example: Generator
reports `model_type: generator` and `/v1/infer`; Reasoner reports
`model_type: reasoner` and `/v1/chat/completions`. If the runtime is wrong, fix
the deployment or `NIM_URL`; do not rewrite the request for the other runtime.

## 3. Prepare a new deployment

Walk through [`prerequisites.md`](../cookbooks/cosmos3/nim/prerequisites.md), [`support-matrix.md`](../cookbooks/cosmos3/nim/support-matrix.md), then [`deployment.md`](../cookbooks/cosmos3/nim/deployment.md)
([`helm.md`](../cookbooks/cosmos3/nim/helm.md) for Kubernetes); open [`configuration.md`](../cookbooks/cosmos3/nim/configuration.md) only for a documented
non-default setting. Use the exact pinned image, never `latest`. Apply these
rules while doing so:

- **Model choice** ([practical recommendation](../cookbooks/cosmos3/nim/support-matrix.md#practical-model-recommendation)):
  Super on H200- and B200-class GPUs when the workload needs it; Nano on H100,
  RTX PRO 6000 Blackwell, slower discrete GPUs, and all unified-memory systems.
  A Super row that fits those systems is compatibility, not a recommendation.
- **Precision** ([precision selection](../cookbooks/cosmos3/nim/deployment.md#precision-selection)):
  leave it unset. Generator prefers FP8; Reasoner prefers BF16 on compute
  capability 8.0–8.8, FP8 on 8.9–9.x, and NVFP4 on 10.0 or newer.
- **Transfer** ([headroom](../cookbooks/cosmos3/nim/support-matrix.md#transfer-headroom)): size
  against the Transfer minimum, not the generation minimum, and recommend an
  RTX PRO 6000 Blackwell 96 GB, H100 80 GB, or faster discrete GPU rather than
  DGX Spark.
- **Reasoner on DGX Spark or Jetson AGX Thor**
  ([memory share](../cookbooks/cosmos3/nim/deployment.md#set-the-reasoner-memory-share-on-unified-memory-systems)):
  the NIM counts 70% of the shared pool toward profile floors and defaults
  `NIM_GPU_MEMORY_UTILIZATION` to `0.70`. Suggest a lower value only from host
  measurements, then include it in every Reasoner launch command. Preflight
  ignores it and always uses the 70% share.
- **NGC login** ([authenticate](../cookbooks/cosmos3/nim/deployment.md#authenticate-to-ngc)):
  mention that Docker stores the login unless a credential helper is configured
  (the helper is optional) and that `--password-stdin` protects only shell
  input. Include `docker logout nvcr.io` in evaluation cleanup; logout does not
  revoke the NGC key.
- **Host software:** recommend NVIDIA driver 595.58.03 or newer and cite the
  validated configuration in [`prerequisites.md`](../cookbooks/cosmos3/nim/prerequisites.md) as tested versions, not
  minimums. Do not invent other fixed CPU, RAM, disk, or Docker minimums.
- **GPU exposure** ([GPU selection](../cookbooks/cosmos3/nim/configuration.md#gpu-selection)):
  select GPUs with Docker `--gpus`, not `CUDA_VISIBLE_DEVICES` (single device: `--gpus device=0`; the docs' nested-quote form `'"device=0"'` is only needed for a comma list, and some agent shells mangle it into `invalid reference format`).
- **Offline or air-gapped**
  ([procedure](../cookbooks/cosmos3/nim/deployment.md#offline-and-air-gapped-deployment)):
  fill the cache online, copy it preserving symlinks, and do not set
  `NIM_DISABLE_MODEL_DOWNLOAD` for Generator.
- **Preflight** ([preflight](../cookbooks/cosmos3/nim/deployment.md#run-the-pre-download-profile-preflight)):
  run it after the pull with the intended selectors and GPUs. A pass means a
  candidate profile; cold start and representative requests confirm the host.
  After startup, run `uv run python examples/inspect_profile.py` to report the
  profile actually selected.

When checking hardware, compare each device's total and free memory with the
per-device floor and never add memory across devices. On discrete GPUs, require
the support matrix's system-memory floor, which profile selection enforces, as
a minimum. The
tested-GPU inventory is the validation scope, not an allowlist.

### Unified-memory decision procedure

1. Run the memory report in
   [prerequisites](../cookbooks/cosmos3/nim/prerequisites.md#inspect-unified-memory-capacity-and-current-state)
   and show `MemFree`, `MemAvailable`, and the cache components. `MemAvailable`
   already includes reclaimable cache; do not add cache to it, and do not add a
   separate host-RAM requirement to the shared pool.
2. For Reasoner, apply the memory-share rule above.
3. Effective capacity is 70% of the device-reported total. If the profile floor
   exceeds it, report **exceeds
   total capacity** (for example, Reasoner Super BF16 TP1 on DGX Spark). If it
   fits but preflight rejects current free memory, report **fits hardware but
   not the current memory state** and ask the operator to stop only the
   identified workloads before rerunning preflight.
4. Recommend model selectors, not image-specific profile IDs.

Never clear page cache, delete model cache, stop unidentified processes, or
reboot automatically. Page-cache reclamation is an approved host-wide
diagnostic only, with explicit operator confirmation and before/after evidence.

## 4. Route the task

| Customer intent                                              | Runtime                                  | Page                                                                                    | Example                                                                                                                                                                                                                         |
| ------------------------------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Text-to-image, text-to-video, image-to-video, video-to-video | Generator                                | [`generation.md`](../cookbooks/cosmos3/nim/generation.md)                               | [`examples/t2i.py`](../cookbooks/cosmos3/nim/examples/t2i.py), [`t2v.py`](../cookbooks/cosmos3/nim/examples/t2v.py), [`i2v.py`](../cookbooks/cosmos3/nim/examples/i2v.py), [`v2v.py`](../cookbooks/cosmos3/nim/examples/v2v.py) |
| Forward, policy, or inverse dynamics                         | Generator                                | [`action.md`](../cookbooks/cosmos3/nim/action.md)                                       | [`examples/action.py`](../cookbooks/cosmos3/nim/examples/action.py)                                                                                                                                                             |
| Edge-/Nano-DROID action-only policy                          | Generator (`edge-droid` or `nano-droid`) | [`action.md`](../cookbooks/cosmos3/nim/action.md)                                       | [`examples/policy_droid.py`](../cookbooks/cosmos3/nim/examples/policy_droid.py)                                                                                                                                                 |
| Controlled video generation                                  | Generator                                | [`transfer.md`](../cookbooks/cosmos3/nim/transfer.md)                                   | [`examples/transfer.py`](../cookbooks/cosmos3/nim/examples/transfer.py)                                                                                                                                                         |
| Image or video understanding                                 | Reasoner                                 | [`reasoning.md`](../cookbooks/cosmos3/nim/reasoning.md)                                 | [`examples/reasoner.py`](../cookbooks/cosmos3/nim/examples/reasoner.py)                                                                                                                                                         |
| Responses API                                                | Reasoner                                 | [`reasoning.md`](../cookbooks/cosmos3/nim/reasoning.md)                                 | [`examples/reasoner_responses.py`](../cookbooks/cosmos3/nim/examples/reasoner_responses.py)                                                                                                                                     |
| Streaming video reasoning                                    | Reasoner                                 | [`streaming.md`](../cookbooks/cosmos3/nim/streaming.md)                                 | [`examples/streaming.py`](../cookbooks/cosmos3/nim/examples/streaming.py)                                                                                                                                                       |
| Kubernetes with Dynamo                                       | Reasoner                                 | [`dynamo/README.md`](../cookbooks/cosmos3/nim/dynamo/README.md)                         | Dynamo deployment guides                                                                                                                                                                                                        |
| Custom checkpoint                                            | Either                                   | [`bring-your-own-checkpoint.md`](../cookbooks/cosmos3/nim/bring-your-own-checkpoint.md) | Runtime-specific launch flow                                                                                                                                                                                                    |
| Metrics and dashboards                                       | Either                                   | [`monitoring.md`](../cookbooks/cosmos3/nim/monitoring.md)                               | Prometheus and Grafana setup                                                                                                                                                                                                    |
| License terms                                                | Either                                   | [`license.md`](../cookbooks/cosmos3/nim/license.md)                                     | —                                                                                                                                                                                                                               |

Initialize the client environment from [`prerequisites.md`](../cookbooks/cosmos3/nim/prerequisites.md) first and add no
dependencies. Do not bypass the example clients' argument and metadata checks.
For Reasoner, pick a case with `examples/reasoner.py --list-cases` or
`--describe <case> --format json`, keep the catalog's prompt text and sampling
values, and use `--guided-output` only when the customer asks for enforced
structured output. Literal `<think>` text requested by a prompt is ordinary
response text.

## 5. Troubleshoot safely

[`operations.md`](../cookbooks/cosmos3/nim/operations.md) owns health, logs, errors, and diagnostics; [`monitoring.md`](../cookbooks/cosmos3/nim/monitoring.md) owns
metrics; task pages own task-specific validation. Collect only sanitized
evidence: image identifier, runtime and variant, redacted `/v1/metadata` (and
`/v1/manifest` when configuration matters), GPU count, compute capability and
per-device memory, HTTP status and error body, and a redacted log excerpt.

Generator requests are synchronous
([long-running requests](../cookbooks/cosmos3/nim/operations.md#long-running-requests)): the
examples use 30-minute (Generator) and 60-minute (Transfer) client timeouts as
ceilings, not expected latency, and the backend still stops at 30 minutes unless
the container sets `NIM_TRITON_REQUEST_TIMEOUT`. Check whether a request is
still running before retrying. Do not recommend an unsafe override as a fix. If
the customer chooses one for a diagnostic, state the risk, limit it to that
diagnostic, and restore the default.

## 6. State the result

Tell the customer what was checked, which runtime, page, and example apply,
which commands are safe to run next, which assumptions remain, and whether the
conclusion comes from the documentation or from observed endpoint output.

## Verify the output

Skill note (not in the source): check what you produced before reporting it; the quotations name their page, and the tool
commands (Pillow / PyAV one-liners from the pinned client environment, `ffplay` / `ffmpeg` from the optional FFmpeg client
tooling of prerequisites.md) are [`references/skill-notes.md`](references/skill-notes.md) § 3.

- Generator (generation.md "Output and Playback"): "T2I returns raw JPEG base64 in `b64_image`. Video modes return raw base64
  VP9-in-MP4 in `b64_video`. The inactive media field is omitted. Examples write `t2i_robot_draping.jpg`,
  `t2v_robot_kitchen.mp4`, `i2v_car_driving.mp4`, or `v2v.mp4` under `examples/outputs/` (`_4step` suffix for specialists)."
  The T2I default `resolution` is "`720_1_1` (960 × 960)", the video default "`720` (1280 × 720)" with `num_frames` 189 and
  `fps` 24.0 — so the `t2i` client's JPEG is 960 × 960, and a video is `vp9` at the request's size and frame count.
- Reasoner (reasoning.md "Artifacts and Validation Boundaries"): "Each run prints the answer and writes
  `examples/outputs/reasoner_<case>/`. Validation only checks that JSON parses. Review semantics yourself." — `output.txt` is the
  answer (`message.content`), `report.md` the case report (every file: § 3). "The runner only warns on another variant, so do
  not present an Edge or Nano result as Super-validated" (reasoning.md "Super BF16 Example Baseline").
- Selected profile (operations.md "Inspect the Running Service"): `curl -fsS "$NIM_URL/v1/metadata" | python3 -m json.tool`
  reports `model_type`, the primary `inference_endpoint`, the profile/checkpoint and the Generator `model_variant`;
  `uv run python examples/inspect_profile.py` prints the selected profile's ID and manifest tags ("Profile IDs and tags are
  image-specific. Do not copy them to another image.").

## Reference files

This skill has no converted notebooks: the NIM cookbook is Markdown pages and scripts, and the pages stay in place. Read the
matching page before the first command (Read this first, item 1); `references/skill-notes.md` holds this skill's own notes.

| File                                                                                                                                                                                                                                                                     | Owns                                                                                                                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`README.md`](../cookbooks/cosmos3/nim/README.md), [`AGENTS.md`](../cookbooks/cosmos3/nim/AGENTS.md)                                                                                                                                                                     | Task → runtime → guide table, quickstart (quoted above), documentation index; the customer-assistant rules (pages are the authority, read-only by default, secrets by name, readiness and metadata before requests)                                                                                                               |
| [`prerequisites.md`](../cookbooks/cosmos3/nim/prerequisites.md), [`support-matrix.md`](../cookbooks/cosmos3/nim/support-matrix.md)                                                                                                                                       | Hardware and software requirements, unified-memory inspection, client tooling and the pinned example environment, NGC access, host verification; model choice and practical recommendation, compute-capability and memory rules, tested GPU inventory, Generator and Reasoner profile floors, Transfer headroom, media and codecs |
| [`deployment.md`](../cookbooks/cosmos3/nim/deployment.md)                                                                                                                                                                                                                | How selection works, NGC login, image pin, unified-memory share, pre-download preflight, model cache, Launch Generator / Reasoner, Edge models, readiness, Verify the Selection, offline deployment                                                                                                                               |
| [`configuration.md`](../cookbooks/cosmos3/nim/configuration.md), [`bring-your-own-checkpoint.md`](../cookbooks/cosmos3/nim/bring-your-own-checkpoint.md), [`helm.md`](../cookbooks/cosmos3/nim/helm.md), [`dynamo/README.md`](../cookbooks/cosmos3/nim/dynamo/README.md) | Every launch-time variable (selection, GPU exposure, logging, advanced profile controls, Generator and Reasoner settings, Dynamo, advanced settings); `NIM_MODEL_PATH` for a local or Hugging Face checkpoint; Kubernetes with the `nim-wfm` Helm chart and the Dynamo deployment guides for the Reasoner                         |
| [`api-reference.md`](../cookbooks/cosmos3/nim/api-reference.md), [`generation.md`](../cookbooks/cosmos3/nim/generation.md)                                                                                                                                               | Runtime routing, the `/v1/infer` envelope, strict JSON types, media inputs, Generator response, errors and live schema; T2I / T2V / I2V / V2V requests and defaults, specialist variants, resolution / frames / FPS, output and playback, common failures                                                                         |
| [`action.md`](../cookbooks/cosmos3/nim/action.md), [`transfer.md`](../cookbooks/cosmos3/nim/transfer.md)                                                                                                                                                                 | Forward dynamics, policy, Edge-/Nano-DROID policy, inverse dynamics, the action object; edge / blur / depth / segmentation / WSM controls, precomputed vs derived controls, VRAM admission                                                                                                                                        |
| [`reasoning.md`](../cookbooks/cosmos3/nim/reasoning.md), [`streaming.md`](../cookbooks/cosmos3/nim/streaming.md)                                                                                                                                                         | Chat Completions and Responses API, the task catalog, structured output, prompting patterns, errors; stateful video-frame sessions over WebSocket and REST ([`examples/streaming.py`](../cookbooks/cosmos3/nim/examples/streaming.py))                                                                                            |
| [`operations.md`](../cookbooks/cosmos3/nim/operations.md), [`monitoring.md`](../cookbooks/cosmos3/nim/monitoring.md)                                                                                                                                                     | Health and startup, inspecting the running service, logging, long-running requests, guardrails, errors, troubleshooting tables; Prometheus and Grafana                                                                                                                                                                            |
| [`release-notes.md`](../cookbooks/cosmos3/nim/release-notes.md), [`license.md`](../cookbooks/cosmos3/nim/license.md), [`acknowledgements/README.md`](../cookbooks/cosmos3/nim/acknowledgements/README.md)                                                                | 2.1.0 / 2.0.0 changes and known issues; license terms and notices; package licenses                                                                                                                                                                                                                                               |
| `../cookbooks/cosmos3/nim/examples/`                                                                                                                                                                                                                                     | The example clients ([`common.py`](../cookbooks/cosmos3/nim/examples/common.py) holds the runtime and profile checks every client runs before a request) and [`reasoner_cases.yaml`](../cookbooks/cosmos3/nim/examples/reasoner_cases.yaml)                                                                                       |
| [`references/skill-notes.md`](references/skill-notes.md)                                                                                                                                                                                                                 | This skill's notes (not from the pages): LFS inputs, headless additions, output checks, uv, the remaining floors rows                                                                                                                                                                                                             |

## Related skills

| Skill                                        | When to use                                                                                                                                                                                                                                        |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../cosmos3-generate-media/SKILL.md`         | Variant 7 — the NIM Generator driven from that skill's bundled prompts and reference images (eval-verified headless launch: preflight, cache-mount check, readiness loop, i2v request), and the six Hugging Face serving paths for the same models |
| `../cosmos3-setup/SKILL.md`                  | Framework environment, when a task also needs the Cosmos Framework checkout (the NIM itself needs only Docker, NGC access and the pinned client environment)                                                                                       |
| `../cosmos3-env-troubleshoot/SKILL.md`       | Host-side CUDA / Docker / import failures; the NIM's own failures are in `../cookbooks/cosmos3/nim/operations.md`                                                                                                                                  |
| `../cosmos3-inference/SKILL.md`              | Framework-native inference reference                                                                                                                                                                                                               |
| `../cosmos3-inference-quantization/SKILL.md` | Quantizing your own checkpoint (the NIM serves its own FP8 profiles — `NIM_PRECISION=fp8` pins the precision; bring-your-own-checkpoint.md covers `NIM_MODEL_PATH`)                                                                                |
| `../cosmos3-skill-authoring/SKILL.md`        | Editing this skill's eval cases                                                                                                                                                                                                                    |
