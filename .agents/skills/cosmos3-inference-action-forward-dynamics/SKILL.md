---
name: cosmos3-inference-action-forward-dynamics
description: >
  Run Cosmos3 action forward dynamics: a start image plus an action trajectory (AV ego poses, camera poses,
  DROID or UMI gripper actions, 57D human hand pose) rolled out into a future video with Cosmos3-Nano (Edge /
  Super where a notebook offers them) through cosmos_framework, diffusers, TensorRT-LLM, vLLM-Omni or SGLang.
  Use when the user asks to "predict future frames from actions", "run forward dynamics with Cosmos3",
  "generate a driving video from ego poses or a trajectory", "roll out a DROID or UMI action chunk
  autoregressively", or asks which backend serves action forward dynamics on their GPU. Block skill of the
  Action row: the cookbook README and every sample asset live in the entry skill ../cosmos3-inference-action.
  Not for inverse dynamics (video to actions: the inverse-dynamics block), policy (the policy block),
  text/image-to-video without actions (cosmos3-generate-media) or quantizing checkpoints
  (cosmos3-inference-quantization).
---

# Cosmos3 Action: Forward Dynamics

"Forward dynamics predicts future visual observations from an initial image and an action trajectory" (framework, vLLM-Omni and SGLang notebooks).
This block of the Action row covers the cookbook's five forward-dynamics notebooks, one variant each: the cosmos_framework
entrypoint, diffusers in-process, and the TensorRT-LLM, vLLM-Omni and SGLang video APIs. The entry skill
`../cosmos3-inference-action/` holds the cookbook README and all vendored sample assets; sibling blocks cover inverse dynamics and policy.

## Read this first

Before the first command of any variant, including the single AV smoke rollout:

1. Read the variant's reference file in full (linked under its heading, with the sections the AV path needs); the variant
   sections below only summarise the notebooks; the files under `references/` are the notebooks migrated 1:1. Also read "Overview",
   "Action Definition" and the backend's Quickstart in `../cosmos3-inference-action/references/action-readme.md` (the README copy).
2. Check the GPU with `nvidia-smi`: the CUDA major version picks the torch backend / dependency group (`cu130` /
   `cu130-train` for a 13.x driver, `cu128` / `cu128-train` for 12.x, per the framework and diffusers notebooks' configuration
   sections, Variants 1 and 2; the server variants pick a container image instead), the memory picks the model (table below).
3. Assets. Skill note (not in the source): the notebooks resolved `assets/` next to themselves (under the cookbook root that
   `find_repo_root` returns, or `COSMOS3_ACTION_ROOT`). This block has no assets of its own: the vendored copies are in the
   entry skill, `../cosmos3-inference-action/assets/` (the cookbook's layout), reachable from this directory because the whole
   `.agents/skills` tree is present. The cookbook's `../cookbooks/cosmos3/generator/action/assets/` still exists, but its
   media (`*.mp4`, `*.png`, `*.jpg`) are git-LFS objects (~130-byte pointer files in a checkout made without git-lfs), which is
   what the notebooks' default paths open. Point every `vision_path` / `action_path` / `asset_path` at the vendored copies.
4. Setup is the shared guide `../cookbooks/cosmos3/README.md` (the notebooks' "Follow the shared ... setup" links) plus
   `../cosmos3-setup/SKILL.md` (framework checkout, uv 0.11.29). Variant 1 (including its § 9 UMI-LeRobot episode) and the DROID and hand-pose
   examples of Variants 2, 4 and 5 need the framework checkout (dataset readers, `pose_utils`); the AV examples need only the backend,
   except that Variant 4's configuration cell has to run inside a framework checkout (quoted there) and the optional trajectory-plot cells import `pose_utils`.
5. Guardrails differ per notebook (quoted in each variant). README "Overview": access to the gated
   `nvidia/Cosmos-1.0-Guardrail` Hugging Face repository is required whenever they stay on; the README names each switch.
6. After a failure, go back to the reference section that covers the failing step before improvising a fix. Do not look for
   the original notebooks; the reference files replace them.
7. Verify what you produced before reporting it (Verify the output below) and record the facts in `result_summary.md`.

## When to use this skill

- Use to roll out future video from a start frame plus an action trajectory: AV ego poses (`domain_name` `av`), camera
  poses (`camera_pose`), DROID multiview manipulation (`droid_lerobot`, autoregressive), UMI (`umi`, autoregressive), human
  hand pose (`hand_pose`, 57D)
- Use to pick the serving path (in-process framework or diffusers vs the TensorRT-LLM / vLLM-Omni / SGLang servers) and the
  checkpoint that fits the GPU
- Inverse dynamics (ego-motion from a video) is `../cosmos3-inference-action-inverse-dynamics/SKILL.md`; policy (future
  observations plus actions from an instruction) is `../cosmos3-inference-action-policy/SKILL.md`; the row's README, Action
  Definition and assets are `../cosmos3-inference-action/SKILL.md`; text/image-to-video without actions is
  `../cosmos3-generate-media/SKILL.md`; environment failures go to `../cosmos3-env-troubleshoot/SKILL.md`

## Bundled assets (vendored in the entry skill)

Paths are relative to this skill directory; "used by" names notebook sections. Skill note (not in the source): row counts
and image sizes were read from the vendored files.

| Example (`domain_name`)          | Start frame                                                                                                             | Actions / prompt                                                                                                                                                                                  | Vendored                                           | Used by                                                                           |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------- |
| AV forward / left / right (`av`) | `../cosmos3-inference-action/assets/images/av_0.jpg` (832x480)                                                          | `../cosmos3-inference-action/assets/actions/av_traj_forward.json`, `av_traj_left.json`, `av_traj_right.json` (60 rows x 9D each); prompt "You are an autonomous vehicle planning system."; 10 fps | yes                                                | all five variants (TensorRT-LLM: the forward trajectory only)                     |
| Camera pose (`camera_pose`)      | `../cosmos3-inference-action/assets/images/lighthouse_720.png`, `solar_720.png`, `mountain_720.png` (720x540 / 720x535) | `../cosmos3-inference-action/assets/actions/camera_action.json` (60 x 9D); prompts `../cosmos3-inference-action/assets/prompts/lighthouse.txt`, `solar.txt`, `mountain.txt`; 30 fps               | yes                                                | framework § 7, diffusers camera cases                                             |
| UMI (`umi`)                      | `../cosmos3-inference-action/assets/images/umi.png` (256x256)                                                           | `../cosmos3-inference-action/assets/actions/umi.json` (32 rows x 10D = two 16-action chunks); prompt "mouse arrangement"; 20 fps                                                                  | yes                                                | diffusers, vLLM-Omni and SGLang UMI sections                                      |
| UMI LeRobot episode (`umi`)      | first frame of the episode                                                                                              | `../cosmos3-inference-action/assets/umi_lerobot_example/` (`UMILeRobotDataset`)                                                                                                                   | yes                                                | framework § 9 UMI                                                                 |
| DROID (`droid_lerobot`)          | first frame of the episode (three tiled views)                                                                          | `../cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/` (`DROIDMergedLeRobotDataset` / `DROIDLeRobotDataset`, `action_normalization="quantile_rot"`)                                | no: LFS clips 3,982,548 to 5,337,025 B             | framework § 8, diffusers DROID, vLLM-Omni and SGLang Robotics                     |
| Human hand pose (`hand_pose`)    | first frame of the episode                                                                                              | `../cookbooks/cosmos3/generator/action/assets/human_hand_pose_lerobot_example/` (`HumanHandPoseLeRobotDataset`, 16 x 57D)                                                                         | no: LFS clip 3,839,215 B (+ a 2,221,102 B parquet) | framework hand-pose section, diffusers Human Hand Pose, vLLM-Omni Human Hand Pose |

Skill note (not in the source): materialising a non-vendored example. The LeRobot `meta/` and `data/` parquet files are
plain git blobs present in every checkout; only the media are LFS objects. Either run
`git lfs pull --include=".agents/skills/cookbooks/cosmos3/generator/action/assets/<example>/**"` in a checkout with git-lfs
installed, or download each LFS object from the public repository at the pinned commit,
`https://media.githubusercontent.com/media/NVIDIA/cosmos-framework/bbd4a67c3cf54cc6699464ad1259f3a0f6537da9/<path>`
(`<path>` = the file's repository path under `.agents/skills/cookbooks/cosmos3/generator/action/assets/`), into the same
relative layout; then point the notebook's `resolve_input(...)` / `asset_path(...)` / dataset `root` at the materialised
directory (the Variant 2 and 4 DROID cells also accept `COSMOS3_DROID_ROOT`).

## Model sizes vs GPU (only what the sources state)

| Checkpoint                                                | What the sources say                                                                                                                                                                                                                                                                                                                                                            | How the notebooks run it                                                                                                   |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Nano`                                     | README: "Cosmos3-Nano action-generation supports three distinct tasks"; the default of every FD notebook (`COSMOS3_CHECKPOINT_PATH` default, diffusers `MODEL_IDS`, every server block); setup guide: "Nano downloads plus CUDA dependencies can take tens of GiB"                                                                                                              | one GPU in every notebook (`CUDA_VISIBLE_DEVICES=0`, `--gpus '"device=0"'`, the `cosmos3-nano-1gpu.yaml` VisualGen config) |
| `nvidia/Cosmos3-Edge`                                     | framework notebook: "set `COSMOS3_CHECKPOINT_PATH=Cosmos3-Edge` ... nothing else changes"; vLLM-Omni notebook: a Nano **or** Edge server, and (FP8) "the launch below serves either tier"; setup guide: "Edge is the compact 4B checkpoint" whose "action weights are not served by this pipeline" (TensorRT-LLM); not in the diffusers `MODEL_IDS`, not in the SGLang notebook | one GPU, the Nano launch with the model name swapped                                                                       |
| `nvidia/Cosmos3-Super`                                    | framework notebook: `COSMOS3_CHECKPOINT_PATH=Cosmos3-Super`; diffusers: "Cosmos3-Super also ships `action_gen=True`, so passing `model="Cosmos3-Super"` works unchanged"; vLLM-Omni FP8: "For Super, add the upstream `--tensor-parallel-size 4` and `--enable-layerwise-offload` options if needed"                                                                            | no FD notebook launches it; the setup guide's Super servers use four GPUs                                                  |

Skill note (not in the source): no FD notebook states a footprint or runtime; the measured figures in `../cosmos3-generate-media/references/skill-notes.md` § 2 are for audiovisual generation.

## Variant 1 — cosmos_framework (`cosmos_framework.scripts.inference`, in the checkout)

Reference: [`references/cosmos-framework.md`](references/cosmos-framework.md) — read § 1 to § 5 (Prerequisites, Configure
Paths, Clone or Reuse Cosmos Framework, Install, Verify) and § 6 "AV Forward Dynamics" before the first command; § 7
Camera, § 8 DROID (autoregressive, non-vendored episode), § 9 UMI (`umi_lerobot_example`) and the hand-pose section
(non-vendored) repeat the pattern. README Quickstart: the README copy's "Run with Cosmos Framework" (the
`torchrun --nproc-per-node=1 ... --parallelism-preset=latency` form; the notebook runs the module directly, below).

"Tested path from the audit": "Framework checkout: `packages/cosmos3`", "Install command:
`uv sync --all-extras --group=cu130-train`", "Models: `Cosmos3-Nano` (default), `Cosmos3-Edge`, and `Cosmos3-Super`";
"Every inference cell below reads the checkpoint from `COSMOS3_CHECKPOINT_PATH` (set in step 2, default `Cosmos3-Nano`)".
§ 2 exports `COSMOS3_REPO`, `COSMOS3_UV_GROUP` (`cu130-train`; "use `cu128-train` for a CUDA 12.x driver"),
`COSMOS3_OUTPUT_ROOT`, `COSMOS3_CHECKPOINT_PATH`, `HF_HOME`, `CUDA_VISIBLE_DEVICES`. § 4 install ("May take several minutes"), plus a skill-added first line:

```bash
uv --version 2>/dev/null | grep -q '^uv 0\.11\.29' || { curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }  # skill note (not in the source): the checkout pins uv ==0.11.29; `uv self update` works only for a standalone-installer uv
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

export GIT_LFS_SKIP_SMUDGE=1
cd "$COSMOS3_REPO"
uv sync --all-extras --group="$COSMOS3_UV_GROUP"
```

Skill note (not in the source): `uv sync` in the checkout needs the uv its `pyproject.toml` pins (`[tool.uv] required-version = "==0.11.29"`);
the notebook only checks that `uv` exists, so the first line above installs 0.11.29 beside a different host uv (`uv self update 0.11.29` applies only
to a standalone-installer uv and exits 2 on a package-managed one: 3 of 4 eval agents, eval 2026-11-08). Setup guide: "Framework inference enables
**guardrails by default**" (system packages plus the gated guardrail); the AV cell below keeps them on; the notebook's DROID, UMI and hand-pose runs pass `--no-guardrails` (the README's switch).

The AV spec (§ 6) is "a JSONL spec, one line per run. Each line shares the same start frame (`vision_path`) but uses a
different ego trajectory (`action_path`)". One record of the notebook's `av_records`:

```python
    {
        "action_chunk_size": 60,
        "action_path": resolve_input(action_rel),
        "domain_name": "av",
        "fps": 10,
        "image_size": 480,
        "view_point": "ego_view",
        "model_mode": "forward_dynamics",
        "name": name,
        "prompt": "You are an autonomous vehicle planning system.",
        "seed": 0,
        "vision_path": av_vision_path,
    }
```

Skill note (not in the source): `resolve_input` returns the absolute path of the cookbook's `av_0.jpg` /
`av_traj_forward.json`; here write the absolute paths of `../cosmos3-inference-action/assets/images/av_0.jpg` and
`../cosmos3-inference-action/assets/actions/av_traj_forward.json` (one line per trajectory; the notebook writes three). The
cell saves the spec as `action_forward_dynamics_av_custom.jsonl` under `COSMOS3_INPUT_DIR` (= `COSMOS3_OUTPUT_ROOT/inputs`)
and exports `COSMOS3_AV_FD_INPUT` (that file) and `COSMOS3_AV_FD_OUTPUT` (`COSMOS3_OUTPUT_ROOT/action_forward_dynamics_av_custom`).

Run (§ 6 "Run AV Forward-Dynamics Inference", verbatim):

```bash
set -euo pipefail

cd "$COSMOS3_REPO"
echo "checkpoint path: $COSMOS3_CHECKPOINT_PATH"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
MASTER_ADDR="$COSMOS3_MASTER_ADDR" MASTER_PORT="$COSMOS3_NANO_TEXT_MASTER_PORT" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
.venv/bin/python -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i "$COSMOS3_AV_FD_INPUT" \
  -o "$COSMOS3_AV_FD_OUTPUT" \
  --checkpoint-path "$COSMOS3_CHECKPOINT_PATH" \
  --video-save-quality 8 \
  --image_size 480 \
  --seed 0 \
  --benchmark
```

Skill note (not in the source): the configuration cell set these through `os.environ`, so export them in the shell that
runs the block: `COSMOS3_REPO`, `COSMOS3_CHECKPOINT_PATH` (default `Cosmos3-Nano`), `CUDA_VISIBLE_DEVICES=0`,
`COSMOS3_MASTER_ADDR=127.0.0.1`, `COSMOS3_NANO_TEXT_MASTER_PORT` (the notebook binds a free local port),
`COSMOS3_AV_FD_INPUT`, `COSMOS3_AV_FD_OUTPUT`. "Each run writes its video to:
`<output>/action_forward_dynamics_av_custom/<name>/vision.mp4`"; the viewer cell reads it with `media.read_video` and
prints `fps` and `num_frames`.

Skill note (not in the source): to run the AV block without guardrails (the eval case's setting), add `--no-guardrails` to the command above — the README's Cosmos Framework switch, which the notebook's DROID (§ 8), UMI (§ 9) and hand-pose cells already pass.

## Variant 2 — diffusers (in-process: `Cosmos3OmniPipeline` + `CosmosActionCondition`)

Reference: [`references/diffusers.md`](references/diffusers.md) — § 1 Prerequisites, § 2 Configure Paths and Environment,
§ 3 Install Diffusers Dependencies, § 5 Verify, § 7 "Define Action Cases, Runner, and Viewer Helpers" (`ACTION_SETS`,
`get_pipe`, `generate_chunk`, `run_action`, `run_rollout`), "Quantized Checkpoints (FP8)", then the use case. README
Quickstart: the README copy's "Run with Diffusers" (the standalone `pipe(prompt=..., action=CosmosActionCondition(...))`).

Notebook: "Every action input is grouped into a `CosmosActionCondition`; `height`, `width`, and `num_frames` stay unset
because the pipeline derives the frame count from `chunk_size + 1` and the conditioning canvas from `resolution_tier`."
"Action prompts are plain task descriptions rather than upsampled JSON ... which is also why `use_system_prompt=False`."
"**Guardrails:** Disabled by default in this notebook. Set `COSMOS3_DIFFUSERS_GUARDRAILS=true` before running the helper
cell to enable them." "**uv version:** use `uv 0.11.29`, matching the version pinned by Cosmos Framework." § 2 exports
`COSMOS3_DIFFUSERS_ACTION_VENV`, `COSMOS3_REPO`, `COSMOS3_TORCH_BACKEND` (`cu130`, or `cu128`), `HF_HOME`, `UV_LINK_MODE=copy`,
`CUDA_VISIBLE_DEVICES`, `COSMOS3_DIFFUSERS_PRECISION` (`bf16  # or fp8`). § 3 install (the venv and install lines plus a skill-added first line; the rest registers a Jupyter kernel):

```bash
uv --version 2>/dev/null | grep -q '^uv 0\.11\.29' || { curl -LsSf https://astral.sh/uv/0.11.29/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" sh; export PATH="$HOME/.local/bin:$PATH"; hash -r; }  # skill note (not in the source): the notebook's uv 0.11.29; inside the checkout any other uv stops at the pyproject pin
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv venv "$COSMOS3_DIFFUSERS_ACTION_VENV" --python 3.13 --seed --managed-python --allow-existing
source "$COSMOS3_DIFFUSERS_ACTION_VENV/bin/activate"

# The LeRobot readers, pose helpers, and trajectory plots add parquet support and plotting on top
# of the diffusers stack.
uv pip install --torch-backend="$COSMOS3_TORCH_BACKEND" \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  "lerobot @ https://github.com/huggingface/lerobot/archive/1a4316c6845330bc552fb982dbc44bdb4f66f2f1.tar.gz" \
  accelerate \
  av \
  cosmos_guardrail \
  datasets \
  draccus \
  huggingface_hub \
  imageio \
  imageio-ffmpeg \
  ipykernel \
  loguru \
  matplotlib \
  mujoco \
  pandas \
  pyarrow \
  scipy \
  torch \
  torchvision \
  transformers \
  "nvidia-modelopt==0.44.0"
```

Skill note (not in the source): run the block as ONE shell call (the `source` does not carry into a later call). Inside a framework
checkout uv applies the pyproject's `required-version` pin from any cwd; the block's first line installs the pinned 0.11.29 beside a different host uv (eval 2026-11-08: a package-managed uv 0.13 cost three of four agents a re-run).
Before the first run on any host without a desktop (every container): `apt-get update && apt-get install -y libxcb1 libgl1 libglib2.0-0` — `Cosmos3OmniPipeline` imports `cosmos_guardrail` -> `cv2` even with guardrails off (§ 1 lists it only as a reactive step).

Run the AV case ("Roll out the `av_traj_forward` ego trajectory from the AV start frame"): the § 2 configuration cell, the
§ 7 helper cell, then `run_action("av_forward")`, as ONE Python file executed with
`"$COSMOS3_DIFFUSERS_ACTION_VENV/bin/python"` from this skill directory (§ 5 and § 7 raise unless `sys.executable` is that
venv's python; `ipykernel` provides the `IPython.display` imports). Skill note (not in the source): the configuration cell
derives `COSMOS3_ACTION_ROOT` as the cookbook directory (the LFS-pointer copy); after `configure_diffusers_environment()`
set `COSMOS3_ACTION_ROOT = Path("../cosmos3-inference-action").resolve()` so that `asset_path(...)` opens the vendored files,
and export `COSMOS3_ACTION_OUTPUT_ROOT="$PWD/outputs/diffusers"` before the cell (its default sits under
`COSMOS3_ACTION_ROOT`). `run_action` prints `frames:  61 at 10 fps, resolution tier 480` (two spaces after the colon) and writes
`<COSMOS3_ACTION_OUTPUT_ROOT>/av_forward/av_forward.mp4` (`export_to_video(frames, str(output_path), fps=spec["fps"], macro_block_size=1)`, 10 fps for the AV cases);
sampling is `FIXED_SAMPLING` (`num_steps` 30, `guidance` 1.0, `shift` 10.0 through `UniPCMultistepScheduler`, `seed` 0).
Other cases: `run_action("av_left" | "av_right" | "camera_lighthouse" | "camera_solar" | "camera_mountain")`; DROID, UMI and
hand pose chain 16-action chunks with `run_rollout` (DROID and hand pose need `COSMOS3_REPO` and the non-vendored episodes).

FP8: "set `COSMOS3_DIFFUSERS_PRECISION=fp8` before loading. Nano and Super use the dedicated Hub repos
`nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`. Use-case cells stay the same." "Optional: point `COSMOS3_FP8_MODEL_PATH` at a local
Diffusers checkpoint directory." The loader (`resolve_pretrained_source`, `ensure_modelopt_state`, `validate_fp8_transformer`) is in § 7; the
notebook's BF16-card and W8A16/W8A8 schedule notes are in the reference's "Quantized Checkpoints (FP8)" section. To quantize your own checkpoint: `../cosmos3-inference-quantization/SKILL.md`.

## Variant 3 — TensorRT-LLM (`trtllm-serve`, `POST /v1/videos/sync`)

Reference: [`references/trt-llm.md`](references/trt-llm.md) — read all of it. README Quickstart: the README copy's "Run
with TensorRT-LLM" (the standalone `requests.post` form; "Both notebooks' current multipart fields were exercised against
TensorRT-LLM source revision `cc63e66daf52644f1441e117bb7ba67d20566016`. Follow the source pin in the shared setup.").

Server: "Follow the shared TensorRT-LLM setup" — `../cookbooks/cosmos3/README.md#tensorrt-llm-generator` (install route: the skill note
below) — "then run this from the TensorRT-LLM checkout. This notebook is a client; keep the server in a separate terminal.":

```bash
export TRTLLM_ROOT="${TRTLLM_ROOT:-$PWD}"
trtllm-serve nvidia/Cosmos3-Nano \
  --visual_gen_args "$TRTLLM_ROOT/examples/visual_gen/configs/cosmos3-nano-1gpu.yaml" \
  --port 8000
```

Setup guide: "TensorRT-LLM serves Edge for text-to-image, text-to-video, and image-to-video only: Edge has no audio tower,
its action weights are not served by this pipeline" — Nano is the TensorRT-LLM forward-dynamics model. Guardrails: the
notebook sends `"use_guardrails": True`; README: `use_guardrails: false` (TensorRT-LLM `extra_params`); setup guide:
`TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` before starting the server. Skill note (not in the source): install with `../cosmos3-generate-media/references/trt-llm.md`
§ 7.2 Route A — the `tensorrt_llm==1.3.0rc27` wheel (`torch==2.13.0+cu130`), the YAML-configs checkout `bca6761` as `TRTLLM_ROOT` and the block's `export CUDA_HOME=...` line — then the
serve block above (§ 7.7 has the same lines): it served this AV request on three eval cells, eval 2026-11-08 (HTTP 200, `video` (61,480,832,3), `action` (60,9), `frame_rate` 10). The README's
devel-container source build at `cc63e66` needs a devel tag whose torch/NCCL match the checkout (`devel:1.4.0rc3` failed at CMake, `undefined symbol: ncclGetPeerDeviceLeInfo`); use it only for a VisualGen change newer than rc27 (`references/trt-llm.md`).

Client: "Install client-side decoding dependencies in this notebook kernel if needed:
`pip install requests safetensors torch imageio imageio-ffmpeg pillow`" — plus `ipython` (skill note: the helper cell imports `IPython.display`; a headless shell has no kernel). The paths cell sets `ACTION_ROOT` to the cookbook
directory, `OUTPUT_ROOT` from `COSMOS3_TRTLLM_ACTION_OUTPUT_ROOT`, `TRTLLM_BASE_URL` from `COSMOS3_TRTLLM_BASE_URL`
(`http://localhost:8000`) and `TRTLLM_API_KEY` from `COSMOS3_TRTLLM_API_KEY` (`tensorrt_llm`). Skill note (not in the
source): set `image_path` / `action_path` to the vendored `../cosmos3-inference-action/assets/images/av_0.jpg` and
`../cosmos3-inference-action/assets/actions/av_traj_forward.json` (the cell asserts 60 rows x 9) and the output root to your outputs.

Readiness: `wait_for_server()` polls `GET <server root>/health` every 10 s for up to 1800 s. Request (`submit_action`):
"The notebooks use the blocking `POST /v1/videos/sync` route. Every Action request sets `format=safetensors` and
`response_format=file`; the response is `application/octet-stream` with named `video`, `action`, and `frame_rate` tensors.
This differs from vLLM-Omni's top-level action metadata." The AV inputs cell:

```python
prompt = "You are an autonomous vehicle planning system."
extra_params = {
    "action_mode": "forward_dynamics",
    "domain_name": "av",
    "action": actions,
    "use_guardrails": True,
}
```

Skill note (not in the source): for a guardrail-free run (the eval case's setting) export `TRTLLM_DISABLE_COSMOS3_GUARDRAILS=1` before `trtllm-serve` (setup guide) or send `"use_guardrails": false` (README); the cell above is the notebook's value.

The form carries `prompt`, `format` `safetensors`, `response_format` `file`, `seed` `"0"` and `extra_params` (compact JSON),
the start frame as the multipart file `image_reference`, with headers `Accept: application/octet-stream` and
`Authorization: Bearer <TRTLLM_API_KEY>`. "The `av` domain preset supplies the trained 480p, 10 fps recipe, so the request
does not duplicate those mode defaults" — "a 60-step, 9D action trajectory conditions a 61-frame rollout". `submit_action`
writes `forward_dynamics_av.safetensors` and prints the `video` / `action` shapes; `save_and_view_video` writes
`forward_dynamics_av.mp4` with `imageio` at the payload's `frame_rate` (expects `[T,H,W,3]`); the last cell prints the first
five `action` rows. The asynchronous `POST /v1/videos` route (202, poll `GET /v1/videos/{id}`, download `GET /v1/videos/{id}/content`) serves the same payload.

## Variant 4 — vLLM-Omni (Docker server, `POST /v1/videos`)

Reference: [`references/vllm-omni.md`](references/vllm-omni.md) — Prerequisites, "Start vLLM-Omni Server" (Nano / Edge,
"Quantized Checkpoints"), "Install Notebook Dependencies", "Configure Notebook Variables" (`build_action_prompt`), "AV";
"Robotics", "UMI" and "Human Hand Pose" follow. README Quickstart: the README copy's "Run with vLLM-Omni"
(`COSMOS3_WORKDIR`, `COSMOS3_HOST_PORT=8001`, the diffusion-defaults table).

Server — "Choose either Cosmos3-Nano or Cosmos3-Edge and start that server in a terminal from the cookbook root
(`.agents/skills`). The commands are mutually exclusive: both use the same container name and publish the container's
port `8000` to host port `8001`, which matches the notebook's default `http://localhost:8001` endpoint." Nano:

```bash
docker rm -f cosmos3-vllm-omni-notebook 2>/dev/null || true

docker run -d --name cosmos3-vllm-omni-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$PWD:/workspace" \
  -v "$PWD:$PWD" \
  -w /workspace \
  -p 8001:8000 --ipc=host \
  vllm/vllm-omni:cosmos3 \
  vllm serve nvidia/Cosmos3-Nano \
    --no-guardrails \
    --omni \
    --model-class-name Cosmos3OmniDiffusersPipeline \
    --allowed-local-media-path / \
    --port 8000 \
    --init-timeout 1800

# Wait until this returns model metadata before running the inference cells.
curl http://localhost:8001/v1/models
```

Edge: the same block with `vllm serve nvidia/Cosmos3-Edge`. Prerequisites: "To disable guardrails server-wide, pass
`--no-guardrails` to `vllm serve`" (both blocks do); "When running with guardrails enabled, also set and pass `HF_TOKEN`
to the container". Setup guide: "always pass `--init-timeout 1800`"; "vLLM-Omni prints `Application startup complete.`
when the API is ready". Skill note (not in the source): the block mounts no Hugging Face cache and passes no `HF_TOKEN`, so
every new container downloads the checkpoint anonymously (`../cosmos3-generate-media/references/skill-notes.md` § 4: headless pattern, token).

Client: "`%pip install matplotlib pandas pyarrow "lerobot[dataset]"`" and "`%pip install av==18.0.0`", plus `requests`. Skill note (not in the source): headless
client venv for the AV request — `uv venv --no-config .venv-client --python 3.12 --seed --managed-python && uv pip install --no-config --python .venv-client/bin/python requests pillow ipython av==18.0.0`,
then run the cells with `.venv-client/bin/python` (`av` 18 needs Python >= 3.11; the request and verify code import `PIL`; `--no-config` skips the checkout's uv pin; `lerobot[dataset]` is for the Robotics section only).
The configuration cell's `find_framework_root` walks up to the directory holding `pyproject.toml` and `cosmos_framework/`
("Launch from the Framework checkout or a notebook directory inside it."), sets `COSMOS_ROOT` to its `.agents/skills`, and
reads `COSMOS3_REPO`, `COSMOS3_VLLM_OUTPUT_ROOT`, `COSMOS3_DROID_ROOT`, `COSMOS3_VLLM_BASE_URL` (`http://localhost:8001`).
Skill note (not in the source): `resolve_input(...)` resolves the notebook's cookbook paths to the LFS copy; use the vendored
`av_0.jpg` / `av_traj_forward.json` under `../cosmos3-inference-action/assets/` as `vision_path` / `action_path`, and export
`COSMOS3_VLLM_OUTPUT_ROOT="$PWD/outputs/vllm_omni"`.

Readiness: `check_vllm_server()` polls `GET /v1/models` every 10 s for up to 600 s. Request (`submit_forward_dynamics`):
multipart `POST /v1/videos` with `files={"input_reference": ...}` and the form fields `prompt`, `num_frames` =
`record["action_chunk_size"] + 1`, `fps`, `size` = `f"{input_width}x{input_height}"` ("The request sets top-level `size` to
the conditioning image resolution so vLLM-Omni returns output at the input resolution without reflection padding"),
`num_inference_steps` 30, `guidance_scale` 1.0, `flow_shift` 10.0, `seed`, `extra_params` = `json.dumps(extra_params)`:

```python
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": record["domain_name"],
        "action_chunk_size": record["action_chunk_size"],
        "image_size": record["image_size"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
    prompt = build_action_prompt(
        record, width=input_width, height=input_height, default_prompt="A robot manipulates an object."
    )
```

`build_action_prompt` turns the record's prompt into the structured JSON caption (viewpoint `framing`, `actions` with the
time span, `duration`, `fps`, `resolution`, `aspect_ratio`). The cell polls `GET /v1/videos/{id}` every 2 s until `status`
is `completed` (`failed` / `cancelled` raise), downloads `GET /v1/videos/{id}/content` to `<run_dir>/vision.mp4` and
writes `response.json`, `final.json` and, when the server returns one, `action.json`. "Each run writes its video to:
`<output>/action_forward_dynamics_av_custom/<name>/vision.mp4`".

FP8 ("Quantized Checkpoints"): "Published Edge FP8, Nano FP8 and Super FP8 checkpoints can be downloaded to a local
directory. Or use an output from the quantization walkthrough. Serve that directory with the original
`--served-model-name` so the request cells keep their existing model IDs. Do not select FP8 with `--revision` alone: the
Cosmos3 vLLM-Omni pipeline must receive a materialized local checkpoint." The reference holds the download blocks (`snapshot_download`
into `FP8_MODEL_DIR`, with `SERVED_MODEL`) and "Serve the selected checkpoint" (mounts `HF_HOME`, the model directory and `FP8_SOURCE_DIR` — "The source mount
preserves exported symlinks" — then `vllm serve "$FP8_MODEL_DIR" --served-model-name "$SERVED_MODEL" --no-guardrails --omni ...`; "For Super, add the upstream `--tensor-parallel-size 4` and `--enable-layerwise-offload` options if needed."). To quantize your own checkpoint: `../cosmos3-inference-quantization/SKILL.md`.

## Variant 5 — SGLang (Docker server, `POST /v1/videos`)

Reference: [`references/sglang.md`](references/sglang.md) — "Set up HF Cache", "Start the SGLang server", "FP8", "Install
Notebook Dependencies", "Configure Notebook Variables", "AV"; "Robotics" and "UMI" follow (this notebook has no hand-pose
section, and the Action README has no SGLang section). Server — the notebook's two cells plus two skill-added `-e` lines ("SGLang container is
run under root by default. User can create a writable cache directory on host, or start the container with host user id."):

```bash
export SGLANG_HF_CACHE="${SGLANG_HF_CACHE:-$HOME/.cache/sglang-huggingface}"
mkdir -p "$SGLANG_HF_CACHE"
chmod 777 "$SGLANG_HF_CACHE"
```

```bash
docker rm -f cosmos3-sglang-notebook 2>/dev/null || true

docker run -d --init --name cosmos3-sglang-notebook \
  --runtime nvidia --gpus '"device=0"' \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1 \
  -e HF_TOKEN \
  -v $SGLANG_HF_CACHE:/root/.cache/huggingface \
  -v "$PWD:/workspace" \
  -p 30000:30000 --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve \
  --model-path nvidia/Cosmos3-Nano \
  --host 0.0.0.0

# Wait until this returns model metadata before running the inference cell.
curl http://localhost:30000/v1/models
```

Skill note (not in the source): the `-e SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` and `-e HF_TOKEN` lines are skill additions (the notebook passes neither; the same two lines as
`../cosmos3-inference-action-inverse-dynamics/SKILL.md` and `../cosmos3-generate-media/SKILL.md` Variant 4): without the first the server loads the gated `nvidia/Cosmos-1.0-Guardrail` at startup and exits with
`GatedRepoError 401` (all four eval agents' verbatim launches, eval 2026-11-08; the per-request `"guardrails": False` below does not prevent it; `../cosmos3-generate-media/references/skill-notes.md` § 3); without `HF_TOKEN` anonymous Hub downloads are rate-limited (`../cosmos3-generate-media/SKILL.md` Variant 4, `../cosmos3-generate-media/references/sglang.md` § 2).

Client: "`%pip install matplotlib pandas pyarrow "lerobot[dataset]"`", "`%pip install av==18.0.0`", `requests` (client venv: the Variant 4 skill note). The
configuration cell reads `COSMOS3_FRAMEWORK_PATH` ("the root of a local checkout of the **cosmos-framework** repository
... that contains the `cosmos_framework/` Python package"), `COSMOS3_SGLANG_OUTPUT_ROOT` and `COSMOS3_SGLANG_BASE_URL`
(`http://localhost:30000`). Skill note (not in the source): paths and output root as in Variant 4 (`COSMOS3_SGLANG_OUTPUT_ROOT="$PWD/outputs/sglang"`).

Readiness: `check_sglang_server()` polls `GET /v1/models` every 10 s for up to 600 s. Skill note (not in the source): it never notices an exited container; poll from the
shell instead: `timeout 600 bash -c 'until curl -sf localhost:30000/v1/models; do sleep 10; done' || docker logs cosmos3-sglang-notebook`. Request (`submit_forward_dynamics`, AV
section): the form fields of Variant 4 with a plain-text `prompt` (`str(record.get("prompt") or "").strip() or "A robot
manipulates an object."`) and these `extra_params`:

```python
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": record["domain_name"],
        "action_view_point": record["view_point"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
```

The Robotics cell rounds `size` down to multiples of 16. Poll and download as in Variant 4 (`response.json`, `final.json`,
`vision.mp4`, `action.json` when returned); output `<output>/action_forward_dynamics_av_custom/<name>/vision.mp4`.
Robotics: "Before running this LeRobot section, PyAV may need to be reinstalled as `av==15.1.0`."

FP8 ("FP8"): "To serve Cosmos3-Nano in FP8, you can either: Download the prebuilt FP8 checkpoint from Hugging Face: `nvidia/Cosmos3-Nano-FP8`.
Quantize a checkpoint yourself by following the quantization guide." (`../cosmos3-inference-quantization/SKILL.md`) — `sglang serve --model-path nvidia/Cosmos3-Nano-FP8 --host 0.0.0.0`;
"To serve a checkpoint you quantized yourself, pass its local directory to `--model-path`. If the server runs in Docker, mount that directory and use its path inside the container."

## Verify the output

Facts the sources state for the AV path: `num_frames` = `action_chunk_size + 1` = 61 frames at 10 fps (README table:
"60 frames @ 10FPS"; TensorRT-LLM notebook: "a 61-frame rollout" with "the trained 480p, 10 fps recipe"; diffusers prints
`frames:  61 at 10 fps, resolution tier 480`); the servers return the start frame's size (`size` = `832x480` for `av_0.jpg`).
Autoregressive examples: "Each autoregressive video chunk includes its conditioning frame at frame 0", i.e. 17 frames per
16-action chunk and "one 80-frame video" after stitching five DROID chunks; UMI: 20 fps, 256x256, two chunks. The viewer
cells check: framework — `media.read_video` (`fps`, `num_frames`); diffusers `view_action` — file size, embedded video;
vLLM-Omni / SGLang — `final.json` `status` `completed`; TensorRT-LLM — `video` shape `[T,H,W,3]`, `action` shape, `frame_rate`.

Skill note (not in the source): check the file before reporting it. Video: `ffprobe -v error -select_streams v:0
-count_frames -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration -of default=nw=1 <file>`
(`ffprobe` is on neither a bare host nor in the venvs: `apt-get update && apt-get install -y ffmpeg` as root, or the `av` decoder one-liner in
`../cosmos3-generate-media/SKILL.md` "Verify the output"). Record codec, size, frames, fps, runtime and GPU memory in `result_summary.md`: framework CLI —
`nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 1000 > gpu_mem.csv &` during the run, `kill %1` after, report the max; diffusers — `torch.cuda.max_memory_allocated()`; vLLM-Omni — `final.json` `peak_memory_mb`.

## Reference files

Every embodiment, the autoregressive loops, stitching and viewer code and the FP8 sections live in `references/`, migrated
1:1 from the forward-dynamics notebooks — read the matching file before the first command. Do not look for the notebooks.

| File                                                      | Content                                                                                                                      |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `references/cosmos-framework.md`                          | Variant 1 — the cosmos_framework forward-dynamics notebook (AV, Camera, DROID, UMI, hand pose)                               |
| `references/diffusers.md`                                 | Variant 2 — the diffusers forward-dynamics notebook (AV x3, Camera x3, DROID, UMI, hand pose; "Quantized Checkpoints (FP8)") |
| `references/trt-llm.md`                                   | Variant 3 — the TensorRT-LLM forward-dynamics notebook (AV)                                                                  |
| `references/vllm-omni.md`                                 | Variant 4 — the vLLM-Omni forward-dynamics notebook (AV, Robotics, UMI, hand pose; "Quantized Checkpoints")                  |
| `references/sglang.md`                                    | Variant 5 — the SGLang forward-dynamics notebook (AV, Robotics, UMI; "FP8")                                                  |
| `../cosmos3-inference-action/references/action-readme.md` | The Action cookbook README: Overview, Action Definition (incl. the AgiBotWorld robot description), Quickstarts, Post-Train   |
| `../cookbooks/cosmos3/README.md`                          | The shared environment setup guide (kept in place)                                                                           |

Origin: the forward-dynamics notebooks of `.agents/skills/cookbooks/cosmos3/generator/action/` (revision bbd4a67), superseded by this skill; the cookbook keeps `assets/` (LFS) and `finetune/`.

## Related skills

| Skill                                                   | When to use                                                              |
| ------------------------------------------------------- | ------------------------------------------------------------------------ |
| `../cosmos3-inference-action/SKILL.md`                  | Entry of the Action row: README, Action Definition, vendored assets      |
| `../cosmos3-inference-action-inverse-dynamics/SKILL.md` | Inverse dynamics (video to ego-motion trajectory)                        |
| `../cosmos3-inference-action-policy/SKILL.md`           | Policy (observations plus action chunk from an instruction)              |
| `../cosmos3-setup/SKILL.md`                             | Framework checkout and venv for Variant 1 and the LeRobot readers        |
| `../cosmos3-env-troubleshoot/SKILL.md`                  | CUDA / import / server failures                                          |
| `../cosmos3-generate-media/SKILL.md`                    | Audiovisual generation on the same backends; measured server notes       |
| `../cosmos3-inference-quantization/SKILL.md`            | Making your own FP8 checkpoint for the FP8 routes of Variants 2, 4 and 5 |
