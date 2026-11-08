---
name: cosmos3-inference-action
description: >
  Entry skill of the Cosmos3 Action row: the action definition and embodiment table, the guardrail rule, the sample
  assets (AV, camera-pose, UMI, Bridge, Fractal, RoboMIND Franka, hand action) and the Action cookbook README's
  per-backend quickstarts, routing each request to a block skill: forward dynamics (start image + actions -> future
  video), inverse dynamics (video -> ego-motion) or policy (observation + instruction -> action chunk) on
  cosmos_framework, diffusers, TensorRT-LLM, vLLM-Omni, SGLang or TensorRT-Edge-LLM. Use when the user asks "run
  Cosmos3 action inference", "which action mode or backend do I need", "where are the Cosmos3 AV / DROID / UMI
  assets", "what is the Cosmos3 action representation", "predict future frames from a trajectory", "recover the
  ego-motion from a video" or "run the Cosmos3 DROID policy", before handing off. Not for audiovisual generation
  without actions (cosmos3-generate-media), post-training (finetune cookbook) or quantizing checkpoints
  (cosmos3-inference-quantization).
---

# Cosmos3 Action Inference: Entry

The Action cookbook (`.agents/skills/cookbooks/cosmos3/generator/action/`, cosmos-framework `main` bbd4a67) became four
skills. This entry holds the cookbook README, migrated verbatim into
[`references/action-readme.md`](references/action-readme.md), and every vendored sample asset under `assets/`; the 14
notebooks and the policy-server guide live, migrated 1:1, in the three block skills this file routes to. README:
"Cosmos3-Nano action-generation supports three distinct tasks:

- **Forward dynamics (`fd`)** — predict future observations from a start image plus an action trajectory.
- **Inverse dynamics (`id`)** — predict ego-motion trajectories from input AV or camera videos using the Cosmos3-Nano.
- **Policy (`policy`)** — predict future observations and action trajectories from a start image, task instruction, and state."

## Read this first

Before the first command of any block, including a single AV smoke run:

1. Pick the block from the routing table below, read its `SKILL.md`, then the variant's reference file in full. The
   reference files are the notebooks migrated 1:1 and replace them: do not look for the original notebooks (the cookbook
   directory keeps only its `assets` and `finetune` subdirectories).
2. Read [`references/action-readme.md`](references/action-readme.md) "Overview", "Action Definition" and the backend's
   "Quickstart" (quoted below). It is the README verbatim; its relative links were re-pointed at the files that moved.
3. Check the GPU with `nvidia-smi`: the driver's CUDA major version picks the torch backend / dependency group (setup
   guide [`../cookbooks/cosmos3/README.md`](../cookbooks/cosmos3/README.md), "CUDA driver and the `cuXXX` backend":
   13.x -> `cu130`, 12.x -> `cu128`); every notebook of the row runs on one GPU (the blocks' checkpoint tables).
4. Assets. Skill note (not in the source): the notebooks resolved `assets/` next to themselves, i.e. under the cookbook
   directory, whose media (`*.mp4`, `*.png`, `*.jpg`) are git-LFS objects (pointer files in a checkout made without
   git-lfs). This skill vendors the small ones as plain files under `assets/` (same layout; "Bundled sample assets"
   below says which) and the blocks reference them under `../cosmos3-inference-action/assets/`; the examples too large
   to vendor stay in `../cookbooks/cosmos3/generator/action/assets/` and need the materialisation step below.
5. Guardrail (README "Overview"): "Generator requires the Guardrail. Request access to the gated
   [nvidia/Cosmos-1.0-Guardrail](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail) HF repository before running these
   examples. To disable the guardrail, set `enable_safety_checker=False` (Diffusers), `use_guardrails: false`
   (TensorRT-LLM `extra_params`), `guardrails: false` (vLLM-Omni `extra_params`/`extra_args`), or `--no-guardrails`
   (Cosmos Framework)." Each block quotes the switch its notebook uses.
6. Setup (README "Overview"): "Environment setup for all backends is centralized in the shared Cosmos3 cookbooks
   environment setup guide; each backend below links to the section you need." The guide is
   [`../cookbooks/cosmos3/README.md`](../cookbooks/cosmos3/README.md); the framework checkout and uv 0.11.29 are
   `../cosmos3-setup/SKILL.md`.
7. Verify what you produced before reporting it ("Verify the output" below and in the block) and record the facts in
   `result_summary.md`. After a failure, go back to the reference section that covers the failing step before
   improvising a fix.

## When to use this skill

- Use to decide which action task (forward dynamics, inverse dynamics, policy) and which serving path (in-process
  cosmos_framework or diffusers; TensorRT-LLM, vLLM-Omni or SGLang servers; TensorRT-Edge-LLM on-device) a request
  needs, then hand off to that block
- Use for the row's shared facts: the action representation and embodiment table, the AgiBotWorld robot description,
  the guardrail rule, the sample assets and where they live, the README quickstarts
- Forward dynamics: `../cosmos3-inference-action-forward-dynamics/SKILL.md`; inverse dynamics:
  `../cosmos3-inference-action-inverse-dynamics/SKILL.md`; policy: `../cosmos3-inference-action-policy/SKILL.md`
- The Cosmos3 NIM serves all three modes through `POST /v1/infer`: the NIM cookbook page
  [`../cookbooks/cosmos3/nim/action.md`](../cookbooks/cosmos3/nim/action.md) and its skill
  `../cosmos3-inference-nim/SKILL.md` (not part of this row)
- Not for audiovisual generation without actions (`../cosmos3-generate-media/SKILL.md`), environment failures
  (`../cosmos3-env-troubleshoot/SKILL.md`), post-training (README "Post-Train" pointer:
  `../cookbooks/cosmos3/generator/action/finetune/README.md`) or quantizing a checkpoint (the
  `cosmos3-inference-quantization` skill, see "FP8 checkpoints")

## Routing: task and backend -> block skill

| Task (README)    | Block skill                                                                | Variants (one per retired notebook)                                                                                                        | Eval cases (the block's eval config)                                                                                       |
| ---------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Forward dynamics | `../cosmos3-inference-action-forward-dynamics/SKILL.md`                    | 1 cosmos_framework, 2 diffusers, 3 TensorRT-LLM, 4 vLLM-Omni, 5 SGLang                                                                     | `framework-nano-fd-av`, `diffusers-nano-fd-av`, `trtllm-nano-fd-av`, `vllm-omni-nano-fd-av`, `sglang-nano-fd-av`           |
| Inverse dynamics | `../cosmos3-inference-action-inverse-dynamics/SKILL.md`                    | 1 diffusers, 2 cosmos_framework, 3 vLLM-Omni, 4 SGLang, 5 TensorRT-LLM                                                                     | `diffusers-nano-id-av`, `framework-nano-id-av`, `vllm-omni-nano-id-av`, `sglang-nano-id-av`, `trtllm-nano-id-av`           |
| Policy (DROID)   | `../cosmos3-inference-action-policy/SKILL.md`                              | 1 diffusers, 2 vLLM-Omni, 3 SGLang, 4 TensorRT-Edge-LLM, 5 Cosmos Framework policy server + RoboLab client (the `.md` guide; no eval case) | `diffusers-nano-policy-droid`, `vllm-omni-edge-policy-droid`, `sglang-nano-policy-droid`, `trt-edge-llm-edge-policy-droid` |
| Any mode via NIM | [`../cookbooks/cosmos3/nim/action.md`](../cookbooks/cosmos3/nim/action.md) | `POST /v1/infer` with `model_mode` `forward_dynamics` / `policy` / `inverse_dynamics`; the DROID specialists "serve policy only"           | none in this row                                                                                                           |

Backend by task, as the sources cover it (FD = forward dynamics, ID = inverse dynamics; the number is the block's variant):

| Backend                       | FD   | ID   | Policy                                       |
| ----------------------------- | ---- | ---- | -------------------------------------------- |
| cosmos_framework (`torchrun`) | FD 1 | ID 2 | policy 5 (server guide)                      |
| diffusers (in-process)        | FD 2 | ID 1 | policy 1                                     |
| TensorRT-LLM (`trtllm-serve`) | FD 3 | ID 5 | - (README: Edge policy is TensorRT-Edge-LLM) |
| vLLM-Omni (Docker)            | FD 4 | ID 3 | policy 2                                     |
| SGLang (Docker)               | FD 5 | ID 4 | policy 3                                     |
| TensorRT-Edge-LLM             | -    | -    | policy 4                                     |

Skill note (not in the source): this entry has no eval cases of its own; its sources are the README and the assets, and
the blocks' cases exercise every notebook. Each block's `SKILL.md` tells the agent to read this README copy first.

## Action definition (README, verbatim)

"Cosmos3 treats action as a modality whose tokens represent transitions between consecutive visual states. This cookbook
follows the unified action interface defined in the paper: ego and end-effector poses are represented as 9D pose deltas,
consisting of 3D translation and a 6D continuous rotation representation. Grasp state encodes the current manipulation
state, using a 1D open-close value for grippers or a 15D human-hand representation with 3D state for each of 5 fingers."

| Embodiment                                | Representation                                                                                      | Dimensionality | Unit  | Post-processing                                     | Generation duration |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------- | ----- | --------------------------------------------------- | ------------------- |
| Autonomous vehicle                        | Ego pose (9D)                                                                                       | 9D             | Meter | Normalization                                       | 60 frames @ 10FPS   |
| [DROID](https://arxiv.org/abs/2403.12945) | End-effector pose (9D) + gripper grasp state (1D)                                                   | 10D            | Meter | Multiview concatenation, `to-OpenCV`, normalization | 16 frames @ 15FPS   |
| UMI                                       | End-effector pose (9D) + gripper grasp state (1D)                                                   | 10D            | Meter | Normalization                                       | 16 frames @ 20FPS   |
| Human hand pose                           | Ego pose (9D) + right wrist (9D) + right fingertips (15D) + left wrist (9D) + left fingertips (15D) | 57D            | Meter | Wrist-frame alignment, normalization                | 16 frames @ 15FPS   |

"Action dimensions are tied to the selected embodiment. `domain_name` chooses a registered action domain, and Cosmos
Framework validates forward-dynamics input width against that domain's canonical raw action dimension. Changing a 10D
DROID vector to 12D or 18D, for example, is not supported by the released checkpoints simply by changing the input shape.
A custom action layout requires a compatible registered embodiment/domain and model weights trained for that action
representation. Some dataset-defined domains, such as hand pose and LIBERO, can resolve their width from dataset
configuration for forward dynamics; this does not make arbitrary widths interchangeable within a trained domain. See the
[Cosmos Framework action-domain registry](https://github.com/NVIDIA/cosmos-framework/blob/main/cosmos_framework/data/generator/action/utils/domain_utils.py)
for the currently registered canonical widths." Samples "can be inspected interactively in the
[Cosmos3 Action Viewer](https://huggingface.co/spaces/nvidia/Cosmos3-Action-Viewer) Hugging Face Space."

AgiBotWorld robot description (README): "The public Cosmos Framework AgiBotWorld forward-kinematics path currently uses
the committed [`G1_omnipicker_calibrated.urdf`](https://github.com/NVIDIA/cosmos-framework/blob/main/cosmos_framework/data/generator/action/robot_assets/G1_omnipicker_calibrated.urdf)
as its calibrated kinematics reference. The URDF contains `package://` references to visual meshes, but those mesh files
are not bundled in the framework's `robot_assets` directory. They are not required by the action FK path: the loader
removes `<visual>` and `<collision>` elements before constructing its MuJoCo kinematics model. Accordingly, the committed
file should be treated as the kinematics reference for the registered omnipicker-based AgiBotWorld action path, not as a
complete renderable robot asset package or a generic description for every AgiBotWorld gripper variant. The public
framework does not currently include a separate CTEK URDF or the referenced mesh package."

## FP8 checkpoints

README (first paragraph): "For Edge/Nano/Super static FP8 checkpoints, see the quantization cookbook and the
**Quantized Checkpoints** section in the vLLM-Omni notebooks." The row's FP8 routes are the notebooks' own sections,
surfaced as the FP8 line of these block variants (read the variant's reference section before using one):

| Block / variant                                       | Notebook section              | Route, as the notebook states it                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Forward dynamics 2, inverse dynamics 1 (diffusers)    | "Quantized Checkpoints (FP8)" | `COSMOS3_DIFFUSERS_PRECISION=fp8` before loading; Nano and Super load `nvidia/Cosmos3-Nano-FP8` and `nvidia/Cosmos3-Super-FP8`; "Do not pass `revision=fp8` on the BF16 cards"; optional `COSMOS3_FP8_MODEL_PATH` for a local Diffusers checkpoint                                                                                                                                           |
| Forward dynamics 4, inverse dynamics 3 (vLLM-Omni)    | "Quantized Checkpoints"       | "Published Edge FP8, Nano FP8 and Super FP8 checkpoints can be downloaded to a local directory ... Serve that directory with the original `--served-model-name` so the request cells keep their existing model IDs. Do not select FP8 with `--revision` alone: the Cosmos3 vLLM-Omni pipeline must receive a materialized local checkpoint."; "The source mount preserves exported symlinks" |
| Forward dynamics 5, inverse dynamics 4 (SGLang)       | "FP8"                         | "To serve the prebuilt checkpoint, pass its Hugging Face ID to `--model-path`" (the notebook's `sglang serve` block with `--model-path nvidia/Cosmos3-Nano-FP8`); "To serve a checkpoint you quantized yourself, pass its local directory to `--model-path`. If the server runs in Docker, mount that directory and use its path inside the container."                                      |
| Policy (all variants), TensorRT-LLM, cosmos_framework | none                          | no FP8 section in these notebooks; the README's pointer above is the row's only FP8 statement for them                                                                                                                                                                                                                                                                                       |

To quantize your own checkpoint: `../cosmos3-inference-quantization/SKILL.md` (the quantization cookbook's own migration;
the path resolves once that row merges — the README copy and the notebooks' FP8 sections link the cookbook README). Skill note
(not in the source): the quantization steps are not copied into the Action skills; one canonical copy, linked.

## Bundled sample assets

The cookbook's `assets/` tree, vendored here as plain files (the skill's `.gitattributes` exempts everything under `assets/` from LFS)
wherever a file is small enough; paths are relative to this skill directory, the blocks use the same paths prefixed with
`../cosmos3-inference-action/`. Skill note (not in the source): the sizes, shapes and dimensions below were read from the
vendored files; "used by" names the retired notebooks (now the blocks' reference files).

| Asset                                     | Path                                                                                                                                                                                                                                 | Facts (skill note)                                                     | Used by                                                                                                                                                                                           |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AV start frames                           | `assets/images/av_0.jpg`, `assets/images/av_1.jpg`                                                                                                                                                                                   | 832x480 JPEG                                                           | every forward-dynamics notebook (`av_0.jpg`); the README quickstarts; `av_1.jpg` is read by no notebook of the row (kept with the tree)                                                           |
| AV ego trajectories                       | `assets/actions/av_traj_forward.json`, `assets/actions/av_traj_left.json`, `assets/actions/av_traj_right.json`                                                                                                                       | 60 rows x 9 values each                                                | every forward-dynamics notebook (TensorRT-LLM: the forward trajectory only); the README quickstarts                                                                                               |
| AV observation clip                       | `assets/videos/av_0.mp4`                                                                                                                                                                                                             | 1,125,838 B                                                            | every inverse-dynamics notebook                                                                                                                                                                   |
| Camera-pose stills, trajectory, prompts   | `assets/images/lighthouse_720.png`, `assets/images/mountain_720.png`, `assets/images/solar_720.png`; `assets/actions/camera_action.json`; `assets/prompts/lighthouse.txt`, `assets/prompts/mountain.txt`, `assets/prompts/solar.txt` | 720x540 / 720x535 / 720x535 PNG; 60 x 9; one caption each              | forward dynamics with cosmos_framework and diffusers (camera-pose cases)                                                                                                                          |
| UMI start frame and action chunk          | `assets/images/umi.png`, `assets/actions/umi.json`, `assets/videos/umi.mp4`                                                                                                                                                          | 256x256 PNG; 32 rows x 10 values (two 16-action chunks); 25,013 B clip | forward dynamics with diffusers, vLLM-Omni and SGLang (`umi.png`, `umi.json`) and with cosmos_framework (`umi.json`, the autoregressive UMI rollout); `umi.mp4` is read by no notebook of the row |
| UMI LeRobot episode                       | `assets/umi_lerobot_example/` (meta, data and videos subdirectories)                                                                                                                                                                 | video 260,846 B                                                        | forward dynamics with cosmos_framework (UMI); inverse dynamics with cosmos_framework and diffusers                                                                                                |
| Bridge, Fractal, RoboMIND Franka episodes | `assets/bridge_lerobot_example/`, `assets/fractal_lerobot_example/`, `assets/robomind_lerobot_example/franka/` (meta, data and videos subdirectories)                                                                                | videos 651,159 B; 184,777 B; 273,435 / 269,874 / 388,411 B             | inverse dynamics with cosmos_framework and diffusers                                                                                                                                              |
| Egocentric hand action example            | `assets/egocentric_hand_action_example/` (cameras, captions and human_annotation JSON; `assets/egocentric_hand_action_example/videos/ESCALE_000374.mp4`)                                                                             | video 977,966 B                                                        | read by no notebook of the row (kept with the tree)                                                                                                                                               |

Not vendored (too large; they stay as git-LFS objects in `../cookbooks/cosmos3/generator/action/assets/`, whose meta
and data files are plain blobs). Skill note (not in the source): sizes are the LFS pointers' `size` fields.

| Example                                                                              | LFS media                                                             | Used by                                                                                                                                                     |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `../cookbooks/cosmos3/generator/action/assets/droid_lerobot_example/`                | three camera clips: 5,337,025 / 4,694,851 / 3,982,548 B               | DROID forward dynamics (cosmos_framework, diffusers, vLLM-Omni, SGLang); every policy notebook; the NIM `../cookbooks/cosmos3/nim/examples/policy_droid.py` |
| `../cookbooks/cosmos3/generator/action/assets/agibotworld_beta_lerobot_example/`     | three clips: 7,090,125 / 7,626,187 / 11,497,483 B                     | inverse dynamics with cosmos_framework and diffusers                                                                                                        |
| `../cookbooks/cosmos3/generator/action/assets/robomind_lerobot_example/franka_dual/` | four clips: 6,953,206 / 9,263,950 / 24,741,792 / 35,905,053 B         | inverse dynamics with cosmos_framework and diffusers                                                                                                        |
| `../cookbooks/cosmos3/generator/action/assets/robomind_lerobot_example/ur/`          | one clip: 1,888,350 B                                                 | inverse dynamics with cosmos_framework and diffusers                                                                                                        |
| `../cookbooks/cosmos3/generator/action/assets/human_hand_pose_lerobot_example/`      | one clip: 3,839,215 B (plus a 2,221,102 B data parquet, a plain blob) | hand-pose forward dynamics (cosmos_framework, diffusers, vLLM-Omni)                                                                                         |
| `../cookbooks/cosmos3/generator/action/assets/videos/av_1.mp4`                       | 1,639,407 B                                                           | the second AV clip of inverse dynamics (cosmos_framework, diffusers, vLLM-Omni, SGLang)                                                                     |
| `../cookbooks/cosmos3/generator/action/assets/videos/robolab_example_rollout.mp4`    | 864,271 B                                                             | referenced by no source in this checkout (an output example)                                                                                                |

Skill note (not in the source): materialising a non-vendored example. Either run
`git lfs install && git lfs pull --include=".agents/skills/cookbooks/cosmos3/generator/action/assets/<example>/**"` in a
framework checkout with git-lfs installed, or download each LFS object from the public repository at the pinned commit,
`https://media.githubusercontent.com/media/NVIDIA/cosmos-framework/bbd4a67c3cf54cc6699464ad1259f3a0f6537da9/.agents/skills/cookbooks/cosmos3/generator/action/assets/<path>`
(the pointer file names the object's `oid sha256` and `size`; check both; verified 2026-11-07 at this commit: `videos/av_1.mp4`
fetched from that URL answered HTTP 200 with 1,639,407 bytes and the pointer's sha256), into the same relative path of the
cookbook's `assets/` directory so the example's meta and data files sit beside it; then point the notebook's `resolve_input(...)`,
`asset_path(...)` or dataset `root` at that directory as the block documents (the policy block carries the DROID download
as one shell block; the forward-dynamics diffusers and vLLM-Omni DROID cells accept `COSMOS3_DROID_ROOT`; the policy notebooks
do not).

## Checkpoints and GPUs, as the sources state them

| Checkpoint                                                                       | What the sources say                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nvidia/Cosmos3-Nano`                                                            | README: "Cosmos3-Nano action-generation supports three distinct tasks"; "Forward dynamics on Nano looks like" (Cosmos Framework quickstart); "launch the VisualGen server with the one-GPU Nano config" (TensorRT-LLM quickstart); setup guide: "**Cosmos3-Nano** (single GPU)" for vLLM-Omni and TensorRT-LLM, "Nano downloads plus CUDA dependencies can take tens of GiB"; the default of every forward- and inverse-dynamics notebook (blocks)        |
| `nvidia/Cosmos3-Edge`                                                            | setup guide: "Edge is the compact 4B checkpoint", "**Cosmos3-Edge** (single GPU)" (vLLM-Omni); "TensorRT-LLM serves Edge for text-to-image, text-to-video, and image-to-video only: Edge has no audio tower, its action weights are not served by this pipeline"; the vLLM-Omni forward/inverse-dynamics notebooks offer a Nano or an Edge server, the cosmos_framework ones `COSMOS3_CHECKPOINT_PATH=Cosmos3-Edge` (blocks)                              |
| `nvidia/Cosmos3-Super`                                                           | README (Diffusers quickstart): "Cosmos3-Super also ships `action_gen=True`, so the forward and inverse calls work against that checkpoint too."; setup guide: "**Cosmos3-Super** (four GPUs; CFG parallelism with Ulysses, plus parallel VAE)" (TensorRT-LLM), "(all GPUs; add tensor parallelism and layerwise offload)" with `--tensor-parallel-size 4 --enable-layerwise-offload` (vLLM-Omni); no action notebook launches it (blocks)                 |
| `nvidia/Cosmos3-Nano-Policy-DROID`, `nvidia/Cosmos3-Edge-Policy-DROID`           | README: "policy, predicting future observations and action trajectories for DROID robot using Cosmos3-Nano-Policy-DROID and Cosmos3-Edge-Policy-DROID"; "On-device TensorRT path for **Cosmos3-Edge-Policy-DROID**" (TensorRT-Edge-LLM); NIM action page: "Edge-DROID and Nano-DROID are BF16-only specialist policy checkpoints", Edge-DROID "is experimental in this release because it has not been validated"; one GPU in every policy source (block) |
| `nvidia/Cosmos3-Edge-FP8`, `nvidia/Cosmos3-Nano-FP8`, `nvidia/Cosmos3-Super-FP8` | the README's FP8 pointer and the notebooks' FP8 sections ("FP8 checkpoints" above)                                                                                                                                                                                                                                                                                                                                                                        |

Skill note (not in the source): no source of the row states a GPU-memory footprint or a runtime for action inference. The
measured figures in `../cosmos3-generate-media/references/skill-notes.md` § 2 are for audiovisual generation on the same
checkpoints, not action measurements.

## Quickstarts (README, verbatim) and the block each belongs to

The README's quickstart blocks are quoted unchanged; their `action_root` is the cookbook directory. Skill note (not in the
source): in a checkout without git-lfs that directory's media are pointer files, so run them from the repository root with
`action_root = Path(".agents/skills/cosmos3-inference-action")` (this skill, same `assets/` layout), or from this skill
directory with `Path(".")`; the notebooks' own paths are handled per variant in the blocks.

### Run with Cosmos Framework -> forward dynamics 1, inverse dynamics 2, policy 5

Reference: [`references/action-readme.md`](references/action-readme.md) — read "Run with Cosmos Framework" (Quickstart;
Cosmos Framework Walkthrough) before the first command; the migrated notebooks behind it are
`../cosmos3-inference-action-forward-dynamics/references/cosmos-framework.md`,
`../cosmos3-inference-action-inverse-dynamics/references/cosmos-framework.md` and the policy guide
`../cosmos3-inference-action-policy/references/cosmos-framework.md`.

"Set up the environment: Cosmos Framework setup ([`../cookbooks/cosmos3/README.md#cosmos-framework`](../cookbooks/cosmos3/README.md#cosmos-framework)).
Activate the framework venv, then run the native inference entrypoint. Forward dynamics on Nano looks like:"

```bash
torchrun --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  --parallelism-preset=latency \
  -i <forward-dynamics input spec>.json \
  -o /tmp/cosmos3_action_fd \
  --checkpoint-path Cosmos3-Nano \
  --seed 0
```

"The input spec pairs a start image with an action trajectory. The notebooks assemble ready-to-run specs for AV, DROID,
UMI, and human hand-pose examples from the checked-in assets under `assets/`. Outputs are written under the framework
checkout." The specs and the single-process run form are in `../cosmos3-inference-action-forward-dynamics/references/cosmos-framework.md` and
`../cosmos3-inference-action-inverse-dynamics/references/cosmos-framework.md`.

### Run with Diffusers -> forward dynamics 2, inverse dynamics 1, policy 1

Reference: [`references/action-readme.md`](references/action-readme.md) — read "Run with Diffusers" (Quickstart; Diffusers
Notebook Walkthrough) before the first command; the migrated notebooks are
`../cosmos3-inference-action-forward-dynamics/references/diffusers.md`,
`../cosmos3-inference-action-inverse-dynamics/references/diffusers.md` and `../cosmos3-inference-action-policy/references/diffusers.md`.

"Set up the environment: Diffusers setup ([`../cookbooks/cosmos3/README.md#diffusers`](../cookbooks/cosmos3/README.md#diffusers)).
Action inputs are grouped into a `CosmosActionCondition`. The pipeline derives the frame count from `chunk_size + 1` and
the conditioning canvas from `resolution_tier`, so `height`, `width`, and `num_frames` stay unset:"

```python
import json
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline, CosmosActionCondition
from diffusers.utils import export_to_video, load_image

action_root = Path(".agents/skills/cookbooks/cosmos3/generator/action")
raw_actions = torch.as_tensor(
    json.load(open(action_root / "assets/actions/av_traj_forward.json")), dtype=torch.float32
)

pipe = Cosmos3OmniPipeline.from_pretrained("nvidia/Cosmos3-Nano", torch_dtype=torch.bfloat16)
pipe.to("cuda")

result = pipe(
    prompt="You are an autonomous vehicle planning system.",
    action=CosmosActionCondition(
        mode="forward_dynamics",
        chunk_size=60,
        domain_name="av",
        resolution_tier=480,
        raw_actions=raw_actions,
        image=load_image(str(action_root / "assets/images/av_0.jpg")),
        view_point="ego_view",
    ),
    fps=10,
    num_inference_steps=30,
    guidance_scale=1.0,
    use_system_prompt=False,
    generator=torch.Generator(device="cuda").manual_seed(0),
)
export_to_video(result.video, "/tmp/cosmos3_action_fd.mp4", fps=10, macro_block_size=1)
```

"For inverse dynamics, pass `mode="inverse_dynamics"` with `video=` instead of `image=` and no `raw_actions`; the
predicted trajectory comes back as `result.action`. `mode="policy"` conditions on a first frame like forward dynamics but
takes no `raw_actions`, predicting the action chunk alongside the video. Cosmos3-Super also ships `action_gen=True`, so
the forward and inverse calls work against that checkpoint too." The notebooks "provision a dedicated venv with the
LeRobot readers, pose helpers, and plotting extras" and "write outputs under outputs/notebooks/diffusers/"; "set
`COSMOS3_REPO` to your framework checkout before running them" and "also set `COSMOS3_DROID_ROOT` to a directory named
after the DROID release you have available."

### Run with TensorRT-LLM -> forward dynamics 3, inverse dynamics 5

Reference: [`references/action-readme.md`](references/action-readme.md) — read "Run with TensorRT-LLM" (Quickstart; TensorRT-LLM
notebook walkthrough) before the first command; the migrated notebooks are
`../cosmos3-inference-action-forward-dynamics/references/trt-llm.md` and `../cosmos3-inference-action-inverse-dynamics/references/trt-llm.md`.

"Set up and launch the VisualGen server with the one-GPU Nano config: TensorRT-LLM setup
([`../cookbooks/cosmos3/README.md#tensorrt-llm-generator`](../cookbooks/cosmos3/README.md#tensorrt-llm-generator)).
Action requests upload a conditioning image as multipart `image_reference` or a video as `video_reference`, and put the
mode-specific fields under `extra_params`. Use the same trained AV prompt as the Cosmos Framework reference; the legacy
`view_point` field is ignored by current TensorRT-LLM. Both notebooks' current multipart fields were exercised against
TensorRT-LLM source revision `cc63e66daf52644f1441e117bb7ba67d20566016`. Follow the source pin in the shared setup."

```python
import json
from pathlib import Path

import requests
from safetensors.torch import load as load_safetensors

action_root = Path(".agents/skills/cookbooks/cosmos3/generator/action")
image_path = action_root / "assets/images/av_0.jpg"
actions = json.loads((action_root / "assets/actions/av_traj_forward.json").read_text())
prompt = "You are an autonomous vehicle planning system."

with image_path.open("rb") as image_file:
    response = requests.post(
        "http://localhost:8000/v1/videos/sync",
        data={
            "prompt": prompt,
            "format": "safetensors",
            "response_format": "file",
            "seed": "0",
            "extra_params": json.dumps(
                {
                    "action_mode": "forward_dynamics",
                    "domain_name": "av",
                    "action": actions,
                    "use_guardrails": True,
                }
            ),
        },
        files={"image_reference": (image_path.name, image_file, "image/jpeg")},
        headers={"Accept": "application/octet-stream"},
    )
response.raise_for_status()
payload = load_safetensors(response.content)
print(payload["video"].shape, payload["action"].shape, payload["frame_rate"].item())
```

"The `av` domain preset supplies its 60-step, 9D, 480p/10 fps recipe; other recognized domains similarly fill omitted
action width, chunk, resolution, and frame rate. Inverse dynamics predicts its action tensor, while forward dynamics
carries the conditioned action alongside the rollout. Both supported modes use a tensor output: `format=auto` selects
`safetensors`, and explicit `mp4`/`avi` is rejected rather than dropping the trajectory. The quickstart uses the blocking
`POST /v1/videos/sync` route with `response_format=file`, which returns the tensor payload bytes directly. The older
`/v1/videos/generations` spelling is a deprecated alias. The asynchronous `POST /v1/videos` route returns HTTP 202; poll
`GET /v1/videos/{id}` and download the same tensor payload from `GET /v1/videos/{id}/content`."

### Run with vLLM-Omni -> forward dynamics 4, inverse dynamics 3, policy 2

Reference: [`references/action-readme.md`](references/action-readme.md) — read "Run with vLLM-Omni" (Quickstart; VLLM-Omni
Notebook Walkthrough) before the first command; the migrated notebooks are
`../cosmos3-inference-action-forward-dynamics/references/vllm-omni.md`,
`../cosmos3-inference-action-inverse-dynamics/references/vllm-omni.md` and `../cosmos3-inference-action-policy/references/vllm-omni.md`.

"Set up the environment and start the server: vLLM-Omni setup
([`../cookbooks/cosmos3/README.md#vllm-omni`](../cookbooks/cosmos3/README.md#vllm-omni)) (Docker recommended). From the
cookbook root (`.agents/skills`), set `export COSMOS3_WORKDIR=$PWD` and `export COSMOS3_HOST_PORT=8001`, then run the
Docker command from the env setup guide. Wait until this succeeds:"

```bash
curl http://localhost:8001/v1/models
```

"Forward-dynamics requests are multipart `POST`s to `/v1/videos` — a start image under `files={"input_reference": ...}`
plus an `extra_params` payload carrying the action trajectory. The vLLM notebooks use these diffusion defaults for action
generation": `num_inference_steps` `30`, `guidance_scale` `1.0`, `flow_shift` `10.0`. "The notebooks build the full request
body for AV, DROID, UMI and human hand-pose examples, including autoregressive chunked generation for the robotics
examples. Policy inference uses async `POST /v1/videos` to retrieve a rollout video plus top-level `action` metadata." The
notebooks "write outputs under outputs/cosmos3_action_vllm/".

### Run with TensorRT-Edge-LLM -> policy 4

Reference: [`references/action-readme.md`](references/action-readme.md) — read "Run with TensorRT-Edge-LLM" (Quickstart; Notebook
walkthrough) before the first command; the migrated notebook is `../cosmos3-inference-action-policy/references/trt-edge-llm.md`.

"On-device TensorRT path for **Cosmos3-Edge-Policy-DROID**. Set up the shared TensorRT-Edge-LLM
([`../cookbooks/cosmos3/README.md#tensorrt-edge-llm`](../cookbooks/cosmos3/README.md#tensorrt-edge-llm)) environment
(upstream install, C++ runtime with `-DBUILD_EXPERIMENTAL_MODELS=ON` and `-DENABLE_CUTE_DSL=ALL`). Then follow
`run_policy_with_trt_edge_llm.ipynb`: `tensorrt-edgellm-build` → `cosmos3_policy_inference`." That notebook is
`../cosmos3-inference-action-policy/references/trt-edge-llm.md`. "This is separate from workstation vLLM-Omni policy
serving." It "builds the same 640×540 DROID `concat_view` as the vLLM-Omni / SGLang policy notebooks (from [the cookbook's]
`../cookbooks/cosmos3/generator/action/assets/droid_lerobot_example`), runs `cosmos3_policy_inference` with `--viewPoint concat_view` and an 8-value `--state`,
and writes `action.json` (`[batch, chunk, action_dimension]`). Optional `--steps` / `--seed` match the upstream VLA guide."

### Post-Train for Cosmos3-Nano-Policy-DROID (pointer, not part of this row)

Reference: [`references/action-readme.md`](references/action-readme.md) "Post-Train for Cosmos3-Nano-Policy-DROID"; the cookbook it
points at is kept in place.

"To reproduce our post-training recipe for Cosmos3-Nano-Policy-DROID, use the Cosmos3-Nano-Policy-DROID SFT cookbook"
([`../cookbooks/cosmos3/generator/action/finetune/README.md`](../cookbooks/cosmos3/generator/action/finetune/README.md),
kept in place). "The same action-policy SFT cookbook also covers **LIBERO-10** (`launch_sft_action_policy_libero_10_nano.sh`)".
Skill note (not in the source): the framework's own docs page for the policy server is
`../../../docs/action_policy_droid_server.md`.

## Verify the output

What the sources state, to quote in `result_summary.md`: the "Action Definition" table's generation duration per
embodiment (autonomous vehicle "60 frames @ 10FPS", DROID "16 frames @ 15FPS", UMI "16 frames @ 20FPS", human hand pose
"16 frames @ 15FPS"); TensorRT-LLM: "Both notebooks decode TensorRT-LLM's `safetensors` response, write the rollout video,
and inspect the named `action` tensor. This response contract is not the top-level action JSON used by vLLM-Omni.";
vLLM-Omni: "Policy inference uses async `POST /v1/videos` to retrieve a rollout video plus top-level `action` metadata.";
TensorRT-Edge-LLM: "writes `action.json` (`[batch, chunk, action_dimension]`)". The per-variant checks (what each
notebook's viewer cell printed, the expected shapes, the tool one-liners) are the "Verify the output" sections of the three
blocks. Skill note (not in the source): `ffprobe` is on neither a bare host nor in the eval's task container; the video,
image, JSON and safetensors check commands are in each block and in `../cosmos3-generate-media/SKILL.md` "Verify the
output". Never report a file you did not inspect.

## Things not obvious from the docs (skill notes, not in the source)

- The README's quickstarts and the notebooks address the cookbook directory (`.agents/skills/cookbooks/cosmos3/generator/action`,
  "From the cookbook root (`.agents/skills`)"); after this migration that directory holds only its `assets` (LFS pointers
  plus the plain-blob meta and data files) and `finetune` subdirectories. The vendored copies are this skill's `assets/`.
- The row has no measured footprints, runtimes or readiness figures of its own; the blocks cite the generate-media
  skill's measurements only as audiovisual figures and label the server-side observations (guardrail gating, token
  pass-through) as unverified for the action routes.
- The NIM serves the same three modes with a different request contract (`POST /v1/infer`, `action_params`); its page
  says of the vLLM-Omni policy notebook: "Do not copy its payload."

## Reference files

The complete cookbook content lives in these files, migrated 1:1 (every markdown and code cell; link targets re-pointed
where a file moved): read the matching one before the first command. Do not look for the notebooks; these files replace them.

| File                                                                          | Content                                                                                                                                                               |
| ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `references/action-readme.md`                                                 | The Action cookbook README: Overview, Action Definition (incl. the AgiBotWorld robot description), per-backend Quickstarts and walkthroughs, Post-Train pointer, TODO |
| `../cosmos3-inference-action-forward-dynamics/references/cosmos-framework.md` | `run_fd_with_cosmos_framework.ipynb` (forward dynamics 1)                                                                                                             |
| `../cosmos3-inference-action-forward-dynamics/references/diffusers.md`        | `run_fd_with_diffusers.ipynb` (forward dynamics 2; "Quantized Checkpoints (FP8)")                                                                                     |
| `../cosmos3-inference-action-forward-dynamics/references/trt-llm.md`          | `run_fd_with_trt_llm.ipynb` (forward dynamics 3)                                                                                                                      |
| `../cosmos3-inference-action-forward-dynamics/references/vllm-omni.md`        | `run_fd_with_vllm_omni.ipynb` (forward dynamics 4; "Quantized Checkpoints")                                                                                           |
| `../cosmos3-inference-action-forward-dynamics/references/sglang.md`           | `run_fd_with_sglang.ipynb` (forward dynamics 5; "FP8")                                                                                                                |
| `../cosmos3-inference-action-inverse-dynamics/references/diffusers.md`        | `run_id_with_diffusers.ipynb` (inverse dynamics 1; "Quantized Checkpoints (FP8)")                                                                                     |
| `../cosmos3-inference-action-inverse-dynamics/references/cosmos-framework.md` | `run_id_with_cosmos_framework.ipynb` (inverse dynamics 2)                                                                                                             |
| `../cosmos3-inference-action-inverse-dynamics/references/vllm-omni.md`        | `run_id_with_vllm_omni.ipynb` (inverse dynamics 3; "Quantized Checkpoints")                                                                                           |
| `../cosmos3-inference-action-inverse-dynamics/references/sglang.md`           | `run_id_with_sglang.ipynb` (inverse dynamics 4; "FP8")                                                                                                                |
| `../cosmos3-inference-action-inverse-dynamics/references/trt-llm.md`          | `run_id_with_trt_llm.ipynb` (inverse dynamics 5)                                                                                                                      |
| `../cosmos3-inference-action-policy/references/diffusers.md`                  | `run_policy_with_diffusers.ipynb` (policy 1)                                                                                                                          |
| `../cosmos3-inference-action-policy/references/vllm-omni.md`                  | `run_policy_with_vllm_omni.ipynb` (policy 2)                                                                                                                          |
| `../cosmos3-inference-action-policy/references/sglang.md`                     | `run_policy_with_sglang.ipynb` (policy 3)                                                                                                                             |
| `../cosmos3-inference-action-policy/references/trt-edge-llm.md`               | `run_policy_with_trt_edge_llm.ipynb` (policy 4)                                                                                                                       |
| `../cosmos3-inference-action-policy/references/cosmos-framework.md`           | `run_policy_with_cosmos_framework.md`, the policy server + RoboLab guide (policy 5)                                                                                   |
| `../cookbooks/cosmos3/README.md`                                              | The shared environment setup guide (kept in place)                                                                                                                    |
| `../cookbooks/cosmos3/nim/action.md`                                          | The NIM Action page (`POST /v1/infer`; kept in place, not part of this row)                                                                                           |
| `../cookbooks/cosmos3/generator/action/finetune/README.md`                    | Post-training cookbook (kept in place, not part of this row)                                                                                                          |

Origin: `.agents/skills/cookbooks/cosmos3/generator/action/` of the public cosmos-framework repository (`main` bbd4a67):
`README.md` -> this skill; the 14 notebooks and `run_policy_with_cosmos_framework.md` -> the three blocks; `assets/` ->
vendored here (the cookbook keeps its `assets` directory as LFS and its `finetune` directory). The migration record is
`../../../docs/cookbook_to_skill_migration.md`.

## Related skills

| Skill                                                   | When to use                                                                                    |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `../cosmos3-inference-action-forward-dynamics/SKILL.md` | Forward dynamics: future video from a start image plus an action trajectory                    |
| `../cosmos3-inference-action-inverse-dynamics/SKILL.md` | Inverse dynamics: ego-motion / end-effector trajectory from a video                            |
| `../cosmos3-inference-action-policy/SKILL.md`           | Policy: action chunk (and rollout) for DROID from one observation and an instruction           |
| `../cosmos3-setup/SKILL.md`                             | Framework checkout and venv (`COSMOS3_REPO`, uv 0.11.29) for the cosmos_framework variants     |
| `../cosmos3-inference/SKILL.md`                         | Framework-native inference reference (`cosmos_framework.scripts.inference` flags)              |
| `../cosmos3-env-troubleshoot/SKILL.md`                  | CUDA / import / server / container failures                                                    |
| `../cosmos3-generate-media/SKILL.md`                    | Audiovisual generation on the same backends; headless-server pattern and measured server notes |
| `../cosmos3-inference-nim/SKILL.md`                     | Operating the Cosmos3 NIM, which serves the action modes through `POST /v1/infer`              |
| `../cosmos3-inference-quantization/SKILL.md`            | Quantizing your own FP8 checkpoint (the released FP8 routes: "FP8 checkpoints" above)          |
| `../cosmos3-skill-authoring/SKILL.md`                   | Editing the blocks' eval cases and running the eval                                            |
