---
name: cosmos3-inference-simulation-bimanual
description: >
  Generate action-conditioned video with Cosmos3 Nano Sim Bimanual in Cosmos Framework. Download the matched release,
  prepare RGB and action inputs, run inference, inspect outputs, or use an exported student from the user's Stage 3
  training. Also serves nvidia/Cosmos3-Nano-Sim-Bimanual through a vLLM-Omni or SGLang server and covers held-out PSNR
  evaluation. Use when the user asks "run Sim-Bimanual inference", "generate a bimanual robot video from an action
  trajectory", "run the AgiBot or mixed-action example", "serve Cosmos3-Nano-Sim-Bimanual with vLLM-Omni or SGLang",
  "run inference with my Sim-Bimanual Stage 3 student" or "score a rollout against held-out video". Not for training or
  distilling the model (use the Sim-Bimanual post-training skill), for choosing robot actions (the model predicts video
  only) or for the Auto and Sim-Transfer Simulation models.
---

# Run inference with Cosmos3-Nano-Sim-Bimanual

Generate a visual future from a task description, an initial RGB observation
and a time-aligned bimanual action trajectory. The model predicts video; it
does not choose robot actions. Rollout length, frame rate and saved geometry
follow the selected input.

## When to use this skill

- Use to run the released Cosmos3-Nano-Sim-Bimanual checkpoint on Cosmos Framework, a vLLM-Omni server or an SGLang server, to run your exported Stage 3 student on Cosmos Framework, and for held-out PSNR evaluation.
- Not for training or distilling the model (see "Adapt the model"), for choosing robot actions (the model predicts video only) or for the Auto and Sim-Transfer models (see "Related skills").

## Paths

This skill's folder replaces the Bimanual folder of the Simulation cookbook (`BIMANUAL_ROOT` in the SGLang reference):
its guide and two notebooks are now `references/bimanual-guide.md`, `references/vllm-omni.md` and
`references/sglang.md`, and its helper is `scripts/bimanual_helpers.py`. No request files are bundled; every variant
reads the published examples from Hugging Face. No file here reads `COSMOS_COOKBOOK_ROOT`: the SGLang reference's setup
cell finds this folder by walking up from the working directory (this folder, `.agents/skills` and the repository root
all work), and `SIM_BIMANUAL_SKILL` in `references/unified-v1-action-schema.md` is this folder. Outputs: the Cosmos
Framework steps write under `$PWD`, the vLLM-Omni reference under `$PWD/outputs/vllm_omni/bimanual` (override:
`COSMOS3_OUTPUT_DIR`), the SGLang reference under an outputs folder inside this folder, sub-folders sglang then bimanual
(override: `BIMANUAL_OUTPUT_DIR`).

## Choose the task

- For generation with the released model, follow the steps below.
- For your own completed training chain, use
  [your exported Stage 3 student](#use-your-own-stage-3-checkpoint).
- For requested held-out PSNR evaluation, use the
  [evaluation protocol](references/evaluation.md). Generation alone does not start scoring.

This workflow follows the [Bimanual inference tutorial](../../../docs/inference_bimanual.md).
Use the current Framework `main` branch with the current default revision of
`nvidia/Cosmos3-Nano-Sim-Bimanual`.

## Choose a variant

| Variant          | Model                                                               | Backend                                                       | GPUs, as the sources state them                                                             | Worked example                                                     | Reference file                                            |
| ---------------- | ------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------- |
| Cosmos Framework | `nvidia/Cosmos3-Nano-Sim-Bimanual`, downloaded to a local directory | Native PyTorch inference: `torchrun`, one process             | No GPU class or memory stated: "The one-process example below is not a GPU-memory minimum." | AgiBot 001 from the unified manifest; the 60-second AgiBot rollout | Steps 1 to 6 of this file; `references/bimanual-guide.md` |
| vLLM-Omni        | The same checkpoint, served from a local snapshot                   | `vllm/vllm-omni:cosmos3` Docker image, synchronous Videos API | "Linux with one NVIDIA GPU"; the launch passes `--gpus '"device=0"'`                        | One unified-action and one mixed-domain example                    | `references/vllm-omni.md`                                 |
| SGLang           | The same checkpoint, downloaded by the server                       | `lmsysorg/sglang:dev` Docker image, asynchronous Videos API   | Not stated; the launch passes `--gpus all`                                                  | The mixed-domain `agibot_mixed_028` example                        | `references/sglang.md`                                    |

## Prerequisites

Framework installation, environment setup and checkpoint download in general: [cosmos3-setup](../cosmos3-setup/SKILL.md).
The model-specific prerequisites follow, with this model's inference-only install; the vLLM-Omni and SGLang sections name
their own server and client requirements.

Use a dedicated Framework checkout with Python 3.13, Git LFS, FFmpeg shared
libraries and a CUDA-compatible GPU allocation. Set `FRAMEWORK_DIR` to that
checkout. If you already use its training environment, reuse it. For an
inference-only environment, follow [setup](../../../docs/setup.md) and run:

```bash
cd "$FRAMEWORK_DIR"
git lfs install --local
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FRAMEWORK_DIR/.uv-cache}"
mkdir -p "$UV_CACHE_DIR"
uvx --from 'uv==0.11.29' uv lock --check
uvx --from 'uv==0.11.29' uv sync --locked --group cu130 --extra guardrail
source .venv/bin/activate
export COSMOS_INTERNAL=0 COSMOS_TRAINING=0
python -m cosmos_framework.scripts.inference --help
```

Use the exact `uv` version shown above: it is the resolver used to generate and
check this Framework revision's lockfile. Do not regenerate `uv.lock` with a
different `uv` version during setup. If the documented `uv lock --check` fails
in a clean checkout at the selected Framework commit, stop and report the checkout
commit and lock-check output. Keep `UV_CACHE_DIR` on a writable workspace or
project filesystem when the home directory has a small quota.

Use the CUDA group matching your driver. Keep the FFmpeg library search path
required by your installation; Framework discovers wheel-provided NPP libraries.
Keep `COSMOS_TRAINING=0` for every inference-only step so imports do not require
training extras. The one-process example below is not a GPU-memory minimum.

Run from the Framework root.
The root lockfile is generated and checked with `uv==0.11.29`. Before setup,
set `UV_CACHE_DIR` to a writable workspace-local directory, then run
`uvx --from 'uv==0.11.29' uv lock --check`; do not regenerate the lock with a
different resolver version. This avoids consuming a quota-limited home cache
without changing the locked environment.

### Hardware and allocation checks

Run CUDA and GPU-count checks inside the target allocation, not only on its
submission host. The inference example launches one process. Check memory for the
selected sample and layout.

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())'
df -h /dev/shm
```

Keep the locked Python packages and the FFmpeg library search path from
Framework setup. Framework discovers wheel-provided NPP libraries.

### Authenticate hosted files without exposing the token

Keep any credentials outside code, manifests and logs;
only remote resources require their corresponding provider access. Gated HF
inference inputs require `HF_TOKEN` in the runtime environment. For Pyxis, pass
the exported variable by name with `--export=ALL,HF_TOKEN` and
`--container-env=HF_TOKEN`; a custom `HF_HOME` cache alone is insufficient for
the direct URL downloader.

Token file, Slurm with Pyxis, in-container check: the [Hugging Face token reference](references/hugging-face-token.md).

## Cosmos Framework

Reference: steps 1 to 5 below and step 6 under "Check the result" are the native workflow; the cookbook guide
([Bimanual guide](references/bimanual-guide.md)) shows the same run on one page.

### 1. Choose the checkpoint path

The default example uses the current schema-5, four-step checkpoint. Keep its
complete configuration and weights together:

```bash
CHECKPOINT_DIR="$PWD/checkpoints/Cosmos3-Nano-Sim-Bimanual"

hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
  --local-dir "$CHECKPOINT_DIR"
```

Authenticate to HF with model read access if required. Inspect the downloaded
metadata before loading weights; require schema 5 and the exact four-step
schedule:

```bash
python - "$CHECKPOINT_DIR/checkpoint.json" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())["cosmos3_nano_sim_bimanual"]
assert metadata["conditioning"]["schema_version"] == 5
assert metadata["conditioning"]["input_contract"]["action_dim"] == 59
assert metadata["fixed_step_sampler_config"]["t_list"] == [1.0, 0.9375, 0.8333333333333334, 0.625]
print("Bimanual checkpoint metadata passed")
PY
```

| Checkpoint                            | Input and prompting                                         | Saved sampling schedule |
| ------------------------------------- | ----------------------------------------------------------- | ----------------------- |
| Current released schema-5 model       | Normalized unified 59D sidecar, per-row domains, plain text | Four-step SDE           |
| Your canonical Stage 3 student export | Matched unified action metadata and training contract       | Four-step SDE           |

An HF export already selects its weight variant. Do not add a DCP EMA-selection
flag to the released-HF command.

### 2. Prepare the data contract and one input row

Each input row supplies:

- One non-empty plain-text prompt with `format_prompt_as_json=false`.
- `model_mode: forward_dynamics` and `action_condition: true`.
- `action_space: normalized_unified_v1` and `T` finite 59D action rows.
- Scalar or per-row domain routing supported by the checkpoint.
- `action_chunk_size: T`, `num_frames: T+1` and the action-aligned `fps`.

The published examples default to 900 actions and 901 frames at 30 FPS. This
is an example length, not a model limit; choose another aligned `T` for your
own trajectory.

The focused mixed AgiBot examples route 300 `agibotworld` rows, 300
`camera_pose` rows, then 300 `agibotworld` rows. The checkpoint maps these
domains to IDs 15, 2 and 15. Prepared values are already normalized and masked;
inference only pads five zero channels to reach 64. Do not apply training
normalization again. Read the
[action/data contract](references/input-contract.md)
and [unified schema](references/unified-v1-action-schema.md)
before preparing your own sidecar.

#### Hugging Face-hosted examples

Download the published manifests and frustum configuration from the model
repository's default branch:

```bash
HF_EXAMPLE_DIR="${HF_EXAMPLE_DIR:-$PWD/hf-examples/Cosmos3-Nano-Sim-Bimanual}"
hf download nvidia/Cosmos3-Nano-Sim-Bimanual \
  examples/inference/sim_bimanual_unifiedv1_hf.jsonl \
  examples/inference/sim_bimanual_mixed_action_hf.jsonl \
  examples/inference/sim_bimanual_agibot_long60_hf.jsonl \
  examples/inference/bimanual_frustum.json \
  --local-dir "$HF_EXAMPLE_DIR"
```

Keep `HF_TOKEN` in the runtime environment when the gated model requires it.
The published manifests point directly to their Hugging Face images, action
sidecars and optional frustum files. Framework materializes those URLs with
`HF_TOKEN` through its standard download hook. Do not rewrite the example URLs
to the checkpoint revision.

Use AgiBot 001 for the worked example. Extract its complete row without
changing the hosted prompt, action routing or input URLs:

```bash
python - "$HF_EXAMPLE_DIR/examples/inference/sim_bimanual_unifiedv1_hf.jsonl" \
  "$HF_EXAMPLE_DIR/agibot_example.jsonl" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
matches = [row for row in rows if row["action_path"].endswith("/agibot_001_normalized_unifiedv1.json")]
if len(matches) != 1:
    raise ValueError("Expected exactly one published AgiBot 001 example")
Path(sys.argv[2]).write_text(json.dumps(matches[0]) + "\n")
PY
```

Additional YAM inputs are available in the
[Hugging Face examples](https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Bimanual/tree/main/examples/inference).
Use the same command with a different `INPUT_MANIFEST`; its complete rows
select the matching image, prompt, actions and domains without source edits.

| Manifest                              | Contents                                                                                      |
| ------------------------------------- | --------------------------------------------------------------------------------------------- |
| `sim_bimanual_unifiedv1_hf.jsonl`     | Unified-action examples, including valid per-row domain routing                               |
| `sim_bimanual_mixed_action_hf.jsonl`  | AgiBot robot/camera/robot routing examples; overlap with the combined manifest is intentional |
| `sim_bimanual_agibot_long60_hf.jsonl` | Focused 60-second AgiBot-scene rollout; also present in the combined manifest                 |

`unifiedv1` names the normalized 59D action schema, not a scalar-routing-only
subset. Do not remove a row merely because its action sidecar uses per-step
robot/camera/robot domain routing. The focused mixed-action manifest exists to
make that behavior easy to select and may repeat rows from the combined
manifest.

Each manifest supplies one plain-text prompt per row with
`format_prompt_as_json=false`; do not add duration or resolution text.

Preserve `T` actions for `T+1` frames and the temporal-compression alignment:
the action count is divisible by four. Check RGB/action timestamps, domains,
units, gripper calibration and pose convention, not only tensor width.
An inference JSONL is not the completed training bundle.

### 3. Review prompting and prompt upsampling

Review the sample's task description against the initial RGB and supplied
actions. For user-authored prompts, describe the objects, starting scene,
arm assignments and task. Use the [prompt guide](references/prompt-guide.md).

The matched schema-5 export selects plain-text prompting automatically.
Prompt upsampling is not part of this recipe. If you use an authorized captioner,
review its output against the actual input and retain the original text.
Keep the selected initial-scene or future-described policy fixed for evaluation.

### 4. Choose important hyperparameter knobs

| Setting                            | Worked value                               | What to keep consistent                                                                |
| ---------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------- |
| Checkpoint revision                | Current default HF revision                | Keep its schema-5 inputs and four-step schedule                                        |
| `--num-steps`                      | `4`                                        | Match the saved schedule; guidance/shift do not replace it                             |
| SDE ladder                         | `[1.0, 0.9375, 0.8333333333333334, 0.625]` | Read it from the selected export rather than reconstructing it                         |
| `action_chunk_size` / `num_frames` | From the input manifest                    | Supply one action per transition and respect compression by four                       |
| `--resolution` / `--image-size`    | `480` / `480`                              | The tier does not upscale arbitrary inputs; validate the selected row's saved geometry |
| `--seed`                           | `2026`                                     | Fix it for comparisons and record the effective value                                  |
| `--parallelism-preset`             | `latency` with one process                 | Recheck memory and outputs if placement changes                                        |
| Compile/cache                      | Compile off, KV cache 64, attention sink 8 | Keep matched when comparing outputs or timing                                          |

### 5. Generate video

From the same activated Framework environment:

```bash
INPUT_MANIFEST="${INPUT_MANIFEST:-$HF_EXAMPLE_DIR/agibot_example.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/sim_bimanual_hf_examples}"
export COSMOS_TRAINING=0
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.scripts.inference \
  -i "$INPUT_MANIFEST" \
  -o "$OUTPUT_DIR" \
  --checkpoint-path "$CHECKPOINT_DIR" \
  --model-mode=FORWARD_DYNAMICS --resolution=480 --image-size=480 \
  --num-steps=4 --seed=2026 \
  --autoregressive=True --kv-cache-inference-size=64 --attention-sink-size=8 \
  --parallelism-preset=latency --max-num-seqs=1 \
  --no-use-torch-compile --no-diffusion-cache --no-guardrails \
  --video-save-quality=4
```

This is the Framework model-validation command; it explicitly disables
guardrails. Use the deployment's guardrail policy for other applications.
Choose a new output directory when changing inputs or settings so existing
results are not mistaken for a new run.

The hosted manifests already contain their matching plain-text captions,
action routing, `action_chunk_size`, `num_frames` and `fps`. Preserve those row
fields and the four-step, KV-64, sink-8
command settings. Select another manifest by changing `INPUT_MANIFEST` and
choosing a new output directory. For your own trajectory, set both length
fields in the manifest; the command does not override them. To run a published
mixed-action example, use its complete row or the focused mixed-action manifest.
For frustum retrieval, copy one manifest row,
add `kv_cache_frustum_path` pointing to
`examples/inference/bimanual_frustum.json`, and use
`--no-use-torch-compile --kv-cache-inference-size=0 --attention-sink-size=0`
for an unbounded physical KV archive.

#### Run the 60-second AgiBot example

Use the published long-trajectory manifest from the same download:

```bash
INPUT_MANIFEST="$HF_EXAMPLE_DIR/examples/inference/sim_bimanual_agibot_long60_hf.jsonl"
OUTPUT_DIR="$PWD/outputs/sim_bimanual_agibot_long60"
```

Rerun the generation command above with these paths. The manifest supplies
the complete initial image, prompt, action trajectory, frame count and FPS;
do not replace its length with the short example's settings. Check the output
against the manifest and inspect early, middle and late portions for drift,
flicker, identity changes and agreement with the actions.

#### Use your own Stage 3 checkpoint

First [export the Stage 3 student](../cosmos3-nano-sim-bimanual/post-training/SKILL.md#export-and-check-inference)
from its saved resolved config, selecting regular or EMA weights explicitly.
Set `CHECKPOINT_DIR` to that complete export and choose a new `OUTPUT_DIR`.

Use an input matching the export's action metadata and training semantics.
For the canonical four-step student, rerun the command above with the new
checkpoint/output paths. Keep its saved four-step SDE ladder rather than
assuming it is identical to the released model's ladder.
The released-example checker is specific to that release's metadata and is
not a general validator for every user export.

## vLLM-Omni

Reference: the [vLLM-Omni reference](references/vllm-omni.md), the cookbook notebook converted 1:1. Read it in full,
then run its sections in order: "1. Prerequisites", "2. Install and start the server", "3. Set up the client",
"4. Download and validate the published examples", "5. Generate the videos", "6. Validate and view".

The cookbook guide ([Bimanual guide](references/bimanual-guide.md), "vLLM-Omni") sums it up: "The
[Bimanual vLLM-Omni notebook](references/vllm-omni.md) shows how to serve the checkpoint through vLLM-Omni's
synchronous Videos API, download the published inputs from Hugging Face, generate unified-action and mixed-domain
rollouts, and validate each output's frame count, frame rate, dimensions, and full decode." "The notebook uses the
`vllm/vllm-omni:cosmos3` Docker image shared by all Simulation notebooks and keeps the checkpoint's four-step schedule."

Requirements (section 1): "**Server host:** Linux with one NVIDIA GPU, Docker with NVIDIA Container Toolkit, and
Hugging Face access to `nvidia/Cosmos3-Nano-Sim-Bimanual`." "**Notebook client:** Python 3.10 or newer, FFmpeg/FFprobe,
and Hugging Face access to the model."

Server (section 2): run the launch block from the reference, unchanged; it downloads the checkpoint snapshot and serves
that directory. "Docker publishes port 8000 on the host loopback interface, matching `http://127.0.0.1:8000`." "This
model-validation command disables guardrails through `"guardrails":false` in the stage `model_config`." Readiness: the
reference has no rule in prose; before the first request its section 5 cell calls `wait_for_model` (defined in
section 3), which polls `/v1/models` until the server answers and requires the model ID in its list.

## SGLang

Reference: the [SGLang reference](references/sglang.md), the cookbook notebook converted 1:1. Read it in full, then run
its sections in order: "1. Model and runtime contract", "2. Start the server", "3. Set up the client" (with "Verify
`/v1/models`"), "4. Download the published example", "5. Generate with `POST /v1/videos`", "6. Verify and view the
output", "7. Next steps".

What it does: "Replay the published `agibot_mixed_028` example of `nvidia/Cosmos3-Nano-Sim-Bimanual` through a
persistent SGLang server: a mixed-domain sequence in which AgiBot manipulation rows alternate with camera-pose rows,
each row tagged with its embodiment." The client imports `scripts/bimanual_helpers.py` (see "Paths").

Server (sections 1 and 2): "The server runs in the published SGLang image, `lmsysorg/sglang:dev`. Pull it on the server
host before each start". "The `dev` tag moves with SGLang's main branch. This notebook was validated against the image
built from SGLang commit
[`ab580b2e`](https://github.com/sgl-project/sglang/commit/ab580b2e060059c0f8387b598a9f648dcec13f93), published as the
fixed tag `lmsysorg/sglang:nightly-dev-20261020-ab580b2e`; use that tag in place of `dev` to reproduce the validated
run." Run the launch block of section 2 from the reference, unchanged.

Readiness (section 2): "The first start downloads the checkpoint into the mounted cache. The HTTP port opens only after
the weights are loaded and the warmup requests finish, so wait for the `Uvicorn running` line." Client (section 3):
"Install FFmpeg with `ffprobe` on the client as well."

## Check the result

Apply the four points of the Simulation checklist,
[Validate a rollout](../cosmos3-inference-simulation/references/simulation-readme.md#validate-a-rollout), then step 6
below. The served references check their own outputs: "6. Validate and view" in the
[vLLM-Omni reference](references/vllm-omni.md), "6. Verify and view the output" in the [SGLang reference](references/sglang.md).

> **Skill note (not in the source):** step 6 calls the `ffprobe` command-line tool, which the "FFmpeg shared libraries" prerequisite does not include. Check with `command -v ffprobe`; if it is missing (`ffprobe: command not found`, exit status 127), install the FFmpeg command-line tools before step 6 (`apt-get install -y ffmpeg` on Debian or Ubuntu).

### 6. Verify outputs and hand off

1. Check the sample's `sample_outputs.json` for success and its generated MP4.
   Every requested row needs a playable output; keep failed or skipped rows
   visible in completeness accounting.
2. Set `OUTPUT_VIDEO` to the generated RGB video reported in that file and
   fully inspect its decoded structure:

```bash
ffprobe -v error -select_streams v:0 -count_frames \
  -show_entries stream=width,height,avg_frame_rate,nb_read_frames:format=duration \
  -of json "$OUTPUT_VIDEO"
```

3. Match the decoded frame count and FPS to that row's `num_frames` and `fps`.
   Validate the saved dimensions against the selected input's declared content geometry.
4. For a mixed-action row, inspect every domain transition at the timestamps
   defined by its action rows and FPS. For every row, inspect the full sequence for frozen
   frames, discontinuities and disagreement with the actions.
5. Preserve the effective prompt, input identity, checkpoint revision/weight
   selection, seed, command and outputs. Use the
   [evaluation guide](references/evaluation.md) for held-out scoring.

The [Framework inference contract](references/inference-contract.md#cosmos-framework)
links the native input checks and current schema rules. Use
[Framework troubleshooting](../../../docs/faq.md) for environment/runtime failures.

## Troubleshooting

Start with the [troubleshooting reference](references/troubleshooting.md): its symptom table leads to the procedures
for runtime and access, input validation, generation and evaluation. For environment problems shared by all models,
use [cosmos3-env-troubleshoot](../cosmos3-env-troubleshoot/SKILL.md). Launch, training-progress and checkpoint symptoms
are not covered here: the opening note of the troubleshooting reference links their procedures, and training itself is
the [Sim-Bimanual post-training skill](../cosmos3-nano-sim-bimanual/post-training/SKILL.md).

## Adapt the model

To train or distill the model on your own data, use the [Sim-Bimanual post-training skill](../cosmos3-nano-sim-bimanual/post-training/SKILL.md).

## Reference files

- [Action and data contract](references/input-contract.md)
- [Unified action schema](references/unified-v1-action-schema.md)
- [Prompt guide](references/prompt-guide.md)
- [Framework inference contract](references/inference-contract.md#cosmos-framework)
- [Held-out evaluation protocol](references/evaluation.md)

| File                                               | When to read it                                                                                                                                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `references/bimanual-guide.md`                     | The cookbook guide, verbatim: supported backends (with the FlashDreams links), the Cosmos Framework run on one page, the vLLM-Omni summary                                                 |
| `references/vllm-omni.md`                          | Before the vLLM-Omni variant: the cookbook notebook, converted 1:1                                                                                                                         |
| `references/sglang.md`                             | Before the SGLang variant: the cookbook notebook, converted 1:1                                                                                                                            |
| `references/inference-contract.md`                 | Before writing your own manifest row: manifest fields, the action sidecar, checkpoint and sampling, placement, output verification                                                         |
| `references/input-contract.md`                     | Before preparing your own action sidecar: action mapping, Domain 2 (`camera_pose`) conversion, timing, the sidecar shape                                                                   |
| `references/unified-v1-action-schema.md`           | For the 59-value action schema: dimensions, pose convention, gripper calibration, normalization, masks, a worked example, the validation command                                           |
| `references/unified-v1-action-schema-receipt.json` | The machine-readable companion of the action schema; the validation command reads it                                                                                                       |
| `references/model-contract.md`                     | For the task definition, the 59-value layout, the four training stages and the latent-frame arithmetic                                                                                     |
| `references/prompt-guide.md`                       | In step 3, when you write or review a plain-text prompt                                                                                                                                    |
| `references/evaluation.md`                         | For held-out PSNR evaluation: held-out data, the scoring command, checkpoint comparison, the evaluation record                                                                             |
| `references/troubleshooting.md`                    | When a run fails: symptom table, then runtime and access, input validation, generation and evaluation procedures                                                                           |
| `references/hugging-face-token.md`                 | When the token is in a file or the run is under Slurm with Pyxis: the complete token procedure                                                                                             |
| `references/quick-start.md`                        | The source's four-way router to generation, post-training, troubleshooting and evaluation                                                                                                  |
| `scripts/bimanual_helpers.py`                      | Imported by the SGLang reference. Its docstring: "CPU-only input preparation and HTTP client for the SGLang Bimanual notebook."                                                            |
| `scripts/validate_output_video.py`                 | Its docstring: "Validate one generated video against its resolved inference request." Named in `references/inference-contract.md`, "Output verification"                                   |
| `scripts/validate_unified_v1.py`                   | Its docstring: "Validate the unified-v1 schema receipt, example masks, and normalization." Run from `references/unified-v1-action-schema.md`, "Validate the schema, masks, and round trip" |

## Related skills

| Skill                                                                                      | When to use                                                                                                       |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| [cosmos3-inference-simulation](../cosmos3-inference-simulation/SKILL.md)                   | Entry skill of the Simulation models: model chooser, shared prerequisites, validation checklist                   |
| [cosmos3-inference-simulation-auto](../cosmos3-inference-simulation-auto/SKILL.md)         | Transfer-Auto and Sim-Auto: multi-camera driving video                                                            |
| [cosmos3-inference-simulation-transfer](../cosmos3-inference-simulation-transfer/SKILL.md) | Sim-Transfer: 480p video from a caption and one control video                                                     |
| [Sim-Bimanual post-training skill](../cosmos3-nano-sim-bimanual/post-training/SKILL.md)    | Training and distillation of this model, and the export of a Stage 3 student                                      |
| [cosmos3-setup](../cosmos3-setup/SKILL.md)                                                 | Framework installation, environment setup, checkpoint download                                                    |
| [cosmos3-inference](../cosmos3-inference/SKILL.md)                                         | Framework inference in general: offline batch generation, parallelism options, input formats, sampling parameters |
| [cosmos3-env-troubleshoot](../cosmos3-env-troubleshoot/SKILL.md)                           | Environment, installation and runtime errors                                                                      |
